"""Glyph SVG -> hanzi-writer JSON.

usage:
  python -m strokegen.predict glyph.svg -o 永.json
  python -m strokegen.predict glyph_dir/ -o out_dir/
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import numpy as np
import torch

from . import features, order
from ._batch import MISSING, stage_a_from_contours, stage_a_from_file, stage_a_from_svg, stage_bc_prep
from .data import collate, to_device
from .decode import assemble_strokes, partition_from_predictions
from .medians import polygon_parts, stroke_geometry, stroke_median
from .model import CutModel, OrderNet
from .order import N_PFEAT, N_SFEAT, decode_order, pair_features, stroke_features
from .paths import CKPT
from .svg_io import load_svg_glyph, segs_to_path

_WORKER_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


@contextmanager
def worker_env():
    """Single-threaded BLAS for worker processes started inside this block (one per core already).

    The variables are only set while workers are being started, so this process and the
    other subprocesses it starts keep their own settings; values the user set are kept.
    """
    old = {k: os.environ.get(k) for k in _WORKER_ENV}
    for k, v in _WORKER_ENV.items():
        os.environ.setdefault(k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class GlyphError(ValueError):
    """A glyph could not be processed (unreadable SVG, degenerate outline, ...)."""


def _empty_result(lab: np.ndarray | None = None, error: str | None = None) -> dict:
    empty = np.zeros(0, np.int64)
    part = empty if lab is None else lab
    seg = empty if lab is None else np.full(len(lab), -1, np.int64)
    out = dict(strokes=[], medians=[], _partition=part, _order=empty, _seg_stroke=seg)
    if error:
        out["_error"] = error
    return out


def _compose(lab, kept, paths, medians, stroke_order, flip) -> dict:
    """hanzi-writer result from the assembled strokes, in writing order and direction."""
    out_s, out_m = [], []
    for k in stroke_order:
        out_s.append(paths[k])
        m = medians[k][::-1] if flip[k] else medians[k]
        out_m.append([[int(x), int(y)] for x, y in m])
    # segment -> index into the returned strokes (-1 = unassigned)
    pos = {kept[int(r)]: p for p, r in enumerate(stroke_order)}
    seg_stroke = np.array([pos.get(int(v), -1) for v in lab], np.int64)
    return dict(strokes=out_s, medians=out_m, _partition=lab, _order=np.asarray(stroke_order),
                _seg_stroke=seg_stroke)


def _cut_batch_size(max_n: int) -> int:
    if max_n <= 96:
        return 64
    if max_n <= 192:
        return 32
    if max_n <= 384:
        return 8
    if max_n <= 768:
        return 2
    return 1


def load_checkpoint(path: str, feature_version: int, train_cmd: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    have = ck.get("args", {}).get("feat_version", 1)
    if have != feature_version:
        raise RuntimeError(
            f"{path} was trained on feature version {have}, but this code computes version {feature_version}. "
            f"Retrain it (python -m {train_cmd}) or use the code it was trained with.")
    return ck


class StrokeGenerator:
    def __init__(self, cut_ckpt: str | None = None, order_ckpt: str | None = None, device: str | None = None):
        cut_ckpt = cut_ckpt or os.path.join(CKPT, "cut.pt")
        order_ckpt = order_ckpt or os.path.join(CKPT, "order.pt")
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = load_checkpoint(cut_ckpt, features.FEATURE_VERSION, "strokegen.train_cut")
        a = ck["args"]
        self.cut = CutModel(d=a.get("d", 256), layers=a.get("layers", 8)).to(self.dev).eval()
        self.cut.load_state_dict(ck["model"])
        ok = load_checkpoint(order_ckpt, order.FEATURE_VERSION, "strokegen.train_order")
        self.order = OrderNet(layers=ok["args"].get("layers", 6)).to(self.dev).eval()
        self.order.load_state_dict(ok["model"])
        self._pool: ProcessPoolExecutor | None = None
        self._pool_workers = 0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._pool_workers = 0

    def _map(self, fn, items: list, workers: int | None) -> list:
        if not items:
            return []
        w = workers if workers is not None else min(os.cpu_count() or 4, 16)
        if w <= 1 or len(items) <= 2:
            return [fn(x) for x in items]
        if self._pool is None or self._pool_workers != w:
            self.close()
            self._pool = ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn"))
            self._pool_workers = w
        chunksize = max(1, len(items) // (w * 4))
        with worker_env():  # workers are started while the tasks are submitted
            return list(self._pool.map(fn, items, chunksize=chunksize))

    # ------------------------------------------------------------ models
    @torch.no_grad()
    def _cut_probs(self, feats: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(cut_p, aff_p) per glyph, batching glyphs of similar size."""
        out: list = [None] * len(feats)
        by_size = sorted(range(len(feats)), key=lambda i: len(feats[i]["feat"]))
        i = 0
        while i < len(by_size):
            sub: list[int] = []  # glyphs sorted by size; the largest one in a batch sets its limit
            while i < len(by_size) and len(sub) < _cut_batch_size(len(feats[by_size[i]]["feat"])):
                sub.append(by_size[i])
                i += 1
            batch = to_device(collate([dict(feats[j], index=b) for b, j in enumerate(sub)]), self.dev)
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
                res = self.cut(batch)
            cut_all = torch.sigmoid(res["cut"].float()).cpu().numpy()
            aff_all = torch.sigmoid(res["aff"].float()).cpu().numpy()
            for b, j in enumerate(sub):
                n = len(feats[j]["feat"])
                out[j] = (cut_all[b, :n].copy(), aff_all[b, :n, :n].copy())
        return out

    @torch.no_grad()
    def _order_logits(self, items: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(dir_logits, prec_logits) per glyph from (stroke_feats, pair_feats)."""
        out: list = [None] * len(items)
        by_size = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        for k in range(0, len(by_size), 128):
            sub = by_size[k:k + 128]
            B, max_s = len(sub), max(len(items[i][0]) for i in sub)
            x = torch.zeros(B, max_s, N_SFEAT, dtype=torch.float32)
            pair = torch.zeros(B, max_s, max_s, N_PFEAT, dtype=torch.float32)
            pad = torch.ones(B, max_s, dtype=torch.bool)
            for b, i in enumerate(sub):
                sfeat, pfeat = items[i]
                s = len(sfeat)
                x[b, :s] = torch.from_numpy(sfeat)
                pair[b, :s, :s] = torch.from_numpy(pfeat)
                pad[b, :s] = False
            res = self.order(x.to(self.dev), pad.to(self.dev), pair.to(self.dev))
            dl, pl = res["dir"].float().cpu().numpy(), res["prec"].float().cpu().numpy()
            for b, i in enumerate(sub):
                s = len(items[i][0])
                out[i] = (dl[b, :s], pl[b, :s, :s])
        return out

    def partition(self, contours) -> np.ndarray:
        """Stage A only: stroke label per outline segment."""
        f = features.glyph_features(contours)
        cut_p, aff_p = self._cut_probs([f])[0]
        return partition_from_predictions(contours, cut_p, aff_p)

    def order_strokes(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Stage C only: (order, flip) for the given stroke records."""
        dl, pl = self._order_logits([(stroke_features(records), pair_features(records))])[0]
        return decode_order(pl), dl > 0

    # ------------------------------------------------------------ prediction
    def predict_contours(self, contours, labels: np.ndarray | None = None) -> dict:
        """hanzi-writer dict for one glyph. Pass `labels` to skip Stage A (e.g. GT partition)."""
        r = self.predict_contours_batch([contours], workers=1, labels=None if labels is None else [labels])[0]
        if "_error" in r:
            raise GlyphError(r["_error"])
        return r

    def _predict_batch(self, stage_a_fn, inputs: list, workers: int | None = None, chunk_size: int = 512,
                       labels: list | None = None) -> list[dict | None]:
        """Results per input: a hanzi-writer dict (with '_error' on failure), or None for missing input."""
        results: list[dict | None] = [None] * len(inputs)
        for start in range(0, len(inputs), chunk_size):
            idx = range(start, min(start + chunk_size, len(inputs)))
            bc_keys, bc_inputs = [], []
            if labels is not None:  # Stage A is given: skip feature extraction and the cut model
                for g in idx:
                    if not inputs[g]:
                        results[g] = _empty_result()
                    else:
                        bc_keys.append(g)
                        bc_inputs.append((inputs[g], None, None, labels[g]))
            else:
                prepped = self._map(stage_a_fn, [inputs[g] for g in idx], workers)
                valid = []
                for g, (contours, feat, err) in zip(idx, prepped):
                    if err == MISSING:
                        results[g] = None
                    elif err:
                        results[g] = _empty_result(error=err)
                    elif not contours or feat is None:
                        results[g] = _empty_result()
                    else:
                        valid.append((g, contours, feat))
                del prepped
                probs = self._cut_probs([f for _, _, f in valid])
                for (g, contours, _), (cut_p, aff_p) in zip(valid, probs):
                    bc_keys.append(g)
                    bc_inputs.append((contours, cut_p, aff_p, None))
                del valid, probs

            bc_out = self._map(stage_bc_prep, bc_inputs, workers)
            del bc_inputs
            pending = []
            for g, (lab, kept, paths, medians, sfeat, pfeat, err) in zip(bc_keys, bc_out):
                if err:
                    results[g] = _empty_result(error=err)
                elif not kept:
                    results[g] = _empty_result(lab)
                else:
                    pending.append((g, lab, kept, paths, medians, sfeat, pfeat))
            del bc_out
            logits = self._order_logits([(t[5], t[6]) for t in pending])
            for (g, lab, kept, paths, medians, _, _), (dl, pl) in zip(pending, logits):
                results[g] = _compose(lab, kept, paths, medians, decode_order(pl), dl > 0)
        return results

    def predict_contours_batch(self, contours_list: list, workers: int | None = None,
                               labels: list | None = None) -> list[dict]:
        """Bulk parallel prediction for a list of parsed glyph contours.

        labels: optional segment labels per glyph (skips Stage A). Failed glyphs get '_error'.
        """
        return [r if r is not None else _empty_result()
                for r in self._predict_batch(stage_a_from_contours, contours_list, workers=workers, labels=labels)]

    def predict_svgs_batch(self, svgs: list[str], workers: int | None = None) -> list[dict]:
        """Bulk parallel prediction for a list of SVG strings. Failed glyphs get '_error'."""
        return [r if r is not None else _empty_result()
                for r in self._predict_batch(stage_a_from_svg, svgs, workers=workers)]

    def predict_files_batch(self, paths: list[str | None], workers: int | None = None) -> list[dict | None]:
        """Bulk parallel prediction for SVG file paths: None for missing files, '_error' for failures."""
        return self._predict_batch(stage_a_from_file, paths, workers=workers)

    def predict_svg(self, svg_text: str) -> dict:
        try:
            contours = load_svg_glyph(svg_text)
        except ValueError as e:
            raise GlyphError(str(e)) from e
        r = self.predict_contours(contours)
        return dict(strokes=r["strokes"], medians=r["medians"])


MERGE_END_TOL = 80.0  # skeleton ends must land this close to the merged strokes' outer ends


def _oriented(med: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """med running from where `first` starts to where `second` ends (or reversed if that fits better)."""
    fwd = np.hypot(*(med[0] - first[0])) + np.hypot(*(med[-1] - second[-1]))
    rev = np.hypot(*(med[-1] - first[0])) + np.hypot(*(med[0] - second[-1]))
    return med if fwd <= rev else med[::-1]


def merge_strokes(contours, seg_stroke, parts, first, second) -> dict:
    """Join strokes into one, the way predict_contours would have built it.

    seg_stroke labels every outline segment with a stroke id; all segments whose id is in
    `parts` become one stroke whose outline is reassembled from the font segments. `first`
    and `second` are the (oriented) medians of the two strokes being merged, in writing order;
    the new median runs from the start of `first` to the end of `second`. When the outline's
    skeleton cannot do that (disconnected pieces, crossing strokes, a jump between the two
    strokes) the median is just `first` followed by `second`.
    """
    lab = np.asarray(seg_stroke, np.int64)
    n_segs = sum(len(c) for c in contours)
    if lab.shape != (n_segs,):
        raise ValueError(f"segment labels do not match this glyph ({len(lab)} labels, {n_segs} segments)")
    mask = np.isin(lab, np.asarray(list(parts), np.int64))
    if not mask.any():
        raise ValueError("those strokes have no outline segments")
    loops = assemble_strokes(contours, np.where(mask, 0, -1), 1)[0]
    first, second = np.asarray(first, float), np.asarray(second, float)

    pieces = [p for p in polygon_parts(stroke_geometry(loops)) if p.area > 1]
    med = None
    if len(pieces) == 1:
        med = _oriented(stroke_median(loops, poly=pieces[0]), first, second)
        if max(np.hypot(*(med[0] - first[0])), np.hypot(*(med[-1] - second[-1]))) > MERGE_END_TOL:
            med = None
    if med is None:
        med = np.concatenate([first, second])
    med = [[int(round(x)), int(round(y))] for x, y in med]
    med = [p for i, p in enumerate(med) if i == 0 or p != med[i - 1]]
    if len(med) < 2:
        med = med * 2
    return dict(stroke=" ".join(segs_to_path(loop) for loop in loops), median=med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="glyph .svg file or a directory of them")
    ap.add_argument("-o", "--output", help="output .json (file input) or directory (directory input)")
    ap.add_argument("--device")
    ap.add_argument("--workers", type=int, help="CPU worker processes for directory input")
    args = ap.parse_args()
    gen = StrokeGenerator(device=args.device)
    try:
        if os.path.isdir(args.input):
            out_dir = args.output or os.path.join("outputs", "predicted")
            os.makedirs(out_dir, exist_ok=True)
            names = sorted(n for n in os.listdir(args.input) if n.endswith(".svg"))
            results = gen.predict_files_batch([os.path.join(args.input, n) for n in names], workers=args.workers)
            failed = 0
            for n, res in zip(names, results):
                if res is None or "_error" in res or not res["strokes"]:
                    failed += 1
                    why = "unreadable" if res is None else res.get("_error", "no outline found")
                    print(f"{n}: skipped ({why})", file=sys.stderr)
                    continue
                with open(os.path.join(out_dir, n[:-4] + ".json"), "w", encoding="utf-8") as f:
                    json.dump(dict(strokes=res["strokes"], medians=res["medians"]), f, ensure_ascii=False)
            print(f"wrote {len(names) - failed} files to {out_dir}" + (f", {failed} skipped" if failed else ""))
            return 1 if failed else 0
        with open(args.input, encoding="utf-8") as f:
            svg = f.read()
        try:
            res = gen.predict_svg(svg)
        except (GlyphError, ValueError) as e:
            print(f"{args.input}: {e}", file=sys.stderr)
            return 1
        text = json.dumps(res, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            sys.stdout.write(text + "\n")
        return 0
    finally:
        gen.close()


if __name__ == "__main__":
    sys.exit(main())
