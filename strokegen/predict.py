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

import numpy as np
import shapely
import torch
from shapely.geometry import Polygon
from shapely.validation import make_valid

from ._batch import stage_a_from_contours, stage_a_from_file, stage_a_from_svg, stage_bc_prep
from .data import collate, to_device
from .decode import assemble_strokes, partition_from_predictions
from .features import glyph_features
from .medians import stroke_median
from .model import CutModel
from .order import N_PFEAT, N_SFEAT, decode_order, pair_features, stroke_features, stroke_record
from .svg_io import contour_polygon, load_svg_glyph, segs_to_path
from .train_cut import CKPT
from .train_order import OrderNet


def _empty_result(lab: np.ndarray | None = None) -> dict:
    empty = np.zeros(0, np.int64)
    part = empty if lab is None else lab
    seg = empty if lab is None else np.full(len(lab), -1, np.int64)
    return dict(strokes=[], medians=[], _partition=part, _order=empty, _seg_stroke=seg)


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


class StrokeGenerator:
    def __init__(self, cut_ckpt: str = os.path.join(CKPT, "cut.pt"), order_ckpt: str = os.path.join(CKPT, "order.pt"),
                 device: str | None = None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(cut_ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.cut = CutModel(d=a.get("d", 256), layers=a.get("layers", 8)).to(self.dev).eval()
        self.cut.load_state_dict(ck["model"])
        ok = torch.load(order_ckpt, map_location="cpu", weights_only=False)
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
        return list(self._pool.map(fn, items, chunksize=chunksize))

    @torch.no_grad()
    def partition(self, contours) -> np.ndarray:
        f = glyph_features(contours)
        f["index"] = 0
        batch = to_device(collate([f]), self.dev)
        with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
            out = self.cut(batch)
        cut_p = torch.sigmoid(out["cut"].float())[0].cpu().numpy()
        aff_p = torch.sigmoid(out["aff"].float())[0].cpu().numpy()
        return partition_from_predictions(contours, cut_p, aff_p)

    @torch.no_grad()
    def order_strokes(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (order, flip) for the given stroke records."""
        x = torch.from_numpy(stroke_features(records))[None].to(self.dev)
        pair = torch.from_numpy(pair_features(records))[None].to(self.dev)
        pad = torch.zeros(1, len(records), dtype=torch.bool, device=self.dev)
        out = self.order(x, pad, pair)
        flip = (out["dir"][0] > 0).cpu().numpy()
        order = decode_order(out["prec"][0].float().cpu().numpy())
        return order, flip

    def predict_contours(self, contours, labels: np.ndarray | None = None) -> dict:
        """hanzi-writer dict for one glyph. Pass `labels` to skip Stage A (e.g. GT partition)."""
        if not contours:
            return _empty_result()
        lab = self.partition(contours) if labels is None else labels
        assembled = assemble_strokes(contours, lab)
        kept = [k for k, s in enumerate(assembled) if s]
        if not kept:
            return _empty_result(lab)
        strokes = [assembled[k] for k in kept]
        records = [stroke_record(loops) for loops in strokes]
        order, flip = self.order_strokes(records)
        out_s, out_m = [], []
        for k in order:
            out_s.append(" ".join(segs_to_path(loop) for loop in strokes[k]))
            m = records[k]["median"][::-1] if flip[k] else records[k]["median"]
            out_m.append([[int(x), int(y)] for x, y in m])
        # segment -> index into the returned strokes (-1 = unassigned)
        pos = {kept[int(r)]: p for p, r in enumerate(order)}
        seg_stroke = np.array([pos.get(int(v), -1) for v in lab], np.int64)
        return dict(strokes=out_s, medians=out_m, _partition=lab, _order=order, _seg_stroke=seg_stroke)

    @torch.no_grad()
    def _predict_batch(self, stage_a_fn, inputs: list, workers: int | None = None,
                       chunk_size: int = 512) -> list[dict | None]:
        results: list[dict | None] = [None] * len(inputs)
        for start in range(0, len(inputs), chunk_size):
            chunk = inputs[start : start + chunk_size]
            prepped = self._map(stage_a_fn, chunk, workers)
            valid: list[tuple[int, list, dict]] = []
            for local_i, (contours, feat) in enumerate(prepped):
                g_idx = start + local_i
                if contours is None:
                    results[g_idx] = None
                elif not contours or feat is None:
                    results[g_idx] = _empty_result()
                else:
                    valid.append((g_idx, contours, feat))
            del prepped
            if not valid:
                continue

            valid.sort(key=lambda t: len(t[2]["feat"]))
            bc_inputs: list[tuple[int, tuple[list, np.ndarray, np.ndarray]]] = []
            i = 0
            while i < len(valid):
                j = i
                sub = []
                while j < len(valid):
                    limit = _cut_batch_size(len(valid[j][2]["feat"]))
                    if sub and len(sub) >= limit:
                        break
                    sub.append(valid[j])
                    j += 1
                    if len(sub) >= limit:
                        break
                i = j
                feats = []
                for b, (_, _, f) in enumerate(sub):
                    f["index"] = b
                    feats.append(f)
                batch = to_device(collate(feats), self.dev)
                with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
                    out = self.cut(batch)
                cut_all = torch.sigmoid(out["cut"].float()).cpu().numpy()
                aff_all = torch.sigmoid(out["aff"].float()).cpu().numpy()
                for b, (g_idx, contours, f) in enumerate(sub):
                    n = len(f["feat"])
                    bc_inputs.append((g_idx, (contours, cut_all[b, :n].copy(), aff_all[b, :n, :n].copy())))
            del valid

            bc_out = self._map(stage_bc_prep, [x[1] for x in bc_inputs], workers)
            order_valid = []
            for (g_idx, _), (lab, kept, paths, medians, sfeat, pfeat) in zip(bc_inputs, bc_out):
                if not kept:
                    results[g_idx] = _empty_result(lab)
                else:
                    order_valid.append((g_idx, lab, kept, paths, medians, sfeat, pfeat))
            del bc_inputs, bc_out
            if not order_valid:
                continue

            order_valid.sort(key=lambda t: len(t[2]))
            for k in range(0, len(order_valid), 128):
                sub = order_valid[k : k + 128]
                B = len(sub)
                max_s = max(len(t[2]) for t in sub)
                x = torch.zeros(B, max_s, N_SFEAT, dtype=torch.float32)
                pair = torch.zeros(B, max_s, max_s, N_PFEAT, dtype=torch.float32)
                pad = torch.ones(B, max_s, dtype=torch.bool)
                for b, (_, _, kept, _, _, sfeat, pfeat) in enumerate(sub):
                    s = len(kept)
                    x[b, :s] = torch.from_numpy(sfeat)
                    pair[b, :s, :s] = torch.from_numpy(pfeat)
                    pad[b, :s] = False
                out = self.order(x.to(self.dev), pad.to(self.dev), pair.to(self.dev))
                dl = out["dir"].float().cpu().numpy()
                pl = out["prec"].float().cpu().numpy()
                for b, (g_idx, lab, kept, paths, medians, _, _) in enumerate(sub):
                    s = len(kept)
                    flip = dl[b, :s] > 0
                    order = decode_order(pl[b, :s, :s])
                    out_s, out_m = [], []
                    for idx in order:
                        out_s.append(paths[idx])
                        m = medians[idx][::-1] if flip[idx] else medians[idx]
                        out_m.append([[int(px), int(py)] for px, py in m])
                    pos = {kept[int(r)]: p for p, r in enumerate(order)}
                    seg_stroke = np.array([pos.get(int(v), -1) for v in lab], np.int64)
                    results[g_idx] = dict(strokes=out_s, medians=out_m, _partition=lab, _order=order,
                                          _seg_stroke=seg_stroke)
        return results

    def predict_contours_batch(self, contours_list: list, workers: int | None = None) -> list[dict]:
        """Bulk parallel prediction for a list of parsed glyph contours."""
        return [r if r is not None else _empty_result()
                for r in self._predict_batch(stage_a_from_contours, contours_list, workers=workers)]

    def predict_svgs_batch(self, svgs: list[str], workers: int | None = None) -> list[dict]:
        """Bulk parallel prediction for a list of SVG strings."""
        return [r if r is not None else _empty_result()
                for r in self._predict_batch(stage_a_from_svg, svgs, workers=workers)]

    def predict_files_batch(self, paths: list[str | None], workers: int | None = None) -> list[dict | None]:
        """Bulk parallel prediction for a list of SVG file paths (None for missing files)."""
        return self._predict_batch(stage_a_from_file, paths, workers=workers)

    def predict_svg(self, svg_text: str) -> dict:
        r = self.predict_contours(load_svg_glyph(svg_text))
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

    geoms = [make_valid(Polygon(p)) for p in (contour_polygon(loop) for loop in loops) if len(p) >= 3]
    union = shapely.union_all(geoms) if geoms else None
    pieces = [] if union is None else [g for g in getattr(union, "geoms", [union]) if g.geom_type == "Polygon" and g.area > 1]
    med = None
    if len(pieces) == 1:
        med = _oriented(stroke_median(loops), first, second)
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
    args = ap.parse_args()
    gen = StrokeGenerator(device=args.device)
    if os.path.isdir(args.input):
        out_dir = args.output or "outputs/predicted"
        os.makedirs(out_dir, exist_ok=True)
        names = sorted(n for n in os.listdir(args.input) if n.endswith(".svg"))
        for n in names:
            res = gen.predict_svg(open(os.path.join(args.input, n), encoding="utf-8").read())
            with open(os.path.join(out_dir, n[:-4] + ".json"), "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False)
        print(f"wrote {len(names)} files to {out_dir}")
    else:
        res = gen.predict_svg(open(args.input, encoding="utf-8").read())
        text = json.dumps(res, ensure_ascii=False)
        if args.output:
            open(args.output, "w", encoding="utf-8").write(text)
        else:
            sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
