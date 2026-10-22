"""Glyph SVG -> hanzi-writer JSON.

usage:
  python -m strokegen.predict glyph.svg -o 永.json
  python -m strokegen.predict glyph_dir/ -o out_dir/
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import shapely
import torch
from shapely.geometry import Polygon
from shapely.validation import make_valid

from .data import collate, to_device
from .decode import assemble_strokes, partition_from_predictions
from .features import glyph_features
from .medians import stroke_median
from .model import CutModel
from .order import decode_order, pair_features, stroke_features, stroke_record
from .svg_io import contour_polygon, load_svg_glyph, segs_to_path
from .train_cut import CKPT
from .train_order import OrderNet


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
            empty = np.zeros(0, np.int64)
            return dict(strokes=[], medians=[], _partition=empty, _order=empty, _seg_stroke=empty)
        lab = self.partition(contours) if labels is None else labels
        assembled = assemble_strokes(contours, lab)
        kept = [k for k, s in enumerate(assembled) if s]
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
