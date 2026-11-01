#!/usr/bin/env python3
"""Check one glyph's hanzi-writer data against its font SVG.

    uv run python tools/verify_glyph.py data.json --char Ḕ [--font noto-sans]

Prints the union IoU, the stroke area outside the font ("extra") and the font area no stroke
covers ("missing"), then per stroke: number of loops, median points, share of median points
inside the stroke, and median coverage (share of the stroke within 1.4 half-widths of its
median). Accents above the em box are included. Exit status 1 when a check fails:
IoU < 0.99, a median point outside its stroke, or coverage < 0.9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.raster import iou  # noqa: E402
from strokegen.svg_io import load_svg_glyph, parse_hw_path  # noqa: E402
from tools.fix_hole_strokes import FIT_A, FIT_B, SIZE, median_coverage, raster  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--char", required=True)
    ap.add_argument("--font", default="noto-sans")
    a = ap.parse_args()
    d = json.load(open(a.data, encoding="utf-8"))
    font = load_svg_glyph(open(os.path.join(ROOT, "training_data", "font_glyphs", a.font, a.char + ".svg"),
                               encoding="utf-8").read())
    rf = raster(font)
    if len(d["strokes"]) != len(d["medians"]):
        print(f"FAIL: {len(d['strokes'])} strokes but {len(d['medians'])} medians")
        sys.exit(1)
    union = np.zeros_like(rf)
    ok = True
    rows = []
    s = SIZE / 1024
    for i, (st, med) in enumerate(zip(d["strokes"], d["medians"])):
        loops = parse_hw_path(st)
        r = raster(loops)
        union = np.maximum(union, r)
        inside = 0
        for x, y in med:
            q = np.array([x, y], float) @ FIT_A.T + FIT_B
            px, py = int(q[0] * s), int((900 - q[1]) * s)
            if 0 <= px < SIZE and 0 <= py < SIZE and r[max(0, py - 2):py + 3, max(0, px - 2):px + 3].max() > 0.5:
                inside += 1
        frac = inside / max(1, len(med))
        cov = median_coverage(loops, med)
        bad = frac < 1.0 or cov < 0.9 or len(med) < 2
        ok &= not bad
        rows.append(f"  stroke {i}: loops={len(loops)} median_pts={len(med)} median_inside={frac:.2f} "
                    f"coverage={cov:.3f} start={med[0]} end={med[-1]}{'  <-- check' if bad else ''}")
    A, B = rf > 0.5, union > 0.5
    score = iou(rf, union)
    ok &= score >= 0.99
    print(f"{a.char}: {len(d['strokes'])} strokes, union IoU {score:.4f}, extra_px {int((B & ~A).sum())}, "
          f"missing_px {int((A & ~B).sum())} (of {int(A.sum())})")
    print("\n".join(rows))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
