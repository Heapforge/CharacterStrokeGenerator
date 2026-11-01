#!/usr/bin/env python3
"""Make cut ends flat: replace the curved bridges that close each stroke with straight lines.

assemble_strokes closes a stroke across a cut with a tangent-continuous cubic (C_RATIO), which
gives the stroke a rounded end. A segment of a stroke that is not one of the font's own outline
segments is a bridge; each curved bridge becomes a straight line when that line stays inside the
glyph (otherwise the curve is kept). Font outline segments are never touched.

    uv run python tools/flatten_bridges.py in.json out.json [--font noto-sans]

in.json maps char -> {"strokes": [...], "medians": [...], ...}; out.json gets the same with
flattened strokes, plus "_flattened": number of bridges changed per char.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.svg_io import Seg, load_svg_glyph, parse_hw_path, segs_to_path  # noqa: E402
from tools.fix_hole_strokes import raster  # noqa: E402

TOL = 0.2  # paths are written with 0.1 precision


def font_segment_index(contours):
    """Endpoint pairs of every font segment (both directions), for spotting bridges."""
    idx = []
    for c in contours:
        for s in c:
            idx.append((s.kind, s.p0, s.p1))
    return idx


def is_font_segment(s: Seg, idx) -> bool:
    for kind, a, b in idx:
        if kind != s.kind:
            continue
        if (np.abs(a - s.p0).max() < TOL and np.abs(b - s.p1).max() < TOL) or \
           (np.abs(a - s.p1).max() < TOL and np.abs(b - s.p0).max() < TOL):
            return True
    return False


def flatten_stroke(path: str, idx, outside: np.ndarray) -> tuple[str, int]:
    loops = parse_hw_path(path)
    base = int(((raster(loops) > 0.5) & outside).sum())
    n = 0
    for loop in loops:
        for i, s in enumerate(loop):
            if s.kind == "L" or is_font_segment(s, idx):
                continue
            loop[i] = Seg("L", np.stack([s.p0, s.p1]))
            if int(((raster(loops) > 0.5) & outside).sum()) > base + 6:
                loop[i] = s  # the straight cut would leave the glyph: keep the curve
            else:
                n += 1
    return " ".join(segs_to_path(l) for l in loops), n


def flatten(char: str, strokes: list[str], font: str = "noto-sans") -> tuple[list[str], int]:
    contours = load_svg_glyph(open(os.path.join(ROOT, "training_data", "font_glyphs", font, char + ".svg"),
                                   encoding="utf-8").read())
    idx = font_segment_index(contours)
    outside = raster(contours) <= 0.5
    out, total = [], 0
    for p in strokes:
        q, n = flatten_stroke(p, idx, outside)
        out.append(q)
        total += n
    return out, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--font", default="noto-sans")
    ap.add_argument("--chars", default="")
    a = ap.parse_args()
    data = json.load(open(a.inp, encoding="utf-8"))
    res = {}
    for c, d in data.items():
        if a.chars and c not in a.chars:
            continue
        s, n = flatten(c, d["strokes"], a.font)
        res[c] = dict(d, strokes=s, _flattened=n)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(res)} chars, {sum(r['_flattened'] for r in res.values())} bridges flattened in "
          f"{sum(1 for r in res.values() if r['_flattened'])} chars")


if __name__ == "__main__":
    main()
