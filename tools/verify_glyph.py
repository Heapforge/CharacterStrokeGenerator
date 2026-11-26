#!/usr/bin/env python3
"""Check one glyph's hanzi-writer data against its font SVG.

    uv run python tools/verify_glyph.py data.json --char Ḕ [--font noto-sans] [--reveal]

Prints the union IoU, the stroke area outside the font ("extra") and the font area no stroke
covers ("missing"), then per stroke: number of loops, median points, share of median points
inside the stroke, and median coverage (share of the stroke within 1.4 half-widths of its
median). Accents above the em box are included. Exit status 1 when a check fails:
IoU < 0.99, a median point outside its stroke, or coverage < 0.9. --reveal also fails strokes
that hanzi-writer leaves partly undrawn (tools/reveal_check.py).

Python: check_glyph(font_contours, data, reveal=True) -> dict is the checker the attempt store
and the dashboard use (strokegen.attempts); this CLI prints the same dict.
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

MIN_IOU = 0.99
MIN_MEDIAN_COVERAGE = 0.9


def check_glyph(font_contours, data: dict, reveal: bool = True, masks: bool = False) -> dict:
    """Checker result for hanzi-writer `data` ({strokes, medians}) cut from `font_contours`.

    Returns {passed, iou, extra_px, missing_px, font_px, n_strokes, failures, strokes}; failures
    lists glyph-level problems ("stroke_median_count", "low_iou"), and each stroke row has its own
    failures ("short_median", "median_outside", "low_coverage", and with reveal "hidden").
    masks=True adds the boolean rasters "missing_mask" / "extra_mask" (SIZE x SIZE, fix_hole_strokes
    frame) for callers that locate the deficit; they are not JSON-serializable.
    """
    strokes, medians = data.get("strokes", []), data.get("medians", [])
    if len(strokes) != len(medians):
        return dict(passed=False, iou=0.0, extra_px=0, missing_px=0, font_px=0, n_strokes=len(strokes),
                    n_medians=len(medians), failures=["stroke_median_count"], strokes=[])
    rf = raster(font_contours)
    union = np.zeros_like(rf)
    rows = []
    s = SIZE / 1024
    for st, med in zip(strokes, medians):
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
        fails = []
        if len(med) < 2:
            fails.append("short_median")
        if frac < 1.0:
            fails.append("median_outside")
        if cov < MIN_MEDIAN_COVERAGE:
            fails.append("low_coverage")
        row = dict(loops=len(loops), median_pts=len(med), median_inside=frac, coverage=cov,
                   start=list(med[0]) if len(med) else None, end=list(med[-1]) if len(med) else None)
        if reveal:
            from tools.reveal_check import MIN_AREA, MIN_SHARE, hidden
            area, share = hidden(st, med) if len(med) else (0.0, 0.0)
            row.update(hidden_area=area, hidden_share=share)
            if area > MIN_AREA and share > MIN_SHARE:
                fails.append("hidden")
        row["failures"] = fails
        rows.append(row)
    A, B = rf > 0.5, union > 0.5
    score = iou(rf, union)
    glyph_fails = ["low_iou"] if score < MIN_IOU else []
    out = dict(
        passed=not glyph_fails and not any(r["failures"] for r in rows),
        iou=float(score), extra_px=int((B & ~A).sum()), missing_px=int((A & ~B).sum()), font_px=int(A.sum()),
        n_strokes=len(strokes), failures=glyph_fails, strokes=rows,
    )
    if masks:
        out.update(missing_mask=A & ~B, extra_mask=B & ~A)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--char", required=True)
    ap.add_argument("--font", default="noto-sans")
    ap.add_argument("--reveal", action="store_true", help="also fail strokes hanzi-writer leaves partly undrawn")
    a = ap.parse_args()
    d = json.load(open(a.data, encoding="utf-8"))
    font = load_svg_glyph(open(os.path.join(ROOT, "training_data", "font_glyphs", a.font, a.char + ".svg"),
                               encoding="utf-8").read())
    res = check_glyph(font, d, reveal=a.reveal)
    if "stroke_median_count" in res["failures"]:
        print(f"FAIL: {res['n_strokes']} strokes but {res['n_medians']} medians")
        sys.exit(1)
    rows = []
    for i, r in enumerate(res["strokes"]):
        hid = f" hidden={r['hidden_share']:.3f}" if "hidden_share" in r else ""
        rows.append(f"  stroke {i}: loops={r['loops']} median_pts={r['median_pts']} "
                    f"median_inside={r['median_inside']:.2f} coverage={r['coverage']:.3f}{hid} "
                    f"start={r['start']} end={r['end']}"
                    f"{'  <-- check' if r['failures'] else ''}")
    print(f"{a.char}: {res['n_strokes']} strokes, union IoU {res['iou']:.4f}, extra_px {res['extra_px']}, "
          f"missing_px {res['missing_px']} (of {res['font_px']})")
    print("\n".join(rows))
    print("PASS" if res["passed"] else "FAIL")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
