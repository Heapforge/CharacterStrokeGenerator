#!/usr/bin/env python3
"""Find stroke geometry that hanzi-writer never shows.

hanzi-writer draws a stroke by clipping a 200-unit-wide brush (STROKE_WIDTH in hanzi-writer
3.5.0) to the stroke outline. The brush is an SVG stroke along the median with its first point
moved 100 units further back, round line caps, and MITER joins with the default miter limit 4
(checked on the rendered page). So any part of the outline more than 100 units from the median
stays invisible (a median cutting across e's counter leaves the top of the arch undrawn), and at a
turn sharper than about 29 degrees the join is bevelled: nothing beyond the turning point is
drawn. Before drawing, hanzi-writer also drops every median point whose two segments are exactly
parallel (z of the cross product == 0, filterParallelPoints), which includes an exact 180 degree
turn: a median that goes straight out along ε's middle tongue and straight back loses its tip
point, so the end of the tongue is chipped off. Both effects are simulated here.

    uv run python tools/reveal_check.py data.json report.json   # data: char -> {strokes, medians}
    uv run python tools/reveal_check.py --curated report.json   # the training data

Per stroke it reports the hidden area (hw units^2) and share; a stroke is flagged when more than
MIN_AREA units^2 and MIN_SHARE of it is hidden.
"""
from __future__ import annotations

import json
import os
import sys

import cairo
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.svg_io import parse_hw_path  # noqa: E402
from tools.fix_hole_strokes import FIT_A, FIT_B, SIZE, raster  # noqa: E402

BRUSH = 200.0
MIN_AREA = 600.0   # hw units^2 (about a 25 x 25 patch)
MIN_SHARE = 0.01
PX_AREA = (1024 / SIZE / FIT_A[0, 0]) ** 2  # hw units^2 per pixel


def extend_start(med, dist=BRUSH / 2):
    """hanzi-writer moves the first point `dist` further away from the second one."""
    m = np.asarray(med, float).copy()
    if len(m) < 2:
        return m
    d = m[0] - m[1]
    n = np.linalg.norm(d)
    if n > 1e-9:
        m[0] = m[0] + d / n * dist
    return m


def filter_parallel_points(med):
    """hanzi-writer's filterParallelPoints: drop a point when the segment into it and the segment
    out of it are exactly parallel (same or opposite direction)."""
    pts = [list(map(float, p)) for p in med]
    if len(pts) < 3:
        return pts
    out = [pts[0], pts[1]]
    for p in pts[2:]:
        cur = (p[0] - out[-1][0], p[1] - out[-1][1])
        prev = (out[-1][0] - out[-2][0], out[-1][1] - out[-2][1])
        if cur[1] * prev[0] - cur[0] * prev[1] == 0:
            out.pop()
        out.append(p)
    return out


def brush(med) -> np.ndarray:
    surf = cairo.ImageSurface(cairo.FORMAT_A8, SIZE, SIZE)
    ctx = cairo.Context(surf)
    s = SIZE / 1024
    ctx.set_matrix(cairo.Matrix(s, 0, 0, -s, 0, 900 * s))
    pts = extend_start(filter_parallel_points(med)) @ FIT_A.T + FIT_B
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    if len(pts) == 1:
        ctx.line_to(*pts[0])
    ctx.set_line_width(BRUSH * FIT_A[0, 0])
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    ctx.set_miter_limit(4.0)
    ctx.stroke()
    surf.flush()
    return np.frombuffer(surf.get_data(), np.uint8).reshape(SIZE, surf.get_stride())[:, :SIZE] > 127


def hidden(stroke_path: str, med) -> tuple[float, float]:
    """(hidden area in hw units^2, hidden share of the stroke)."""
    r = raster(parse_hw_path(stroke_path)) > 0.5
    if not r.any():
        return 0.0, 0.0
    h = r & ~brush(med)
    return float(h.sum() * PX_AREA), float(h.sum() / r.sum())


def check(data: dict) -> dict:
    out = {}
    for c, d in data.items():
        rows = []
        for i, (s, m) in enumerate(zip(d["strokes"], d["medians"])):
            a, share = hidden(s, m)
            if a > MIN_AREA and share > MIN_SHARE:
                rows.append({"stroke": i, "hidden_area": round(a), "hidden_share": round(share, 3)})
        if rows:
            out[c] = rows
    return out


def main():
    if sys.argv[1] == "--curated":
        D = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
        data = {}
        for n in sorted(os.listdir(D)):
            if n.endswith(".json") and not n.endswith(("-jp.json", "-ko.json")):
                try:
                    data[n[:-5]] = json.load(open(os.path.join(D, n), encoding="utf-8"))
                except ValueError:
                    pass
    else:
        data = json.load(open(sys.argv[1], encoding="utf-8"))
    rep = check(data)
    json.dump(rep, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    worst = sorted(rep.items(), key=lambda kv: -max(r["hidden_share"] for r in kv[1]))
    print(f"{len(data)} glyphs checked, {len(rep)} with hidden geometry")
    print(" ".join(f"{c}:{max(r['hidden_share'] for r in rows):.0%}" for c, rows in worst[:80]))


if __name__ == "__main__":
    main()
