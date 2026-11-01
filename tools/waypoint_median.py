#!/usr/bin/env python3
"""Build a median that follows a stroke's centre line through given waypoints, in order.

Each waypoint snaps to the nearest point of the stroke's skeleton and consecutive waypoints are
joined by the shortest skeleton path, so the median stays inside the stroke, never crosses a
counter, and retraces where the pen goes back (waypoints bottom-left -> arch top on η run back
up the stem). The first and last points are the stroke's terminals nearest the first and last
waypoints.

    uv run python tools/waypoint_median.py --char α --stroke 0 --points "470,600 300,610 ..."
    uv run python tools/waypoint_median.py --char α --leaves      # list skeleton terminals

Python: waypoint_median(stroke_path, [(x, y), ...]) -> [[x, y], ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.medians import rdp  # noqa: E402
from strokegen.svg_io import load_svg_glyph, parse_hw_path, segs_to_path  # noqa: E402
from tools.median_repair import holey_polygon, skeleton  # noqa: E402


def _graph(stroke_path):
    P = holey_polygon(parse_hw_path(stroke_path))
    sk = skeleton(P) if P is not None else None
    if sk is None:
        raise ValueError("no skeleton for this stroke")
    return sk


def leaves(stroke_path):
    V, G, used, clear = _graph(stroke_path)
    deg = np.diff(G.indptr)
    return [tuple(np.round(V[i]).astype(int)) for i in used if deg[i] == 1]


def waypoint_median(stroke_path, points, eps=3.0):
    V, G, used, clear = _graph(stroke_path)
    U = V[used]
    snap = [int(used[np.argmin(np.linalg.norm(U - np.asarray(p, float), axis=1))]) for p in points]
    path = [snap[0]]
    for a, b in zip(snap[:-1], snap[1:]):
        if a == b:
            continue
        dist, pred = dijkstra(G, indices=a, return_predecessors=True)
        if not np.isfinite(dist[b]):
            raise ValueError(f"waypoints {a} and {b} are not connected inside the stroke")
        seg = [b]
        while seg[-1] != a:
            seg.append(pred[seg[-1]])
        path.extend(seg[::-1][1:])
    pts = rdp(V[path], eps)
    out = [[int(round(x)), int(round(y))] for x, y in pts]
    return [p for i, p in enumerate(out) if i == 0 or p != out[i - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True)
    ap.add_argument("--font", default="noto-sans")
    ap.add_argument("--data", help="json with strokes (default: the whole font glyph as one stroke)")
    ap.add_argument("--stroke", type=int, default=0)
    ap.add_argument("--points", default="")
    ap.add_argument("--leaves", action="store_true")
    a = ap.parse_args()
    if a.data:
        stroke = json.load(open(a.data, encoding="utf-8"))["strokes"][a.stroke]
    else:
        g = load_svg_glyph(open(os.path.join(ROOT, "training_data", "font_glyphs", a.font, a.char + ".svg"),
                                encoding="utf-8").read())
        stroke = " ".join(segs_to_path(c) for c in g)
    if a.leaves:
        print(leaves(stroke))
        return
    pts = [tuple(map(float, p.split(","))) for p in a.points.split()]
    print(json.dumps(waypoint_median(stroke, pts)))


if __name__ == "__main__":
    main()
