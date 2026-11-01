#!/usr/bin/env python3
"""Split B-like glyphs (B, в, ʙ, ᴮ, ...) into a stem stroke and a bowls stroke.

The rule: write the vertical stem first (top to bottom), then the rest in one stroke: top bar,
round the upper bowl, back along the middle bar, round the lower bowl, back along the bottom bar.

A glyph qualifies when its largest contour has a straight vertical left edge over most of its
height, the counters inside it are stacked and start at the same x (the stem's inner edge), and
the edges at the top and bottom of the stem are horizontal lines. The cut runs straight down the
stem's inner edge; the two horizontal edges are split there (collinear pieces, so the outline is
unchanged).

    uv run python tools/split_stem.py out.json --chars Bвʙ
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from shapely.geometry import LineString, Point

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.svg_io import Seg, load_svg_glyph, segs_to_path  # noqa: E402
from tools.fix_hole_strokes import poly  # noqa: E402
from tools.median_repair import holey_polygon  # noqa: E402

FONT = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")


def L(a, b):
    return Seg("L", np.array([a, b], float))


def _hits(ray, geom, c):
    g = ray.intersection(geom)
    pts = [g] if g.geom_type == "Point" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Point"]
    return sorted(pts, key=lambda p: p.distance(Point(c)))


def bowl_median(rest_loops, holes, x1, inset):
    """Top-left -> round each counter clockwise on its right side -> back to the stem edge."""
    P = holey_polygon(rest_loops)
    out = []
    for h in sorted(holes, key=lambda p: -p.centroid.y):
        c = np.array(h.centroid.coords[0])
        seq = []
        for th in np.linspace(np.pi / 2, -np.pi / 2, 21):
            ray = LineString([c, c + 3000 * np.array([np.cos(th), np.sin(th)])])
            a = _hits(ray, h.exterior, c)
            b = _hits(ray, P.exterior, c)
            if not a or not b:
                return None
            a = a[0]
            b = [q for q in b if q.distance(Point(c)) > a.distance(Point(c)) + 1]
            if not b:
                return None
            seq.append([(a.x + b[0].x) / 2, (a.y + b[0].y) / 2])
        if not out:
            out.append([x1 + inset, seq[0][1]])
        out.extend(seq)
        out.append([x1 + inset, seq[-1][1]])
    pts = [[int(round(x)), int(round(y))] for x, y in out]
    return [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]


def split_b(contours, min_holes=2):
    """[stem, rest] strokes and medians for the B-like part of a glyph, plus the contour ids used."""
    polys = [poly(c) for c in contours]
    order = sorted((i for i, p in enumerate(polys) if p is not None), key=lambda i: -polys[i].area)
    for oi in order:
        outer, op = contours[oi], polys[oi]
        holes = [i for i, p in enumerate(polys) if i != oi and p is not None and op.contains(p.representative_point())]
        if len(holes) < min_holes:
            continue
        x0 = min(s.pts[:, 0].min() for s in outer)
        ys = np.concatenate([s.pts[:, 1] for s in outer])
        H = ys.max() - ys.min()
        vi = [k for k, s in enumerate(outer) if s.kind == "L" and abs(s.p0[0] - x0) < 0.5 and abs(s.p1[0] - x0) < 0.5
              and abs(s.p1[1] - s.p0[1]) > 0.8 * H]
        if len(vi) != 1:
            continue
        k = vi[0]
        n = len(outer)
        v, prev, nxt = outer[k], outer[(k - 1) % n], outer[(k + 1) % n]
        x1 = min(min(s.pts[:, 0].min() for s in contours[h]) for h in holes)
        hole_lefts = [min(s.pts[:, 0].min() for s in contours[h]) for h in holes]
        if max(hole_lefts) - x1 > 1.0 or x1 - x0 < 20:
            continue
        # B bowls: curved counters with a waist on the right between them (not Ɇ's slash
        # holes, not a barred D)
        if not all(any(s.kind != "L" for s in contours[h]) for h in holes):
            continue
        hs = sorted(holes, key=lambda h: -polys[h].centroid.y)
        waist_ok = True
        for ha, hb in zip(hs, hs[1:]):
            ya = min(s.pts[:, 1].min() for s in contours[ha])
            yb = max(s.pts[:, 1].max() for s in contours[hb])
            ym = (ya + yb) / 2
            r_mid = op.intersection(LineString([(x0 - 10, ym), (x0 + 5000, ym)])).bounds[2]
            r_a = op.intersection(LineString([(x0 - 10, polys[ha].centroid.y), (x0 + 5000, polys[ha].centroid.y)])).bounds[2]
            r_b = op.intersection(LineString([(x0 - 10, polys[hb].centroid.y), (x0 + 5000, polys[hb].centroid.y)])).bounds[2]
            if r_mid > min(r_a, r_b) - 15:
                waist_ok = False
        if not waist_ok:
            continue
        ok = all(s.kind == "L" and abs(s.p0[1] - s.p1[1]) < 0.5 and
                 min(s.p0[0], s.p1[0]) <= x0 + 0.5 and max(s.p0[0], s.p1[0]) > x1 + 1 for s in (prev, nxt))
        if not ok:
            continue
        y_prev, y_next = prev.p0[1], nxt.p0[1]
        cut_prev, cut_next = np.array([x1, y_prev]), np.array([x1, y_next])
        stem = [v, L(v.p1, cut_next), L(cut_next, cut_prev), L(cut_prev, v.p0)]
        rest = [L(cut_next, nxt.p1)] + [outer[(k + 2 + j) % n] for j in range(n - 3)] + \
               [L(prev.p0, cut_prev), L(cut_prev, cut_next)]
        rest_loops = [rest] + [contours[h] for h in holes]
        inset = (x1 - x0) / 2
        top, bot = max(y_prev, y_next), min(y_prev, y_next)
        stem_med = [[int(round(x0 + inset)), int(round(top - inset))], [int(round(x0 + inset)), int(round(bot + inset))]]
        med = bowl_median(rest_loops, [polys[h] for h in holes], x1, inset * 0.6)
        if med is None:
            continue
        return {"strokes": [segs_to_path(stem), " ".join(segs_to_path(l) for l in rest_loops)],
                "medians": [stem_med, med]}, [oi] + holes
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--chars", required=True)
    a = ap.parse_args()
    res = {}
    for c in a.chars:
        p = os.path.join(FONT, c + ".svg")
        if not os.path.exists(p):
            continue
        cont = load_svg_glyph(open(p, encoding="utf-8").read())
        r = split_b(cont)
        if r and len(r[1]) == len(cont):
            res[c] = r[0]
        print(c, "ok" if c in res else "no")
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
