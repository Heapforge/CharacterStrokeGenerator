#!/usr/bin/env python3
"""Extend medians that skip a limb of their stroke.

Example: u is one stroke, but a median that runs from the top-left terminal round the bowl and up
the right stem stops at the top and never draws the stem's foot below the bowl. The pen goes up,
then back down to the foot. This finds the stroke's skeleton, the longest skeleton branch the
median does not reach, and the point J where that branch leaves the median. If the median's last
run from J continues straight in the opposite direction of the branch (u, ɥ), the median is
extended at its end: back to J, then down the branch. Otherwise the branch is drawn as an
out-and-back detour at J. Branches shorter than 2.5 stroke half-widths (corner spurs, blobs)
are ignored.

    uv run python tools/median_repair.py in.json out.json [--chars uɥ]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Point

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.medians import rdp  # noqa: E402
from strokegen.svg_io import parse_hw_path  # noqa: E402
from tools.fix_hole_strokes import median_coverage, poly  # noqa: E402

STEP = 5.0
MIN_GAIN = 0.02


def holey_polygon(loops):
    g = None
    for p in sorted((p for p in (poly(l) for l in loops) if p is not None), key=lambda p: -p.area):
        g = p if g is None else g.symmetric_difference(p)
    if g is not None and g.geom_type != "Polygon":
        parts = [q for q in getattr(g, "geoms", []) if q.geom_type == "Polygon"]
        g = max(parts, key=lambda q: q.area) if parts else None
    return g


def skeleton(P):
    pts = []
    for ring in [P.exterior, *P.interiors]:
        pts.append(np.asarray(shapely.segmentize(ring, STEP).coords)[:-1])
    pts = np.concatenate(pts)
    vor = Voronoi(pts + np.random.default_rng(0).normal(0, 1e-3, pts.shape))
    V = vor.vertices
    inside = shapely.contains_xy(P, V[:, 0], V[:, 1])
    E = np.array([e for e in vor.ridge_vertices if e[0] >= 0 and e[1] >= 0 and inside[e[0]] and inside[e[1]]])
    if len(E) == 0:
        return None
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) + 1e-6
    n = len(V)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    used = np.unique(E)
    clear = np.array([P.boundary.distance(Point(v)) for v in V[used]])
    return V, G, used, dict(zip(used.tolist(), clear))


def _path(pred, a, b):
    out = [b]
    while out[-1] != a and pred[out[-1]] >= 0:
        out.append(pred[out[-1]])
    return out[::-1]


def repair_one(loops, med, mode):
    """mode "end": extend the end back down a limb the last run of the median skipped (u, ɥ).
    mode "detour": draw the longest unreached limb as an out-and-back detour."""
    P = holey_polygon(loops)
    if P is None or len(med) < 2:
        return None
    sk = skeleton(P)
    if sk is None:
        return None
    V, G, used, clear = sk
    half = max(clear.values())
    M = LineString(med)
    d_med = {i: M.distance(Point(V[i])) for i in used}
    near = [i for i in used if d_med[i] < 0.6 * half]
    if not near:
        return None
    dist, pred, _ = dijkstra(G, indices=near, min_only=True, return_predecessors=True)
    med = np.asarray(med, float)
    far = [i for i in used if np.isfinite(dist[i]) and d_med[i] > (0.8 if mode == "end" else 1.2) * half]
    for L in sorted(far, key=lambda i: -dist[i]):
        if dist[L] < (1.0 if mode == "end" else 2.5) * half:
            return None
        chain = [L]
        while pred[chain[-1]] >= 0:
            chain.append(pred[chain[-1]])
        J = chain[-1]
        branch = V[chain[::-1]]  # J -> L
        k = int(np.argmin(np.linalg.norm(med - V[J], axis=1)))
        if mode == "detour":
            new = np.concatenate([med[:k + 1], branch[1:], branch[::-1][1:], med[k + 1:]])
        else:
            tail = med[k:]
            d_tail = med[-1] - V[J]
            d_branch = branch[-1] - branch[0]
            if len(tail) < 1 or np.linalg.norm(d_tail) < 2 * half:
                continue
            straight = LineString(np.vstack([V[J], tail])).length < 1.25 * np.linalg.norm(d_tail) + 1
            cos = np.dot(d_branch, d_tail) / (np.linalg.norm(d_branch) * np.linalg.norm(d_tail) + 1e-9)
            if not (straight and cos < -0.8):
                continue
            new = np.concatenate([med, [V[J]], branch[1:]])  # ... end, straight back to J, down the limb
        new = rdp(new, 3.0)
        return [[int(round(x)), int(round(y))] for x, y in new]
    return None


def repair(stroke_path, med, detours=True, max_rounds=3):
    """Repaired median, or the original when nothing is missing or the repair does not help.

    The end retrace is a writing rule, so it applies whenever its shape test passes. Detours only
    apply below 0.97 coverage and when they add coverage; detours=False skips them (used for
    medians copied from curated data, which follow their own conventions)."""
    loops = parse_hw_path(stroke_path)
    cur = med
    new = repair_one(loops, cur, "end")
    if new is not None and median_coverage(loops, new) >= median_coverage(loops, cur):
        cur = new
    if not detours:
        return cur
    cov = median_coverage(loops, cur)
    for _ in range(max_rounds):
        if cov >= 0.97:
            break
        new = repair_one(loops, cur, "detour")
        if new is None:
            break
        c2 = median_coverage(loops, new)
        if c2 < cov + MIN_GAIN:
            break
        cur, cov = new, c2
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--chars", default="")
    a = ap.parse_args()
    data = json.load(open(a.inp, encoding="utf-8"))
    res, changed = {}, []
    for c, d in data.items():
        if a.chars and c not in a.chars:
            continue
        tmpl = set(d.get("template_strokes", []))
        meds = [repair(s, m, detours=i not in tmpl) for i, (s, m) in enumerate(zip(d["strokes"], d["medians"]))]
        if meds != d["medians"]:
            changed.append(c)
        res[c] = dict(d, medians=meds)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(res)} chars, medians extended in {len(changed)}: {''.join(changed)}")


if __name__ == "__main__":
    main()
