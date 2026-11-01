#!/usr/bin/env python3
"""Reroute medians that cut across a counter so they follow the stroke instead.

e's median runs along the crossbar and then cuts through the counter instead of going up and
over the arch, so hanzi-writer (which only shows the stroke within 100 units of its median, see
tools/reveal_check.py) leaves the top of the arch undrawn. For every stretch of a median that
leaves its stroke, this replaces the stretch by a path along the stroke's skeleton between the
last point before and the first point after it. A counter can be passed on either side, so both
ways round are tried (the shortest skeleton path, and the shortest one that avoids it) and the
one that leaves the least geometry hidden wins. The rest of the median (start, end, direction)
is kept. A stroke is only changed when the hidden area goes down. A median that still leaves
over 30% of its stroke hidden (it sits inside a counter, or on the wrong dot) is replaced by a
ring median (O.json style) or the stroke's centre line when that halves the hidden area.

    uv run python tools/median_reroute.py in.json out.json [--chars eg]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, Point

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.medians import rdp  # noqa: E402
from strokegen.svg_io import parse_hw_path  # noqa: E402
from tools.fix_hole_strokes import holey_median, oriented, ring_median  # noqa: E402
from tools.median_repair import holey_polygon, skeleton  # noqa: E402
from tools.reveal_check import hidden  # noqa: E402

STEP = 3.0


def _resample(med, step=STEP):
    m = np.asarray(med, float)
    out = [m[0]]
    for a, b in zip(m[:-1], m[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * t for t in np.arange(1, n + 1) / n)
    return np.array(out)


def _outside_runs(P, pts, tol=2.0):
    """(i, j): pts[i] is the last point inside before a run outside the stroke, pts[j] the first after."""
    inside = [P.buffer(tol).contains(Point(p)) for p in pts]
    runs, i = [], 0
    while i < len(pts):
        if not inside[i]:
            j = i
            while j < len(pts) and not inside[j]:
                j += 1
            if i > 0 and j < len(pts):
                runs.append((i - 1, j))
            i = j
        else:
            i += 1
    return runs


def _graph_path(G, a, b, avoid=None, V=None, radius=0.0):
    H = G
    if avoid is not None:
        H = G.tolil(copy=True)
        rows, cols = G.nonzero()
        near = np.zeros(G.shape[0], bool)
        for k in avoid:
            near |= np.linalg.norm(V - V[k], axis=1) < radius
        near[a] = near[b] = False
        for r, c in zip(rows, cols):
            if near[r] or near[c]:
                H[r, c] = G[r, c] * 50
        H = H.tocsr()
    dist, pred = dijkstra(H, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        return None
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return path[::-1]


def reroute(stroke_path: str, med):
    loops = parse_hw_path(stroke_path)
    P = holey_polygon(loops)
    if P is None or len(med) < 2:
        return med
    pts = _resample(med)
    runs = _outside_runs(P, pts)
    if not runs:
        return med
    sk = skeleton(P)
    if sk is None:
        return med
    V, G, used, clear = sk
    half = max(clear.values())
    U = V[used]
    nearest = lambda p: int(used[np.argmin(np.linalg.norm(U - p, axis=1))])
    best = list(map(list, med))
    best_h = hidden(stroke_path, med)[0]
    # handle runs from the last one back, so earlier indices stay valid
    cur = pts
    for i, j in reversed(runs):
        a, b = nearest(cur[i]), nearest(cur[j])
        if a == b:
            continue
        p1 = _graph_path(G, a, b)
        cands = [p1] if p1 else []
        if p1:
            p2 = _graph_path(G, a, b, avoid=p1[1:-1], V=V, radius=half)
            if p2 and p2 != p1:
                cands.append(p2)
        options = []
        for p in cands:
            new = np.vstack([cur[:i + 1], V[p], cur[j:]])
            simp = rdp(new, 3.0)
            m = [[int(round(x)), int(round(y))] for x, y in simp]
            options.append((hidden(stroke_path, m)[0], len(p), new, m))
        if not options:
            continue
        h, _, new, m = min(options, key=lambda t: (t[0], t[1]))
        cur = new
        if h < best_h - 1:
            best, best_h = m, h
    return best


def dehook(stroke_path: str, med, max_angle=60.0, slack=30.0):
    """Drop points where the median turns back sharply (a hook left where a reroute joins),
    as long as that hides no more than `slack` units^2 of the stroke. Real retraces (u's
    descent, R's lower bar) hide much more when removed, so they stay."""
    m = [list(p) for p in med]
    h0 = hidden(stroke_path, m)[0]
    changed = True
    while changed and len(m) > 2:
        changed = False
        for k in range(1, len(m) - 1):
            a, b, c = (np.array(m[k - 1], float), np.array(m[k], float), np.array(m[k + 1], float))
            u, v = a - b, c - b
            cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
            if np.degrees(np.arccos(np.clip(cosang, -1, 1))) < max_angle:
                cand = m[:k] + m[k + 1:]
                if hidden(stroke_path, cand)[0] <= h0 + slack:
                    m, changed = cand, True
                    break
    return m


def extend_tips(stroke_path: str, med, max_angle=30.0, margin=-2.0):
    """hanzi-writer bevels joins sharper than ~29 degrees (miter limit 4), so nothing beyond a
    sharp turning point is drawn. Move each such point outwards (along the bisector, i.e. the way
    the pen was going for a straight retrace) until it is `margin` units from the stroke edge
    (negative: just past it; the brush is clipped to the outline, so that is harmless), so the
    tip of the limb (ε's tongue, ω's middle prong, u's stem top) is drawn."""
    P = holey_polygon(parse_hw_path(stroke_path))
    if P is None:
        return med
    m = [list(map(float, p)) for p in med]
    for k in range(1, len(m) - 1):
        a, b, c = (np.array(m[k - 1]), np.array(m[k]), np.array(m[k + 1]))
        u, v = a - b, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        u, v = u / nu, v / nv
        if np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))) >= max_angle:
            continue
        d = -(u + v)
        d = d / np.linalg.norm(d)
        if not P.contains(Point(b)):
            continue
        hit = LineString([b, b + d * 400]).intersection(P.boundary)
        pts = [hit] if hit.geom_type == "Point" else [g for g in getattr(hit, "geoms", []) if g.geom_type == "Point"]
        if not pts:
            continue
        t = min(Point(b).distance(q) for q in pts) - margin
        if t > 0.5:
            m[k] = list(b + d * t)
    out = [[int(round(x)), int(round(y))] for x, y in m]
    return out if hidden(stroke_path, out)[0] <= hidden(stroke_path, med)[0] else med


def keep_turns(med):
    """hanzi-writer drops a point whose in and out segments are exactly parallel, which also
    deletes the tip of an exact out-and-back retrace. Nudge the point after such a turn by one
    unit sideways so the turn survives (plain collinear points may still be dropped; harmless)."""
    pts = [list(map(int, p)) for p in med]
    for i in range(2, len(pts)):
        a, b, c = pts[i - 2], pts[i - 1], pts[i]
        prev = (b[0] - a[0], b[1] - a[1])
        cur = (c[0] - b[0], c[1] - b[1])
        if cur[1] * prev[0] - cur[0] * prev[1] == 0 and cur[0] * prev[0] + cur[1] * prev[1] < 0:
            # perpendicular unit step (integer)
            px, py = -prev[1], prev[0]
            n = max(abs(px), abs(py)) or 1
            c[0] += int(round(px / n))
            c[1] += int(round(py / n))
    return pts


def hw_safe(stroke_path: str, med):
    """Tips reach the end of their limb and survive hanzi-writer's point filter."""
    return keep_turns(extend_tips(stroke_path, med))


def fallback(stroke_path: str, med, share_limit=0.3):
    """For a median that barely touches its stroke (inside a counter, on the wrong dot): use the
    O.json-style ring median for a ring, else the stroke's own centre line, whichever hides less."""
    h0, share = hidden(stroke_path, med)
    if share <= share_limit:
        return med
    loops = parse_hw_path(stroke_path)
    cands = []
    rm = ring_median(loops)
    if rm:
        cands.append(rm)
    hm = holey_median(loops)
    if len(hm) >= 2:
        cands.append(oriented(hm, med))
    scored = [(hidden(stroke_path, m)[0], m) for m in cands]
    if not scored:
        return med
    h, m = min(scored, key=lambda t: t[0])
    return m if h < 0.5 * h0 else med


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
        meds = []
        for s, m in zip(d["strokes"], d["medians"]):
            r = fallback(s, reroute(s, m))
            r = dehook(s, r) if r != m else m
            meds.append(hw_safe(s, r))
        if meds != d["medians"]:
            changed.append(c)
        res[c] = dict(d, medians=meds)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(res)} glyphs, medians rerouted in {len(changed)}: {''.join(changed)}")


if __name__ == "__main__":
    main()
