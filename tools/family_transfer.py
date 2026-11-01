#!/usr/bin/env python3
"""Give accented / look-alike glyphs the strokes of a reviewed base letter.

Many glyphs are a base letter plus marks (Ḕ = E + macron + acute, Ѐ = E + grave), and the base
letter's outline appears in them unchanged, only shifted. For each target glyph this finds the
template letter whose font contours all appear in the target with one common shift and that
covers the most of it, copies the template's strokes and medians (shifted), and adds one stroke
per remaining mark (a contour plus the counters inside it). Marks above the template go first,
marks below go last, which is how the curated data orders them (È, Ŗ, ş). A mark keeps the
predicted median when one predicted stroke covers exactly that mark.

    uv run python tools/family_transfer.py current.json out.json [--exclude Bu] [--templates extra.json]

current.json: char -> {"strokes", "medians", "segStroke"?} (the predictions being corrected).
Templates: training_data/.../data/{char}.json minus --exclude, plus --templates (char -> data,
used first). Only templates whose own strokes match their font glyph (IoU >= 0.99) are used.
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
from strokegen.svg_io import contour_polygon, load_svg_glyph, parse_hw_path, segs_to_path  # noqa: E402
from tools.fix_hole_strokes import holey_median, poly, raster, ring_median, union_raster  # noqa: E402

FONT = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")
DATA = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
TOL = 1.0
MIN_SHARE = 0.5  # the template must be at least half of the target's ink


def glyph(char):
    p = os.path.join(FONT, char + ".svg")
    return load_svg_glyph(open(p, encoding="utf-8").read()) if os.path.exists(p) else None


def signature(contour):
    pts = np.round(np.concatenate([s.pts for s in contour]), 1)
    return len(contour), pts


def area(contours):
    return sum(abs(p.area) for p in (poly(c) for c in contours) if p is not None)


def find_shift(tmpl, target):
    """(shift, matched target contour ids) if every template contour appears in target with one shift."""
    t0 = tmpl[0]
    n0, p0 = signature(t0)
    for j, c in enumerate(target):
        n, p = signature(c)
        if n != n0:
            continue
        shift = p.min(0) - p0.min(0)
        matched = []
        for tc in tmpl:
            tn, tp = signature(tc)
            hit = None
            for k, gc in enumerate(target):
                if k in matched:
                    continue
                gn, gp = signature(gc)
                if gn == tn and gp.shape == tp.shape and _same_points(tp + shift, gp):
                    hit = k
                    break
            if hit is None:
                break
            matched.append(hit)
        else:
            return shift, matched
    return None


def _same_points(a, b):
    """Same closed outline, allowing a different start segment or direction."""
    if np.abs(a - b).max() < TOL:
        return True
    sa, sb = a[np.lexsort(a.T)], b[np.lexsort(b.T)]
    return np.abs(sa - sb).max() < TOL


def shift_path(path, s):
    loops = [[seg.transformed(np.eye(2), s) for seg in loop] for loop in parse_hw_path(path)]
    return " ".join(segs_to_path(l) for l in loops)


def mark_groups(contours, ids):
    """Group leftover contours: each outer contour with the counters inside it."""
    polys = {i: poly(contours[i]) for i in ids}
    outer = [i for i in ids if polys[i] is not None and not any(
        j != i and polys[j] is not None and polys[j].area > polys[i].area and
        polys[j].contains(polys[i].representative_point()) for j in ids)]
    groups = {i: [i] for i in outer}
    for i in ids:
        if i in groups or polys[i] is None:
            continue
        host = min((o for o in outer if polys[o].contains(polys[i].representative_point())),
                   key=lambda o: polys[o].area, default=None)
        if host is not None:
            groups[host].append(i)
    return list(groups.values())


def load_templates(exclude, extra):
    out = {}
    for c, d in (extra or {}).items():
        out[c] = d
    for name in sorted(os.listdir(DATA)):
        if not name.endswith(".json") or name.endswith(("-jp.json", "-ko.json")):
            continue
        c = name[:-5]
        if c in exclude or c in out or len(c) != 1:
            continue
        try:
            out[c] = json.load(open(os.path.join(DATA, name), encoding="utf-8"))
        except ValueError:
            continue
    good = {}
    for c, d in out.items():
        g = glyph(c)
        if not g:
            continue
        try:
            if iou(raster(g), union_raster(d["strokes"])) >= 0.99:
                good[c] = (g, d)
        except Exception:
            continue
    return good


def _mark_order(marks, tol=30.0):
    """Stacked marks top one first (Ấ: acute, circumflex; Ǖ: macron, dots), marks side by side
    left to right (double grave, dieresis), as in the curated data."""
    rows = []
    for m in sorted(marks, key=lambda t: -t[1]):
        if rows and rows[-1][0][1] - m[1] <= tol:
            rows[-1].append(m)
        else:
            rows.append([m])
    return [m for row in rows for m in sorted(row, key=lambda t: t[0])]


def transfer(char, pred, templates):
    target = glyph(char)
    if not target:
        return None
    total = area(target)
    best = None
    for tc, (tg, td) in templates.items():
        if tc == char or len(tg) > len(target):
            continue
        r = find_shift(tg, target)
        if r is None:
            continue
        share = area(tg) / total if total else 0
        if share >= MIN_SHARE and (best is None or share > best[0]):
            best = (share, tc, r)
    if best is None:
        return None
    share, tc, (shift, matched) = best
    tg, td = templates[tc]
    t_strokes = [shift_path(s, shift) for s in td["strokes"]]
    t_meds = [[[int(round(x + shift[0])), int(round(y + shift[1]))] for x, y in m] for m in td["medians"]]

    rest = [i for i in range(len(target)) if i not in matched]
    top = max(p[1] for c in (target[i] for i in matched) for s in c for p in s.pts)
    above, below = [], []
    pred_loops = [parse_hw_path(s) for s in pred["strokes"]]
    for grp in mark_groups(target, rest):
        loops = [target[i] for i in grp]
        path = " ".join(segs_to_path(l) for l in loops)
        med = None
        g_r = raster(loops) > 0.5
        for ps, pm in zip(pred_loops, pred["medians"]):
            pr = raster(ps) > 0.5
            if (pr | g_r).sum() and (pr & g_r).sum() / (pr | g_r).sum() > 0.97:
                med = pm
                break
        if med is None:
            med = ring_median(loops) or [[int(x), int(y)] for x, y in holey_median(loops)]
        pts = np.concatenate([s.pts for l in loops for s in l])
        cx, cy, ct = pts[:, 0].min(), pts[:, 1].mean(), pts[:, 1].max()
        (above if cy > top - 50 else below).append((cx, ct, path, med))
    above = [(p, m) for _, _, p, m in _mark_order(above)]
    below = [(p, m) for _, _, p, m in _mark_order(below)]
    strokes = [p for p, _ in above] + t_strokes + [p for p, _ in below]
    meds = [m for _, m in above] + t_meds + [m for _, m in below]
    score = iou(raster(target), union_raster(strokes))
    if score < 0.99:
        return None
    return {"strokes": strokes, "medians": meds, "template": tc, "iou": round(score, 4),
            "template_strokes": list(range(len(above), len(above) + len(t_strokes)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("current")
    ap.add_argument("out")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--templates", default="")
    ap.add_argument("--chars", default="")
    a = ap.parse_args()
    cur = json.load(open(a.current, encoding="utf-8"))
    extra = json.load(open(a.templates, encoding="utf-8")) if a.templates else None
    templates = load_templates(set(a.exclude), extra)
    print(f"{len(templates)} usable templates")
    res = {}
    for c, d in cur.items():
        if a.chars and c not in a.chars:
            continue
        r = transfer(c, d, templates)
        if r:
            res[c] = r
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    from collections import Counter
    print(f"{len(res)} glyphs rebuilt from templates")
    print(" ".join(f"{t}:{n}" for t, n in Counter(r["template"] for r in res.values()).most_common()))


if __name__ == "__main__":
    main()
