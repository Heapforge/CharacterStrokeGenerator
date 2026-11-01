#!/usr/bin/env python3
"""Fix predicted glyphs whose counters (holes) got filled in.

Two common failures. First, an inner contour (the hole of O, D, Q, ...) comes out as its own stroke, or
sits in a stroke that does not enclose it, so the union of strokes fills the counter. This moves
each hole contour into the stroke that best restores the font shape (greedy, scored by union IoU
against the font), rebuilds the changed strokes from the font segments (outline segments are
never changed), drops strokes left empty, and recomputes the medians of changed strokes. A
stroke that ends up as a ring gets a closed-loop median like data/O.json: from the top,
counter-clockwise, all the way round. Second, the cubic bridges that close a stroke can bulge
out of the glyph (П, Е, ...); those are straightened when that removes area outside the font.
Medians of every changed stroke are recomputed.

    uv run python tools/fix_hole_strokes.py page_export.json out_dir [--chars ÖÕ]

page_export.json is a list of {char, strokes, medians, segStroke} as the dashboard predicts
them (segStroke indexes the strokes list). Writes out_dir/{char}.json for every char it
improved and out_dir/_report.json with before/after scores for every char.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cairo
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.validation import make_valid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.decode import assemble_strokes  # noqa: E402
from strokegen.medians import RDP_EPS, TIP_FRAC, _extend, rdp, skeleton_path, stroke_median, trim_spurs  # noqa: E402
from strokegen.raster import iou, render  # noqa: E402
from strokegen.svg_io import Seg, contour_polygon, load_svg_glyph, parse_hw_path, segs_to_path  # noqa: E402

FONT = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")
SIZE = 768
GOOD_IOU = 0.98
# Stacked accents reach far above the hw frame (y > 900); shrink x [-300, 1400], y [-400, 1300]
# into it before rasterizing so the whole glyph is compared.
FIT_A, FIT_B = np.eye(2) * 0.6, np.array([180.0, 116.0])


def fit(loops):
    return [[s.transformed(FIT_A, FIT_B) for s in loop] for loop in loops]


def raster(loops) -> np.ndarray:
    return render(fit(loops), SIZE)


def poly(loop) -> Polygon | None:
    p = contour_polygon(loop)
    if len(p) < 3:
        return None
    g = make_valid(Polygon(p))
    return g if g.area > 1 else None


def hole_contours(contours) -> list[int]:
    """Indices of contours nested at odd depth (counters)."""
    polys = [poly(c) for c in contours]
    out = []
    for i, p in enumerate(polys):
        if p is None:
            continue
        pt = p.representative_point()
        depth = sum(1 for j, q in enumerate(polys) if j != i and q is not None and q.area > p.area and q.contains(pt))
        if depth % 2:
            out.append(i)
    return out


def union_raster(stroke_paths: list[str]) -> np.ndarray:
    u = np.zeros((SIZE, SIZE), np.float32)
    for s in stroke_paths:
        u = np.maximum(u, raster(parse_hw_path(s)))
    return u


def ring_median(loops) -> list[list[int]] | None:
    """Closed-loop midline of a stroke with exactly one hole, or None if it is not ring-like."""
    try:
        return _ring_median(loops)
    except (AttributeError, ValueError):  # odd geometry (multipolygons, ...): not a plain ring
        return None


def _ring_median(loops) -> list[list[int]] | None:
    polys = sorted((p for p in (poly(l) for l in loops) if p is not None), key=lambda p: -p.area)
    if len(polys) != 2 or not polys[0].contains(polys[1].representative_point()):
        return None
    outer, inner = polys
    ring = outer.difference(inner)
    c = np.array(inner.centroid.coords[0])
    R = 2048.0
    pts = []
    for th in np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 33):
        ray = LineString([c, c + R * np.array([np.cos(th), np.sin(th)])])
        hit_in, hit_out = ray.intersection(inner.exterior), ray.intersection(outer.exterior)
        if hit_in.is_empty or hit_out.is_empty:
            return None
        a = min((Point(p) for p in _coords(hit_in)), key=lambda q: q.distance(Point(c)))
        b = min((Point(p) for p in _coords(hit_out)), key=lambda q: q.distance(Point(c)))
        pts.append([(a.x + b.x) / 2, (a.y + b.y) / 2])
    med = LineString(pts)
    half = max(ring.boundary.distance(Point(p)) for p in pts) * 1.6
    if ring.difference(med.buffer(half)).area > 0.1 * ring.area:
        return None  # the stroke has a tail (b, d, q, ...): not a plain ring
    out = [[int(round(x)), int(round(y))] for x, y in pts]
    return [p for i, p in enumerate(out) if i == 0 or p != out[i - 1]]


def _coords(g):
    if g.geom_type == "Point":
        return [g.coords[0]]
    return [p for part in getattr(g, "geoms", [g]) for p in (part.coords if part.geom_type != "Point" else [part.coords[0]])]


def holey_median(loops) -> np.ndarray:
    """stroke_median, but on the stroke shape with its counters cut out.

    stroke_median unions a stroke's loops into one solid shape, so for a stroke that owns a
    counter its median would cross the hole. Loops are combined even-odd here instead.
    """
    g = None
    for p in sorted((p for p in (poly(l) for l in loops) if p is not None), key=lambda p: -p.area):
        g = p if g is None else g.symmetric_difference(p)
    if g is not None and g.geom_type != "Polygon":
        parts = [q for q in getattr(g, "geoms", []) if q.geom_type == "Polygon"]
        g = max(parts, key=lambda q: q.area) if parts else None
    if g is None or not g.interiors:
        return stroke_median(loops)
    path = skeleton_path(g)
    if path is None or len(path) < 2:
        return stroke_median(loops)
    return np.round(rdp(_extend(trim_spurs(path, g), g, TIP_FRAC), RDP_EPS))


def oriented(med: np.ndarray, old: list) -> list[list[int]]:
    old = np.asarray(old, float)
    med = np.asarray(med, float)
    fwd = np.hypot(*(med[0] - old[0])) + np.hypot(*(med[-1] - old[-1]))
    rev = np.hypot(*(med[-1] - old[0])) + np.hypot(*(med[0] - old[-1]))
    m = med if fwd <= rev else med[::-1]
    return [[int(round(x)), int(round(y))] for x, y in m]


def outside_px(path: str, outside: np.ndarray) -> int:
    return int(((raster(parse_hw_path(path)) > 0.5) & outside).sum())


def straighten_bridges(path: str, outside: np.ndarray) -> str:
    """Replace cubic bridges that bulge out of the glyph with straight lines.

    Noto outlines only have L/Q segments, so every C in a predicted stroke is a bridge added by
    assemble_strokes; straightening one never touches a font segment.
    """
    loops = parse_hw_path(path)
    cur = outside_px(path, outside)
    for loop in loops:
        for i, sg in enumerate(loop):
            if sg.kind != "C":
                continue
            loop[i] = Seg("L", np.stack([sg.p0, sg.p1]))
            cand = " ".join(segs_to_path(l) for l in loops)
            n = outside_px(cand, outside)
            if n < cur - 8:
                cur = n
            else:
                loop[i] = sg
    return " ".join(segs_to_path(l) for l in loops)


def fix(item: dict) -> dict:
    char = item["char"]
    contours = load_svg_glyph(open(os.path.join(FONT, char + ".svg"), encoding="utf-8").read())
    rf = raster(contours)
    strokes, medians = list(item["strokes"]), list(item["medians"])
    lab = np.asarray(item["segStroke"], np.int64)
    before = iou(rf, union_raster(strokes))
    res = {"char": char, "before": round(before, 4), "after": round(before, 4), "changed": False}
    if before >= GOOD_IOU:
        return res

    n = len(strokes)
    paths, loops, changed = list(strokes), [parse_hw_path(s) for s in strokes], set()

    # Pass 1: give each counter to the stroke that restores the font shape best.
    if lab.shape == (sum(len(c) for c in contours),):
        offs = np.cumsum([0] + [len(c) for c in contours])

        def build(l):
            ch = {k for k in range(n) if not np.array_equal(l == k, lab == k)}
            lp = assemble_strokes(contours, l, n)
            return [(" ".join(segs_to_path(x) for x in lp[k]) if k in ch else strokes[k]) for k in range(n)], lp, ch

        best_l, best_score = lab.copy(), before
        for h in hole_contours(contours):
            cand = []
            for k in range(n):
                l = best_l.copy()
                l[offs[h]:offs[h + 1]] = k
                if np.array_equal(l, best_l):
                    continue
                cand.append((iou(rf, union_raster([p for p in build(l)[0] if p])), k, l))
            if cand:
                s, k, l = max(cand, key=lambda t: t[0])
                if s > best_score + 1e-4:
                    best_score, best_l = s, l
        if best_score > before + 1e-4:
            paths, lp, changed = build(best_l)
            for k in changed:
                loops[k] = lp[k]

    # Pass 2: straighten bridges that stick out of the glyph.
    outside = rf <= 0.5
    for k in range(n):
        if not paths[k]:
            continue
        p2 = straighten_bridges(paths[k], outside)
        if p2 != paths[k]:
            paths[k], loops[k] = p2, parse_hw_path(p2)
            changed.add(k)

    after = iou(rf, union_raster([p for p in paths if p]))
    if not changed or after <= before + 1e-4:
        return res
    new_s, new_m, how, weak = [], [], [], []
    for k in range(n):
        if not paths[k]:
            continue  # the stroke was only a counter; it now belongs to its enclosing stroke
        if k in changed:
            # Same shape as a curated stroke (accented copies of a base letter): reuse its median.
            # Else a ring gets the O.json loop (top, counter-clockwise); else a skeleton median.
            m, src = curated_median(loops[k]), "curated"
            if m is None:
                m, src = ring_median(loops[k]), "ring"
            if m is None:
                m, src = oriented(holey_median(loops[k]), medians[k]), "skeleton"
            how.append(src)
            if src != "curated" and median_coverage(loops[k], m) < MIN_COVERAGE:
                weak.append(len(new_s))
        else:
            m = medians[k]
        new_m.append(m)
        new_s.append(paths[k])
    res.update(after=round(after, 4), changed=True, n_before=n, n_after=len(new_s), medians_from=how,
               weak_medians=weak, data={"strokes": new_s, "medians": new_m})
    return res


MIN_COVERAGE = 0.9
DATA = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
_curated: list | None = None


def _bbox(loops) -> np.ndarray:
    pts = np.concatenate([sg.pts for loop in loops for sg in loop])
    return np.concatenate([pts.min(0), pts.max(0)])


def _norm_raster(loops, box, size=128) -> np.ndarray:
    span = max(box[2] - box[0], box[3] - box[1], 1.0) * 1.05
    A = np.eye(2) * (1024 / span)
    b = -box[:2] * (1024 / span) + np.array([10.0, -114.0])
    return render([[sg.transformed(A, b) for sg in loop] for loop in loops], size)


def curated_median(loops) -> list[list[int]] | None:
    """Median of a training-data stroke with the same shape (up to a shift), shifted onto this one."""
    global _curated
    if _curated is None:
        _curated = []
        for name in sorted(os.listdir(DATA)):
            if not name.endswith(".json") or name.endswith(("-jp.json", "-ko.json")):
                continue
            try:
                d = json.load(open(os.path.join(DATA, name), encoding="utf-8"))
                for st, med in zip(d["strokes"], d["medians"]):
                    lp = parse_hw_path(st)
                    _curated.append((_bbox(lp), lp, med))
            except (ValueError, KeyError, IndexError):
                continue
    box = _bbox(loops)
    size = box[2:] - box[:2]
    mine = None
    best = (0.985, None)
    for cb, lp, med in _curated:
        if np.abs((cb[2:] - cb[:2]) - size).max() > 3:
            continue
        mine = _norm_raster(loops, box) if mine is None else mine
        s = iou(mine, _norm_raster(lp, cb))
        if s > best[0]:
            best = (s, np.asarray(med, float) + (box[:2] - cb[:2]))
    if best[1] is None:
        return None
    return [[int(round(x)), int(round(y))] for x, y in best[1]]


def median_coverage(loops, med) -> float:
    """Share of the stroke within reach of its median (reach = 1.4 x the stroke's half-width)."""
    from scipy.ndimage import distance_transform_edt
    r = raster(loops) > 0.5
    if not r.any():
        return 1.0
    half = distance_transform_edt(r).max()
    surf = cairo.ImageSurface(cairo.FORMAT_A8, SIZE, SIZE)
    ctx = cairo.Context(surf)
    s = SIZE / 1024
    ctx.set_matrix(cairo.Matrix(s, 0, 0, -s, 0, 900 * s))
    pts = np.asarray(med, float) @ FIT_A.T + FIT_B
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.set_line_width(2 * 1.4 * half / s)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke()
    surf.flush()
    band = np.frombuffer(surf.get_data(), np.uint8).reshape(SIZE, surf.get_stride())[:, :SIZE] > 127
    return float((r & band).sum() / r.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("out_dir")
    ap.add_argument("--chars", default="")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    items = json.load(open(a.export, encoding="utf-8"))
    if a.chars:
        items = [i for i in items if i["char"] in a.chars]
    report = []
    for it in items:
        try:
            r = fix(it)
        except Exception as e:
            r = {"char": it["char"], "error": f"{type(e).__name__}: {e}"}
        if r.get("changed"):
            with open(os.path.join(a.out_dir, f"{it['char']}.json"), "w", encoding="utf-8") as f:
                json.dump(r.pop("data"), f, ensure_ascii=False)
        report.append(r)
    json.dump(report, open(os.path.join(a.out_dir, "_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    ok = [r for r in report if "before" in r]
    print(f"{len(report)} chars, {sum(r.get('changed', False) for r in report)} changed, "
          f"{sum(1 for r in report if 'error' in r)} errors")
    print(f"below {GOOD_IOU}: before {sum(r['before'] < GOOD_IOU for r in ok)}, after {sum(r['after'] < GOOD_IOU for r in ok)}")


if __name__ == "__main__":
    main()
