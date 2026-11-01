"""Build supervised training records from font glyph SVGs + hanzi-writer stroke data.

For each plain {char}.json whose strokes were cut from one of the exported font outlines,
assign every outline segment to the stroke that uses it and record the bridge segments
that close each stroke.

usage: python -m strokegen.build_dataset [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
from collections import Counter

import numpy as np

from .svg_io import Seg, load_svg_glyph, parse_hw_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLYPHS = os.path.join(ROOT, "training_data", "font_glyphs")
HWDATA = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
OUT = os.path.join(ROOT, "data", "processed")
FONT_PRIORITY = [
    "gkai00mp", "bkai00mp", "ukai-cn", "ukai-hk", "ukai-tw", "ukai-tw-mbe",
    "noto-sans", "ubuntu", "liberation-serif", "merriweather",
    "noto-sans-cjk-sc", "noto-sans-cjk-hk", "noto-sans-cjk-tc", "noto-sans-cjk-jp", "noto-sans-cjk-kr",
    "noto-serif-cjk-sc", "noto-serif-cjk-hk", "noto-serif-cjk-tc", "noto-serif-cjk-jp", "noto-serif-cjk-kr",
]
TOL = 1.01  # hw data is integer-rounded
SHAPE_TOL = 5.0  # max distance between an hw segment and a font segment to count as the same one
EDGE_TOL = 2.0  # max distance of an off-vertex bridge end from the font outline
# Quality gates. Strokes cut straight from a font outline score ~1.0 on both; hand-corrected
# strokes that split font segments mid-edge score lower (>= 0.75 / 0.62 in the current data);
# hw data drawn from a different glyph scores < 0.25 / 0.15.
MIN_VERTEX_MATCH = 0.7  # share of hw vertices that are font vertices
MIN_COVERAGE = 0.6  # share of font segments used verbatim by some hw stroke


def _key(p) -> tuple[int, int]:
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def _vertex_match_frac(hw_pts: np.ndarray, font_pts: np.ndarray) -> float:
    d = np.abs(hw_pts[:, None, :] - font_pts[None, :, :]).max(-1).min(1)
    return float((d <= TOL).mean())


def _edge_dist(pts: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distance from each of pts to the nearest of the line pieces a[i] -> a[i] + d[i]."""
    t = np.clip(((pts[:, None] - a) * d).sum(-1) / np.maximum((d**2).sum(-1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + t[..., None] * d - pts[:, None], axis=-1).min(1)


def _path_dist(pts: np.ndarray, poly: np.ndarray) -> float:
    """Largest distance from any of pts to the polyline poly."""
    return float(_edge_dist(pts, poly[:-1], np.diff(poly, axis=0)).max())


class _VertexIndex:
    """Find font vertices near a point (tolerates +-1 rounding)."""

    def __init__(self, pts: np.ndarray):
        self.pts = pts
        self.grid: dict[tuple[int, int], list[int]] = {}
        for i, p in enumerate(pts):
            self.grid.setdefault(_key(p), []).append(i)

    def near(self, p) -> list[int]:
        kx, ky = _key(p)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for i in self.grid.get((kx + dx, ky + dy), ()):
                    if np.abs(self.pts[i] - p).max() <= TOL:
                        out.append(i)
        return out


def decompose(contours: list[list[Seg]], hw: dict) -> dict:
    """Match hw strokes to font segments. Returns the labels plus match-quality counters for process()."""
    # vertex table: vertex v = start point of segment (ci, si)
    seg_ids = [(ci, si) for ci, c in enumerate(contours) for si in range(len(c))]
    vpts = np.array([contours[ci][si].p0 for ci, si in seg_ids])
    vindex = _VertexIndex(vpts)
    flat = {sid: n for n, sid in enumerate(seg_ids)}

    def next_vertex(n):
        ci, si = seg_ids[n]
        return flat[(ci, (si + 1) % len(contours[ci]))]

    # segment n goes from vertex n to vertex next(n)
    seg_end = np.array([next_vertex(n) for n in range(len(seg_ids))])
    by_pair: dict[tuple[int, int], list[int]] = {}
    for n in range(len(seg_ids)):
        by_pair.setdefault((n, int(seg_end[n])), []).append(n)

    seg_stroke = np.full(len(seg_ids), -1, np.int64)
    loops = []  # per stroke: list of ("F", n, reversed) | ("B", Seg)
    n_bridge_bad = 0  # bridge ends off the font outline
    n_bridge_on_edge = 0  # bridge ends on a font edge but not at a vertex
    # outline as line pieces (a -> b), for point-to-outline distance
    poly = [s.flatten(0.5) for c in contours for s in c]
    edge_a = np.concatenate([p[:-1] for p in poly])
    edge_d = np.concatenate([p[1:] for p in poly]) - edge_a

    def outline_dist(p) -> float:
        return float(_edge_dist(np.asarray(p, float)[None], edge_a, edge_d)[0])

    def same_shape(s: Seg, n: int) -> bool:
        # endpoints already match; also require the two paths to agree so a straight hw bridge
        # is not mistaken for a font curve between the same two vertices (and vice versa).
        # Compared geometrically, not by parameter: hw data often writes lines as degenerate Qs.
        ci, si = seg_ids[n]
        f = contours[ci][si]
        return max(_path_dist(s.sample(5)[1:-1], f.flatten(0.5)),
                   _path_dist(f.sample(5)[1:-1], s.flatten(0.5))) <= SHAPE_TOL

    for k, sd in enumerate(hw["strokes"]):
        sc = parse_hw_path(sd)
        loop = []
        for c in sc:
            for s in c:
                va, vb = vindex.near(s.p0), vindex.near(s.p1)
                hit = None
                for a in va:
                    for b in vb:
                        for n in by_pair.get((a, b), ()):
                            if hit is None and seg_stroke[n] == -1 and same_shape(s, n):
                                hit = (n, False)
                        for n in by_pair.get((b, a), ()):
                            if hit is None and seg_stroke[n] == -1 and same_shape(s, n):
                                hit = (n, True)
                if hit is not None:
                    seg_stroke[hit[0]] = k
                    loop.append(("F", hit[0], hit[1]))
                else:
                    # a bridge end off every font vertex is fine if it sits on a font edge (a
                    # hand-made cut partway along a segment); off the outline means the GT
                    # was not cut from this glyph
                    for p, v in ((s.p0, va), (s.p1, vb)):
                        if not v:
                            if outline_dist(p) <= EDGE_TOL:
                                n_bridge_on_edge += 1
                            else:
                                n_bridge_bad += 1
                    # snap bridge ends to exact font vertices
                    pts = s.pts.copy()
                    if va:
                        pts[0] = vpts[va[0]]
                    if vb:
                        pts[-1] = vpts[vb[0]]
                    loop.append(("B", Seg(s.kind, pts), va[0] if va else -1, vb[0] if vb else -1))
        loops.append(loop)

    covered = (seg_stroke >= 0).mean()
    # fill isolated gaps whose neighbours along the contour agree
    for n in np.where(seg_stroke < 0)[0]:
        ci, si = seg_ids[n]
        L = len(contours[ci])
        prv = seg_stroke[flat[(ci, (si - 1) % L)]]
        nxt = seg_stroke[flat[(ci, (si + 1) % L)]]
        if prv >= 0 and prv == nxt:
            seg_stroke[n] = prv
    # anything still -1 stays unknown: features.targets masks it out of the cut loss
    return dict(
        seg_stroke=seg_stroke,
        loops=loops,
        coverage=float(covered),
        n_bridge_bad=n_bridge_bad,
        n_bridge_on_edge=n_bridge_on_edge,
    )


def outline_hash(contours) -> str:
    h = hashlib.sha1()
    for c in sorted(contours, key=lambda c: tuple(np.round(c[0].p0).astype(int))):
        for s in c:
            h.update(s.kind.encode())
            h.update(np.round(s.pts).astype(np.int32).tobytes())
    return h.hexdigest()


def process(item, min_coverage: float = MIN_COVERAGE, min_vertex_match: float = MIN_VERTEX_MATCH):
    char, entry = item
    path = os.path.join(HWDATA, f"{char}.json")
    if f"{char}.json" not in entry["data_files"] or not os.path.exists(path):
        return char, None, "no_plain_json"
    hw = json.load(open(path, encoding="utf-8"))
    if not hw.get("strokes"):
        return char, None, "empty_hw"
    hw_pts = []
    for sd in hw["strokes"]:
        for c in parse_hw_path(sd):
            hw_pts += [s.p0 for s in c]
    if not hw_pts:
        return char, None, "empty_hw"
    hw_pts = np.array(hw_pts)
    best = None
    for f in sorted(entry["fonts"], key=FONT_PRIORITY.index):
        svg_path = os.path.join(GLYPHS, f, f"{char}.svg")
        if not os.path.exists(svg_path):
            continue
        svg = open(svg_path, encoding="utf-8").read()
        contours = load_svg_glyph(svg)
        if not contours:
            continue
        fp = np.array([s.p0 for c in contours for s in c])
        frac = _vertex_match_frac(hw_pts, fp)
        if best is None or frac > best[0] + 1e-9:
            best = (frac, f, contours)
        if frac >= 0.999:
            break
    if best is None:
        return char, None, "no_font_svg"
    frac, font, contours = best
    if frac < min_vertex_match:
        return char, None, "no_exact_font"
    lab = decompose(contours, hw)
    if lab["coverage"] < min_coverage:
        return char, None, "low_coverage"
    if lab["n_bridge_bad"] > 0:
        return char, None, "bridge_off_outline"
    rec = dict(
        char=char,
        font=font,
        contours=contours,
        seg_stroke=lab["seg_stroke"],
        loops=lab["loops"],
        coverage=lab["coverage"],
        vertex_match=frac,
        # GT is an exact partition of the outline segments (no mid-edge cuts, nothing unassigned)
        exact=bool(frac >= 0.99 and lab["coverage"] >= 0.97 and lab["n_bridge_on_edge"] == 0),
        medians=[np.array(m, float) for m in hw["medians"]],
        strokes_gt=hw["strokes"],
        hash=outline_hash(contours),
    )
    return char, rec, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    man = json.load(open(os.path.join(GLYPHS, "manifest.json"), encoding="utf-8"))["characters"]
    items = [(c, e) for c, e in man.items() if e["fonts"]]
    if args.limit:
        random.Random(0).shuffle(items)
        items = items[: args.limit]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(process, items, chunksize=32)
    status = Counter(s for _, _, s in results)
    print("status:", dict(status))
    recs = [r for _, r, _ in sorted(results, key=lambda t: t[0]) if r is not None]
    print(f"kept {len(recs)} records")
    print("fonts:", dict(Counter(r["font"] for r in recs)))
    full = sum(1 for r in recs if (r["seg_stroke"] >= 0).all())
    print(f"fully covered: {full}/{len(recs)}")
    # characters with identical outlines (e.g. Latin P / Greek Rho) are all kept, but go to the
    # same split so a test glyph is never also a training glyph
    groups: dict[str, list[dict]] = {}
    for r in recs:
        groups.setdefault(r["hash"], []).append(r)
    print(f"{len(groups)} distinct outlines ({len(recs) - len(groups)} records share an outline)")
    order = list(groups.values())
    random.Random(0).shuffle(order)
    n = len(recs)
    n_val = n_test = max(1, n // 20)
    splits = {"test": [], "val": [], "train": []}
    for g in order:
        name = "test" if len(splits["test"]) < n_test else "val" if len(splits["val"]) < n_val else "train"
        splits[name] += g
    os.makedirs(OUT, exist_ok=True)
    for name, rs in splits.items():
        with open(os.path.join(OUT, f"{name}.pkl"), "wb") as f:
            pickle.dump(rs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(name, len(rs))


if __name__ == "__main__":
    main()
