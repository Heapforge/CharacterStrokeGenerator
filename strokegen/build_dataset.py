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
FONT_PRIORITY = ["gkai00mp", "bkai00mp", "ukai-cn", "ukai-hk", "ukai-tw", "ukai-tw-mbe", "noto-sans"]
TOL = 1.01  # hw data is integer-rounded


def _key(p) -> tuple[int, int]:
    return (int(round(float(p[0]))), int(round(float(p[1]))))


def _vertex_match_frac(hw_pts: np.ndarray, font_pts: np.ndarray) -> float:
    d = np.abs(hw_pts[:, None, :] - font_pts[None, :, :]).max(-1).min(1)
    return float((d <= TOL).mean())


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


def decompose(contours: list[list[Seg]], hw: dict) -> dict | None:
    """Match hw strokes to font segments. Returns label dict or None if they don't correspond."""
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
    n_bridge_bad = 0
    n_reuse = 0
    for k, sd in enumerate(hw["strokes"]):
        sc = parse_hw_path(sd)
        loop = []
        for c in sc:
            for s in c:
                va, vb = vindex.near(s.p0), vindex.near(s.p1)
                hit = None
                if s.kind != "C":
                    for a in va:
                        for b in vb:
                            for n in by_pair.get((a, b), ()):
                                if hit is None and seg_stroke[n] == -1:
                                    hit = (n, False)
                            for n in by_pair.get((b, a), ()):
                                if hit is None and seg_stroke[n] == -1:
                                    hit = (n, True)
                if hit is not None and seg_stroke[hit[0]] == -1:
                    seg_stroke[hit[0]] = k
                    loop.append(("F", hit[0], hit[1]))
                else:
                    if hit is not None:
                        n_reuse += 1
                    if not va or not vb:
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
    return dict(
        seg_stroke=seg_stroke,
        loops=loops,
        coverage=float(covered),
        n_bridge_bad=n_bridge_bad,
        n_reuse=n_reuse,
    )


def outline_hash(contours) -> str:
    h = hashlib.sha1()
    for c in sorted(contours, key=lambda c: tuple(np.round(c[0].p0).astype(int))):
        for s in c:
            h.update(s.kind.encode())
            h.update(np.round(s.pts).astype(np.int32).tobytes())
    return h.hexdigest()


def process(item, min_coverage: float = 0.97):
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
    hw_pts = np.array(hw_pts)
    best = None
    for f in sorted(entry["fonts"], key=FONT_PRIORITY.index):
        svg = open(os.path.join(GLYPHS, f, f"{char}.svg"), encoding="utf-8").read()
        contours = load_svg_glyph(svg)
        if not contours:
            continue
        fp = np.array([s.p0 for c in contours for s in c])
        frac = _vertex_match_frac(hw_pts, fp)
        if best is None or frac > best[0] + 1e-9:
            best = (frac, f, contours)
        if frac >= 0.999:
            break
    if best is None or best[0] < 0.99:
        return char, None, "no_exact_font"
    _, font, contours = best
    lab = decompose(contours, hw)
    if lab["coverage"] < min_coverage:
        return char, None, "low_coverage"
    if lab["n_bridge_bad"] > 0:
        return char, None, "bridge_off_vertex"
    rec = dict(
        char=char,
        font=font,
        contours=contours,
        seg_stroke=lab["seg_stroke"],
        loops=lab["loops"],
        coverage=lab["coverage"],
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
    recs, seen, dup = [], set(), 0
    for _, r, s in sorted(results, key=lambda t: t[0]):
        if r is None:
            continue
        if r["hash"] in seen:
            dup += 1
            continue
        seen.add(r["hash"])
        recs.append(r)
    print(f"kept {len(recs)} records ({dup} duplicate outlines dropped)")
    print("fonts:", dict(Counter(r["font"] for r in recs)))
    full = sum(1 for r in recs if (r["seg_stroke"] >= 0).all())
    print(f"fully covered: {full}/{len(recs)}")
    rng = random.Random(0)
    rng.shuffle(recs)
    n = len(recs)
    n_val = n_test = max(1, n // 20)
    splits = {"test": recs[:n_test], "val": recs[n_test:n_test + n_val], "train": recs[n_test + n_val:]}
    os.makedirs(OUT, exist_ok=True)
    for name, rs in splits.items():
        with open(os.path.join(OUT, f"{name}.pkl"), "wb") as f:
            pickle.dump(rs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(name, len(rs))


if __name__ == "__main__":
    main()
