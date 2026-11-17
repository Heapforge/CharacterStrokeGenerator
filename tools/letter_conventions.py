#!/usr/bin/env python3
"""Stroke conventions for letters the model splits wrongly, applied by cutting the outline along vertical lines.

    E family  (E, Ǝ, ᴇ, ...):  | stem (top to bottom), then the bars top, middle, bottom (left to right)
    N family  (N, ɴ, ᴺ, ...):  | left stem, then one stroke down the diagonal and up the right stem
    И family  (И, и, ...):     | left stem, then one stroke up the diagonal and down the right stem
    Ж family  (Ж, ж, Җ, ...):  )-  left arms with the bar to the stem, | stem, -(  right arms with the bar

`cut_vertical(contours, xs)` does the geometry: straight segments that cross a cut line are split there
(collinear pieces, so the outline is unchanged); segments lying on a cut line are dropped; the pieces between
two lines are closed with straight bridges along the lines. The cut points on each line are paired bottom-up
into ink intervals, so a notch next to the line stays background and every piece is a separate stroke
(the three bars of E are three strokes). A whole contour that a cut does not cross stays with the piece that
encloses it (a counter) or becomes its own piece (a mark).

    uv run python tools/letter_conventions.py in.json out.json [--chars ENИЖ] [--font noto-sans]

in.json: char -> {"strokes", "medians"} (or a list of {"char", ...}). Glyphs whose family is not recognised
or whose outline does not have the expected stems are passed through unchanged. Marks (accents) keep their
own stroke: above the letter first, below the letter last. Medians: stroke_median, oriented per convention,
then made hanzi-writer safe (median_reroute.hw_safe, median_repair.repair).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata

import numpy as np
from shapely.geometry import Point, Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.medians import stroke_median  # noqa: E402
from strokegen.svg_io import Seg, contour_polygon, load_svg_glyph, segs_to_path  # noqa: E402
from tools.median_repair import repair  # noqa: E402
from tools.median_reroute import hw_safe  # noqa: E402
from tools.reveal_check import MIN_AREA, MIN_SHARE, hidden  # noqa: E402

EPS = 0.5
MIN_STEM = 120.0  # a stem edge is a vertical straight run at least this long


# ---------------------------------------------------------------- geometry
def _split_at(seg: Seg, xc: float) -> list[Seg]:
    if seg.kind != "L":
        return [seg]
    (x0, y0), (x1, y1) = seg.pts
    if (x0 - xc) * (x1 - xc) >= 0 or abs(x1 - x0) < 1e-9 or min(abs(x0 - xc), abs(x1 - xc)) < EPS:
        return [seg]
    t = (xc - x0) / (x1 - x0)
    m = np.array([xc, y0 + t * (y1 - y0)])
    return [Seg("L", np.array([seg.pts[0], m])), Seg("L", np.array([m, seg.pts[1]]))]


def _slab(seg: Seg, xs: list[float]) -> int | None:
    """Index of the slab (between consecutive cut lines) the segment lies in; None when it lies on a line."""
    px = seg.pts[:, 0]
    for x in xs:
        if np.all(np.abs(px - x) < EPS):
            return None
    mid = float(np.mean(px)) if seg.kind == "L" else float(seg.pts[[0, -1], 0].mean() * 0.5 + px.mean() * 0.5)
    return int(np.searchsorted(xs, mid))


def _on_line(p, xs) -> int | None:
    for i, x in enumerate(xs):
        if abs(p[0] - x) < EPS:
            return i
    return None


def cut_vertical(contours: list[list[Seg]], xs: list[float]) -> list[dict]:
    """Pieces between vertical cut lines: [{"slab": k, "loops": [[Seg]], "poly": Polygon}]."""
    xs = sorted(xs)
    pieces, closed = [], []
    for contour in contours:
        segs = [s for seg in contour for s in _iter_split(seg, xs)]
        slabs = [_slab(s, xs) for s in segs]
        if all(k == slabs[0] for k in slabs) and slabs[0] is not None:
            closed.append((slabs[0], segs))
            continue
        n = len(segs)
        start = next((i for i in range(n) if slabs[i] is not None and slabs[i - 1] != slabs[i]), None)
        if start is None:
            raise ValueError("contour lies on a cut line")
        segs, slabs = segs[start:] + segs[:start], slabs[start:] + slabs[:start]
        chains, cur, ck = [], [], None
        for s, k in zip(segs, slabs):
            if k != ck and cur:
                if ck is not None:
                    chains.append((ck, cur))
                cur = []
            ck = k
            cur.append(s)
        if cur and ck is not None:
            chains.append((ck, cur))
        for k, ch in chains:
            pieces.append({"slab": k, "chain": ch})
    out = []
    for k in sorted({p["slab"] for p in pieces}):
        out.extend(_close_slab([p["chain"] for p in pieces if p["slab"] == k], xs, k))
    for k, segs in closed:  # whole contours: counters join the enclosing piece, others are pieces
        poly = _poly([segs])
        host = None
        for p in out:
            if p["slab"] == k and p["poly"].contains(poly.representative_point()) and p["poly"].area > poly.area:
                host = p if host is None or p["poly"].area < host["poly"].area else host
        if host is not None:
            host["loops"].append(segs)
            host["poly"] = _poly(host["loops"])
        else:
            out.append({"slab": k, "loops": [segs], "poly": poly})
    return out


def _iter_split(seg, xs):
    parts = [seg]
    for x in xs:
        parts = [q for p in parts for q in _split_at(p, x)]
    return parts


def _close_slab(chains, xs, k):
    """Join a slab's open chains into closed loops with bridges along the cut lines."""
    ends = []  # (line, y, kind, chain id)
    for ci, ch in enumerate(chains):
        a, b = ch[0].p0, ch[-1].p1
        la, lb = _on_line(a, xs), _on_line(b, xs)
        if la is None or lb is None:
            raise ValueError(f"chain end not on a cut line: {a} {b}")
        ends.append((la, a[1], "start", ci))
        ends.append((lb, b[1], "end", ci))
    partner = {}
    for line in {e[0] for e in ends}:
        pts = sorted((e for e in ends if e[0] == line), key=lambda e: e[1])
        if len(pts) % 2:
            raise ValueError("odd number of cut points on a line")
        for i in range(0, len(pts), 2):
            p, q = pts[i], pts[i + 1]
            if p[2] == q[2]:
                raise ValueError("cut interval joins two chain starts")
            s, e = (p, q) if p[2] == "end" else (q, p)
            partner[e[3]] = s[3]
    loops, used = [], set()
    for ci in range(len(chains)):
        if ci in used:
            continue
        loop, c = [], ci
        while c not in used:
            used.add(c)
            loop.extend(chains[c])
            nxt = partner[c]
            p0, p1 = chains[c][-1].p1, chains[nxt][0].p0
            if np.linalg.norm(p0 - p1) > 1e-6:
                loop.append(Seg("L", np.array([p0, p1])))
            c = nxt
        loops.append(loop)
    # loops may be nested (a piece with a counter): group each loop under its smallest container
    polys = [Polygon(contour_polygon(lp)).buffer(0) for lp in loops]
    pieces = []
    order = sorted(range(len(loops)), key=lambda i: -polys[i].area)
    for i in order:
        host = None
        for p in pieces:
            if p["poly"].contains(polys[i].representative_point()):
                host = p
        if host is not None and _orient(loops[i]) != _orient(host["loops"][0]):
            host["loops"].append(loops[i])
            host["poly"] = _poly(host["loops"])
        else:
            pieces.append({"slab": k, "loops": [loops[i]], "poly": polys[i]})
    return pieces


def _orient(loop):
    p = contour_polygon(loop)
    return np.sign(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def _poly(loops):
    g = None
    for lp in loops:
        q = Polygon(contour_polygon(lp)).buffer(0)
        g = q if g is None else g.symmetric_difference(q)
    return g


# ---------------------------------------------------------------- stems
def vertical_edges(contours, min_len=MIN_STEM):
    """x -> total length of vertical straight segments at that x (runs >= min_len), clustered within 2 units."""
    runs = []
    for c in contours:
        for s in c:
            if s.kind == "L" and abs(s.pts[0, 0] - s.pts[1, 0]) < EPS and abs(s.pts[1, 1] - s.pts[0, 1]) >= min_len:
                runs.append((float(s.pts[0, 0]), abs(s.pts[1, 1] - s.pts[0, 1])))
    out: dict[float, float] = {}
    for x, ln in sorted(runs):
        key = next((k for k in out if abs(k - x) < 2), x)
        out[key] = out.get(key, 0) + ln
    return dict(sorted(out.items()))


def _bbox(contours):
    p = np.concatenate([s.pts for c in contours for s in c])
    return p.min(0), p.max(0)


# ---------------------------------------------------------------- families
_N = {"N", "Ν", "ɴ", "ᴺ", "ᶰ", "ℕ", "Ꞑ", "Ꞥ"}
_I = {"И", "и", "ᴎ"}
_E = {"E", "ᴇ", "ᴱ", "Ǝ", "ᴲ", "ⱻ", "Е", "Ε"}
_ZHE = {"Ж", "ж", "Җ", "җ", "ᷬ"}


def family(char: str) -> str | None:
    """Convention family of a character: its base letter (NFD/compat decomposition) or its Unicode name."""
    base = unicodedata.normalize("NFKD", char)[:1] or char
    try:
        name = unicodedata.name(char)
    except ValueError:
        name = ""
    for fam, members, pat in (("N", _N, None), ("I", _I, None), ("E", _E, None), ("ZHE", _ZHE, "ZHE")):
        if char in members or base in members or (pat and pat in name.split()):
            return fam
    if ("CAPITAL LETTER N WITH" in name and "LONG RIGHT LEG" not in name) or name.endswith("SMALL CAPITAL N"):
        return "N"
    if "CYRILLIC" in name and ("LETTER I WITH" in name or "LETTER SHORT I" in name):
        return "I"
    if "LETTER E WITH" in name and "CAPITAL" in name:
        return "E"
    return None


def _letter_contours(contours):
    """Split contours into (letter, marks above, marks below): the letter is the largest connected group."""
    polys = [Polygon(contour_polygon(c)).buffer(0) for c in contours]
    big = max(range(len(contours)), key=lambda i: polys[i].area)
    lo, hi = polys[big].bounds[1], polys[big].bounds[3]
    letter, above, below = [], [], []
    for i, c in enumerate(contours):
        if i == big or polys[big].contains(polys[i].representative_point()) or polys[big].intersects(polys[i]):
            letter.append(c)
            continue
        b = polys[i].bounds
        cy = (b[1] + b[3]) / 2
        # a contour beside the letter's vertical extent (not above/below) belongs to the letter (e.g. Җ hook)
        (above if cy > hi else below if cy < lo else letter).append(c)
    return letter, above, below


def _plan(fam, letter):
    """Cut lines + per-piece role for a family, or None if the outline does not fit."""
    edges = vertical_edges(letter)
    (x0, y0), (x1, y1) = _bbox(letter)
    w, h = x1 - x0, y1 - y0
    xs_e = list(edges)
    if fam in ("N", "I"):
        # inner edge of the left stem: a long vertical run right of the outer edge, in the left third
        left = [x for x in xs_e if x0 + 2 < x < x0 + 0.35 * w]
        return [max(left)] if left and any(abs(x - x0) < 2 for x in xs_e) else None
    if fam == "E":
        # stem on the left (E) or on the right (Ǝ): the outer edge runs the full height, the inner edge
        # is the x right next to it with the most vertical run length
        outer_l = any(abs(x - x0) < 2 and ln > 0.8 * h for x, ln in edges.items())
        outer_r = any(abs(x - x1) < 2 and ln > 0.8 * h for x, ln in edges.items())
        if outer_l and not outer_r:
            cand = [x for x in xs_e if x0 + 2 < x < x0 + 0.45 * w]
        elif outer_r and not outer_l:
            cand = [x for x in xs_e if x1 - 0.45 * w < x < x1 - 2]
        else:
            return None
        return [max(cand, key=lambda x: edges[x])] if cand else None
    if fam == "ZHE":
        mid = [x for x in xs_e if x0 + 0.3 * w < x < x1 - 0.3 * w]
        if len(mid) < 2:
            return None
        cx = (x0 + x1) / 2
        a = max((x for x in mid if x < cx), default=None)
        b = min((x for x in mid if x > cx), default=None)
        return [a, b] if a is not None and b is not None else None
    return None


def _median(piece, start_hint):
    """Median of a piece, oriented to start at the end nearest start_hint."""
    m = np.asarray(stroke_median(piece["loops"]), float)
    if np.hypot(*(m[-1] - start_hint)) < np.hypot(*(m[0] - start_hint)):
        m = m[::-1]
    return m


def _straighten_tips(m, max_len=80.0, max_turn=20.0):
    """stroke_median bends the last few units of a tip into a corner of a flat stroke end; a hand-drawn
    median runs straight out instead (E's stem: [[145, 812], [145, 81]]). Replace such a bent tip by the
    straight continuation of the segment before it, keeping the tip's reach."""
    m = [np.asarray(p, float) for p in m]
    for _ in range(2):
        if len(m) >= 3:
            a, b, c = m[0], m[1], m[2]
            u, v = b - a, c - b
            if np.linalg.norm(u) < max_len:
                cosang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
                if np.degrees(np.arccos(np.clip(cosang, -1, 1))) > max_turn:
                    d = -v / (np.linalg.norm(v) + 1e-9)
                    reach = float(np.dot(a - b, d))
                    m = [b + d * max(reach, 0.0)] + m[1:] if reach > 5 else m[1:]
        m = m[::-1]
    return m


def _hidden_ok(path, m):
    a, share = hidden(path, m)
    return not (a > MIN_AREA and share > MIN_SHARE)


def _clamp_tips(loops, m, inset=4.0):
    """Pull an end point that lies outside the stroke back along its segment until it is inside."""
    g = _poly(loops).buffer(-inset)
    m = [np.asarray(p, float) for p in m]
    for i, j in ((0, 1), (-1, -2)):
        a, b = m[i], m[j]
        if g.contains(Point(*a)):
            continue
        for t in np.linspace(0.05, 0.95, 19):
            q = a + (b - a) * t
            if g.contains(Point(*q)):
                m[i] = q
                break
    return m


def _fix_median(path, m, loops=None):
    m = _straighten_tips(m)
    if loops is not None:
        m = _clamp_tips(loops, m)
    m = hw_safe(path, [[float(x), float(y)] for x, y in m])
    if not _hidden_ok(path, m):
        r = repair(path, m, detours=True)
        if _hidden_ok(path, r) and np.allclose(r[0], m[0]):
            m = r
    return [[int(round(x)), int(round(y))] for x, y in m]


def _bar_detour(m, piece, xc):
    """Ж: the arms' median runs tip -> junction -> tip; add the bar as junction -> cut line -> junction."""
    m = np.asarray(m, float)
    k = int(np.argmin(np.abs(m[:, 0] - xc)))
    # centre of the bar where it meets the cut line
    ys = [p[1] for lp in piece["loops"] for s in lp for p in s.pts if abs(p[0] - xc) < EPS]
    if not ys or k in (0, len(m) - 1):
        return m
    yb = (min(ys) + max(ys)) / 2
    end = np.array([xc + (12 if m[k, 0] > xc else -12), yb])
    return np.vstack([m[:k + 1], [end], m[k:]])


def apply(char, contours):
    """{strokes, medians} under the family convention, or None if not applicable."""
    fam = family(char)
    if fam is None:
        return None
    letter, above, below = _letter_contours(contours)
    xs = _plan(fam, letter)
    if not xs:
        return None
    pieces = cut_vertical(letter, xs)
    for p in pieces:
        b = p["poly"].bounds
        p["box"] = b
        p["cx"], p["cy"] = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    (gx0, gy0), (gx1, gy1) = _bbox(letter)
    strokes = []  # (loops, start hint)
    if fam in ("N", "I"):
        stem = [p for p in pieces if p["slab"] == 0]
        rest = [p for p in pieces if p["slab"] == 1]
        if len(stem) != 1 or len(rest) != 1:
            return None
        s, r = stem[0], rest[0]
        strokes.append((s["loops"], (s["cx"], gy1)))
        rb = r["box"]
        strokes.append((r["loops"], (rb[0], rb[3]) if fam == "N" else (rb[0], rb[1])))
    elif fam == "E":
        xc = xs[0]
        stem_slab = 0 if abs(xc - gx0) < abs(xc - gx1) and xc < (gx0 + gx1) / 2 else 1
        stem = [p for p in pieces if p["slab"] == stem_slab]
        bars = sorted((p for p in pieces if p["slab"] != stem_slab), key=lambda p: -p["cy"])
        if len(stem) != 1 or len(bars) != 3:
            return None
        strokes.append((stem[0]["loops"], (stem[0]["cx"], gy1)))
        for b in bars:
            strokes.append((b["loops"], (b["box"][0], b["cy"])))
    elif fam == "ZHE":
        L = [p for p in pieces if p["slab"] == 0]
        M = [p for p in pieces if p["slab"] == 1]
        R = [p for p in pieces if p["slab"] == 2]
        if not (len(L) == len(M) == len(R) == 1):
            return None
        strokes.append((L[0]["loops"], (L[0]["box"][0], L[0]["box"][3]), (L[0], xs[0])))
        strokes.append((M[0]["loops"], (M[0]["cx"], gy1)))
        strokes.append((R[0]["loops"], (R[0]["box"][2], R[0]["box"][3]), (R[0], xs[1])))
    marks_above = _marks(above)
    marks_below = _marks(below)
    out_s, out_m = [], []
    for loops, hint, *bar in marks_above + strokes + marks_below:
        path = " ".join(segs_to_path(lp) for lp in loops)
        m = _median({"loops": loops}, np.asarray(hint, float))
        if bar:
            m = _bar_detour(_straighten_tips(m), *bar[0])
        out_s.append(path)
        out_m.append(_fix_median(path, m, loops))
    return {"strokes": out_s, "medians": out_m, "family": fam, "cuts": [round(x, 1) for x in xs]}


def _marks(contours):
    """One stroke per mark (a contour plus the counters inside it), left to right."""
    if not contours:
        return []
    polys = [Polygon(contour_polygon(c)).buffer(0) for c in contours]
    groups = []
    for i in sorted(range(len(contours)), key=lambda i: -polys[i].area):
        host = next((g for g in groups if g["poly"].contains(polys[i].representative_point())), None)
        if host:
            host["loops"].append(contours[i])
        else:
            groups.append({"loops": [contours[i]], "poly": polys[i]})
    groups.sort(key=lambda g: g["poly"].bounds[0])
    return [(g["loops"], (g["poly"].bounds[0], g["poly"].bounds[3])) for g in groups]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--chars", default=None)
    ap.add_argument("--font", default="noto-sans")
    a = ap.parse_args()
    data = json.load(open(a.inp, encoding="utf-8"))
    if isinstance(data, list):
        data = {d["char"]: {"strokes": d["strokes"], "medians": d["medians"]} for d in data}
    chars = a.chars or "".join(data)
    changed = []
    for c in chars:
        p = os.path.join(ROOT, "training_data", "font_glyphs", a.font, c + ".svg")
        if not os.path.exists(p):
            continue
        try:
            r = apply(c, load_svg_glyph(open(p, encoding="utf-8").read()))
        except Exception as e:  # noqa: BLE001
            print(f"{c} U+{ord(c):04X}: failed: {e}")
            continue
        if r:
            data[c] = {"strokes": r["strokes"], "medians": r["medians"]}
            changed.append(c)
            print(f"{c} U+{ord(c):04X}: {r['family']} cuts {r['cuts']} -> {len(r['strokes'])} strokes")
    json.dump(data, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(changed)} glyphs rewritten: {''.join(changed)}")


if __name__ == "__main__":
    main()
