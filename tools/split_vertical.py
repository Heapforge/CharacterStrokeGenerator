#!/usr/bin/env python3
"""Cut a glyph along a vertical line into a left and a right stroke (stem | bowl of R, я, ʀ, ...).

Straight segments that cross the line are split there (collinear pieces, so the outline is
unchanged); every other segment goes to the side its midpoint is on. Segments lying on the line
itself only bound the ink on one side and are dropped: the cut closes each side with straight
bridges along the line instead (flat cut ends). Counters go to the side they lie on.

    from tools.split_vertical import split_vertical
    left, right = split_vertical(contours, 191.5)          # hw path strings
"""
from __future__ import annotations

import numpy as np

from strokegen.svg_io import Seg, segs_to_path

EPS = 0.5


def _split_at(seg: Seg, xc: float) -> list[Seg]:
    if seg.kind != "L":
        return [seg]
    (x0, y0), (x1, y1) = seg.pts
    if (x0 - xc) * (x1 - xc) >= 0 or abs(x1 - x0) < 1e-9:
        return [seg]
    t = (xc - x0) / (x1 - x0)
    m = np.array([xc, y0 + t * (y1 - y0)])
    return [Seg("L", np.array([seg.pts[0], m])), Seg("L", np.array([m, seg.pts[1]]))]


def _side(seg: Seg, xc: float) -> int:
    xs = seg.pts[:, 0]
    if np.all(np.abs(xs - xc) < EPS):
        return 0  # on the cut line
    mid = seg.sample(3)[1][0] if hasattr(seg, "sample") else xs.mean()
    return -1 if mid < xc else 1


def _clip(contour: list[Seg], xc: float, side: int) -> str | None:
    segs = [p for s in contour for p in _split_at(s, xc)]
    keep = [_side(s, xc) == side for s in segs]
    if not any(keep):
        return None
    if all(keep):
        return segs_to_path(segs)
    # rotate so the list starts at the beginning of a kept run
    n = len(segs)
    start = next(i for i in range(n) if keep[i] and not keep[i - 1])
    segs, keep = segs[start:] + segs[:start], keep[start:] + keep[:start]
    chains, cur = [], []
    for s, k in zip(segs, keep):
        if k:
            cur.append(s)
        elif cur:
            chains.append(cur)
            cur = []
    if cur:
        chains.append(cur)
    out = []
    for i, ch in enumerate(chains):
        out.extend(ch)
        nxt = chains[(i + 1) % len(chains)][0].p0
        if np.linalg.norm(ch[-1].p1 - nxt) > 1e-6:
            out.append(Seg("L", np.array([ch[-1].p1, nxt])))
    return segs_to_path(out)


def split_vertical(contours, xc: float) -> tuple[str, str]:
    left, right = [], []
    for c in contours:
        for side, acc in ((-1, left), (1, right)):
            p = _clip(c, xc, side)
            if p:
                acc.append(p)
    return " ".join(left), " ".join(right)
