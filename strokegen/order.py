"""Stroke-level features and decoding for the order/direction model (Stage C)."""
from __future__ import annotations

import numpy as np

from .medians import resample, stroke_median, stroke_polygon

N_MED = 16
CX, CY, SC = 512.0, 388.0, 512.0


def canonical(median: np.ndarray) -> np.ndarray:
    """Deterministic orientation (start = the more top-left end); the model predicts whether to flip it."""
    a, b = median[0], median[-1]
    return median if (a[0] - a[1]) <= (b[0] - b[1]) else median[::-1]


def stroke_record(loops) -> dict:
    """Geometry the order model needs for one stroke."""
    med = canonical(stroke_median(loops))
    poly = stroke_polygon(loops)
    area = poly.area if poly is not None else 1.0
    if poly is not None:
        x0, y0, x1, y1 = poly.bounds
    else:
        x0, y0 = med.min(0)
        x1, y1 = med.max(0)
    return dict(median=med, area=float(area), bbox=np.array([x0, y0, x1, y1], float))


def stroke_features(strokes: list[dict]) -> np.ndarray:
    """(S, F) float32 features for a glyph's strokes (records from stroke_record)."""
    out = []
    for st in strokes:
        m = st["median"].astype(float)
        r = resample(m, N_MED)
        rn = np.stack([(r[:, 0] - CX) / SC, (r[:, 1] - CY) / SC], 1)
        seg = np.linalg.norm(np.diff(m, axis=0), axis=1)
        L = max(seg.sum(), 1.0)
        d0 = r[1] - r[0]
        d1 = r[-1] - r[-2]
        d0 = d0 / (np.hypot(*d0) + 1e-9)
        d1 = d1 / (np.hypot(*d1) + 1e-9)
        x0, y0, x1, y1 = st["bbox"]
        bb = np.array([(x0 - CX) / SC, (y0 - CY) / SC, (x1 - CX) / SC, (y1 - CY) / SC, (x1 - x0) / SC, (y1 - y0) / SC])
        width = st["area"] / L
        out.append(np.concatenate([rn.reshape(-1), d0, d1, bb, [L / 1024.0, width / 100.0, np.log1p(st["area"]) / 12.0, len(m) / 10.0]]))
    return np.array(out, np.float32)


N_SFEAT = 2 * N_MED + 4 + 6 + 4


def augment_strokes(strokes: list[dict], rng: np.random.Generator) -> list[dict]:
    ang = np.radians(rng.uniform(-4, 4))
    sx, sy = rng.uniform(0.9, 1.1, 2)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ np.diag([sx, sy])
    c = np.array([CX, CY])
    t = rng.uniform(-25, 25, 2)
    out = []
    for st in strokes:
        m = (st["median"] - c) @ R.T + c + t + rng.normal(0, 2.0, st["median"].shape)
        bb = st["bbox"].reshape(2, 2)
        corners = np.array([[bb[0, 0], bb[0, 1]], [bb[1, 0], bb[0, 1]], [bb[0, 0], bb[1, 1]], [bb[1, 0], bb[1, 1]]])
        corners = (corners - c) @ R.T + c + t
        out.append(dict(median=m, area=st["area"] * sx * sy, bbox=np.r_[corners.min(0), corners.max(0)]))
    return out


def decode_order(prec_logits: np.ndarray, refine: bool = True) -> np.ndarray:
    """Most likely total order under pairwise precedence probabilities.

    Starts from a soft Borda count, then improves sum_{i before j} log P(i before j) by
    moving single strokes (insertion local search)."""
    P = 1.0 / (1.0 + np.exp(-np.clip(prec_logits, -50, 50)))
    np.fill_diagonal(P, 0.0)
    order = list(np.argsort(-P.sum(1), kind="stable"))
    if not refine or len(order) < 3:
        return np.array(order)
    L = np.log(np.clip(P, 1e-6, 1.0))
    np.fill_diagonal(L, 0.0)
    D = L - L.T  # gain of putting i before j instead of j before i
    n = len(order)
    improved = True
    while improved:
        improved = False
        for a in range(n):
            x = order[a]
            rest = order[:a] + order[a + 1:]
            # score of inserting x at position p relative to inserting at the front:
            # each element placed before x flips its pair from (x, y) to (y, x)
            gains = np.concatenate([[0.0], np.cumsum([-D[x, y] for y in rest])])
            p = int(np.argmax(gains))
            if p != a and gains[p] > gains[a] + 1e-9:
                order = rest[:p] + [x] + rest[p:]
                improved = True
    return np.array(order)


def kendall_tau(pred_order: np.ndarray, gt_rank: np.ndarray) -> float:
    """pred_order: list of stroke ids in predicted order; gt_rank: GT position of each stroke id."""
    r = gt_rank[pred_order]
    n = len(r)
    if n < 2:
        return 1.0
    s = 0
    for i in range(n):
        s += np.sign(r[i + 1:] - r[i]).sum()
    return float(s / (n * (n - 1) / 2))


def _seg_cross(p, q) -> bool:
    """Do polylines p and q properly intersect?"""
    def cr(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    a, b = p[:-1, None], p[1:, None]
    c, d = q[None, :-1], q[None, 1:]
    d1, d2 = cr(b - a, c - a), cr(b - a, d - a)
    d3, d4 = cr(d - c, a - c), cr(d - c, b - c)
    return bool(((d1 * d2 < 0) & (d3 * d4 < 0)).any())


def pair_features(strokes: list[dict]) -> np.ndarray:
    """(S, S, N_PFEAT) relational features between strokes."""
    S = len(strokes)
    meds = [resample(st["median"].astype(float), 12) for st in strokes]
    cen = np.array([m.mean(0) for m in meds])
    st_ = np.array([m[0] for m in meds])
    en_ = np.array([m[-1] for m in meds])
    bb = np.array([st["bbox"] for st in strokes])
    out = np.zeros((S, S, N_PFEAT), np.float32)
    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            ox = max(0.0, min(bb[i, 2], bb[j, 2]) - max(bb[i, 0], bb[j, 0]))
            oy = max(0.0, min(bb[i, 3], bb[j, 3]) - max(bb[i, 1], bb[j, 1]))
            wi, hi = bb[i, 2] - bb[i, 0] + 1, bb[i, 3] - bb[i, 1] + 1
            wj, hj = bb[j, 2] - bb[j, 0] + 1, bb[j, 3] - bb[j, 1] + 1
            dmin = np.linalg.norm(meds[i][:, None] - meds[j][None], axis=-1).min()
            out[i, j] = np.concatenate([
                (cen[i] - cen[j]) / SC,
                (st_[i] - st_[j]) / SC, (en_[i] - en_[j]) / SC,
                (st_[i] - en_[j]) / SC, (en_[i] - st_[j]) / SC,
                [dmin / 256.0, ox / min(wi, wj), oy / min(hi, hj), ox * oy / (wi * hi + wj * hj - ox * oy),
                 float(_seg_cross(strokes[i]["median"], strokes[j]["median"]))],
            ])
    return out


N_PFEAT = 15
