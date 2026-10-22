"""Per-segment features for the cutting model, plus training-time augmentation."""
from __future__ import annotations

import numpy as np

from .raster import render
from .svg_io import Seg, contour_polygon, signed_area

IMG = 256
MASK_RES = 512
CX, CY, SC = 512.0, 388.0, 512.0  # hw -> normalized coords
N_BUCKETS = 11
KINDS = {"L": 0, "Q": 1, "C": 2}


def norm_xy(p: np.ndarray) -> np.ndarray:
    return np.stack([(p[..., 0] - CX) / SC, (p[..., 1] - CY) / SC], -1)


def img_xy(p: np.ndarray) -> np.ndarray:
    """hw -> grid_sample coords in [-1, 1] (x right, y down)."""
    return np.stack([p[..., 0] / 512.0 - 1.0, (900.0 - p[..., 1]) / 512.0 - 1.0], -1)


# ---------------------------------------------------------------- augmentation
def augment(contours, labels: np.ndarray, rng: np.random.Generator):
    """Random affine + random curve subdivision. Returns (contours, labels)."""
    ang = np.radians(rng.uniform(-5, 5))
    sx, sy = rng.uniform(0.88, 1.12, 2)
    sh = rng.uniform(-0.1, 0.1)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    A = R @ np.array([[sx, sh], [0, sy]])
    c = np.array([512.0, 388.0])
    b = c - A @ c + rng.uniform(-30, 30, 2)
    out, lab_out, off = [], [], 0
    p_split = rng.choice([0.0, 0.1, 0.25])
    for cont in contours:
        new = []
        for s in cont:
            t = s.transformed(A, b)
            if p_split and rng.random() < p_split and t.length() > 12:
                a1, a2 = t.split(rng.uniform(0.35, 0.65))
                new += [a1, a2]
                lab_out += [labels[off], labels[off]]
            else:
                new.append(t)
                lab_out.append(labels[off])
            off += 1
        # random rotation of the contour start (keeps labels aligned)
        k = int(rng.integers(len(new)))
        n0 = len(lab_out) - len(new)
        seg_l = lab_out[n0:]
        out.append(new[k:] + new[:k])
        lab_out[n0:] = seg_l[k:] + seg_l[:k]
    return out, np.array(lab_out)


# ---------------------------------------------------------------- features
def _lookup(mask: np.ndarray, p_hw: np.ndarray) -> np.ndarray:
    s = MASK_RES / 1024.0
    x = np.clip((p_hw[..., 0] * s).astype(int), 0, MASK_RES - 1)
    y = np.clip(((900.0 - p_hw[..., 1]) * s).astype(int), 0, MASK_RES - 1)
    return mask[y, x]


def _thickness(mask, pts, normals, max_d=420.0, step=4.0, auto_inward=True):
    """Distance from boundary points along a direction until leaving the glyph.

    With auto_inward the direction is flipped to point into the glyph."""
    if auto_inward:
        inside = _lookup(mask, pts + 4 * normals) >= _lookup(mask, pts - 4 * normals)
        n = np.where(inside[:, None], normals, -normals)
    else:
        n = normals
    ts = np.arange(step, max_d + step, step)
    ray = pts[:, None, :] + ts[None, :, None] * n[:, None, :]
    occ = _lookup(mask, ray) > 0.5
    occ[:, 0] = True  # we start on the boundary
    out = np.argmin(occ, axis=1).astype(float)
    out[occ.all(1)] = len(ts)
    return out * step / 1024.0, n


def glyph_features(contours) -> dict:
    """Compute model inputs for one glyph (contours already in hw coords)."""
    segs = [s for c in contours for s in c]
    N = len(segs)
    cid = np.concatenate([np.full(len(c), i) for i, c in enumerate(contours)])
    idx = np.concatenate([np.arange(len(c)) for c in contours])
    clen = np.concatenate([np.full(len(c), len(c)) for c in contours])
    offs = np.cumsum([0] + [len(c) for c in contours])[:-1]
    nxt = np.array([offs[cid[i]] + (idx[i] + 1) % clen[i] for i in range(N)])
    prv = np.array([offs[cid[i]] + (idx[i] - 1) % clen[i] for i in range(N)])

    cub = np.stack([s.cubic() for s in segs])  # (N, 4, 2)
    mid = np.stack([s.sample(3)[1] for s in segs])
    lens = np.array([s.length() for s in segs])
    t0 = np.stack([s.tangent_start() for s in segs])
    t1 = np.stack([s.tangent_end() for s in segs])
    kind = np.eye(3)[[KINDS[s.kind] for s in segs]]

    # turning at the start vertex (prev end -> this start) and at the end vertex
    def turn(a, b):
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        dot = (a * b).sum(1)
        return np.stack([cross, dot], 1)

    turn_s = turn(t1[prv], t0)
    turn_e = turn(t1, t0[nxt])

    # contour-level
    polys = [contour_polygon(c, 4.0) for c in contours]
    areas = np.array([signed_area(p) for p in polys])
    outer_sign = np.sign(areas[np.argmax(np.abs(areas))]) or 1.0
    hole = (np.sign(areas) != outer_sign).astype(float)[cid]
    carea = np.log1p(np.abs(areas))[cid] / 14.0
    cum = np.concatenate([np.cumsum(np.concatenate([[0], [s.length() for s in c][:-1]])) / max(1e-6, sum(s.length() for s in c)) for c in contours])
    per = np.array([sum(s.length() for s in c) for c in contours])[cid] / 4096.0

    # thickness probes from the raster
    mask = render(contours, MASK_RES)
    nrm_mid = np.stack([s.sample(5)[3] - s.sample(5)[1] for s in segs])
    nrm_mid = nrm_mid / (np.linalg.norm(nrm_mid, axis=1, keepdims=True) + 1e-9)
    nrm_mid = np.stack([-nrm_mid[:, 1], nrm_mid[:, 0]], 1)
    th_mid, inward = _thickness(mask, mid, nrm_mid)
    # at the start vertex: probe along the bisector of the two inward normals
    n_prev = np.stack([-t1[prv][:, 1], t1[prv][:, 0]], 1)
    n_this = np.stack([-t0[:, 1], t0[:, 0]], 1)
    sgn = np.sign((inward * nrm_mid).sum(1))[:, None]
    bis = (n_prev + n_this) * sgn
    bis = bis / (np.linalg.norm(bis, axis=1, keepdims=True) + 1e-9)
    th_vtx, _ = _thickness(mask, cub[:, 0], bis)
    # tangential probes from the start vertex (does the glyph continue straight on?)
    th_fwd, _ = _thickness(mask, cub[:, 0], t1[prv], auto_inward=False)
    th_bwd, _ = _thickness(mask, cub[:, 0], -t0, auto_inward=False)

    f = np.concatenate(
        [
            norm_xy(cub).reshape(N, 8),
            norm_xy(mid),
            (lens / 1024.0)[:, None],
            np.log(lens + 1.0)[:, None] / 7.0,
            t0, t1, turn_s, turn_e, kind,
            hole[:, None], carea[:, None], per[:, None], (clen / 100.0)[:, None],
            np.stack([np.sin(2 * np.pi * cum), np.cos(2 * np.pi * cum)], 1),
            th_mid[:, None], th_vtx[:, None], th_fwd[:, None], th_bwd[:, None],
            sgn * inward,  # inward normal (orientation-free)
        ],
        1,
    ).astype(np.float32)

    # relative position buckets
    same = cid[:, None] == cid[None, :]
    delta = (idx[None, :] - idx[:, None]) % clen[:, None]
    delta = np.where(delta > clen[:, None] / 2, delta - clen[:, None], delta)
    ad = np.abs(delta)
    b = np.where(ad <= 3, delta + 3, np.where(ad <= 6, 7, np.where(ad <= 12, 8, 9)))
    bucket = np.where(same, b, 10).astype(np.int8)

    img = render(contours, IMG)
    probe = img_xy(np.stack([cub[:, 0], mid, cub[:, 3], mid + inward * th_mid[:, None] * 512.0], 1)).astype(np.float32)
    return dict(feat=f, bucket=bucket, img=img[None].astype(np.float32), probe=probe, nxt=nxt, prv=prv)


N_FEAT = 35


def targets(contours, labels: np.ndarray, nxt: np.ndarray) -> dict:
    lab = np.asarray(labels)
    valid = lab >= 0
    cut = (lab != lab[nxt]).astype(np.float32)
    cut_mask = (valid & valid[nxt]).astype(np.float32)
    return dict(lab=lab.astype(np.int64), cut=cut, cut_mask=cut_mask)
