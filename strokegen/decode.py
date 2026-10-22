"""Turn a segment -> stroke assignment into closed stroke outlines.

Chains are maximal runs of consecutive outline segments that belong to the same stroke.
Each stroke is closed by visiting its chains in a cycle and joining the end of one chain
to the start of the next with a bridge segment. Outline segments are never modified, so
the union of the strokes covers exactly the original glyph.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .svg_io import Seg

C_RATIO = 0.85  # control point distance / chord, fit on GT bridges
L_MAX_TURN = 8.0  # degrees: use a straight bridge when both tangents line up with the chord


@dataclass
class Chain:
    stroke: int
    segs: list[int]  # flat segment indices, in contour order
    closed: bool  # whole contour


def flat_segments(contours) -> list[Seg]:
    return [s for c in contours for s in c]


def contour_offsets(contours) -> list[int]:
    off, out = 0, []
    for c in contours:
        out.append(off)
        off += len(c)
    return out


def build_chains(contours, seg_stroke: np.ndarray) -> list[Chain]:
    chains = []
    for ci, (c, off) in enumerate(zip(contours, contour_offsets(contours))):
        lab = np.asarray(seg_stroke[off:off + len(c)])
        n = len(c)
        if (lab == lab[0]).all():
            chains.append(Chain(int(lab[0]), list(range(off, off + n)), True))
            continue
        # start right after a cut
        start = next(i for i in range(n) if lab[i] != lab[i - 1])
        cur = [start]  # local indices within the contour
        for k in range(1, n):
            i = (start + k) % n
            if lab[i] == lab[cur[-1]]:
                cur.append(i)
            else:
                chains.append(Chain(int(lab[cur[0]]), [off + j for j in cur], False))
                cur = [i]
        chains.append(Chain(int(lab[cur[0]]), [off + j for j in cur], False))
    return chains


def make_bridge(a_end: np.ndarray, t_in: np.ndarray, b_start: np.ndarray, t_out: np.ndarray) -> Seg:
    """Bridge from a chain end (incoming tangent t_in) to the next chain start (outgoing tangent t_out)."""
    d = b_start - a_end
    L = float(np.hypot(*d))
    if L < 1e-6:
        return Seg("L", np.stack([a_end, b_start]))
    u = d / L
    turn1 = np.degrees(np.arccos(np.clip(t_in @ u, -1, 1)))
    turn2 = np.degrees(np.arccos(np.clip(t_out @ u, -1, 1)))
    if turn1 < L_MAX_TURN and turn2 < L_MAX_TURN:
        return Seg("L", np.stack([a_end, b_start]))
    k = C_RATIO * L
    return Seg("C", np.stack([a_end, a_end + k * t_in, b_start - k * t_out, b_start]))


def _cycle_cost(order, ends, starts):
    return sum(np.hypot(*(starts[order[(i + 1) % len(order)]] - ends[order[i]])) for i in range(len(order)))


def order_chains(chs: list[Chain], segs: list[Seg]) -> list[int]:
    """Cyclic visiting order of open chains that minimizes total bridge length."""
    n = len(chs)
    if n <= 2:
        return list(range(n))
    ends = [segs[c.segs[-1]].p1 for c in chs]
    starts = [segs[c.segs[0]].p0 for c in chs]
    if n <= 7:
        best, best_cost = None, np.inf
        for perm in itertools.permutations(range(1, n)):
            o = (0,) + perm
            cost = _cycle_cost(o, ends, starts)
            if cost < best_cost:
                best, best_cost = list(o), cost
        return best
    # greedy nearest neighbour + 2-opt style improvement on the cycle
    left = set(range(1, n))
    o = [0]
    while left:
        j = min(left, key=lambda k: np.hypot(*(starts[k] - ends[o[-1]])))
        o.append(j)
        left.remove(j)
    improved = True
    while improved:
        improved = False
        base = _cycle_cost(o, ends, starts)
        for i in range(1, n):
            for j in range(i + 1, n):
                cand = o[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = _cycle_cost(cand, ends, starts)
                if c < base - 1e-9:
                    o, base, improved = cand, c, True
    return o


def _assemble(contours, seg_stroke: np.ndarray, n_strokes: int | None = None):
    """(per stroke closed loops, per stroke bridge segments)."""
    segs = flat_segments(contours)
    chains = build_chains(contours, seg_stroke)
    ids = sorted({c.stroke for c in chains if c.stroke >= 0})
    if n_strokes is None:
        n_strokes = (max(ids) + 1) if ids else 0
    out: list[list[list[Seg]]] = [[] for _ in range(n_strokes)]
    bridges: list[list[Seg]] = [[] for _ in range(n_strokes)]
    for k in ids:
        mine = [c for c in chains if c.stroke == k]
        for c in mine:
            if c.closed:
                out[k].append([segs[i] for i in c.segs])
        opn = [c for c in mine if not c.closed]
        if not opn:
            continue
        order = order_chains(opn, segs)
        loop: list[Seg] = []
        for idx, ci in enumerate(order):
            ch = opn[ci]
            nxt = opn[order[(idx + 1) % len(order)]]
            loop += [segs[i] for i in ch.segs]
            a, ta = segs[ch.segs[-1]].p1, segs[ch.segs[-1]].tangent_end()
            b, tb = segs[nxt.segs[0]].p0, segs[nxt.segs[0]].tangent_start()
            loop.append(make_bridge(a, ta, b, tb))
            bridges[k].append(loop[-1])
        out[k].append(loop)
    return out, bridges


def assemble_strokes(contours, seg_stroke: np.ndarray, n_strokes: int | None = None) -> list[list[list[Seg]]]:
    """Return per stroke a list of closed loops (each a list of Seg). Stroke ids are kept (empty -> [])."""
    return _assemble(contours, seg_stroke, n_strokes)[0]


def fill_unassigned(contours, seg_stroke: np.ndarray) -> np.ndarray:
    """Give unassigned (-1) segments the label of the previous assigned segment on their contour."""
    lab = np.array(seg_stroke).copy()
    for c, off in zip(contours, contour_offsets(contours)):
        n = len(c)
        sub = lab[off:off + n]
        if (sub < 0).all():
            continue
        for _ in range(2):
            for i in range(n):
                if sub[i] < 0 and sub[i - 1] >= 0:
                    sub[i] = sub[i - 1]
        lab[off:off + n] = sub
    return lab


# ---------------------------------------------------------------- from model predictions
def chains_from_cuts(contours, cut_p: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Chain id per flat segment, cutting after segment i where cut_p[i] > thr."""
    ids = np.zeros(sum(len(c) for c in contours), np.int64)
    nid = 0
    for c, off in zip(contours, contour_offsets(contours)):
        n = len(c)
        cuts = np.asarray(cut_p[off:off + n]) > thr
        if not cuts.any():
            ids[off:off + n] = nid
            nid += 1
            continue
        start = (int(np.argmax(cuts)) + 1) % n  # first segment after a cut
        for k in range(n):
            i = (start + k) % n
            ids[off + i] = nid
            if cuts[i]:
                nid += 1
        if not cuts[(start - 1) % n]:
            nid += 1
    # compact ids
    _, ids = np.unique(ids, return_inverse=True)
    return ids


def cluster_chains(chain_id: np.ndarray, aff_p: np.ndarray, w: np.ndarray, thr: float = 0.5,
                   valid=None) -> np.ndarray:
    """Average-linkage agglomeration of chains on (length weighted) mean pairwise affinity.

    valid(list_of_chain_ids) -> bool can veto a merge."""
    K = int(chain_id.max()) + 1
    M = np.zeros((K, len(chain_id)))
    M[chain_id, np.arange(len(chain_id))] = w
    A = 0.5 * (aff_p + aff_p.T)
    S = M @ A @ M.T  # weighted affinity sums between chains
    W = M.sum(1)
    members = {k: [k] for k in range(K)}
    alive = list(range(K))
    vetoed: set[tuple[int, int]] = set()
    while len(alive) > 1:
        idx = np.array(alive)
        avg = S[np.ix_(idx, idx)] / np.outer(W[idx], W[idx])
        np.fill_diagonal(avg, -1)
        for x, y in vetoed:
            if x in members and y in members:
                avg[alive.index(x), alive.index(y)] = avg[alive.index(y), alive.index(x)] = -1
        a, b = np.unravel_index(np.argmax(avg), avg.shape)
        if avg[a, b] <= thr:
            break
        ka, kb = idx[a], idx[b]
        if valid is not None and not valid(members[ka] + members[kb]):
            vetoed.add((int(ka), int(kb)))
            continue
        S[ka, :] += S[kb, :]
        S[:, ka] += S[:, kb]
        W[ka] += W[kb]
        members[ka] += members.pop(kb)
        alive.remove(kb)
    lab_of_chain = np.zeros(K, np.int64)
    for new, k in enumerate(alive):
        for c in members[k]:
            lab_of_chain[c] = new
    return lab_of_chain[chain_id]


def glyph_mask(contours, res: int = 512, grow: int = 5):
    """inside(points (n, 2)) -> bool (n,): point lies in the (slightly grown) glyph."""
    from scipy.ndimage import maximum_filter

    from .raster import render

    mask = maximum_filter(render(contours, res) > 0.5, size=grow)

    def inside(p):
        s = res / 1024.0
        x = np.clip((p[:, 0] * s).astype(int), 0, res - 1)
        y = np.clip(((900.0 - p[:, 1]) * s).astype(int), 0, res - 1)
        return mask[y, x]

    return inside


def bridge_validator(contours, chain_id: np.ndarray):
    """Returns valid(chain_ids): True if all bridges of the would-be stroke stay inside the glyph."""
    inside = glyph_mask(contours)
    segs = flat_segments(contours)
    chains = {}
    for k in np.unique(chain_id):
        idx = np.where(chain_id == k)[0]
        # restore contour order for chains that wrap around the contour start
        if len(idx) > 1 and (np.diff(idx) > 1).any():
            cut = int(np.argmax(np.diff(idx) > 1)) + 1
            idx = np.r_[idx[cut:], idx[:cut]]
        chains[int(k)] = Chain(int(k), [int(i) for i in idx], False)

    t = np.linspace(0.1, 0.9, 9)[:, None]

    def valid(ids):
        chs = [chains[i] for i in ids]
        if len(chs) < 2:
            return True
        order = order_chains(chs, segs)
        for i, ci in enumerate(order):
            a = segs[chs[ci].segs[-1]].p1
            b = segs[chs[order[(i + 1) % len(order)]].segs[0]].p0
            if not inside(a * (1 - t) + b * t).all():
                return False
        return True

    return valid


def _partition(contours, cut_p, aff_p, w, cut_thr, merge_thr, validate) -> np.ndarray:
    chain = chains_from_cuts(contours, cut_p, cut_thr)
    lab = cluster_chains(chain, aff_p, w, merge_thr)
    if not validate:
        return lab
    # Re-cluster any stroke whose bridges would leave the glyph, vetoing invalid merges step by step.
    # (Validating only final clusters: partial subsets of a correct stroke can need outside bridges.)
    valid = bridge_validator(contours, chain)
    out = lab.copy()
    nxt = int(lab.max()) + 1
    for k in np.unique(lab):
        sel = lab == k
        ids = sorted(set(chain[sel].tolist()))
        if valid(ids):
            continue
        sub_chain = np.searchsorted(ids, chain[sel])
        sub = cluster_chains(sub_chain, aff_p[np.ix_(sel, sel)], w[sel], merge_thr,
                             lambda m: valid([ids[i] for i in m]))
        out[sel] = sub + nxt
        nxt += int(sub.max()) + 1
    _, out = np.unique(out, return_inverse=True)
    return out


def bridge_overflow(contours, lab: np.ndarray, inside=None) -> np.ndarray:
    """Per stroke, how many sample points of its bridge curves fall outside the glyph."""
    inside = inside or glyph_mask(contours)
    _, bridges = _assemble(contours, lab)
    return np.array([sum(int((~inside(b.sample(11)[1:-1])).sum()) for b in bs) for bs in bridges])


def partition_from_predictions(contours, cut_p, aff_p, cut_thr=0.5, merge_thr=0.5, validate=True,
                               refine_min=0.1, refine_steps=4) -> np.ndarray:
    segs = flat_segments(contours)
    w = np.array([max(s.length(), 1.0) for s in segs])
    cut_p = np.asarray(cut_p, float)
    lab = _partition(contours, cut_p, aff_p, w, cut_thr, merge_thr, validate)
    if not validate or refine_steps <= 0:
        return lab
    # A missed cut leaves a piece of one stroke glued to another, and the bridges that close the two
    # outlines then swing across the glyph. When a bridge curve leaves the glyph, try the most likely
    # sub-threshold cuts inside that stroke and keep one if it reduces the overflow.
    inside = glyph_mask(contours)
    nxt = np.concatenate([np.roll(np.arange(len(c)), -1) + o for c, o in zip(contours, contour_offsets(contours))])
    over = bridge_overflow(contours, lab, inside)
    tried: set[int] = set()
    for _ in range(refine_steps):
        if over.sum() == 0:
            break
        bad = set(np.nonzero(over)[0].tolist())
        cand = [i for i in np.argsort(-cut_p)
                if refine_min <= cut_p[i] <= cut_thr and i not in tried
                and lab[i] in bad and lab[nxt[i]] == lab[i]]
        accepted = False
        for i in cand:
            tried.add(int(i))
            trial_p = cut_p.copy()
            trial_p[i] = 1.0
            trial = _partition(contours, trial_p, aff_p, w, cut_thr, merge_thr, validate)
            trial_over = bridge_overflow(contours, trial, inside)
            if trial_over.sum() < over.sum():
                cut_p, lab, over, accepted = trial_p, trial, trial_over, True
                break
        if not accepted:
            break
    return lab


def match_strokes(pred: np.ndarray, gt: np.ndarray, w: np.ndarray) -> dict:
    """Hungarian match of predicted vs GT strokes by length-weighted segment-set IoU (GT -1 ignored)."""
    from scipy.optimize import linear_sum_assignment

    valid = gt >= 0
    pred, gt, w = pred[valid], gt[valid], w[valid]
    P, G = int(pred.max()) + 1, int(gt.max()) + 1
    inter = np.zeros((P, G))
    np.add.at(inter, (pred, gt), w)
    ps, gs = inter.sum(1), inter.sum(0)
    iou = inter / (ps[:, None] + gs[None, :] - inter + 1e-9)
    r, c = linear_sum_assignment(-iou)
    ious = iou[r, c]
    exact = int((ious > 0.9999).sum())
    return dict(n_gt=G, n_pred=int((ps > 0).sum()), exact=exact, miou=float(ious.sum() / max(G, P)),
                glyph_exact=bool(exact == G and int((ps > 0).sum()) == G))
