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
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.validation import make_valid

from .medians import stroke_geometry
from .svg_io import HW_BASE, HW_EM, Seg

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
    for c, off in zip(contours, contour_offsets(contours)):
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


_CYCLES: dict[int, np.ndarray] = {}


def _cycles(n: int) -> np.ndarray:
    """All visiting orders of n chains that start with chain 0, shape ((n - 1)!, n)."""
    if n not in _CYCLES:
        _CYCLES[n] = np.array([(0, *p) for p in itertools.permutations(range(1, n))], np.int64)
    return _CYCLES[n]


def _cycle_cost(order, D: np.ndarray) -> float:
    o = np.asarray(order)
    return float(D[o, np.roll(o, -1)].sum())


def order_chains(chs: list[Chain], segs: list[Seg]) -> list[int]:
    """Cyclic visiting order of open chains that minimizes total bridge length."""
    n = len(chs)
    if n <= 2:
        return list(range(n))
    ends = np.array([segs[c.segs[-1]].p1 for c in chs])
    starts = np.array([segs[c.segs[0]].p0 for c in chs])
    D = np.linalg.norm(ends[:, None] - starts[None], axis=-1)  # D[i, j]: bridge from chain i to chain j
    if n <= 7:
        P = _cycles(n)
        cost = D[P, np.roll(P, -1, 1)].sum(1)
        return P[int(np.argmin(cost))].tolist()
    # greedy nearest neighbour + 2-opt style improvement on the cycle
    left = set(range(1, n))
    o = [0]
    while left:
        j = min(left, key=lambda k: D[o[-1], k])
        o.append(j)
        left.remove(j)
    improved = True
    while improved:
        improved = False
        base = _cycle_cost(o, D)
        for i in range(1, n):
            for j in range(i + 1, n):
                cand = o[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = _cycle_cost(cand, D)
                if c < base - 1e-9:
                    o, base, improved = cand, c, True
    return o


def _spills(b: Seg, inside) -> bool:
    return not inside(b.sample(11)[1:-1]).all()


def _straight(b: Seg) -> Seg:
    return Seg("L", np.stack([b.p0, b.p1]))


def _lens_mostly_outside(b: Seg, glyph) -> bool:
    """Does the region between curve b and its chord lie mostly outside the glyph (glyph() -> shape)?

    That region is what straightening b takes away from its stroke.
    """
    g = glyph()
    lens = make_valid(Polygon(b.sample(17)))  # the curve, closed back along its chord
    if g is None or lens.is_empty or lens.area < 1.0:
        return False
    outside = lens.difference(g).area
    return outside > lens.area - outside


def _assemble(contours, seg_stroke: np.ndarray, n_strokes: int | None = None, inside=None,
              straighten: bool = True):
    """(per stroke closed loops, per stroke bridge segments).

    straighten: a curved bridge that bulges out of the glyph becomes straight. Tangent-continuous
    cubics fit cuts inside a glyph, but at a square stroke end (sans fonts) the tangents point
    outwards. When two strokes close across the same cut (a bridge p -> q in one, q -> p in the
    other) both become the chord, which splits the ink between them exactly; a bridge without a
    partner is straightened when the curve's bulge lies mostly outside the glyph. inside is the
    glyph_mask to test against (computed on demand).
    """
    segs = flat_segments(contours)
    chains = build_chains(contours, seg_stroke)
    ids = sorted({c.stroke for c in chains if c.stroke >= 0})
    shape = []

    def glyph():
        if not shape:
            shape.append(stroke_geometry(contours))
        return shape[0]

    if n_strokes is None:
        n_strokes = (max(ids) + 1) if ids else 0
    out: list[list[list[Seg]]] = [[] for _ in range(n_strokes)]
    bridges: list[list[Seg]] = [[] for _ in range(n_strokes)]
    placed: list[tuple[int, list[Seg], int]] = []  # (stroke, loop, index of the bridge in the loop)
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
            placed.append((k, loop, len(loop) - 1))
        out[k].append(loop)
    if straighten and any(loop[i].kind == "C" for _, loop, i in placed):
        if inside is None:
            inside = glyph_mask(contours)
        by_ends: dict[tuple, list[tuple[int, list[Seg], int]]] = {}
        for k, loop, i in placed:
            by_ends.setdefault((*loop[i].p0, *loop[i].p1), []).append((k, loop, i))
        for k, loop, i in placed:
            br = loop[i]
            if br.kind != "C" or not _spills(br, inside):
                continue
            partners = [(k2, l2, i2) for k2, l2, i2 in by_ends.get((*br.p1, *br.p0), []) if k2 != k]
            if partners:
                for _, l2, i2 in partners:
                    l2[i2] = _straight(l2[i2])
                loop[i] = _straight(br)
            elif _lens_mostly_outside(br, glyph):
                loop[i] = _straight(br)
    for k, loop, i in placed:
        bridges[k].append(loop[i])
    return out, bridges


def assemble_strokes(contours, seg_stroke: np.ndarray, n_strokes: int | None = None, inside=None,
                     straighten: bool = True) -> list[list[list[Seg]]]:
    """Return per stroke a list of closed loops (each a list of Seg). Stroke ids are kept (empty -> [])."""
    return _assemble(contours, seg_stroke, n_strokes, inside, straighten)[0]


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
        for k in range(n):  # the last segment visited is a cut, so the contour ends a chain
            i = (start + k) % n
            ids[off + i] = nid
            if cuts[i]:
                nid += 1
    # compact ids
    _, ids = np.unique(ids, return_inverse=True)
    return ids


def cluster_chains(chain_id: np.ndarray, aff_p: np.ndarray, w: np.ndarray, thr: float = 0.5,
                   valid=None, on_veto=None) -> np.ndarray:
    """Average-linkage agglomeration of chains on (length weighted) mean pairwise affinity.

    valid(list_of_chain_ids) -> bool can veto a merge. A veto lasts until one of the two
    clusters grows: a bigger cluster can make a merge valid that was not before.
    on_veto(chains_a, chains_b) is told about every vetoed merge (it does not change the result)."""
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
            if on_veto is not None:
                on_veto(members[ka], members[kb])
            continue
        S[ka, :] += S[kb, :]
        S[:, ka] += S[:, kb]
        W[ka] += W[kb]
        members[ka] += members.pop(kb)
        alive.remove(kb)
        vetoed = {v for v in vetoed if ka not in v and kb not in v}
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
    s = res / HW_EM

    def inside(p):
        x = np.clip((p[:, 0] * s).astype(int), 0, res - 1)
        y = np.clip(((HW_BASE - p[:, 1]) * s).astype(int), 0, res - 1)
        return mask[y, x]

    return inside


def bridge_validator(contours, chain_id: np.ndarray, inside=None):
    """Returns valid(chain_ids): True if all bridges of the would-be stroke stay inside the glyph.

    Mirrors _assemble: a chain that is a whole contour stays a closed loop of its own and gets
    no bridge, so only the open chains are joined (and checked). A whole contour still has to
    touch the rest of the stroke through ink.
    """
    inside = inside or glyph_mask(contours)
    segs = flat_segments(contours)
    contour_len = np.concatenate([np.full(len(c), len(c)) for c in contours])
    chains = {}
    for k in np.unique(chain_id):
        idx = np.where(chain_id == k)[0]
        # restore contour order for chains that wrap around the contour start
        if len(idx) > 1 and (np.diff(idx) > 1).any():
            cut = int(np.argmax(np.diff(idx) > 1)) + 1
            idx = np.r_[idx[cut:], idx[:cut]]
        closed = len(idx) == contour_len[idx[0]]
        chains[int(k)] = Chain(int(k), [int(i) for i in idx], bool(closed))

    t = np.linspace(0.1, 0.9, 9)[:, None]
    memo: dict[frozenset, bool] = {}
    pts: dict[int, np.ndarray] = {}

    def outline_pts(k: int) -> np.ndarray:
        if k not in pts:
            pts[k] = np.concatenate([segs[i].flatten(4.0) for i in chains[k].segs])
        return pts[k]

    def line_inside(a, b) -> bool:
        return bool(inside(a * (1 - t) + b * t).all())

    def valid(ids):
        key = frozenset(ids)
        if key not in memo:
            memo[key] = _valid(sorted(key))
        return memo[key]

    def _valid(ids):
        if len(ids) < 2:
            return True
        # a whole contour needs no bridge, but it has to be joined to the rest of the stroke through
        # ink (a counter and the outline around it), not across the background (two separate parts)
        for k in ids:
            if chains[k].closed and not _attached(k, [j for j in ids if j != k]):
                return False
        chs = [chains[k] for k in ids if not chains[k].closed]
        if len(chs) < 2:
            return True
        order = order_chains(chs, segs)
        for i, ci in enumerate(order):
            a = segs[chs[ci].segs[-1]].p1
            b = segs[chs[order[(i + 1) % len(order)]].segs[0]].p0
            if not line_inside(a, b):
                return False
        return True

    def _attached(k, others, tries: int = 3) -> bool:
        """Is one of the closest point pairs between contour k and the other chains joined by ink?"""
        p = outline_pts(k)
        q = np.concatenate([outline_pts(j) for j in others])
        d, nearest = cKDTree(q).query(p)
        for i in np.argsort(d)[:tries]:
            if line_inside(p[i], q[nearest[i]]):
                return True
        return False

    return valid


def _partition(contours, cut_p, aff_p, w, cut_thr, merge_thr, validate, inside=None,
               vetoed: set | None = None) -> np.ndarray:
    """vetoed: when given, collects the segments of chains whose merge the bridge validator refused."""
    chain = chains_from_cuts(contours, cut_p, cut_thr)
    lab = cluster_chains(chain, aff_p, w, merge_thr)
    if not validate:
        return lab
    # Re-cluster any stroke whose bridges would leave the glyph, vetoing invalid merges step by step.
    # (Validating only final clusters: partial subsets of a correct stroke can need outside bridges.)
    valid = bridge_validator(contours, chain, inside)
    refused: set[int] = set()
    out = lab.copy()
    nxt = int(lab.max()) + 1
    for k in np.unique(lab):
        sel = lab == k
        ids = sorted(set(chain[sel].tolist()))
        if valid(ids):
            continue
        sub_chain = np.searchsorted(ids, chain[sel])
        sub = cluster_chains(sub_chain, aff_p[np.ix_(sel, sel)], w[sel], merge_thr,
                             lambda m, ids=ids: valid([ids[i] for i in m]),
                             lambda a, b, ids=ids: refused.update(ids[i] for i in a + b))
        out[sel] = sub + nxt
        nxt += int(sub.max()) + 1
    if vetoed is not None and refused:
        vetoed.update(np.nonzero(np.isin(chain, sorted(refused)))[0].tolist())
    _, out = np.unique(out, return_inverse=True)
    return out


def absorb_counters(contours, lab: np.ndarray, aff_p: np.ndarray) -> np.ndarray:
    """Give every stroke made only of whole counters to a stroke that owns the outline around them.

    A counter is a contour whose own region the glyph leaves empty (the total winding number there
    is 0: the hole of 口 or O). Pieces of a counter's outline are often all of a stroke (the middle
    bar of 日 lies between two counters), but a whole counter filled on its own covers background,
    so it can only be part of a stroke that also has the ink around it. Such strokes join the
    stroke, among those owning the contour that encloses the counter, with the highest mean
    affinity to them.
    """
    from .medians import _winding
    from .svg_io import contour_polygon, signed_area

    offs = contour_offsets(contours)
    cid = np.concatenate([np.full(len(c), i) for i, c in enumerate(contours)])
    whole = {}  # stroke -> contours, for strokes that are a union of whole contours
    for k in np.unique(lab):
        touched = set(cid[lab == k].tolist())
        if all((lab[offs[i]:offs[i] + len(contours[i])] == k).all() for i in touched):
            whole[int(k)] = touched
    if not whole:
        return lab
    polys = [contour_polygon(c, 4.0) for c in contours]
    areas = np.abs([signed_area(p) if len(p) >= 3 else 0.0 for p in polys])
    shapes: dict[int, object] = {}

    def shape(i):
        if i not in shapes:
            shapes[i] = make_valid(Polygon(polys[i])) if len(polys[i]) >= 3 else None
        return shapes[i]

    def probe(i):
        """A point inside contour i but outside the smaller contours (islands) within it."""
        if shape(i) is None or shape(i).is_empty or areas[i] <= 1.0:
            return None
        region = shape(i)
        for j in range(len(contours)):
            if j != i and areas[j] < areas[i] and shape(j) is not None and region.intersects(shape(j)):
                region = region.difference(shape(j))
        return None if region.is_empty else region.representative_point().coords[0]

    def counter(i) -> bool:
        p = probe(i)
        return p is not None and _winding(*p, polys) == 0

    def parent(i):  # smallest other contour that encloses the counter
        x, y = probe(i)
        enc = [j for j in range(len(contours)) if j != i and areas[j] > areas[i] and not counter(j)
               and _winding(x, y, [polys[j]]) != 0]
        return min(enc, key=lambda j: areas[j], default=None)

    lab = lab.copy()
    for k, touched in whole.items():
        if not all(counter(i) for i in touched):
            continue
        sel = lab == k
        owners = {int(v) for i in touched if (j := parent(i)) is not None
                  for v in lab[offs[j]:offs[j] + len(contours[j])] if v != k}
        if owners:
            lab[sel] = max(owners, key=lambda o: aff_p[np.ix_(sel, lab == o)].mean())
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def bridge_overflow(contours, lab: np.ndarray, inside=None) -> np.ndarray:
    """Per stroke, how many sample points of its bridge curves fall outside the glyph."""
    inside = inside or glyph_mask(contours)
    _, bridges = _assemble(contours, lab, inside=inside)
    return np.array([sum(int((~inside(b.sample(11)[1:-1])).sum()) for b in bs) for bs in bridges])


def partition_from_predictions(contours, cut_p, aff_p, cut_thr=0.5, merge_thr=0.5, validate=True,
                               refine_min=0.1, refine_steps=4, refine_candidates=8,
                               report: dict | None = None) -> np.ndarray:
    """Stroke label per segment from Stage A's predictions.

    report: when given, report["vetoed"] is set to the segments (sorted flat indices) of chains whose
    merge the bridge validator refused in the returned partition (refiner feedback; no effect on it)."""
    segs = flat_segments(contours)
    w = np.array([max(s.length(), 1.0) for s in segs])
    cut_p = np.asarray(cut_p, float)
    inside = glyph_mask(contours) if validate else None
    vetoed: set[int] = set()
    lab = _partition(contours, cut_p, aff_p, w, cut_thr, merge_thr, validate, inside, vetoed)
    if report is not None:
        report["vetoed"] = sorted(vetoed)
    if validate:
        lab = absorb_counters(contours, lab, aff_p)
    if not validate or refine_steps <= 0:
        return lab
    # A missed cut leaves a piece of one stroke glued to another, and the bridges that close the two
    # outlines then swing across the glyph. When a bridge curve leaves the glyph, try the most likely
    # sub-threshold cuts inside that stroke (at most refine_candidates per step) and keep one if it
    # reduces the overflow.
    nxt = np.concatenate([np.roll(np.arange(len(c)), -1) + o for c, o in zip(contours, contour_offsets(contours))])
    over = bridge_overflow(contours, lab, inside)
    tried: set[int] = set()
    for _ in range(refine_steps):
        if over.sum() == 0:
            break
        bad = set(np.nonzero(over)[0].tolist())
        cand = [i for i in np.argsort(-cut_p)
                if refine_min <= cut_p[i] <= cut_thr and i not in tried
                and lab[i] in bad and lab[nxt[i]] == lab[i]][:refine_candidates]
        accepted = False
        for i in cand:
            tried.add(int(i))
            trial_p = cut_p.copy()
            trial_p[i] = 1.0
            trial_vetoed: set[int] = set()
            trial = absorb_counters(contours, _partition(contours, trial_p, aff_p, w, cut_thr, merge_thr, validate,
                                                         inside, trial_vetoed), aff_p)
            trial_over = bridge_overflow(contours, trial, inside)
            if trial_over.sum() < over.sum():
                cut_p, lab, over, accepted = trial_p, trial, trial_over, True
                if report is not None:
                    report["vetoed"] = sorted(trial_vetoed)
                break
        if not accepted:
            break
    return lab


def match_strokes(pred: np.ndarray, gt: np.ndarray, w: np.ndarray, assign_min: float = 0.3) -> dict:
    """Hungarian match of predicted vs GT strokes by length-weighted segment-set IoU (GT -1 ignored).

    assignment maps a predicted stroke id to its matched GT stroke id when their IoU > assign_min.
    """
    from scipy.optimize import linear_sum_assignment

    pred, gt, w = np.asarray(pred), np.asarray(gt), np.asarray(w, float)
    valid = (gt >= 0) & (pred >= 0)
    pred, gt, w = pred[valid], gt[valid], w[valid]
    if not len(gt):
        return dict(n_gt=0, n_pred=0, exact=0, miou=1.0, glyph_exact=True, assignment={})
    P, G = int(pred.max()) + 1, int(gt.max()) + 1
    inter = np.zeros((P, G))
    np.add.at(inter, (pred, gt), w)
    ps, gs = inter.sum(1), inter.sum(0)
    n_pred, n_gt = int((ps > 0).sum()), int((gs > 0).sum())
    iou = inter / (ps[:, None] + gs[None, :] - inter + 1e-9)
    r, c = linear_sum_assignment(-iou)
    ious = iou[r, c]
    exact = int((ious > 0.9999).sum())
    return dict(n_gt=n_gt, n_pred=n_pred, exact=exact, miou=float(ious.sum() / max(n_gt, n_pred)),
                glyph_exact=bool(exact == n_gt and n_pred == n_gt),
                assignment={int(a): int(b) for a, b in zip(r, c) if iou[a, b] > assign_min})
