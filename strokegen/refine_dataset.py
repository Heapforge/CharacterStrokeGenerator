"""Refiner training pairs from the attempt store: (a version of a glyph + checker feedback) -> its approved version.

    uv run python -m strokegen.refine_dataset [--identity-cap 0.3] [--per-source-cap N] [--per-tag-cap N]
        [--recency-decay 0.5 | --no-recency] [--synthetic 2] [--include-imported] [--seed 0]

Writes data/processed/refine_{train,val,test}.pkl, refine_negatives.pkl and refine_report.json.

- Every earlier version of an approved glyph is paired with the approved version (not version n -> n+1).
- Glyphs approved on the first try give identity pairs (input = target), capped at --identity-cap of all pairs.
- Unresolved and still-pending glyphs give no pairs; the report lists them.
- Rejected versions that passed the checker go to refine_negatives.pkl (not used for training yet).
- The split is build_dataset.split_of(glyph_key), so every version of a glyph lands in the same split.
- --per-source-cap / --per-tag-cap limit the pairs from any one producer (fix-up script, LLM model, ...) or error tag.
- Recency: a model / refiner version made by a checkpoint d checkpoints older than the newest gets weight decay**d.
- --synthetic N adds up to N corruptions per approved glyph (merge adjacent strokes, split a stroke at a vertex,
  move a counter to the wrong stroke, flip a direction, swap two strokes in the order), tagged synthetic.

A record extends a build_dataset record (contours, seg_stroke = target labels, medians, hash) with the input
version: input_seg_stroke, input_flip, input_order, input_medians, seg_flags (strokegen.attempts.check.SEG_FLAGS),
plus weight, source_type, source_key, tags, attempt_id, target_attempt_id, identity.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from .attempts.check import SEG_FLAGS, segment_flags
from .attempts.labels import same_partition
from .attempts.store import AttemptStore
from .build_dataset import MIN_COVERAGE, split_of
from .decode import contour_offsets
from .paths import ATTEMPTS, PROCESSED

SYNTHETIC_KINDS = ("merge_adjacent", "split_at_vertex", "move_counter", "flip_direction", "swap_order")


# ---------------------------------------------------------------- helpers
def source_key(att: dict) -> str:
    """The producer a pair is balanced by: a fix-up script, an LLM model, a checkpoint family, ..."""
    r = att.get("source_ref") or {}
    t = att["source_type"]
    if t == "fixup_script":
        return f"fixup:{r.get('script', '?')}"
    if t == "llm_repair":
        return f"llm:{r.get('llm_model', '?')}"
    return t


def recency_weight(att: dict, ckpt_rank: dict[str, int], newest: int, decay: float | None) -> float:
    if decay is None:
        return 1.0
    sha = (att.get("source_ref") or {}).get("cut_sha")
    if sha is None or sha not in ckpt_rank:
        return 1.0
    return float(decay ** max(0, newest - ckpt_rank[sha]))


def _renumber(lab: np.ndarray) -> np.ndarray:
    """Stroke ids 0..k-1 in order of first appearance of each id's minimum (keeps -1)."""
    out = lab.copy()
    ids = sorted({int(v) for v in lab if v >= 0})
    for new, old in enumerate(ids):
        out[lab == old] = new
    return out


def _nxt(contours) -> np.ndarray:
    return np.concatenate([np.roll(np.arange(len(c)), -1) + o for c, o in zip(contours, contour_offsets(contours))])


def corrupt(kind: str, contours, lab, flip, rng: np.random.Generator):
    """A plausible wrong version of correct labels, or None when `kind` does not apply to this glyph.

    Returns (seg_stroke, flip, order); stroke ids stay writing-order positions.
    """
    lab = np.asarray(lab, np.int64).copy()
    flip = list(map(bool, flip))
    k = int(lab.max()) + 1 if (lab >= 0).any() else 0
    order = list(range(k))
    if kind == "flip_direction":
        if k < 1:
            return None
        i = int(rng.integers(k))
        flip[i] = not flip[i]
        return lab, flip, order
    if kind == "swap_order":
        if k < 2:
            return None
        i, j = sorted(rng.choice(k, 2, replace=False).tolist())
        out = lab.copy()
        out[lab == i], out[lab == j] = j, i
        flip[i], flip[j] = flip[j], flip[i]
        order[i], order[j] = order[j], order[i]
        return out, flip, order
    nxt = _nxt(contours)
    if kind == "merge_adjacent":
        pairs = sorted({(int(min(lab[a], lab[b])), int(max(lab[a], lab[b])))
                        for a, b in zip(range(len(lab)), nxt) if lab[a] >= 0 and lab[b] >= 0 and lab[a] != lab[b]})
        if not pairs:
            return None
        a, b = pairs[int(rng.integers(len(pairs)))]
        out = lab.copy()
        out[lab == b] = a
        out = _renumber(out)
        return out, [f for s, f in enumerate(flip) if s != b], list(range(int(out.max()) + 1))
    if kind == "split_at_vertex":
        # a stroke's run of >= 2 consecutive segments along one contour, cut at an inner vertex
        runs = []
        for c, o in zip(contours, contour_offsets(contours)):
            idx = list(range(o, o + len(c)))
            for s in range(len(idx)):
                if lab[idx[s]] >= 0 and lab[idx[s - 1]] != lab[idx[s]]:  # a run starts here
                    run = [idx[s]]
                    t = s + 1
                    while lab[idx[t % len(idx)]] == lab[idx[s]] and len(run) < len(idx):
                        run.append(idx[t % len(idx)])
                        t += 1
                    if len(run) >= 2:
                        runs.append(run)
        if not runs:
            return None
        run = runs[int(rng.integers(len(runs)))]
        cut = int(rng.integers(1, len(run)))
        out = lab.copy()
        out[run[cut:]] = k
        return out, flip + [flip[int(lab[run[0]])]], order + [k]
    if kind == "move_counter":
        # a whole contour inside a stroke that also owns other segments (a counter or an island) goes elsewhere
        cands = []
        for c, o in zip(contours, contour_offsets(contours)):
            seg = lab[o:o + len(c)]
            if (seg >= 0).all() and len(set(seg.tolist())) == 1 and (lab == seg[0]).sum() > len(c):
                cands.append((o, len(c), int(seg[0])))
        if not cands:
            return None
        o, n, s = cands[int(rng.integers(len(cands)))]
        others = [j for j in range(k) if j != s]
        out = lab.copy()
        if others and rng.random() < 0.5:
            out[o:o + n] = others[int(rng.integers(len(others)))]  # absorbed by the wrong stroke
            out = _renumber(out)
            return out, flip[:int(out.max()) + 1], list(range(int(out.max()) + 1))
        out[o:o + n] = k  # split off as a stroke of its own
        return out, flip + [False], order + [k]
    raise ValueError(f"unknown corruption {kind!r}")


def _record(contours, glyph, final, inp_labels, inp_checker, **extra) -> dict:
    tl = final["labels"]
    lab_in = np.asarray(inp_labels["seg_stroke"], np.int64)
    return dict(
        char=glyph[1], font=glyph[0], contours=contours, hash=final["glyph_key"],
        seg_stroke=np.asarray(tl["seg_stroke"], np.int64),
        medians=[np.asarray(m, float) for m in tl["medians"]], flip=np.asarray(tl["flip"], bool),
        strokes_gt=final["hw"]["strokes"], coverage=tl["match"]["coverage"],
        exact=bool(tl["match"]["unassigned"] == 0 and tl["match"]["n_bridge_on_edge"] == 0),
        input_seg_stroke=lab_in, input_flip=np.asarray(inp_labels["flip"], bool),
        input_order=np.asarray(inp_labels.get("order", range(len(inp_labels["flip"]))), np.int64),
        input_medians=None if inp_labels.get("medians") is None else [np.asarray(m, float)
                                                                      for m in inp_labels["medians"]],
        seg_flags=segment_flags(lab_in, inp_checker, inp_labels.get("vetoed")),
        target_attempt_id=final["attempt_id"], **extra)


# ---------------------------------------------------------------- builder
def build_pairs(store: AttemptStore, identity_cap: float = 0.3, per_source_cap: int | None = None,
                per_tag_cap: int | None = None, recency_decay: float | None = 0.5, synthetic: int = 0,
                include_imported: bool = False, seed: int = 0) -> dict:
    """{"splits": {train/val/test: [records]}, "negatives": [...], "report": {...}}"""
    rng = np.random.default_rng(seed)
    ckpts = [c for c in store.checkpoints() if c.get("kind", "cut") == "cut"]
    ckpt_rank = {c["sha"]: i for i, c in enumerate(ckpts)}
    newest = len(ckpts) - 1
    store.sync()

    pairs, identity, negatives = [], [], []
    rep: dict = defaultdict(Counter)
    lists: dict[str, list] = defaultdict(list)
    for font, char in store.glyphs():
        hist = store.history(font, char, sync=False)
        for a in hist:  # valid-but-wrong versions: the checker passed them, a reviewer did not
            if a["verdict"] == "rejected" and a["checker"] and a["checker"].get("passed"):
                negatives.append(dict(font=font, char=char, attempt_id=a["attempt_id"], glyph_key=a["glyph_key"],
                                      split=split_of(a["glyph_key"]), source_type=a["source_type"],
                                      source_key=source_key(a), reason=a["reason"], tags=a["verdict_tags"],
                                      labels=a["labels"], hw=a["hw"], checker=a["checker"]))
        status, final = store.status(hist)
        rep["glyphs"][status] += 1
        if status in ("unresolved", "pending"):
            lists[status].append(dict(font=font, char=char, versions=len(hist),
                                      reason=final["reason"] if final else None))
            continue
        if status != "approved":
            continue
        if final["source_type"] == "imported" and not include_imported:
            rep["skipped"]["imported_final"] += 1
            continue
        m = final["labels"]["match"]
        if m["coverage"] < MIN_COVERAGE or m["n_bridge_bad"] > 0:
            rep["skipped"]["final_off_outline"] += 1
            lists["final_off_outline"].append(dict(font=font, char=char, attempt_id=final["attempt_id"]))
            continue
        contours = store.contours(font, char, final["glyph_key"])
        split = split_of(final["glyph_key"])
        earlier = [a for a in hist if a["attempt_id"] != final["attempt_id"] and a["created_at"] <= final["created_at"]]
        kept = 0
        for a in earlier:
            if a["glyph_key"] != final["glyph_key"]:
                rep["skipped"]["outline_changed"] += 1
                continue
            if a["source_type"] == "imported" and not include_imported:
                rep["skipped"]["imported_input"] += 1
                continue
            same = (same_partition(a["labels"]["seg_stroke"], final["labels"]["seg_stroke"])
                    and a["labels"]["flip"] == final["labels"]["flip"] and a["content_hash"] == final["content_hash"])
            pairs.append(_record(contours, (font, char), final, a["labels"], a["checker"], split=split,
                                 identity=same, weight=recency_weight(a, ckpt_rank, newest, recency_decay),
                                 source_type=a["source_type"], source_key=source_key(a),
                                 tags=sorted(set(a["tags"]) | set(a["verdict_tags"])), attempt_id=a["attempt_id"]))
            kept += 1
        if not kept:  # approved on the first try: the input is the approved version itself
            identity.append(_record(contours, (font, char), final, final["labels"], final["checker"], split=split,
                                    identity=True, weight=recency_weight(final, ckpt_rank, newest, recency_decay),
                                    source_type="identity", source_key="identity", tags=[],
                                    attempt_id=final["attempt_id"]))
        for kind in rng.permutation(SYNTHETIC_KINDS)[:synthetic] if synthetic else []:
            c = corrupt(str(kind), contours, final["labels"]["seg_stroke"], final["labels"]["flip"], rng)
            if c is None:
                rep["synthetic_skipped"][str(kind)] += 1
                continue
            lab, flip, order = c
            pairs.append(_record(contours, (font, char), final,
                                 dict(seg_stroke=lab, flip=flip, order=order, medians=None), None, split=split,
                                 identity=False, weight=1.0, source_type="synthetic", source_key=f"synthetic:{kind}",
                                 tags=["synthetic", str(kind)], attempt_id=None))

    # balance: at most per_source_cap pairs per producer, per_tag_cap per error tag
    order = rng.permutation(len(pairs)).tolist()
    by_src: Counter = Counter()
    by_tag: Counter = Counter()
    balanced = []
    for i in order:
        p = pairs[i]
        if per_source_cap is not None and by_src[p["source_key"]] >= per_source_cap:
            rep["dropped_by_source_cap"][p["source_key"]] += 1
            continue
        if per_tag_cap is not None and any(by_tag[t] >= per_tag_cap for t in p["tags"]):
            rep["dropped_by_tag_cap"][",".join(p["tags"])] += 1
            continue
        by_src[p["source_key"]] += 1
        for t in p["tags"]:
            by_tag[t] += 1
        balanced.append(p)
    # identity pairs: at most identity_cap of all pairs
    n_id_in = sum(p["identity"] for p in balanced)
    n_non = len(balanced) - n_id_in
    room = int(np.floor(identity_cap / (1 - identity_cap) * n_non)) - n_id_in if identity_cap < 1 else len(identity)
    take = [identity[i] for i in sorted(rng.permutation(len(identity))[:max(0, room)].tolist())]
    rep["identity"]["available"] = len(identity)
    rep["identity"]["kept"] = len(take)
    final_pairs = sorted(balanced, key=lambda p: (p["font"], p["char"])) + take

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for p in final_pairs:
        splits[p["split"]].append(p)
    report = {k: dict(v) for k, v in rep.items()}
    report.update(
        pairs={s: len(v) for s, v in splits.items()},
        by_source=dict(Counter(p["source_key"] for p in final_pairs)),
        by_tag=dict(Counter(t for p in final_pairs for t in p["tags"])),
        by_script=dict(Counter(_script(p["char"]) for p in final_pairs)),
        negatives=len(negatives), seg_flags=list(SEG_FLAGS),
        settings=dict(identity_cap=identity_cap, per_source_cap=per_source_cap, per_tag_cap=per_tag_cap,
                      recency_decay=recency_decay, synthetic=synthetic, include_imported=include_imported, seed=seed),
        unresolved=lists["unresolved"], pending=lists["pending"], final_off_outline=lists["final_off_outline"],
    )
    return dict(splits=splits, negatives=negatives, report=report)


def _script(char: str) -> str:
    from .attempts.labels import script_of
    return script_of(char)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=ATTEMPTS, help="attempt store")
    ap.add_argument("--out", default=PROCESSED)
    ap.add_argument("--identity-cap", type=float, default=0.3, help="max share of identity pairs (0..1)")
    ap.add_argument("--per-source-cap", type=int, help="max pairs from one producer (script, LLM model, ...)")
    ap.add_argument("--per-tag-cap", type=int, help="max pairs carrying one error tag")
    ap.add_argument("--recency-decay", type=float, default=0.5, help="weight = decay ** checkpoints-behind-newest")
    ap.add_argument("--no-recency", action="store_true", help="weight every pair 1")
    ap.add_argument("--synthetic", type=int, default=0, metavar="N", help=f"corruptions per approved glyph, of "
                    f"{', '.join(SYNTHETIC_KINDS)}")
    ap.add_argument("--include-imported", action="store_true",
                    help="also use versions backfilled from the training data's git history")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    store = AttemptStore(a.root)
    res = build_pairs(store, a.identity_cap, a.per_source_cap, a.per_tag_cap,
                      None if a.no_recency else a.recency_decay, a.synthetic, a.include_imported, a.seed)
    os.makedirs(a.out, exist_ok=True)
    for name, recs in res["splits"].items():
        with open(os.path.join(a.out, f"refine_{name}.pkl"), "wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(a.out, "refine_negatives.pkl"), "wb") as f:
        pickle.dump(res["negatives"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(a.out, "refine_report.json"), "w", encoding="utf-8") as f:
        json.dump(res["report"], f, ensure_ascii=False, indent=1)
    r = res["report"]
    print(f"pairs {r['pairs']} · identity kept {r['identity']['kept']} of {r['identity']['available']} · "
          f"negatives {r['negatives']} · unresolved {len(r['unresolved'])} · pending {len(r['pending'])}")
    print(f"by source {r['by_source']}")
    print(f"wrote {a.out}/refine_*.pkl and refine_report.json")


if __name__ == "__main__":
    main()
