"""Refiner training pairs from the attempt store (Phase 2)."""
from collections import Counter

import numpy as np
from glyph_fixtures import PLUS_D, PLUS_LAB, hw_from_labels, plus

from strokegen.svg_io import parse_path

# ---------------------------------------------------------------- refinement pairs (Phase 2)

def shifted_plus(dx: int):
    """Plus signs at different positions: different outlines, so different glyph keys and splits."""
    return parse_path(" ".join(f"{float(t) + dx:g}" if i % 3 == 1 else t
                               for i, t in enumerate(PLUS_D.replace("Z", "").split())) + " Z")


def _wrong(hw):
    return {"strokes": hw["strokes"], "medians": [m[::-1] for m in hw["medians"]]}


def _populate(store, n=24):
    """n glyphs: a third first-try, a third repaired by a fix-up script, the rest pending / unresolved."""
    ref_old, ref_new = {"cut_sha": "old"}, {"cut_sha": "new"}
    store.note_checkpoint("old", {"kind": "cut"})
    store.note_checkpoint("new", {"kind": "cut"})
    keys = {}
    for i in range(n):
        c = shifted_plus(i * 3)
        ch = chr(0x4E00 + i)
        hw = hw_from_labels(c, PLUS_LAB)
        if i % 3 == 0:  # first try
            aid, _ = store.record("f", ch, c, hw, "model", ref_new, seg_stroke=PLUS_LAB)
            store.check(aid)
            store.approve(aid)
        elif i % 3 == 1:  # model wrong (old checkpoint) -> rejected although it passes -> fix-up -> approved
            m, _ = store.record("f", ch, c, _wrong(hw), "model", ref_old)
            store.check(m)
            store.set_verdict(m, "rejected", "medians run backwards", ["direction"])
            f, _ = store.record("f", ch, c, hw, "fixup_script", {"script": f"tools/fixups/r/fix{i % 2}.py"},
                                parent=m, tags=["direction"])
            store.approve(f)
        elif i % 6 == 2:
            store.record("f", ch, c, _wrong(hw), "model", ref_new)
        else:
            m, _ = store.record("f", ch, c, _wrong(hw), "model", ref_new)
            store.set_verdict(m, "unresolved", "no idea")
        keys[ch] = hist_key(store, ch)
    return keys


def hist_key(store, ch):
    return store.history("f", ch)[0]["glyph_key"]


def test_refine_pairs_follow_the_outline_split(store):
    from strokegen.build_dataset import split_of
    from strokegen.refine_dataset import build_pairs
    keys = _populate(store)
    assert len({split_of(k) for k in keys.values()}) > 1  # the glyphs really spread over several splits
    res = build_pairs(store, identity_cap=1.0, recency_decay=0.5)
    seen: dict[str, set] = {}
    for split, recs in res["splits"].items():
        for r in recs:
            assert r["split"] == split == split_of(r["hash"]) == split_of(keys[r["char"]])
            seen.setdefault(r["char"], set()).add(split)
    assert all(len(s) == 1 for s in seen.values())
    for r in res["negatives"]:
        assert r["split"] == split_of(keys[r["char"]])


def test_refine_pairs_content_caps_and_weights(store):
    from strokegen.attempts.check import SEG_FLAGS
    from strokegen.refine_dataset import build_pairs
    _populate(store)
    res = build_pairs(store, identity_cap=1.0, recency_decay=0.5)
    recs = [r for v in res["splits"].values() for r in v]
    rep = res["report"]
    # 8 first-try glyphs -> identity pairs; 8 repaired glyphs -> model version -> approved fix-up version
    assert rep["identity"] == {"available": 8, "kept": 8}
    assert rep["by_source"] == {"identity": 8, "model": 8}
    assert all(r["target_attempt_id"] != r["attempt_id"] for r in recs if r["source_type"] == "model")
    model = [r for r in recs if r["source_type"] == "model"]
    assert all(r["weight"] == 0.5 and r["tags"] == ["direction"] and not r["identity"] for r in model)
    assert all(r["seg_flags"].shape == (12, len(SEG_FLAGS)) for r in recs)
    assert all(np.array_equal(r["input_flip"], ~r["flip"]) for r in model)  # the wrong medians run backwards
    # unresolved / pending glyphs are reported, never paired; checker-passing rejections are negatives
    assert len(rep["unresolved"]) == 4 and len(rep["pending"]) == 4
    paired = {r["char"] for r in recs}
    assert not paired & {u["char"] for u in rep["unresolved"] + rep["pending"]}
    assert len(res["negatives"]) == 8 and all(n["reason"] == "medians run backwards" for n in res["negatives"])

    capped = build_pairs(store, identity_cap=0.3, per_source_cap=5, recency_decay=None)
    crecs = [r for v in capped["splits"].values() for r in v]
    n_id = sum(r["identity"] for r in crecs)
    assert n_id <= 0.3 * len(crecs) + 1e-9
    assert max(Counter(r["source_key"] for r in crecs if r["source_type"] != "identity").values()) <= 5
    assert all(r["weight"] == 1.0 for r in crecs)
    tagged = build_pairs(store, identity_cap=1.0, per_tag_cap=3)
    assert sum("direction" in r["tags"] for v in tagged["splits"].values() for r in v) == 3


def test_synthetic_corruptions_change_the_labels(store):
    from strokegen.refine_dataset import SYNTHETIC_KINDS, build_pairs, corrupt
    glyphs = [(parse_path(RING_TWO_BARS), np.array([0] * 8 + [1] * 4)),  # ring + counter = 0, separate bar = 1
              (plus(), PLUS_LAB)]
    rng = np.random.default_rng(0)
    for kind in SYNTHETIC_KINDS:
        applied = 0
        for c, lab in glyphs:
            out = corrupt(kind, c, lab, [False, False], rng)
            if out is None:
                continue
            applied += 1
            new, flip, order = out
            assert len(new) == len(lab) and len(flip) == int(new.max()) + 1 == len(order)
            assert not (np.array_equal(new, lab) and flip == [False, False] and order == [0, 1]), kind
        assert applied, kind
    assert corrupt("merge_adjacent", *glyphs[0], [False, False], rng) is None  # the bar touches nothing
    _populate(store)
    res = build_pairs(store, identity_cap=1.0, synthetic=2)
    syn = [r for v in res["splits"].values() for r in v if r["source_type"] == "synthetic"]
    assert syn and all(r["tags"][0] == "synthetic" and r["tags"][1] in SYNTHETIC_KINDS for r in syn)


RING_TWO_BARS = ("M 100 500 L 500 500 L 500 900 L 100 900 Z M 200 600 L 200 800 L 400 800 L 400 600 Z "
                 "M 600 100 L 900 100 L 900 200 L 600 200 Z")
