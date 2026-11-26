"""Checker, attempt store, staging view and attempts CLI (synthetic glyphs only)."""
import json
import os
import shutil

import numpy as np
import pytest
from glyph_fixtures import MODEL_REF, PLUS_LAB, ROOT, hw_from_labels, plus, svg_of

# ---------------------------------------------------------------- checker


def test_check_glyph_passes_a_clean_split_and_names_failures():
    from tools.verify_glyph import check_glyph

    c = plus()
    good = hw_from_labels(c, PLUS_LAB)
    res = check_glyph(c, good)
    assert res["passed"], res
    assert res["iou"] > 0.99 and len(res["strokes"]) == 2
    assert all(r["failures"] == [] and "hidden_share" in r for r in res["strokes"])

    # one stroke missing and a median outside the remaining one
    bad = {"strokes": good["strokes"][:1], "medians": [[[0, 0], [10, 10]]]}
    res = check_glyph(c, bad)
    assert not res["passed"]
    assert res["failures"] == ["low_iou"]
    assert "median_outside" in res["strokes"][0]["failures"]

    res = check_glyph(c, {"strokes": good["strokes"], "medians": good["medians"][:1]})
    assert not res["passed"] and res["failures"] == ["stroke_median_count"]


# ---------------------------------------------------------------- store


def test_labels_from_hw_reproduce_the_segment_partition():
    from strokegen.attempts.labels import labels_from_hw, same_partition

    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    lab = labels_from_hw(c, hw)
    assert same_partition(lab["seg_stroke"], PLUS_LAB)
    assert lab["match"]["coverage"] == 1.0 and lab["match"]["unassigned"] == 0
    assert lab["n_strokes"] == 2 and lab["order"] == [0, 1]
    assert all(len(b) == 2 for b in lab["bridges"])  # each bar is closed by two bridges across the crossing
    # a reversed median flips the direction bit and nothing else
    rev = {"strokes": hw["strokes"], "medians": [hw["medians"][0][::-1], hw["medians"][1]]}
    lab2 = labels_from_hw(c, rev)
    assert lab2["flip"] == [not lab["flip"][0], lab["flip"][1]] and lab2["seg_stroke"] == lab["seg_stroke"]


def test_record_dedups_and_survives_reindex(store):
    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    aid, created = store.record("font-a", "十", c, hw, "model", MODEL_REF, seg_stroke=PLUS_LAB)
    assert created
    assert store.record("font-a", "十", c, hw, "model", MODEL_REF, seg_stroke=PLUS_LAB) == (aid, False)
    # the same content from another producer is a separate version
    aid2, created = store.record("font-a", "十", c, hw, "fixup_script", {"script": "x.py"}, parent=aid)
    assert created
    with pytest.raises(ValueError):
        store.record("font-a", "十", c, hw, "magic", {})
    with pytest.raises(ValueError):
        store.record("../etc", "十", c, hw, "model", {})

    hist = store.history("font-a", "十")
    assert [h["attempt_id"] for h in hist] == [aid, aid2]
    assert hist[1]["parent_attempt_id"] == aid and hist[0]["verdict"] == "pending"
    from strokegen.build_dataset import outline_hash, split_of
    assert hist[0]["glyph_key"] == outline_hash(c) and hist[0]["script"] == "cjk"
    # the stored outline is exact: same hash, same split
    again = store.contours("font-a", "十", hist[0]["glyph_key"])
    assert outline_hash(again) == outline_hash(c) and split_of(outline_hash(again)) == split_of(outline_hash(c))

    log = store.log_path("font-a", "十")
    assert log.endswith(os.path.join("log", "font-a", "U+5341.jsonl"))
    lines = open(log, encoding="utf-8").read().splitlines()
    assert [json.loads(x)["type"] for x in lines] == ["glyph", "attempt", "attempt"]

    os.unlink(os.path.join(store.root, "index.sqlite"))
    from strokegen.attempts import AttemptStore
    s2 = AttemptStore(store.root, hwdata=store.hwdata)
    assert [h["attempt_id"] for h in s2.history("font-a", "十")] == [aid, aid2]
    s2.close()


def test_appends_from_another_process_are_picked_up(store):
    """Two stores on one root stand for the dashboard and a CLI run (or a git pull)."""
    from strokegen.attempts import AttemptStore
    c = plus()
    other = AttemptStore(store.root, hwdata=store.hwdata)
    aid, _ = other.record("font-a", "A", c, hw_from_labels(c, PLUS_LAB), "model", MODEL_REF)
    other.close()
    assert [h["attempt_id"] for h in store.history("font-a", "A")] == [aid]
    assert [r["char"] for r in store.staged_list("font-a")] == ["A"]


def test_staging_approval_and_supersede(store):
    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    a1, _ = store.record("font-a", "十", c, hw, "model", MODEL_REF, seg_stroke=PLUS_LAB)
    staged = json.load(open(store.staging_path("font-a", "十"), encoding="utf-8"))
    assert staged["attempt_id"] == a1 and staged["status"] == "pending" and staged["segStroke"] == PLUS_LAB.tolist()
    assert staged["strokes"] == hw["strokes"] and staged["has_training_file"] is False

    store.set_verdict(a1, "rejected", "order is wrong", ["order"])
    bad = {"strokes": hw["strokes"][::-1], "medians": hw["medians"][::-1]}
    a2, _ = store.record("font-a", "十", c, bad, "llm_repair", {"llm_model": "m", "run_id": "r"}, parent=a1)
    a3, _ = store.record("font-a", "十", c, {"strokes": hw["strokes"], "medians": [m[::-1] for m in hw["medians"]]},
                         "fixup_script", {"script": "y.py"}, parent=a2)
    staged = json.load(open(store.staging_path("font-a", "十"), encoding="utf-8"))
    assert staged["attempt_id"] == a3 and staged["versions"] == 3
    assert [h["verdict"] for h in staged["history"]] == ["rejected", "pending", "pending"]

    # the same character approved in another font first
    b1, _ = store.record("font-b", "十", c, hw, "model", MODEL_REF)
    store.approve(b1, by="tester")

    done = store.approve(a2, by="tester")
    assert set(done) == {a3, b1}
    v = {h["attempt_id"]: h["verdict"] for h in store.history("font-a", "十")}
    assert v == {a1: "rejected", a2: "approved", a3: "superseded"}  # a rejection keeps its reason
    assert store.get(b1)["verdict"] == "superseded" and "replaced by font-a" in store.get(b1)["reason"]
    assert not os.path.exists(store.staging_path("font-a", "十"))
    assert store.staged("font-a", "十") is None and store.status(store.history("font-a", "十"))[0] == "approved"

    # the staging view is derived: rebuilding it gives the same files
    store.record("font-a", "A", c, hw, "model", MODEL_REF)
    before = sorted(os.listdir(os.path.join(store.staging_dir, "font-a")))
    shutil.rmtree(store.staging_dir)
    assert store.stage() == 1
    assert sorted(os.listdir(os.path.join(store.staging_dir, "font-a"))) == before == ["U+0041.json"]


def test_rejecting_a_checker_passing_version_needs_a_reason(store):
    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    aid, _ = store.record("font-a", "十", c, hw, "model", MODEL_REF, seg_stroke=PLUS_LAB)
    with pytest.raises(ValueError):  # not checked yet: might be valid-but-wrong
        store.set_verdict(aid, "rejected")
    res = store.check(aid)
    assert res["passed"] and res["version"] == 1 and store.get(aid)["checker"]["passed"]
    with pytest.raises(ValueError):
        store.set_verdict(aid, "rejected", "  ")
    store.set_verdict(aid, "rejected", "strokes should be written vertical first", ["order"])
    assert store.get(aid)["reason"].startswith("strokes should")

    broken = {"strokes": hw["strokes"][:1], "medians": hw["medians"][:1]}
    bid, _ = store.record("font-a", "十", c, broken, "model", {"cut_sha": "x"})
    assert not store.check(bid)["passed"]
    store.set_verdict(bid, "rejected")  # the checker already says why
    with pytest.raises(ValueError):
        store.set_verdict(bid, "unresolved")


def test_segment_flags_point_at_the_failing_stroke(store):
    from strokegen.attempts.check import SEG_FLAGS, segment_flags
    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    broken = {"strokes": hw["strokes"][:1], "medians": [[[0, 0], [5, 5]]]}
    aid, _ = store.record("font-a", "十", c, broken, "model", {})
    store.check(aid)
    a = store.get(aid)
    f = segment_flags(a["labels"]["seg_stroke"], a["checker"])
    assert f.shape == (12, len(SEG_FLAGS))
    lab = np.asarray(a["labels"]["seg_stroke"])
    assert (f[lab == 0, 0] == 1).all() and (f[lab == 0, 2] == 1).all()  # failing stroke, median off
    assert (f[lab < 0, 4] == 1).all() and (lab < 0).any()  # the missing bar's segments are unassigned
    assert f[:, 1].any()  # and some segments border the missing area


# ---------------------------------------------------------------- CLI

def test_cli_export_import_history_stats(store, tmp_path, capsys):
    from strokegen.attempts.__main__ import main
    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    aid, _ = store.record("font-a", "十", c, {"strokes": hw["strokes"], "medians": hw["medians"][::-1]}, "model",
                          MODEL_REF)
    store.check(aid)
    root = ["--root", store.root]

    q = tmp_path / "queue.json"
    main(root + ["export", "--font", "font-a", "-o", str(q)])
    from strokegen.paths import HWDATA
    if os.path.exists(os.path.join(HWDATA, "十.json")):  # glyphs with a training file only on request
        assert json.load(open(q, encoding="utf-8")) == {}
    main(root + ["export", "--font", "font-a", "--include-existing", "-o", str(q)])
    exported = json.load(open(q, encoding="utf-8"))
    assert list(exported) == ["十"] and exported["十"]["attempt_id"] == aid

    # a "fix-up script" output: the right medians, plus an unchanged glyph and a report entry
    out = {"十": {"strokes": hw["strokes"], "medians": hw["medians"]}, "_report": {"n": 1}}
    o = tmp_path / "out.json"
    json.dump(out, open(o, "w", encoding="utf-8"))
    script = os.path.join(ROOT, "tools", "verify_glyph.py")
    main(root + ["import", str(o), "--font", "font-a", "--source", "fixup_script", "--script", script,
                 "--tags", "median_mismatch"])
    assert "recorded 1" in capsys.readouterr().out
    hist = store.history("font-a", "十")
    assert len(hist) == 2 and hist[1]["parent_attempt_id"] == aid and hist[1]["source_type"] == "fixup_script"
    assert hist[1]["source_ref"]["script"] == os.path.join("tools", "verify_glyph.py")
    assert hist[1]["source_ref"]["blob_sha"] and hist[1]["checker"] is not None
    assert hist[1]["tags"] == ["median_mismatch"]

    # importing the same output again changes nothing; llm_repair needs its model and run
    main(root + ["import", str(o), "--font", "font-a", "--source", "fixup_script", "--script", script])
    assert "unchanged 1" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(root + ["import", str(o), "--font", "font-a", "--source", "llm_repair"])

    main(root + ["history", "十", "--font", "font-a"])
    text = capsys.readouterr().out
    assert "2 versions" in text and f"<- {aid}" in text

    store.approve(hist[1]["attempt_id"], by="tester")
    main(root + ["stats", "--by", "source"])
    text = capsys.readouterr().out
    assert "after_fixup" in text and text.strip().splitlines()[-1].split()[-1] == "1"

    from strokegen.attempts.__main__ import glyph_stats
    assert glyph_stats(store) == [dict(font="font-a", char="十", script="cjk", outcome="after_fixup", versions=2,
                                       fixups=[os.path.join("tools", "verify_glyph.py")])]


def test_backfill_imports_git_history_without_inventing_attempts(store, tmp_path, monkeypatch):
    import subprocess

    from strokegen.attempts import backfill
    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    repo = tmp_path / "hwrepo"
    (repo / "data").mkdir(parents=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for meds in (hw["medians"][::-1], hw["medians"]):
        (repo / "data" / "十.json").write_text(json.dumps({"strokes": hw["strokes"][::-1] if meds is not hw["medians"]
                                                           else hw["strokes"], "medians": meds}), encoding="utf-8")
        subprocess.run(git + ["add", "-A"], cwd=repo, check=True)
        subprocess.run(git + ["commit", "-q", "-m", "v"], cwd=repo, check=True)
    glyphs = tmp_path / "glyphs"
    (glyphs / "font-a").mkdir(parents=True)
    (glyphs / "font-a" / "十.svg").write_text(svg_of(c), encoding="utf-8")
    json.dump({"characters": {"十": {"fonts": ["font-a"], "data_files": ["十.json"]}}},
              open(glyphs / "manifest.json", "w", encoding="utf-8"))
    monkeypatch.setattr(backfill, "GLYPHS", str(glyphs))

    counts = backfill.backfill_hwdata(store, hwdata=str(repo / "data"))
    assert counts["imported"] == 1 and counts["versions"] == 2
    hist = store.history("font-a", "十")
    assert [h["source_type"] for h in hist] == ["imported", "imported"]
    assert [h["verdict"] for h in hist] == ["superseded", "approved"]
    assert hist[1]["parent_attempt_id"] == hist[0]["attempt_id"] and len(hist[0]["source_ref"]["commit"]) == 40
    # running it again adds nothing
    again = backfill.backfill_hwdata(store, hwdata=str(repo / "data"))
    assert again["already_imported"] == 2 and len(store.history("font-a", "十")) == 2
