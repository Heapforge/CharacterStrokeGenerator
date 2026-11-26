"""Stage A refiner (Phase 3), the refinement loop (Phase 4) and its evaluation (Phase 5), on CPU.

Nothing here trains a model: the generators are tiny, randomly initialised checkpoints, and the loop tests
replace the refiner with a stub that returns chosen decompositions.
"""
import numpy as np
import pytest
import torch
from glyph_fixtures import MODEL_REF, PLUS_LAB, hw_from_labels, plus

from strokegen import features, order
from strokegen.decode import partition_from_predictions
from strokegen.features import glyph_features
from strokegen.model import CutModel, OrderNet, RefineCutModel, cut_losses
from strokegen.refine import N_RFEAT, REFINE_VERSION, RefineDataset, collate_refine, refine_item
from strokegen.svg_io import parse_path

TWO_BARS = "M 100 100 L 900 100 L 900 200 L 100 200 Z M 100 700 L 900 700 L 900 800 L 100 800 Z"
TINY = dict(d=32, layers=2)


def _items(rng):
    out = []
    for c in (plus(), parse_path(TWO_BARS)):
        f = glyph_features(c)
        n = len(f["feat"])
        chk = dict(failures=["low_iou"], strokes=[dict(failures=["hidden"])], deficit_segments=[0])
        out.append(refine_item(f, rng.integers(-1, 3, n), chk, vetoed=[1, 2]))
    return out


def test_refiner_equals_stage_a_at_init():
    torch.manual_seed(0)
    base = CutModel(**TINY).eval()
    ref = RefineCutModel.from_cut(base.state_dict(), n_rfeat=N_RFEAT, **TINY).eval()
    batch = collate_refine([dict(it, index=i) for i, it in enumerate(_items(np.random.default_rng(0)))])
    assert batch["rfeat"].abs().sum() > 0 and (batch["in_lab"] >= 0).any()  # the new inputs are not trivial
    with torch.no_grad():
        a, b = base(batch), ref(batch)
    assert torch.equal(a["cut"], b["cut"]) and torch.equal(a["aff"], b["aff"])

    # ... and the zero-initialised inputs can still learn: they get gradients
    ref.train()
    out = ref(batch)
    target = dict(batch, lab=torch.where(batch["pad"], -1, 0), cut=torch.ones_like(out["cut"]),
                  cut_mask=(~batch["pad"]).float())
    sum(cut_losses(out, target)).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in ref.new_parameters())


def test_refiner_rejects_a_state_of_another_size():
    with pytest.raises(RuntimeError):
        RefineCutModel.from_cut(CutModel(d=32, layers=1).state_dict(), n_rfeat=N_RFEAT, d=64, layers=1)


def test_weighted_losses_with_unit_weights_are_the_plain_losses():
    torch.manual_seed(1)
    B, N = 2, 6
    out = dict(cut=torch.randn(B, N), aff=torch.randn(B, N, N))
    batch = dict(cut=(torch.rand(B, N) > 0.5).float(), cut_mask=torch.ones(B, N),
                 lab=torch.tensor([[0, 0, 1, 1, -1, 2], [0, 1, 1, 0, 0, 1]]), pad=torch.zeros(B, N, dtype=torch.bool))
    plain = cut_losses(out, batch)
    weighted = cut_losses(out, batch, sample_weight=torch.ones(B))
    assert all(torch.allclose(torch.as_tensor(p), torch.as_tensor(w)) for p, w in zip(plain, weighted))
    # a zero weight removes a glyph
    only0 = cut_losses(out, batch, sample_weight=torch.tensor([1.0, 0.0]))
    first = cut_losses(dict(cut=out["cut"][:1], aff=out["aff"][:1]), {k: v[:1] for k, v in batch.items()})
    assert all(torch.allclose(torch.as_tensor(p), torch.as_tensor(w)) for p, w in zip(only0, first))


def test_decoder_reports_refused_merges_without_changing_the_result():
    bars = parse_path(TWO_BARS)
    rep: dict = {}
    lab = partition_from_predictions(bars, np.zeros(8), np.ones((8, 8)), report=rep)
    assert np.array_equal(lab, partition_from_predictions(bars, np.zeros(8), np.ones((8, 8))))
    assert rep["vetoed"] == list(range(8))  # the two bars wanted to merge across the background
    rep = {}
    partition_from_predictions(plus(), np.zeros(12), np.ones((12, 12)), report=rep)
    assert rep["vetoed"] == []


def _pair_record(contours, target, inp, weight=1.0):
    from strokegen.attempts.check import SEG_FLAGS
    n = len(target)
    return dict(contours=contours, seg_stroke=np.asarray(target), input_seg_stroke=np.asarray(inp),
                seg_flags=np.zeros((n, len(SEG_FLAGS)), np.float32), weight=weight, char="十")


def test_refine_dataset_keeps_inputs_aligned_under_augmentation():
    from strokegen.attempts.labels import same_partition
    torch.manual_seed(3)
    ds = RefineDataset([_pair_record(plus(), PLUS_LAB, PLUS_LAB, 0.25)], train=True)
    for _ in range(5):
        it = ds[0]
        assert len(it["in_lab"]) == len(it["feat"]) == len(it["lab"]) == len(it["rfeat"])
        assert same_partition(it["in_lab"], it["lab"])  # identity pair stays identity after subdivision
        assert it["weight"] == 0.25
        assert np.array_equal(it["rfeat"][:, -1], (it["in_lab"] != it["in_lab"][it["nxt"]]).astype(np.float32))


def test_train_epoch_runs_on_cpu():
    """One optimiser step on two pairs: checks the training code runs, not that the refiner improves."""
    from torch.utils.data import DataLoader

    from strokegen.train_cut import lr_schedule
    from strokegen.train_refine import input_baseline, make_optimizer, train_epoch
    torch.manual_seed(0)
    recs = [_pair_record(plus(), PLUS_LAB, np.zeros(12, int)), _pair_record(plus(), PLUS_LAB, PLUS_LAB, 0.5)]
    model = RefineCutModel.from_cut(CutModel(**TINY).state_dict(), n_rfeat=N_RFEAT, **TINY)
    dl = DataLoader(RefineDataset(recs, True), 2, collate_fn=collate_refine)
    opt = make_optimizer(model, 1e-4, 1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule(1, 1))
    before = [p.detach().clone() for p in model.new_parameters()]
    losses, steps = train_epoch(model, dl, opt, sched, torch.amp.GradScaler(enabled=False), torch.device("cpu"))
    assert steps == 1 and np.isfinite(losses).all()
    assert any(not torch.equal(b, p) for b, p in zip(before, model.new_parameters()))
    base = input_baseline(recs)
    assert base["glyph_exact"] == 0.5


# ---------------------------------------------------------------- the refinement loop

def _tiny_checkpoints(tmp_path, refine: bool = True):
    torch.manual_seed(0)
    cut = CutModel(**TINY)
    paths = dict(cut_ckpt=str(tmp_path / "cut.pt"), order_ckpt=str(tmp_path / "order.pt"))
    torch.save(dict(model=cut.state_dict(), args=dict(TINY, feat_version=features.FEATURE_VERSION), epoch=0),
               paths["cut_ckpt"])
    torch.save(dict(model=OrderNet(layers=1).state_dict(), args=dict(layers=1, feat_version=order.FEATURE_VERSION),
                    epoch=0), paths["order_ckpt"])
    if refine:
        ref = RefineCutModel.from_cut(cut.state_dict(), n_rfeat=N_RFEAT, **TINY)
        paths["refine_ckpt"] = str(tmp_path / "refine.pt")
        torch.save(dict(model=ref.state_dict(), args=dict(TINY, feat_version=features.FEATURE_VERSION,
                                                          refine_version=REFINE_VERSION), epoch=0),
                   paths["refine_ckpt"])
    return paths


GLYPHS = [plus(), parse_path(TWO_BARS)]


def test_zero_init_refiner_leaves_predictions_unchanged(tmp_path):
    from strokegen.predict import StrokeGenerator
    p = _tiny_checkpoints(tmp_path)
    plain = StrokeGenerator(p["cut_ckpt"], p["order_ckpt"], device="cpu")
    loop = StrokeGenerator(p["cut_ckpt"], p["order_ckpt"], device="cpu", refine_iters=3,
                           refine_ckpt=p["refine_ckpt"])
    try:
        a = plain.predict_contours_batch(GLYPHS, workers=1)
        b = loop.predict_contours_batch(GLYPHS, workers=1)
        ref = loop.refine_ref(1)
    finally:
        plain.close()
        loop.close()
    for x, y in zip(a, b):
        assert x["strokes"] == y["strokes"] and np.array_equal(x["_seg_stroke"], y["_seg_stroke"])
        assert "_rounds" not in x
        assert y["_refine"]["rounds"] == 0 and y["_refine"]["stop"] in ("passed", "unchanged")
        assert y["_rounds"][0]["checker"] is not None
    assert ref["round"] == 1 and ref["refine_iters"] == 3 and "refine_sha" in ref and "refine_sha" not in \
        loop.source_ref()


class _Scripted(torch.nn.Module):
    """Stands in for a cut model: returns logits that decode to the next labelling of `script`."""

    def __init__(self, script):
        super().__init__()
        self.script = list(script)
        self.calls = 0

    def forward(self, batch):
        lab = torch.as_tensor(self.script[min(self.calls, len(self.script) - 1)])
        self.calls += 1
        B, N = batch["pad"].shape
        n = len(lab)
        cut = torch.full((B, N), -10.0)
        aff = torch.full((B, N, N), -10.0)
        nxt = batch["nxt"][0, :n]
        cut[:, :n] = torch.where(lab != lab[nxt], 10.0, -10.0)
        aff[:, :n, :n] = torch.where(lab[:, None] == lab[None, :], 10.0, -10.0)
        return dict(cut=cut, aff=aff)


ONE = np.zeros(12, np.int64)  # the whole plus as one stroke: its median cannot cover it
EVERY = np.arange(12)  # every segment a stroke of its own


def _scripted_generator(tmp_path, base, refined, k=3):
    from strokegen.predict import StrokeGenerator
    p = _tiny_checkpoints(tmp_path)
    gen = StrokeGenerator(p["cut_ckpt"], p["order_ckpt"], device="cpu", refine_iters=k, refine_ckpt=p["refine_ckpt"])
    gen.cut, gen.refiner = _Scripted([base]), _Scripted(refined)
    return gen


def test_loop_stops_when_the_checker_passes(tmp_path):
    from strokegen.attempts.labels import same_partition
    gen = _scripted_generator(tmp_path, ONE, [PLUS_LAB])
    try:
        r = gen.predict_contours_batch([plus()], workers=1)[0]
    finally:
        gen.close()
    assert r["_refine"] == dict(rounds=1, stop="passed", passed=True)
    assert not r["_rounds"][0]["checker"]["passed"] and r["_rounds"][1]["checker"]["passed"]
    assert same_partition(r["_seg_stroke"], PLUS_LAB) and len(r["strokes"]) == 2


def test_loop_stops_on_oscillation_and_after_k_rounds(tmp_path):
    gen = _scripted_generator(tmp_path, ONE, [EVERY, ONE])
    try:
        r = gen.predict_contours_batch([plus()], workers=1)[0]
    finally:
        gen.close()
    assert r["_refine"]["stop"] == "oscillation" and r["_refine"]["rounds"] == 2  # ONE -> EVERY -> ONE
    assert all(rd["checker"] is not None for rd in r["_rounds"])

    gen = _scripted_generator(tmp_path, ONE, [EVERY, np.arange(12) // 2], k=2)
    try:
        r = gen.predict_contours_batch([plus()], workers=1)[0]
    finally:
        gen.close()
    assert r["_refine"] == dict(rounds=2, stop="max_iters", passed=False)  # ONE -> EVERY -> pairs, K = 2
    assert gen.refiner.calls == 2


# ---------------------------------------------------------------- dashboard + evaluation

def test_dashboard_records_every_round(store):
    from test_pipeline import _load_tool
    dashboard = _load_tool("dashboard", "tools/dashboard.py")
    c = plus()
    bad, good = hw_from_labels(c, ONE), hw_from_labels(c, PLUS_LAB)
    chk = [{"passed": False, "failures": [], "strokes": []}, {"passed": True, "failures": [], "strokes": []}]
    pred = {**good, "segStroke": PLUS_LAB.tolist(), "_rounds": [
        {"round": 0, **bad, "seg_stroke": ONE.tolist(), "vetoed": [3], "checker": chk[0]},
        {"round": 1, **good, "seg_stroke": PLUS_LAB.tolist(), "vetoed": [], "checker": chk[1]}]}

    class Stub:
        def source_ref(self):
            return dict(MODEL_REF)

        def refine_ref(self, round_):
            return dict(MODEL_REF, refine_sha="r" * 16, round=round_)

    rec = dashboard.Recorder(store, Stub())
    aid = rec.record_now("font-a", "十", c, pred)
    rec.flush("font-a", "十")
    hist = store.history("font-a", "十")
    assert [a["source_type"] for a in hist] == ["model", "refiner"]
    assert hist[1]["attempt_id"] == aid and hist[1]["parent_attempt_id"] == hist[0]["attempt_id"]
    assert hist[0]["labels"]["vetoed"] == [3] and hist[1]["source_ref"]["round"] == 1
    assert [a["checker"]["passed"] for a in hist] == [False, True]  # the loop's checks, not re-run
    assert store.staged("font-a", "十")["attempt_id"] == aid
    assert dashboard.public({"strokes": [], "_rounds": [], "refine": {}}) == {"strokes": [], "refine": {}}


def test_eval_refine_report(tmp_path):
    from strokegen.eval_refine import collect, markdown, summarize
    from strokegen.predict import StrokeGenerator
    c = plus()
    hw = hw_from_labels(c, PLUS_LAB)
    recs = [dict(char=ch, font="f", contours=c, seg_stroke=PLUS_LAB, medians=[np.asarray(m, float)
                                                                            for m in hw["medians"]])
            for ch in ("十", "a")]
    p = _tiny_checkpoints(tmp_path)
    gen = StrokeGenerator(p["cut_ckpt"], p["order_ckpt"], device="cpu", refine_iters=2, refine_ckpt=p["refine_ckpt"])
    try:
        rows = collect(gen, recs, 2, workers=1)
    finally:
        gen.close()
    summ = summarize(rows, 2)
    assert set(summ) == {"all", "cjk", "latin", "lowercase_a"}
    for table in summ.values():
        assert [t["K"] for t in table] == [0, 1, 2]
        # a zero-init refiner changes nothing: every K equals the single pass, and nothing regresses
        assert all({k: v for k, v in t.items() if k != "K"} == {k: v for k, v in table[0].items() if k != "K"}
                   for t in table)
        assert table[0]["regressed"] == 0 and table[0]["oscillated"] == 0
    md = markdown({"split test": dict(summary=summ, stops={"unchanged": 2}, note="two glyphs")},
                  dict(k=2, refine_ckpt="refine.pt", refine_sha="r", refine_epoch=0, cut_sha="c",
                       commands=["uv run python -m strokegen.eval_refine --k 2"]))
    assert "### lowercase_a (1 glyphs)" in md and "uv run python -m strokegen.eval_refine --k 2" in md
