import importlib.util
import io
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from strokegen.decode import (
    _assemble,
    assemble_strokes,
    cluster_chains,
    fill_unassigned,
    match_strokes,
    order_chains,
    partition_from_predictions,
)
from strokegen.medians import stroke_median, stroke_polygon
from strokegen.raster import iou, render, render_union
from strokegen.svg_io import load_svg_glyph, parse_path, segs_to_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_DATA = os.path.exists(os.path.join(ROOT, "data", "processed", "val.pkl"))

# outer square with a counter wound the other way (nonzero winding leaves it open)
RING = "M 100 100 L 900 100 L 900 900 L 100 900 Z M 700 700 L 700 300 L 300 300 L 300 700 Z"


def oracle(contours, lab):
    """Perfect cut / affinity predictions for a segment labelling."""
    lab = np.asarray(lab)
    nxt = np.concatenate([np.roll(np.arange(len(c)), -1) + o for c, o in
                          zip(contours, np.cumsum([0] + [len(c) for c in contours])[:-1])])
    return (lab != lab[nxt]).astype(float), (lab[:, None] == lab[None, :]).astype(float)


def test_parse_relative_and_shorthand():
    c = parse_path("M10 10l10 0h5v5q5 5 0 10t-10 0c-1 -2 -3 -4 -5 -5s-2 -2 -5 -10z")
    assert len(c) == 1
    kinds = "".join(s.kind for s in c[0])
    assert kinds == "LLLQQCCL"
    for a, b in zip(c[0], c[0][1:] + c[0][:1]):
        assert np.allclose(a.p1, b.p0)
    # T reflects the previous Q control point
    assert np.allclose(c[0][4].pts[1], 2 * c[0][3].p1 - c[0][3].pts[1])


def test_serialize_roundtrip():
    c = parse_path("M 0 0 L 100 0 Q 120 50 100 100 C 60 120 40 120 0 100 Z")[0]
    again = parse_path(segs_to_path(c))[0]
    assert len(again) == len(c)
    for a, b in zip(c, again):
        assert a.kind == b.kind and np.allclose(a.pts, b.pts)


def test_svg_transform_matches_hanzi_writer_frame():
    svg = '<svg viewBox="-10 0 1034 1024"><path d="M117 499L200 499L200 600Z"/></svg>'
    c = load_svg_glyph(svg)
    assert np.allclose(c[0][0].p0, [117, 401])  # y_hw = 900 - y_svg


def test_median_of_bar_is_center_line():
    bar = parse_path("M 100 400 L 900 400 L 900 460 L 100 460 Z")[0]
    m = stroke_median([bar])
    assert len(m) >= 2
    assert np.all(np.abs(m[:, 1] - 430) <= 30)  # inside the bar
    mid = m[(m[:, 0] > 200) & (m[:, 0] < 800)]
    assert np.all(np.abs(mid[:, 1] - 430) <= 3)  # on the centre line away from the ends
    assert m[:, 0].max() - m[:, 0].min() > 650


def test_cross_split_into_two_strokes():
    # a plus sign as one contour; cut it into horizontal + vertical bars
    plus = parse_path("M 400 100 L 600 100 L 600 400 L 900 400 L 900 600 L 600 600 L 600 900 L 400 900 "
                      "L 400 600 L 100 600 L 100 400 L 400 400 Z")
    # segments: 0 bottom of vertical, 1 right-lower inner, 2 right-arm bottom, 3 right end, 4 right-arm top ...
    lab = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1])
    strokes = assemble_strokes(plus, lab)
    assert len(strokes) == 2
    union = np.maximum(render(strokes[0], 256), render(strokes[1], 256))
    assert iou(union, render(plus, 256)) > 0.99
    horiz = render(strokes[0], 256)
    ys, xs = np.nonzero(horiz > 0.5)
    assert xs.max() - xs.min() > ys.max() - ys.min()


# square whose right side is a curve
_GLYPH = parse_path("M 0 0 L 100 0 Q 150 50 100 100 L 0 100 Z")


def test_decompose_matches_degenerate_quad_to_line():
    from strokegen.build_dataset import decompose

    # hw data often writes straight lines as Qs whose control point is not the midpoint
    lab = decompose(_GLYPH, {"strokes": ["M 0 0 Q 90 0 100 0 Q 150 50 100 100 L 0 100 Q 0 10 0 0 Z"]})
    assert lab["coverage"] == 1.0 and (lab["seg_stroke"] == 0).all()


def test_decompose_line_is_not_matched_to_curve():
    from strokegen.build_dataset import decompose

    lab = decompose(_GLYPH, {"strokes": ["M 0 0 L 100 0 L 100 100 L 0 100 Z"]})
    # the straight side is a bridge, not a match for the font curve
    assert lab["coverage"] == 0.75
    assert [e[0] for e in lab["loops"][0]] == ["F", "B", "F", "F"]
    assert lab["n_bridge_bad"] == 0


def test_decompose_bridge_ends_on_edge_vs_off_outline():
    from strokegen.build_dataset import decompose

    on_edge = decompose(_GLYPH, {"strokes": ["M 50 0 L 100 0 Q 150 50 100 100 L 0 100 L 50 0 Z"]})
    assert on_edge["n_bridge_on_edge"] > 0 and on_edge["n_bridge_bad"] == 0
    off = decompose(_GLYPH, {"strokes": ["M 50 50 L 100 0 Q 150 50 100 100 L 50 50 Z"]})
    assert off["n_bridge_bad"] > 0


@pytest.mark.skipif(not HAS_DATA, reason="run python -m strokegen.build_dataset first")
def test_dataset_records_pass_quality_gates():
    from strokegen.build_dataset import MIN_COVERAGE, MIN_VERTEX_MATCH
    from strokegen.data import load_split

    test, val = load_split("test"), load_split("val")
    for r in test + val:
        assert r["vertex_match"] >= MIN_VERTEX_MATCH and r["coverage"] >= MIN_COVERAGE, r["char"]
    # identical outlines never straddle splits
    assert not {r["hash"] for r in test} & {r["hash"] for r in val}


@pytest.mark.skipif(not HAS_DATA, reason="run python -m strokegen.build_dataset first")
def test_gt_partition_roundtrip():
    from strokegen.data import load_split

    # records with mid-edge cuts or unassigned segments can't round-trip by construction;
    # test_dataset_records_pass_quality_gates covers that those are still sane
    for r in [r for r in load_split("val") if r["exact"]][:40]:
        lab = fill_unassigned(r["contours"], r["seg_stroke"])
        st = assemble_strokes(r["contours"], lab)
        assert iou(render_union(st), render(r["contours"], 256)) > 0.99, r["char"]
        # oracle predictions decode back to the GT partition
        pred = partition_from_predictions(r["contours"], *oracle(r["contours"], lab))
        w = np.array([s.length() for c in r["contours"] for s in c])
        assert match_strokes(pred, r["seg_stroke"], w)["glyph_exact"]


def test_merge_strokes_rebuilds_outline_and_median():
    from strokegen.predict import merge_strokes

    # an L shape (one contour) cut at the corner into a vertical bar (0) and a horizontal foot (1)
    ell = parse_path("M 100 900 L 100 100 L 900 100 L 900 300 L 300 300 L 300 900 Z")
    lab = np.array([0, 1, 1, 1, 0, 0])
    first = [[200, 880], [200, 300]]  # top -> down
    second = [[300, 200], [880, 200]]  # left -> right
    r = merge_strokes(ell, lab, [0, 1], first, second)
    merged = parse_path(r["stroke"])
    assert len(merged) == 1 and len(merged[0]) == len(ell[0])  # cut bridges gone, font segments kept
    assert iou(render(merged, 256), render(ell, 256)) > 0.99
    m = np.array(r["median"])
    assert np.hypot(*(m[0] - first[0])) < 150 and np.hypot(*(m[-1] - second[-1])) < 150
    # same strokes in the other writing order -> reversed median
    r2 = merge_strokes(ell, lab, [1, 0], second[::-1], first[::-1])
    assert np.array_equal(np.array(r2["median"]), m[::-1])


def test_merge_strokes_disconnected_concatenates_medians():
    from strokegen.predict import merge_strokes

    bars = parse_path("M 100 100 L 900 100 L 900 200 L 100 200 Z M 100 700 L 900 700 L 900 800 L 100 800 Z")
    r = merge_strokes(bars, [0] * 4 + [1] * 4, [0, 1], [[120, 150], [880, 150]], [[120, 750], [880, 750]])
    assert len(parse_path(r["stroke"])) == 2
    assert r["median"] == [[120, 150], [880, 150], [120, 750], [880, 750]]
    # crossing strokes: the skeleton can't follow "first, then second", so the medians are joined
    plus = parse_path("M 400 100 L 600 100 L 600 400 L 900 400 L 900 600 L 600 600 L 600 900 L 400 900 "
                      "L 400 600 L 100 600 L 100 400 L 400 400 Z")
    lab = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
    r = merge_strokes(plus, lab, [0, 1], [[120, 500], [880, 500]], [[500, 880], [500, 120]])
    assert r["median"] == [[120, 500], [880, 500], [500, 880], [500, 120]]
    with pytest.raises(ValueError):
        merge_strokes(bars, [0] * 3, [0, 1], [[0, 0]], [[1, 1]])


def _checkpoints_usable() -> bool:
    try:
        import torch

        from strokegen import features, order

        for name, version in (("cut.pt", features.FEATURE_VERSION), ("order.pt", order.FEATURE_VERSION)):
            ck = torch.load(os.path.join(ROOT, "checkpoints", name), map_location="cpu", weights_only=True)
            if ck["args"].get("feat_version", 1) != version:
                return False
        return True
    except (OSError, KeyError):
        return False


HAS_CKPT = _checkpoints_usable()


@pytest.mark.skipif(not HAS_CKPT, reason="requires checkpoints/{cut,order}.pt trained on the current features")
def test_predict_contours_batch_matches_single():
    from strokegen.predict import StrokeGenerator

    bar = parse_path("M 100 450 L 900 450 L 900 530 L 100 530 Z")
    plus = parse_path("M 460 120 L 540 120 L 540 460 L 880 460 L 880 540 L 540 540 L 540 880 L 460 880 "
                      "L 460 540 L 120 540 L 120 460 L 460 460 Z")
    two_bars = parse_path("M 150 700 L 850 700 L 850 770 L 150 770 Z M 100 220 L 900 220 L 900 290 L 100 290 Z")
    glyphs = [bar, [], plus, two_bars]

    gen = StrokeGenerator(device="cpu")
    try:
        singles = [gen.predict_contours(g) for g in glyphs]
        batched = gen.predict_contours_batch(glyphs, workers=2)
    finally:
        gen.close()

    assert len(batched) == len(singles)
    for s, b in zip(singles, batched):
        assert b["strokes"] == s["strokes"]
        assert b["medians"] == s["medians"]
        assert np.array_equal(b["_seg_stroke"], s["_seg_stroke"])


def _load_tool(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_fontforge() -> bool:
    for py in ("/usr/bin/python3", shutil.which("python3")):
        if py and os.path.exists(py) and subprocess.run([py, "-I", "-c", "import fontforge"],
                                                        capture_output=True, check=False).returncode == 0:
            return True
    return False


class DummySock:
    """Enough of a socket for BaseHTTPRequestHandler to serve one request."""

    def __init__(self, data: bytes):
        self._r = io.BytesIO(data)
        self.w = io.BytesIO()

    def makefile(self, mode: str, *args, **kwargs):
        return self._r if "r" in mode else self.w

    def sendall(self, data: bytes):
        self.w.write(data)


def _request(handler, method: str, path: str, body: bytes = b"", headers: dict | None = None):
    """Serve one raw request with `handler`; returns (status, parsed JSON body)."""
    head = {"Host": "127.0.0.1:8417", "Content-Length": str(len(body)), "Content-Type": "application/json",
            **(headers or {})}
    raw = f"{method} {path} HTTP/1.1\r\n".encode() + b"".join(
        f"{k}: {v}\r\n".encode() for k, v in head.items() if v is not None) + b"\r\n" + body
    sock = DummySock(raw)
    handler(sock, ("127.0.0.1", 12345), None)
    status_line, _, rest = sock.w.getvalue().partition(b"\r\n")
    return int(status_line.split()[1]), json.loads(rest.split(b"\r\n\r\n", 1)[1] or b"null")


@pytest.mark.skipif(not _has_fontforge(), reason="needs the FontForge Python bindings")
def test_dashboard_predict_batch_unpacks_once(tmp_path, monkeypatch):
    dashboard = _load_tool("dashboard", "tools/dashboard.py")

    glyphs_dir = tmp_path / "font_glyphs"
    hwdata_dir = tmp_path / "hwdata"
    glyphs_dir.mkdir()
    hwdata_dir.mkdir()
    existing_A = {"strokes": ["M 0 0 L 10 0 L 10 10 Z"], "medians": [[[0, 0], [10, 10]]]}
    (hwdata_dir / "A.json").write_text(json.dumps(existing_A), encoding="utf-8")

    monkeypatch.setattr(dashboard, "GLYPHS", str(glyphs_dir))
    monkeypatch.setattr(dashboard, "HWDATA", str(hwdata_dir))

    export_calls = []
    real_run = subprocess.run

    def spy_run(cmd, *args, **kwargs):
        if dashboard.EXPORT_SCRIPT in cmd:
            export_calls.append(list(cmd))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(dashboard.subprocess, "run", spy_run)

    class StubModel:
        def __init__(self):
            self.batch_calls = []

        def predict_files_batch(self, paths):
            self.batch_calls.append(list(paths))
            out = []
            for p in paths:
                if p is None or not os.path.exists(p):
                    out.append(None)
                else:
                    contours = load_svg_glyph(open(p, encoding="utf-8").read())
                    n_segs = sum(len(c) for c in contours)
                    out.append({
                        "strokes": [segs_to_path(contours[0])] if contours else [],
                        "medians": [[[100, 100], [200, 200]]] if contours else [],
                        "segStroke": [0] * n_segs,
                    })
            return out

    stub = StubModel()
    monkeypatch.setattr(dashboard.Handler, "model", stub, raising=False)

    # 105 characters (> 100 threshold), plus 1 character absent from noto-sans
    chars = (
        [chr(c) for c in range(ord("0"), ord("9") + 1)]
        + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        + [chr(c) for c in range(0xC0, 0xC0 + 43)]
        + ["\u9fff"]
    )
    assert len(chars) == 106
    payload = json.dumps({"font": "noto-sans", "chars": chars}).encode("utf-8")
    raw_req = (
        b"POST /api/predict_batch HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        + f"Content-Length: {len(payload)}\r\n".encode("ascii")
        + b"Content-Type: application/json\r\n\r\n"
        + payload
    )

    sock = DummySock(raw_req)
    dashboard.Handler(sock, ("127.0.0.1", 12345), None)
    resp = sock.w.getvalue().split(b"\r\n\r\n", 1)[1]
    body = json.loads(resp.decode("utf-8"))

    # Font unpacked once across all 106 characters, and predicted in a single batch call
    assert len(export_calls) == 1
    assert len(stub.batch_calls) == 1
    assert len(body["results"]) == 106

    idx_A = chars.index("A")
    assert body["results"][idx_A]["strokes"]
    assert body["results"][idx_A]["existing"] == existing_A
    idx_B = chars.index("B")
    assert body["results"][idx_B]["strokes"]
    assert body["results"][idx_B]["existing"] is None
    assert body["results"][-1] == {"error": "no such glyph"}

    # Second call with the same characters does not unpack the font again
    sock2 = DummySock(raw_req)
    dashboard.Handler(sock2, ("127.0.0.1", 12345), None)
    assert len(export_calls) == 1


# ---------------------------------------------------------------- decoding


def test_ring_stays_one_stroke_with_its_counter():
    ring = parse_path(RING)
    n = sum(len(c) for c in ring)
    # a whole contour needs no bridge, so the counter may join the outline around it
    lab = partition_from_predictions(ring, np.zeros(n), np.ones((n, n)))
    assert (lab == 0).all()
    assert iou(render_union(assemble_strokes(ring, lab)), render(ring, 256)) > 0.99


def test_separate_contours_are_not_merged_across_the_background():
    bars = parse_path("M 100 100 L 900 100 L 900 200 L 100 200 Z M 100 700 L 900 700 L 900 800 L 100 800 Z")
    lab = partition_from_predictions(bars, np.zeros(8), np.ones((8, 8)))
    assert len(set(lab[:4])) == 1 and len(set(lab[4:])) == 1 and lab[0] != lab[4]


def test_shared_cut_with_spilling_curves_becomes_straight():
    # L shape cut diagonally at its outer corner: the tangent-continuous bridges of both strokes
    # would bulge out below / left of the corner, so both take the shared straight chord
    ell = parse_path("M 100 100 L 900 100 L 900 300 L 300 300 L 300 900 L 100 900 Z")
    lab = np.array([1, 1, 1, 0, 0, 0])
    loops, bridges = _assemble(ell, lab)
    assert [b.kind for bs in bridges for b in bs] == ["L", "L"]
    assert iou(render_union(loops), render(ell, 256)) > 0.995
    # without straightening the curves stick out of the glyph
    curved, _ = _assemble(ell, lab, straighten=False)
    assert iou(render_union(curved), render(ell, 256)) < iou(render_union(loops), render(ell, 256))


def test_cluster_veto_is_lifted_when_a_cluster_grows():
    aff = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.8], [0.8, 0.8, 1.0]])
    # 0 + 1 alone is invalid, but fine once 2 joined them
    valid = lambda ids: set(ids) != {0, 1}  # noqa: E731
    lab = cluster_chains(np.arange(3), aff, np.ones(3), 0.5, valid)
    assert (lab == lab[0]).all()


def test_order_chains_finds_the_shortest_cycle():
    import itertools

    from strokegen.decode import Chain, _cycle_cost
    from strokegen.svg_io import Seg

    rng = np.random.default_rng(1)
    for n in (3, 5, 7):
        segs = [Seg("L", rng.uniform(0, 1000, (2, 2))) for _ in range(n)]
        chs = [Chain(0, [i], False) for i in range(n)]
        ends = np.array([s.p1 for s in segs])
        starts = np.array([s.p0 for s in segs])
        D = np.linalg.norm(ends[:, None] - starts[None], axis=-1)
        best = min(_cycle_cost((0, *p), D) for p in itertools.permutations(range(1, n)))
        assert abs(_cycle_cost(order_chains(chs, segs), D) - best) < 1e-6


def test_match_strokes_without_ground_truth():
    m = match_strokes(np.array([0, 0, 1]), np.array([-1, -1, -1]), np.ones(3))
    assert m["n_gt"] == 0 and m["assignment"] == {}
    m = match_strokes(np.array([1, 1, 0]), np.array([0, 0, 1]), np.ones(3))
    assert m["glyph_exact"] and m["assignment"] == {1: 0, 0: 1}


# ---------------------------------------------------------------- geometry and features


def test_stroke_polygon_keeps_the_counter():
    ring = parse_path(RING)
    poly = stroke_polygon(ring)
    assert len(poly.interiors) == 1 and abs(poly.area - (800**2 - 400**2)) < 1.0
    # the median goes round the counter, not across it
    m = stroke_median(ring)
    assert not any(300 < x < 700 and 300 < y < 700 for x, y in m)


def test_inward_normal_feature_points_into_the_glyph():
    from strokegen.features import glyph_features

    ring = parse_path(RING)  # straight segments: the midpoint normal is exact
    f = glyph_features(ring)["feat"]
    mids = np.stack([s.sample(3)[1] for c in ring for s in c])
    probe = mids + 20 * f[:, -2:]
    poly = stroke_polygon(ring)
    from shapely import contains_xy

    assert contains_xy(poly, probe[:, 0], probe[:, 1]).all()


def test_pair_features_match_the_pairwise_definition():
    from strokegen.medians import resample
    from strokegen.order import N_PFEAT, SC, _seg_cross, pair_features

    rng = np.random.default_rng(0)
    strokes = []
    for _ in range(4):
        med = rng.uniform(100, 900, (int(rng.integers(2, 6)), 2))
        lo, hi = med.min(0) - 20, med.max(0) + 20
        strokes.append(dict(median=med, area=1000.0, bbox=np.r_[lo, hi]))
    got = pair_features(strokes)
    meds = [resample(s["median"], 12) for s in strokes]
    for i in range(4):
        assert not got[i, i].any()
        for j in range(4):
            if i == j:
                continue
            bi, bj = strokes[i]["bbox"], strokes[j]["bbox"]
            ox = max(0.0, min(bi[2], bj[2]) - max(bi[0], bj[0]))
            oy = max(0.0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
            wi, hi, wj, hj = bi[2] - bi[0] + 1, bi[3] - bi[1] + 1, bj[2] - bj[0] + 1, bj[3] - bj[1] + 1
            want = np.concatenate([
                (meds[i].mean(0) - meds[j].mean(0)) / SC,
                (meds[i][0] - meds[j][0]) / SC, (meds[i][-1] - meds[j][-1]) / SC,
                (meds[i][0] - meds[j][-1]) / SC, (meds[i][-1] - meds[j][0]) / SC,
                [np.linalg.norm(meds[i][:, None] - meds[j][None], axis=-1).min() / 256.0, ox / min(wi, wj),
                 oy / min(hi, hj), ox * oy / (wi * hi + wj * hj - ox * oy),
                 float(_seg_cross(strokes[i]["median"], strokes[j]["median"]))],
            ])
            assert got.shape[-1] == N_PFEAT and np.allclose(got[i, j], want, atol=1e-6)


def test_decode_order_recovers_a_consistent_order():
    from strokegen.order import decode_order

    perm = np.array([3, 0, 4, 1, 2])
    rank = np.argsort(perm)
    logits = np.where(rank[:, None] < rank[None, :], 4.0, -4.0)
    np.fill_diagonal(logits, 0)
    assert decode_order(logits).tolist() == perm.tolist()


def test_single_quoted_svg_attributes():
    c = load_svg_glyph("<svg viewBox='0 0 1024 1024'><path d='M117 499L200 499L200 600Z'/></svg>")
    assert np.allclose(c[0][0].p0, [117, 401])


# ---------------------------------------------------------------- data


def test_split_depends_only_on_the_outline():
    import hashlib

    from strokegen.build_dataset import split_of

    hashes = [hashlib.sha1(str(i).encode()).hexdigest() for i in range(4000)]
    splits = [split_of(h) for h in hashes]
    assert splits == [split_of(h) for h in hashes]
    frac = {k: splits.count(k) / len(splits) for k in ("train", "val", "test")}
    assert 0.03 < frac["test"] < 0.07 and 0.03 < frac["val"] < 0.07


def test_unknown_fonts_sort_last():
    from strokegen.fonts import FONT_PRIORITY, font_rank

    assert sorted(["new-font", FONT_PRIORITY[1], FONT_PRIORITY[0]], key=font_rank) == [
        FONT_PRIORITY[0], FONT_PRIORITY[1], "new-font"]


def test_lr_schedule_survives_tiny_runs():
    from strokegen.train_cut import lr_schedule

    f = lr_schedule(3, 1000)
    assert all(0.0 <= f(s) <= 1.0 for s in range(5))


# ---------------------------------------------------------------- tools


def test_scratch_corrector_import_has_no_side_effects():
    mod = _load_tool("fix_svgs", "tools/scratch/fix_svgs.py")
    assert callable(mod.check_iou) and callable(mod.main) and not hasattr(mod, "corrections")


@pytest.mark.parametrize("headers,why", [
    ({"Origin": "https://evil.example"}, "cross-origin"),
    ({"Host": "evil.example:8417"}, "Host"),
    ({"Content-Type": "text/plain"}, "application/json"),
    ({"Sec-Fetch-Site": "cross-site"}, "cross-site"),
])
def test_dashboard_refuses_foreign_writes(monkeypatch, headers, why):
    dashboard = _load_tool("dashboard", "tools/dashboard.py")
    monkeypatch.setattr(dashboard.Handler, "allowed_hosts", dashboard.host_header_names(["127.0.0.1"], 8417))
    body = json.dumps({"char": "x", "data": {}}).encode()
    status, res = _request(dashboard.Handler, "POST", "/api/commit", body, headers)
    assert status == 403 and why in res["error"]


def test_dashboard_accepts_its_own_page(monkeypatch, tmp_path):
    dashboard = _load_tool("dashboard", "tools/dashboard.py")
    monkeypatch.setattr(dashboard.Handler, "allowed_hosts", dashboard.host_header_names(["127.0.0.1"], 8417))
    monkeypatch.setattr(dashboard, "HWDATA", str(tmp_path))
    status, res = _request(dashboard.Handler, "POST", "/api/commit", b"{}", {"Origin": "http://127.0.0.1:8417"})
    assert status == 400  # past the guard: rejected for the missing character, not refused
    status, _ = _request(dashboard.Handler, "GET", "/api/data?char=x", headers={"Content-Type": None})
    assert status == 404


def test_dashboard_batch_survives_odd_character_names(monkeypatch, tmp_path):
    """Latin fonts export glyphs named by whitespace and backslash; one of them must not fail the batch."""
    dashboard = _load_tool("dashboard", "tools/dashboard.py")
    font_dir = tmp_path / "glyphs" / "some-font"
    font_dir.mkdir(parents=True)
    names = [" ", "\r", " ", " ", "A"] + (["\\"] if os.sep == "/" else [])
    svg = '<svg viewBox="0 0 1024 1024"><path d="M100 400L900 400L900 480L100 480Z"/></svg>'
    for c in names:
        (font_dir / f"{c}.svg").write_text(svg, encoding="utf-8")
    monkeypatch.setattr(dashboard, "GLYPHS", str(tmp_path / "glyphs"))
    monkeypatch.setattr(dashboard, "HWDATA", str(tmp_path))

    class StubModel:
        def predict_files_batch(self, paths):
            return [None if p is None else {"strokes": ["M 0 0 Z"], "medians": [[[0, 0], [1, 1]]],
                                            "segStroke": [0], "error": None} for p in paths]

    monkeypatch.setattr(dashboard.Handler, "model", StubModel(), raising=False)
    chars = names + ["a/b", ".", ""]
    status, res = _request(dashboard.Handler, "POST", "/api/predict_batch",
                           json.dumps({"font": "some-font", "chars": chars}).encode())
    assert status == 200
    assert all(r.get("strokes") for r in res["results"][:len(names)])
    assert all("invalid character name" in r["error"] for r in res["results"][len(names):])

    for c in names:
        assert dashboard.safe_char(c) == c
    for c in ["/", "\0", ".", "..", " x", "x ", "a/b"]:
        with pytest.raises(ValueError):
            dashboard.safe_char(c)


@pytest.mark.skipif(not HAS_CKPT, reason="requires checkpoints/{cut,order}.pt trained on the current features")
def test_batch_reports_bad_glyphs_without_failing_the_rest():
    from strokegen.predict import GlyphError, StrokeGenerator

    good = '<svg viewBox="0 0 1024 1024"><path d="M100 400L900 400L900 480L100 480Z"/></svg>'
    bad = '<svg viewBox="0 0 1024 1024"><path d="M 0 0 A 1 1 0 0 0 1 1 Z"/></svg>'
    gen = StrokeGenerator(device="cpu")
    try:
        res = gen.predict_svgs_batch([good, bad, good], workers=1)
        with pytest.raises(GlyphError):
            gen.predict_svg(bad)
    finally:
        gen.close()
    assert res[0]["strokes"] and res[2]["strokes"] and "arc" in res[1]["_error"]


def test_dashboard_unpacks_a_new_font_in_the_background(monkeypatch, tmp_path):
    import threading

    dashboard = _load_tool("dashboard", "tools/dashboard.py")
    monkeypatch.setattr(dashboard, "GLYPHS", str(tmp_path))
    monkeypatch.setattr(dashboard, "HWDATA", str(tmp_path))
    release = threading.Event()

    def slow_export(font, chars=None):  # stands in for the FontForge export
        release.wait(5)
        os.makedirs(tmp_path / font, exist_ok=True)
        (tmp_path / font / "A.svg").write_text("<svg/>")
        return str(tmp_path / font)

    monkeypatch.setattr(dashboard, "unpack_font_chars", slow_export)
    font = next(f for f in dashboard.font_names() if dashboard._font_binary_exists(dashboard.FONT_FILES[f]))
    status, res = _request(dashboard.Handler, "GET", f"/api/glyphs?font={font}", headers={"Content-Type": None})
    assert status == 202 and res["unpacking"]
    release.set()
    for _ in range(100):
        status, res = _request(dashboard.Handler, "GET", f"/api/glyphs?font={font}", headers={"Content-Type": None})
        if status == 200:
            break
        threading.Event().wait(0.05)
    assert status == 200 and res["chars"] == ["A"]


def test_a_whole_counter_is_never_a_stroke_of_its_own():
    from strokegen.decode import absorb_counters

    # a square with a counter, and an island (a bar) inside the counter
    glyph = parse_path(RING + " M 350 480 L 650 480 L 650 520 L 350 520 Z")
    lab = np.array([0] * 4 + [1] * 4 + [2] * 4)  # the counter came out as its own stroke
    n = len(lab)
    out = absorb_counters(glyph, lab, np.ones((n, n)))
    assert out[4] == out[0] and out[8] != out[0] and len(set(out.tolist())) == 2
    # the island alone is ink, not a counter: it keeps its stroke
    assert (absorb_counters(glyph, np.array([0] * 8 + [1] * 4), np.ones((n, n))) == [0] * 8 + [1] * 4).all()
