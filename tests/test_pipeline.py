import os

import numpy as np
import pytest

from strokegen.decode import assemble_strokes, fill_unassigned, match_strokes, partition_from_predictions
from strokegen.medians import stroke_median
from strokegen.raster import iou, render
from strokegen.svg_io import Seg, load_svg_glyph, parse_path, segs_to_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_DATA = os.path.exists(os.path.join(ROOT, "data", "processed", "val.pkl"))


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
        g = render(r["contours"], 256)
        u = np.zeros_like(g)
        for loops in st:
            if loops:
                u = np.maximum(u, render(loops, 256))
        assert iou(u, g) > 0.99
        # oracle predictions decode back to the GT partition
        nxt = np.concatenate([np.roll(np.arange(len(c)), -1) + o for c, o in
                              zip(r["contours"], np.cumsum([0] + [len(c) for c in r["contours"]])[:-1])])
        cut = (lab != lab[nxt]).astype(float)
        aff = (lab[:, None] == lab[None, :]).astype(float)
        pred = partition_from_predictions(r["contours"], cut, aff)
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


HAS_CKPT = (
    os.path.exists(os.path.join(ROOT, "checkpoints", "cut.pt"))
    and os.path.exists(os.path.join(ROOT, "checkpoints", "order.pt"))
)


@pytest.mark.skipif(not HAS_CKPT, reason="requires checkpoints/cut.pt and checkpoints/order.pt")
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


def test_dashboard_predict_batch_unpacks_once(tmp_path, monkeypatch):
    import importlib.util
    import io
    import json
    import subprocess

    spec = importlib.util.spec_from_file_location("dashboard", os.path.join(ROOT, "tools", "dashboard.py"))
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)

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

    class DummySock:
        def __init__(self, data: bytes):
            self._r = io.BytesIO(data)
            self.w = io.BytesIO()

        def makefile(self, mode: str, *args, **kwargs):
            return self._r if "r" in mode else self.w

        def sendall(self, data: bytes):
            self.w.write(data)

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
