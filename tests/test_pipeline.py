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


@pytest.mark.skipif(not HAS_DATA, reason="run python -m strokegen.build_dataset first")
def test_gt_partition_roundtrip():
    from strokegen.data import load_split

    for r in load_split("val")[:40]:
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
