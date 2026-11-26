"""Synthetic glyphs and a throwaway attempt store, shared by the attempt-store tests."""
import os
import sys

import numpy as np
import pytest

from strokegen.decode import assemble_strokes
from strokegen.medians import stroke_median
from strokegen.svg_io import parse_path, segs_to_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # tools/ is not an installed package

# a plus sign with 100-unit arms: one contour, cut into a horizontal (0) and a vertical (1) bar
PLUS_D = ("M 450 150 L 550 150 L 550 450 L 850 450 L 850 550 L 550 550 L 550 850 L 450 850 "
          "L 450 550 L 150 550 L 150 450 L 450 450 Z")
PLUS_LAB = np.array([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1])


def plus():
    return parse_path(PLUS_D)


def hw_from_labels(contours, lab) -> dict:
    """hanzi-writer data the way the pipeline builds it from a segment labelling."""
    st = assemble_strokes(contours, np.asarray(lab))
    return {"strokes": [" ".join(segs_to_path(loop) for loop in s) for s in st],
            "medians": [[[int(x), int(y)] for x, y in stroke_median(s)] for s in st]}


def svg_of(contours) -> str:
    """Font-style SVG (y down) for hw-frame contours made of lines."""
    d = " ".join(" ".join(("M" if i == 0 else "L") + f" {s.p0[0]:g} {900 - s.p0[1]:g}" for i, s in enumerate(c)) + " Z"
                 for c in contours)
    return f'<svg viewBox="0 0 1024 1024"><path d="{d}"/></svg>'



@pytest.fixture
def store(tmp_path):
    from strokegen.attempts import AttemptStore
    (tmp_path / "hw").mkdir()
    s = AttemptStore(str(tmp_path / "attempts"), hwdata=str(tmp_path / "hw"))
    yield s
    s.close()


MODEL_REF = {"cut_sha": "c" * 16, "order_sha": "o" * 16}
