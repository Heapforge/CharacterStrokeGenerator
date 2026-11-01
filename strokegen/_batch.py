"""Pure-CPU worker functions for bulk parallel glyph prediction.

Kept separate from predict.py so spawned worker processes do not import torch or initialize CUDA,
and BLAS thread pools stay single-threaded per worker process.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from .decode import assemble_strokes, partition_from_predictions  # noqa: E402
from .features import glyph_features  # noqa: E402
from .order import pair_features, stroke_features, stroke_record  # noqa: E402
from .svg_io import load_svg_glyph, segs_to_path  # noqa: E402


def stage_a_from_contours(contours) -> tuple[list | None, dict | None]:
    if contours is None:
        return None, None
    if not contours:
        return [], None
    f = glyph_features(contours)
    f["index"] = 0
    return contours, f


def stage_a_from_svg(svg_text: str | None) -> tuple[list | None, dict | None]:
    if not svg_text:
        return None, None
    return stage_a_from_contours(load_svg_glyph(svg_text))


def stage_a_from_file(path: str | None) -> tuple[list | None, dict | None]:
    if not path or not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as fh:
        svg_text = fh.read()
    return stage_a_from_svg(svg_text)


def stage_bc_prep(args: tuple[list, np.ndarray, np.ndarray]) -> tuple:
    contours, cut_p, aff_p = args
    lab = partition_from_predictions(contours, cut_p, aff_p)
    assembled = assemble_strokes(contours, lab)
    kept = [k for k, s in enumerate(assembled) if s]
    if not kept:
        return lab, kept, [], [], None, None
    strokes = [assembled[k] for k in kept]
    records = [stroke_record(loops) for loops in strokes]
    paths = [" ".join(segs_to_path(loop) for loop in s) for s in strokes]
    medians = [r["median"] for r in records]
    return lab, kept, paths, medians, stroke_features(records), pair_features(records)
