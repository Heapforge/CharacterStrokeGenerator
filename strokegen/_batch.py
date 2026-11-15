"""Pure-CPU worker functions for bulk parallel glyph prediction.

Kept separate from predict.py so worker processes do not need torch for their own work. The
callers start workers with single-threaded BLAS (see predict.worker_env).

Every function here returns an error string instead of raising, so one broken glyph does not
abort a whole batch.
"""
from __future__ import annotations

import os

import numpy as np

from .decode import assemble_strokes, partition_from_predictions
from .features import glyph_features
from .order import pair_features, stroke_features, stroke_record
from .svg_io import load_svg_glyph, segs_to_path

MISSING = "missing"


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def stage_a_from_contours(contours) -> tuple[list | None, dict | None, str | None]:
    """(contours, Stage A features, error). contours None means there was no input."""
    if contours is None:
        return None, None, MISSING
    if not contours:
        return [], None, None
    try:
        f = glyph_features(contours)
    except Exception as e:  # noqa: BLE001  (reported per glyph)
        return None, None, _err(e)
    f["index"] = 0
    return contours, f, None


def stage_a_from_svg(svg_text: str | None) -> tuple[list | None, dict | None, str | None]:
    if not svg_text:
        return None, None, MISSING
    try:
        contours = load_svg_glyph(svg_text)
    except Exception as e:  # noqa: BLE001
        return None, None, _err(e)
    return stage_a_from_contours(contours)


def stage_a_from_file(path: str | None) -> tuple[list | None, dict | None, str | None]:
    if not path or not os.path.exists(path):
        return None, None, MISSING
    try:
        with open(path, encoding="utf-8") as fh:
            svg_text = fh.read()
    except (OSError, UnicodeError) as e:
        return None, None, _err(e)
    return stage_a_from_svg(svg_text)


def stage_bc_prep(args: tuple) -> tuple:
    """Partition (unless labels are given), assemble strokes, medians and Stage C features.

    args: (contours, cut_p, aff_p, labels); labels None -> decode cut_p / aff_p.
    Returns (lab, kept, paths, medians, stroke_feats, pair_feats, error).
    """
    contours, cut_p, aff_p, labels = args
    try:
        lab = partition_from_predictions(contours, cut_p, aff_p) if labels is None else np.asarray(labels)
        assembled = assemble_strokes(contours, lab)
        kept = [k for k, s in enumerate(assembled) if s]
        if not kept:
            return lab, kept, [], [], None, None, None
        strokes = [assembled[k] for k in kept]
        records = [stroke_record(loops) for loops in strokes]
        paths = [" ".join(segs_to_path(loop) for loop in s) for s in strokes]
        medians = [r["median"] for r in records]
        return lab, kept, paths, medians, stroke_features(records), pair_features(records), None
    except Exception as e:  # noqa: BLE001
        return None, [], [], [], None, None, _err(e)


def decode_partition(args: tuple) -> np.ndarray:
    """partition_from_predictions(contours, cut_p, aff_p), for process pools."""
    contours, cut_p, aff_p = args
    return partition_from_predictions(contours, cut_p, aff_p)
