"""Checker results for stored versions, plus per-segment feedback for the refiner.

Pass/fail comes only from tools/verify_glyph.check_glyph (the dashboard's and the repair passes'
checker); this module just runs it and projects its per-stroke failures onto the font segments.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from ..paths import ROOT

# Bump when check_glyph's meaning changes, so stale results can be found and re-run.
CHECK_VERSION = 1

# per-segment feedback channels (refiner input, strokegen.refine); the last one comes from the decoder, not the checker
SEG_FLAGS = ("in_failing_stroke", "near_iou_deficit", "median_not_covering", "stroke_hidden", "unassigned",
             "in_rejected_merge")
DEFICIT_RADIUS_PX = 3  # a segment is near the deficit when a sample lies this close to a missing/extra pixel


def _checker():
    if ROOT not in sys.path:  # tools/ is not an installed package
        sys.path.insert(0, ROOT)
    from tools.verify_glyph import check_glyph
    return check_glyph


def _deficit_segments(contours, deficit: np.ndarray) -> list[int]:
    """Font segments (flat index) with a sample point next to a missing or extra pixel."""
    from scipy.ndimage import binary_dilation

    from tools.fix_hole_strokes import FIT_A, FIT_B, SIZE

    if not deficit.any():
        return []
    near = binary_dilation(deficit, iterations=DEFICIT_RADIUS_PX)
    s = SIZE / 1024
    out = []
    n = 0
    for c in contours:
        for seg in c:
            q = seg.sample(9) @ FIT_A.T + FIT_B
            px = np.clip((q[:, 0] * s).astype(int), 0, SIZE - 1)
            py = np.clip(((900 - q[:, 1]) * s).astype(int), 0, SIZE - 1)
            if near[py, px].any():
                out.append(n)
            n += 1
    return out


def run_check(contours, hw: dict) -> dict:
    """check_glyph's result, stamped with CHECK_VERSION; a crash is a failed check, not an exception.

    Adds "deficit_segments": the font segments next to area the strokes miss or add.
    """
    t = time.time()
    try:
        res = _checker()(contours, hw, masks=True)
        deficit = res.pop("missing_mask", None)
        extra = res.pop("extra_mask", None)
        res["deficit_segments"] = [] if deficit is None else _deficit_segments(contours, deficit | extra)
    except Exception as e:  # noqa: BLE001  (an unreadable version still gets a verdict-able result)
        res = dict(passed=False, failures=["checker_error"], error=f"{type(e).__name__}: {e}", strokes=[])
    res["version"] = CHECK_VERSION
    res["ms"] = round((time.time() - t) * 1000)
    return res


def segment_flags(seg_stroke, checker: dict | None, vetoed=None) -> np.ndarray:
    """(n_segments, len(SEG_FLAGS)) float array of feedback per font segment.

    vetoed: segments in merges the decoder's bridge validator refused (labels["vetoed"] of a model or
    refiner version; None for versions no decoder made)."""
    lab = np.asarray(seg_stroke, np.int64)
    out = np.zeros((len(lab), len(SEG_FLAGS)), np.float32)
    out[:, 4] = lab < 0
    out[[int(i) for i in (() if vetoed is None else vetoed) if 0 <= i < len(lab)], 5] = 1
    checker = checker or {}
    near = [i for i in checker.get("deficit_segments", []) if 0 <= i < len(lab)]
    out[near, 1] = 1
    glyph_fail = bool(checker.get("failures"))
    for k, row in enumerate(checker.get("strokes") or []):
        sel = lab == k
        f = set(row.get("failures", []))
        out[sel, 0] = bool(f) or glyph_fail
        out[sel, 2] = "low_coverage" in f or "median_outside" in f or "short_median" in f
        out[sel, 3] = "hidden" in f
    return out
