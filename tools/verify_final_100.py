#!/usr/bin/env python3
"""Comprehensive verification of all 100 characters in the review queue."""
import json
import os
import sys
import numpy as np
import cairo

sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, parse_hw_path
from strokegen.raster import render, iou, SPAN, X0, Y_TOP, _trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")
DATA_DIR = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
MANIFEST_PATH = os.path.join(ROOT, "training_data", "font_glyphs", "manifest.json")

GOOD_CHARS = set(["4", "È", "É", "Ê", "Ë", "Ö", "ß", "Ē", "Ĕ", "Ė", "Ę", "Ě", "Ų"])

def eval_char(font_contours, strokes, medians, brush_width=128):
    r_font = render(font_contours)
    s = 256.0 / SPAN
    full_unclipped = np.zeros((256, 256), dtype=np.float32)
    full_clipped = np.zeros((256, 256), dtype=np.float32)
    
    # 1. Unclipped
    surf_all = cairo.ImageSurface(cairo.FORMAT_A8, 256, 256)
    ctx_all = cairo.Context(surf_all)
    ctx_all.scale(s, -s); ctx_all.translate(-X0, -Y_TOP)
    for st_str in strokes:
        for loop in parse_hw_path(st_str): _trace(ctx_all, loop)
    ctx_all.fill(); surf_all.flush()
    stride_all = surf_all.get_stride()
    buf_all = np.frombuffer(surf_all.get_data(), np.uint8).reshape(256, stride_all)[:, :256].astype(np.float32) / 255.0
    raw_iou = float(iou(r_font, buf_all))
    
    # 2. Clipped
    for st_str, med in zip(strokes, medians):
        surf_out = cairo.ImageSurface(cairo.FORMAT_A8, 256, 256)
        ctx_out = cairo.Context(surf_out)
        ctx_out.scale(s, -s); ctx_out.translate(-X0, -Y_TOP)
        for loop in parse_hw_path(st_str): _trace(ctx_out, loop)
        ctx_out.fill(); surf_out.flush()
        stride_out = surf_out.get_stride()
        buf_out = np.frombuffer(surf_out.get_data(), np.uint8).reshape(256, stride_out)[:, :256].astype(np.float32) / 255.0
        
        surf_med = cairo.ImageSurface(cairo.FORMAT_A8, 256, 256)
        ctx_med = cairo.Context(surf_med)
        ctx_med.scale(s, -s); ctx_med.translate(-X0, -Y_TOP)
        ctx_med.set_line_width(brush_width)
        ctx_med.set_line_cap(cairo.LINE_CAP_ROUND); ctx_med.set_line_join(cairo.LINE_JOIN_ROUND)
        if len(med) > 0:
            ctx_med.move_to(med[0][0], med[0][1])
            for pt in med[1:]: ctx_med.line_to(pt[0], pt[1])
            ctx_med.stroke()
        surf_med.flush()
        stride_med = surf_med.get_stride()
        buf_med = np.frombuffer(surf_med.get_data(), np.uint8).reshape(256, stride_med)[:, :256].astype(np.float32) / 255.0
        
        full_clipped = np.maximum(full_clipped, buf_out * (buf_med > 0.05))
        
    clipped_iou = float(iou(r_font, full_clipped))
    uncovered_px = int(np.sum((r_font > 0.1) & (full_clipped <= 0.05)))
    return raw_iou, clipped_iou, uncovered_px

def main():
    with open("/tmp/queue100.json", encoding="utf-8") as f:
        queue = json.load(f)

    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))

    print(f"Verifying all {len(queue)} characters in queue...\n")

    missing = []
    manifest_missing = []
    format_errs = []
    ring_errs = []
    low_iou = []

    for item in queue:
        c = item["char"]
        data_p = os.path.join(DATA_DIR, f"{c}.json")
        font_p = os.path.join(FONT_DIR, f"{c}.svg")

        if not os.path.exists(data_p):
            missing.append(c)
            continue
        if c not in manifest["characters"]:
            manifest_missing.append(c)

        d = json.load(open(data_p, encoding="utf-8"))
        st = d.get("strokes", [])
        med = d.get("medians", [])

        if len(st) == 0 or len(st) != len(med):
            format_errs.append((c, f"len mismatch: st={len(st)}, med={len(med)}"))
        for m in med:
            if len(m) < 2:
                format_errs.append((c, f"median too short: {len(m)}"))

        if c in ["Ů", "ů", "°"]:
            if len(med[0]) != 24:
                ring_errs.append((c, f"ring points = {len(med[0])}"))

        cnts = load_svg_glyph(open(font_p, encoding="utf-8").read())
        raw_score, clip_score, uncov = eval_char(cnts, st, med)

        # For non-good chars, assert clipped_iou >= 0.98 and raw_iou >= 0.98
        if c not in GOOD_CHARS:
            if raw_score < 0.98 or clip_score < 0.98:
                low_iou.append((c, raw_score, clip_score, uncov))

    print("=" * 60)
    print("VERIFICATION AUDIT REPORT")
    print(f"Total checked: {len(queue)}")
    print(f"Missing data files: {len(missing)} {missing}")
    print(f"Manifest missing: {len(manifest_missing)} {manifest_missing}")
    print(f"Format errors: {len(format_errs)} {format_errs}")
    print(f"Ring 24-point errors: {len(ring_errs)} {ring_errs}")
    print(f"Low IoU (<0.98) in modified chars: {len(low_iou)} {low_iou}")
    print("=" * 60)

    if not missing and not manifest_missing and not format_errs and not ring_errs and not low_iou:
        print("\nALL 100 CHARACTERS PASSED ALL VERIFICATIONS WITH FLYING COLORS!")
    else:
        print("\nSOME CHECKS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
