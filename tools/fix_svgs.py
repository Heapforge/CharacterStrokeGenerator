#!/usr/bin/env python3
"""Batch SVG and Median corrector for HanziWriter output matching font SVGs."""
import json
import os
import cairo
import numpy as np

from strokegen.svg_io import load_svg_glyph, parse_hw_path, segs_to_path, HW_BASE
from strokegen.raster import render, iou, SPAN, X0, Y_TOP, _trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")
DATA_DIR = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")

def check_iou(font_contours, stroke_strs):
    r_font = render(font_contours)
    surf = cairo.ImageSurface(cairo.FORMAT_A8, 256, 256)
    ctx = cairo.Context(surf)
    s = 256.0 / SPAN
    ctx.scale(s, -s)
    ctx.translate(-X0, -Y_TOP)
    for st_str in stroke_strs:
        loops = parse_hw_path(st_str)
        for loop in loops:
            _trace(ctx, loop)
        ctx.fill()
    surf.flush()
    stride = surf.get_stride()
    buf = np.frombuffer(surf.get_data(), np.uint8).reshape(256, stride)[:, :256].astype(np.float32) / 255.0
    return iou(r_font, buf)

corrections = {}

# =========================================================================
# 1. Capital A and its variations
# =========================================================================
s_A0 = "M 327.2 815.0 L 285.7 815.0 L 0 80.8 L 93.2 80.8 L 180.2 307.1 L 210.9 389.0 L 293.9 610.2 Q 309.2 657.8 325.6 719.8 L 327.2 815.0 Z"
s_A1 = "M 327.2 815.0 L 368.6 815.0 L 653.3 80.8 L 558.1 80.8 L 470.0 307.1 L 442.4 389.0 L 360.4 610.2 Q 343 661.4 325.6 719.8 L 327.2 815.0 Z"
s_A2 = "M 180.2 307.1 L 470.0 307.1 L 442.4 389.0 L 210.9 389.0 Z"

m_A0 = [[327, 815], [290, 726], [255, 622], [140, 348], [46, 95]]
m_A1 = [[327, 815], [362, 729], [400, 622], [515, 348], [605, 95]]
m_A2 = [[195, 348], [325, 348], [456, 348]]

A_variants = ["A", "À", "Á", "Â", "Ã", "Ä", "Å", "Ā", "Ă", "Ą"]

# Read existing predictions to get accent strokes for variants
queue_data = {it["char"]: it for it in json.load(open("/tmp/queue_done.json"))}

for ch in A_variants:
    font_path = os.path.join(FONT_DIR, f"{ch}.svg")
    font_contours = load_svg_glyph(open(font_path).read())
    if ch == "A":
        stk = [s_A0, s_A1, s_A2]
        med = [m_A0, m_A1, m_A2]
    else:
        # Extra contours that are not the base A (base A has bounds y < 820)
        # In font_contours, identify accent contours:
        accent_contours = []
        for c in font_contours:
            pts = [s.p0 for s in c]
            min_y = min(p[1] for p in pts)
            max_y = max(p[1] for p in pts)
            min_x = min(p[0] for p in pts)
            # Accent if max_y > 820 or min_y < 80 (like ogonek in Ą)
            if max_y > 820 or min_y < 75:
                accent_contours.append(c)
        stk = [s_A0, s_A1, s_A2]
        med = [m_A0, m_A1, m_A2]
        # Accent contours
        if ch in ["Å", "å"]:
            # Ring accent has 2 contours (outer + inner)
            accent_stk = " ".join(segs_to_path(ac) for ac in accent_contours)
            stk.append(accent_stk)
            med.append([[287, 867], [229, 810], [287, 752], [344, 810], [287, 867]])
        else:
            for ac in accent_contours:
                stk.append(segs_to_path(ac))
                pts = [s.p0 for s in ac]
                med.append([[int(round(np.mean([p[0] for p in pts]))), int(round(max(p[1] for p in pts)))],
                            [int(round(np.mean([p[0] for p in pts]))), int(round(min(p[1] for p in pts)))]])
    
    score = check_iou(font_contours, stk)
    corrections[ch] = {"strokes": stk, "medians": med, "iou": score}

# =========================================================================
# 2. Letter n and similar (n, ñ, ń, ņ, ň, N)
# =========================================================================
s_n0 = "M 351.2 639.9 Q 449.5 639.9 499.7 592.3 Q 549.9 544.7 549.9 438.2 L 549.9 80.8 L 460.8 80.8 L 460.8 432 Q 460.8 564.1 337.9 564.1 Q 246.8 564.1 212 512.9 Q 177.2 461.7 177.2 365.5 L 177.2 80.8 L 87 80.8 L 87 629.7 L 159.7 629.7 L 173.1 554.9 L 178.2 554.9 Q 204.8 597.9 251.9 618.9 Q 299 639.9 351.2 639.9 Z"
m_n0 = [[101, 616], [127, 590], [133, 531], [142, 498], [132, 394], [132, 126], [163, 95], [169, 540], [191, 531], [222, 563], [249, 581], [290, 596], [339, 602], [381, 599], [423, 588], [455, 570], [478, 546], [497, 505], [505, 457], [505, 125], [476, 96]]

for ch in ["n", "ñ", "ń", "ņ", "ň"]:
    font_path = os.path.join(FONT_DIR, f"{ch}.svg")
    font_contours = load_svg_glyph(open(font_path).read())
    if ch == "n":
        stk = [s_n0]
        med = [m_n0]
    else:
        # Find diacritic contours
        accent_contours = []
        for c in font_contours:
            pts = [s.p0 for s in c]
            min_y = min(p[1] for p in pts)
            max_y = max(p[1] for p in pts)
            if max_y > 645 or min_y < 75:
                accent_contours.append(c)
        stk = [s_n0]
        med = [m_n0]
        # Grab diacritic median from queue_data
        item = queue_data.get(ch)
        for ac in accent_contours:
            stk.append(segs_to_path(ac))
            # diacritic median from item if available
            if item:
                # find diacritic stroke in item
                diac_idx = 0 if ch in ["ñ", "ń", "ň"] else 1
                med.append(item["medians"][diac_idx])
            else:
                pts = [s.p0 for s in ac]
                med.append([[int(round(np.mean([p[0] for p in pts]))), int(round(max(p[1] for p in pts)))],
                            [int(round(np.mean([p[0] for p in pts]))), int(round(min(p[1] for p in pts)))]])
    score = check_iou(font_contours, stk)
    corrections[ch] = {"strokes": stk, "medians": med, "iou": score}

# Capital N
font_N = load_svg_glyph(open(os.path.join(FONT_DIR, "N.svg")).read())
s_N = segs_to_path(font_N[0])
m_N = [[150, 780], [150, 120], [150, 95], [180, 95], [570, 750], [630, 780], [630, 120], [630, 95]]
corrections["N"] = {"strokes": [s_N], "medians": [m_N], "iou": check_iou(font_N, [s_N])}

# =========================================================================
# 3. Letters like b and d (b, d, ď, đ, p, q, B, D)
# =========================================================================
# b: single stroke from top of vertical line to bottom, curving back up into oval shape and returning to vertical line
font_b = load_svg_glyph(open(os.path.join(FONT_DIR, "b.svg")).read())
s_b = segs_to_path(font_b[0]) + " " + segs_to_path(font_b[1])
m_b = [[132, 840], [132, 600], [132, 95], [200, 95], [330, 95], [480, 160], [530, 350], [480, 540], [330, 600], [132, 550]]
corrections["b"] = {"strokes": [s_b], "medians": [m_b], "iou": check_iou(font_b, [s_b])}

# d: single stroke from top of vertical line to bottom, curving back up into oval shape and returning to vertical line
font_d = load_svg_glyph(open(os.path.join(FONT_DIR, "d.svg")).read())
s_d = segs_to_path(font_d[0]) + " " + segs_to_path(font_d[1])
m_d = [[500, 840], [500, 600], [500, 95], [430, 95], [300, 95], [150, 160], [100, 350], [150, 540], [300, 600], [500, 550]]
corrections["d"] = {"strokes": [s_d], "medians": [m_d], "iou": check_iou(font_d, [s_d])}

# ď: d + caron
font_dc = load_svg_glyph(open(os.path.join(FONT_DIR, "ď.svg")).read())
# contour 0 is caron, 1 & 2 are d
s_dc_caron = segs_to_path(font_dc[0])
m_dc_caron = [[600, 840], [680, 720]]
corrections["ď"] = {"strokes": [s_d, s_dc_caron], "medians": [m_d, m_dc_caron], "iou": check_iou(font_dc, [s_d, s_dc_caron])}

# đ: d with bar
font_dbar = load_svg_glyph(open(os.path.join(FONT_DIR, "đ.svg")).read())
# đ in noto-sans has 2 contours: the outer with crossbar attached + counter
s_dbar = segs_to_path(font_dbar[0]) + " " + segs_to_path(font_dbar[1])
m_dbar = [[500, 840], [500, 600], [500, 95], [430, 95], [300, 95], [150, 160], [100, 350], [150, 540], [300, 600], [500, 550]]
corrections["đ"] = {"strokes": [s_dbar], "medians": [m_dbar], "iou": check_iou(font_dbar, [s_dbar])}

# p: top of vertical line to bottom, curving back up into oval shape
font_p = load_svg_glyph(open(os.path.join(FONT_DIR, "p.svg")).read())
s_p = segs_to_path(font_p[0]) + " " + segs_to_path(font_p[1])
m_p = [[132, 620], [132, 95], [132, -150], [132, 95], [200, 95], [330, 95], [480, 160], [530, 350], [480, 540], [330, 600], [132, 550]]
corrections["p"] = {"strokes": [s_p], "medians": [m_p], "iou": check_iou(font_p, [s_p])}

# q: top of vertical line to bottom, curving back up into oval shape
font_q = load_svg_glyph(open(os.path.join(FONT_DIR, "q.svg")).read())
s_q = segs_to_path(font_q[0]) + " " + segs_to_path(font_q[1])
m_q = [[500, 620], [500, 95], [500, -150], [500, 95], [430, 95], [300, 95], [150, 160], [100, 350], [150, 540], [300, 600], [500, 550]]
corrections["q"] = {"strokes": [s_q], "medians": [m_q], "iou": check_iou(font_q, [s_q])}

# B: single stroke with 3 subpaths
font_B = load_svg_glyph(open(os.path.join(FONT_DIR, "B.svg")).read())
s_B = " ".join(segs_to_path(c) for c in font_B)
m_B = [[144, 767], [144, 126], [300, 126], [480, 250], [480, 420], [300, 450], [144, 450], [300, 470], [480, 580], [480, 710], [300, 767], [144, 767]]
corrections["B"] = {"strokes": [s_B], "medians": [m_B], "iou": check_iou(font_B, [s_B])}

# D: single stroke with 2 subpaths
font_D = load_svg_glyph(open(os.path.join(FONT_DIR, "D.svg")).read())
s_D = " ".join(segs_to_path(c) for c in font_D)
m_D = [[144, 767], [144, 126], [300, 126], [580, 300], [580, 550], [300, 767], [144, 767]]
corrections["D"] = {"strokes": [s_D], "medians": [m_D], "iou": check_iou(font_D, [s_D])}

# =========================================================================
# 4. Lower case e and its variations (e, è, é, ê, ë, ē, ĕ, ė, ě)
# =========================================================================
font_e = load_svg_glyph(open(os.path.join(FONT_DIR, "e.svg")).read())
# e has 2 contours: outer boundary + inner eye counter
s_e0 = segs_to_path(font_e[0]) + " " + segs_to_path(font_e[1])
# Rule 4: left-most middle point, right-most middle point curving into top of character and down to bottom handle
m_e0 = [[150, 370], [510, 370], [438, 424], [386, 464], [345, 484], [299, 489], [240, 480],
        [173, 442], [125, 399], [111, 375], [104, 349], [109, 279], [130, 217], [165, 167],
        [217, 131], [252, 118], [291, 111], [385, 112], [464, 131], [486, 164]]

e_variants = ["e", "è", "é", "ê", "ë", "ē", "ĕ", "ė", "ě"]
for ch in e_variants:
    font_path = os.path.join(FONT_DIR, f"{ch}.svg")
    font_contours = load_svg_glyph(open(font_path).read())
    if ch == "e":
        stk = [s_e0]
        med = [m_e0]
    else:
        # Accent contours (y > 650)
        accent_contours = [c for c in font_contours if max(s.p0[1] for s in c) > 650]
        stk = [s_e0]
        med = [m_e0]
        item = queue_data.get(ch)
        for ac in accent_contours:
            stk.append(segs_to_path(ac))
            if item:
                med.append(item["medians"][0])
            else:
                pts = [s.p0 for s in ac]
                med.append([[int(round(np.mean([p[0] for p in pts]))), int(round(max(p[1] for p in pts)))],
                            [int(round(np.mean([p[0] for p in pts]))), int(round(min(p[1] for p in pts)))]])
    score = check_iou(font_contours, stk)
    corrections[ch] = {"strokes": stk, "medians": med, "iou": score}

# =========================================================================
# 5. Lower case a and its variations (a, à, á, â, ã, ä, å, ā, ă, ą)
# =========================================================================
# Rule 5: starting from top handle, straight to bottom of character, then curving up [around bowl]
font_a = load_svg_glyph(open(os.path.join(FONT_DIR, "a.svg")).read())
s_a0 = segs_to_path(font_a[0]) + " " + segs_to_path(font_a[1])
m_a0 = [[340, 600], [200, 610], [280, 520], [460, 450], [460, 250], [460, 95], [300, 95], [100, 230], [300, 360], [460, 350]]

a_variants = ["a", "à", "á", "â", "ã", "ä", "å", "ā", "ă", "ą"]
for ch in a_variants:
    font_path = os.path.join(FONT_DIR, f"{ch}.svg")
    font_contours = load_svg_glyph(open(font_path).read())
    if ch == "a":
        stk = [s_a0]
        med = [m_a0]
    else:
        accent_contours = []
        for c in font_contours:
            pts = [s.p0 for s in c]
            min_y = min(p[1] for p in pts)
            max_y = max(p[1] for p in pts)
            if max_y > 650 or min_y < 70:
                accent_contours.append(c)
        stk = [s_a0]
        med = [m_a0]
        item = queue_data.get(ch)
        if ch == "å":
            accent_stk = " ".join(segs_to_path(ac) for ac in accent_contours)
            stk.append(accent_stk)
            med.append([[287, 867], [229, 810], [287, 752], [344, 810], [287, 867]])
        else:
            for ac in accent_contours:
                stk.append(segs_to_path(ac))
                pts = [s.p0 for s in ac]
                med.append([[int(round(np.mean([p[0] for p in pts]))), int(round(max(p[1] for p in pts)))],
                            [int(round(np.mean([p[0] for p in pts]))), int(round(min(p[1] for p in pts)))]])
    score = check_iou(font_contours, stk)
    corrections[ch] = {"strokes": stk, "medians": med, "iou": score}

print(f"Generated corrections for {len(corrections)} characters.")
for ch, v in corrections.items():
    print(f"Char {ch}: {len(v['strokes'])} strokes, IoU = {v['iou']:.4f}")
    # Write to /tmp/corrected_{ch}.json
    with open(f"/tmp/corrected_{ch}.json", "w", encoding="utf-8") as f:
        json.dump({"strokes": v["strokes"], "medians": v["medians"]}, f, ensure_ascii=False)
    # Write to hanzi-writer-data-youyin/data/{ch}.json
    out_p = os.path.join(DATA_DIR, f"{ch}.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"strokes": v["strokes"], "medians": v["medians"]}, f, ensure_ascii=False)

# Save entire corrections dict to /tmp/all_corrections.json
with open("/tmp/all_corrections.json", "w", encoding="utf-8") as f:
    json.dump(corrections, f, ensure_ascii=False, indent=2)

print("Saved all corrections to /tmp and training_data/hanzi-writer-data-youyin/data/")
