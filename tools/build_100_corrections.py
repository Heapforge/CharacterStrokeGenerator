#!/usr/bin/env python3
"""Build and apply corrections for the first 100 characters in the review dashboard."""
import json
import os
import sys
import urllib.request
import numpy as np

sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path, Seg, HW_BASE
from tools.fix_svgs import check_iou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")
DATA_DIR = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
MANIFEST_PATH = os.path.join(ROOT, "training_data", "font_glyphs", "manifest.json")

with open("/tmp/queue100.json", encoding="utf-8") as f:
    queue = json.load(f)

print(f"Loaded {len(queue)} characters from queue.")

def circle_median(center_x, center_y, radius=85.2, n_points=24):
    """Generate 24-point counter-clockwise circular median (starting at top)."""
    thetas = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi - 0.25, n_points)
    return [[int(round(center_x + radius * np.cos(th))), int(round(center_y + radius * np.sin(th)))] for th in thetas]

def fix_acute_median(med):
    """Ensure acute median runs from top-right to bottom-left (p0_y > p1_y and p0_x > p1_x)."""
    if len(med) >= 2:
        p0, p1 = med[0], med[-1]
        # If moving up or moving right, reverse
        if (p1[1] > p0[1]) or (p1[0] > p0[0] and p1[1] >= p0[1] - 20):
            return list(reversed(med))
    return med

# We will collect corrected dicts: char -> {"strokes": [...], "medians": [...]}
corrections = {}

# =========================================================================
# Base E 4-stroke geometry
# =========================================================================
cnts_E = load_svg_glyph(open(os.path.join(FONT_DIR, "E.svg"), encoding="utf-8").read())
s_E_stem = "M 99.328 811.904 L 191.488 811.904 L 191.488 80.8 L 99.328 80.8 Z"
m_E_stem = [[145, 812], [145, 81]]
s_E_top = "M 99.328 811.904 L 507.904 811.904 L 507.904 731.008 L 99.328 731.008 Z"
m_E_top = [[145, 771], [508, 771]]
s_E_mid = "M 191.488 502.656 L 489.472 502.656 L 489.472 422.784 L 191.488 422.784 Z"
m_E_mid = [[191, 463], [489, 463]]
s_E_bot = "M 99.328 161.664 L 507.904 161.664 L 507.904 80.8 L 99.328 80.8 Z"
m_E_bot = [[145, 121], [508, 121]]

strokes_E = [s_E_stem, s_E_top, s_E_mid, s_E_bot]
medians_E = [m_E_stem, m_E_top, m_E_mid, m_E_bot]
corrections["E"] = {"strokes": strokes_E, "medians": medians_E}

# =========================================================================
# Digit 4
# =========================================================================
cnts_4 = load_svg_glyph(open(os.path.join(FONT_DIR, "4.svg"), encoding="utf-8").read())
s_4_stem = "M 371.712 816.032 L 458.752 816.032 L 458.752 80.8 L 371.712 80.8 Z"
m_4_stem = [[415, 816], [415, 81]]
b_4_top = Seg('L', np.array([[365.568, 816.032], [376.832, 719.776]]))
b_4_across_top = Seg('L', np.array([[371.712, 327.584], [458.752, 327.584]]))
b_4_across_bot = Seg('L', np.array([[458.752, 246.688], [371.712, 246.688]]))
s_4_L = segs_to_path([cnts_4[0][4], cnts_4[0][5], cnts_4[0][6], b_4_top] + 
                     cnts_4[1][2:7] + 
                     [b_4_across_top, cnts_4[0][9], cnts_4[0][10], cnts_4[0][0], b_4_across_bot])
m_4_L = [[365, 800], [210, 560], [35, 285], [300, 285], [565, 285]]
corrections["4"] = {"strokes": [s_4_L, s_4_stem], "medians": [m_4_L, m_4_stem]}

# =========================================================================
# Digit 6
# =========================================================================
cnts_6 = load_svg_glyph(open(os.path.join(FONT_DIR, "6.svg"), encoding="utf-8").read())
s_6 = segs_to_path(cnts_6[0]) + " " + segs_to_path(cnts_6[1])
m_6 = [[480, 800], [330, 730], [180, 520], [105, 300], [190, 105], [380, 105], [490, 240], [450, 420], [280, 470], [170, 360]]
corrections["6"] = {"strokes": [s_6], "medians": [m_6]}

# =========================================================================
# Digit 8
# =========================================================================
cnts_8 = load_svg_glyph(open(os.path.join(FONT_DIR, "8.svg"), encoding="utf-8").read())
s_8 = segs_to_path(cnts_8[0]) + " " + segs_to_path(cnts_8[1]) + " " + segs_to_path(cnts_8[2])
m_8 = [[350, 780], [220, 680], [290, 465], [400, 280], [300, 100], [180, 200], [290, 465], [400, 650], [350, 780]]
corrections["8"] = {"strokes": [s_8], "medians": [m_8]}

# =========================================================================
# Digit 9
# =========================================================================
cnts_9 = load_svg_glyph(open(os.path.join(FONT_DIR, "9.svg"), encoding="utf-8").read())
s_9 = segs_to_path(cnts_9[0]) + " " + segs_to_path(cnts_9[1])
m_9 = [[490, 550], [300, 750], [90, 550], [290, 420], [490, 550], [450, 300], [280, 100], [150, 100]]
corrections["9"] = {"strokes": [s_9], "medians": [m_9]}

# =========================================================================
# @ symbol
# =========================================================================
cnts_at = load_svg_glyph(open(os.path.join(FONT_DIR, "@.svg"), encoding="utf-8").read())
s_at = segs_to_path(cnts_at[0]) + " " + segs_to_path(cnts_at[1])
m_at = [[400, 460], [330, 560], [240, 500], [230, 370], [300, 310], [400, 350], [420, 460], [450, 620], [530, 640], [600, 500], [600, 320], [500, 120], [340, 70], [180, 150], [120, 320], [140, 600], [280, 760], [450, 800], [650, 740], [750, 550], [740, 300]]
corrections["@"] = {"strokes": [s_at], "medians": [m_at]}

# =========================================================================
# & symbol
# =========================================================================
cnts_amp = load_svg_glyph(open(os.path.join(FONT_DIR, "&.svg"), encoding="utf-8").read())
s_amp = segs_to_path(cnts_amp[0]) + " " + segs_to_path(cnts_amp[1]) + " " + segs_to_path(cnts_amp[2])
m_amp = [[409, 723], [329, 663], [299, 652], [284, 552], [258, 274], [357, 239], [422, 230], [503, 225], [536, 217], [562, 202], [643, 123]]
corrections["&"] = {"strokes": [s_amp], "medians": [m_amp]}

# =========================================================================
# Q letter
# =========================================================================
cnts_Q = load_svg_glyph(open(os.path.join(FONT_DIR, "Q.svg"), encoding="utf-8").read())
b_close_oval = Seg('L', np.array([[425.984, 71.584], [524.288, 88.992]]))
oval_outer_Q = cnts_Q[0][5:] + cnts_Q[0][:2] + [b_close_oval]
s_Q_oval = segs_to_path(oval_outer_Q) + " " + segs_to_path(cnts_Q[1])
s_Q_tail = f"M {524.288} {88.992} L {699.392} {-93.28} L {567.296} {-93.28} L {425.984} {71.584} Z"
m_Q_oval = circle_median(400, 447, radius=295, n_points=24)
m_Q_tail = [[475, 80], [633, -93]]
corrections["Q"] = {"strokes": [s_Q_oval, s_Q_tail], "medians": [m_Q_oval, m_Q_tail]}

# =========================================================================
# Ö letter
# =========================================================================
cnts_Ö = load_svg_glyph(open(os.path.join(FONT_DIR, "Ö.svg"), encoding="utf-8").read())
s_Ö_dot0 = segs_to_path(cnts_Ö[0])
s_Ö_dot1 = segs_to_path(cnts_Ö[1])
s_Ö_body = segs_to_path(cnts_Ö[2]) + " " + segs_to_path(cnts_Ö[3])
m_Ö_dot0 = [[305, 1010], [305, 910]]
m_Ö_dot1 = [[497, 1010], [497, 910]]
m_Ö_body = circle_median(400, 447, radius=295, n_points=24)
corrections["Ö"] = {"strokes": [s_Ö_dot0, s_Ö_dot1, s_Ö_body], "medians": [m_Ö_dot0, m_Ö_dot1, m_Ö_body]}

# =========================================================================
# R letter
# =========================================================================
cnts_R = load_svg_glyph(open(os.path.join(FONT_DIR, "R.svg"), encoding="utf-8").read())
s_R_stem = "M 99.328 811.936 L 191.488 811.936 L 191.488 80.8 L 99.328 80.8 Z"
m_R_stem = [[145, 812], [145, 81]]
b_R_lobe_in = Seg('L', np.array([[420.864, 408.48], [191.488, 460.704]]))
b_R_lobe_up = Seg('L', np.array([[191.488, 460.704], [191.488, 811.936]]))
b_R_lobe_top = Seg('L', np.array([[191.488, 811.936], [301.056, 811.936]]))
s_R_lobe_outer = segs_to_path([b_R_lobe_top] + cnts_R[0][:5] + [b_R_lobe_in, b_R_lobe_up])
s_R_lobe = s_R_lobe_outer + " " + segs_to_path(cnts_R[1])
m_R_lobe = [[191, 772], [301, 772], [470, 700], [510, 600], [470, 480], [301, 440], [191, 440]]
b_R_leg_stem = Seg('L', np.array([[191.488, 382.88], [191.488, 460.704]]))
b_R_leg_across = Seg('L', np.array([[191.488, 460.704], [420.864, 408.48]]))
s_R_leg = segs_to_path([cnts_R[0][5], cnts_R[0][6], cnts_R[0][7], cnts_R[0][8], b_R_leg_stem, b_R_leg_across])
m_R_leg = [[378, 395], [568, 81]]
corrections["R"] = {"strokes": [s_R_stem, s_R_lobe, s_R_leg], "medians": [m_R_stem, m_R_lobe, m_R_leg]}

# =========================================================================
# Ŕ, Ř, Ŗ
# =========================================================================
cnts_Ŕ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŕ.svg"), encoding="utf-8").read())
s_Ŕ_acute = segs_to_path(cnts_Ŕ[0])
m_Ŕ_acute = [[465, 1037], [340, 894]] # top-right to bottom-left
corrections["Ŕ"] = {"strokes": [s_Ŕ_acute, s_R_stem, s_R_lobe, s_R_leg], "medians": [m_Ŕ_acute, m_R_stem, m_R_lobe, m_R_leg]}

cnts_Ř = load_svg_glyph(open(os.path.join(FONT_DIR, "Ř.svg"), encoding="utf-8").read())
s_Ř_caron = segs_to_path(cnts_Ř[0])
m_Ř_caron = [[250, 1037], [360, 910], [470, 1037]]
corrections["Ř"] = {"strokes": [s_Ř_caron, s_R_stem, s_R_lobe, s_R_leg], "medians": [m_Ř_caron, m_R_stem, m_R_lobe, m_R_leg]}

cnts_Ŗ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŗ.svg"), encoding="utf-8").read())
s_Ŗ_cedilla = segs_to_path(cnts_Ŗ[0])
m_Ŗ_cedilla = [[356, 80], [356, -95]]
corrections["Ŗ"] = {"strokes": [s_R_stem, s_R_lobe, s_R_leg, s_Ŗ_cedilla], "medians": [m_R_stem, m_R_lobe, m_R_leg, m_Ŗ_cedilla]}

# =========================================================================
# E variants: È, É, Ê, Ë, Ē, Ĕ, Ė, Ę, Ě, Ǝ
# =========================================================================
cnts_È = load_svg_glyph(open(os.path.join(FONT_DIR, "È.svg"), encoding="utf-8").read())
s_È_grave = segs_to_path(cnts_È[0])
m_È_grave = [[230, 1037], [410, 894]]
corrections["È"] = {"strokes": [s_È_grave] + strokes_E, "medians": [m_È_grave] + medians_E}

cnts_É = load_svg_glyph(open(os.path.join(FONT_DIR, "É.svg"), encoding="utf-8").read())
s_É_acute = segs_to_path(cnts_É[0])
m_É_acute = [[465, 1037], [282, 894]] # top-right to bottom-left
corrections["É"] = {"strokes": [s_É_acute] + strokes_E, "medians": [m_É_acute] + medians_E}

cnts_Ê = load_svg_glyph(open(os.path.join(FONT_DIR, "Ê.svg"), encoding="utf-8").read())
s_Ê_circ = segs_to_path(cnts_Ê[0])
m_Ê_circ = [[180, 894], [304, 1037], [428, 894]]
corrections["Ê"] = {"strokes": [s_Ê_circ] + strokes_E, "medians": [m_Ê_circ] + medians_E}

cnts_Ë = load_svg_glyph(open(os.path.join(FONT_DIR, "Ë.svg"), encoding="utf-8").read())
s_Ë_dot0 = segs_to_path(cnts_Ë[0])
s_Ë_dot1 = segs_to_path(cnts_Ë[1])
m_Ë_dot0 = [[230, 1010], [230, 910]]
m_Ë_dot1 = [[377, 1010], [377, 910]]
corrections["Ë"] = {"strokes": [s_Ë_dot0, s_Ë_dot1] + strokes_E, "medians": [m_Ë_dot0, m_Ë_dot1] + medians_E}

cnts_Ē = load_svg_glyph(open(os.path.join(FONT_DIR, "Ē.svg"), encoding="utf-8").read())
s_Ē_macron = segs_to_path(cnts_Ē[0])
m_Ē_macron = [[150, 970], [450, 970]]
corrections["Ē"] = {"strokes": [s_Ē_macron] + strokes_E, "medians": [m_Ē_macron] + medians_E}

cnts_Ĕ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ĕ.svg"), encoding="utf-8").read())
s_Ĕ_breve = segs_to_path(cnts_Ĕ[0])
m_Ĕ_breve = [[180, 1030], [304, 910], [428, 1030]]
corrections["Ĕ"] = {"strokes": [s_Ĕ_breve] + strokes_E, "medians": [m_Ĕ_breve] + medians_E}

cnts_Ė = load_svg_glyph(open(os.path.join(FONT_DIR, "Ė.svg"), encoding="utf-8").read())
s_Ė_dot = segs_to_path(cnts_Ė[0])
m_Ė_dot = [[304, 1010], [304, 910]]
corrections["Ė"] = {"strokes": [s_Ė_dot] + strokes_E, "medians": [m_Ė_dot] + medians_E}

cnts_Ě = load_svg_glyph(open(os.path.join(FONT_DIR, "Ě.svg"), encoding="utf-8").read())
s_Ě_caron = segs_to_path(cnts_Ě[0])
m_Ě_caron = [[180, 1037], [304, 910], [428, 1037]]
corrections["Ě"] = {"strokes": [s_Ě_caron] + strokes_E, "medians": [m_Ě_caron] + medians_E}

cnts_Ę = load_svg_glyph(open(os.path.join(FONT_DIR, "Ę.svg"), encoding="utf-8").read())
s_Ę_bot = s_E_bot + " " + segs_to_path(cnts_Ę[0]) # continuous into ogonek
m_Ę_bot = [[145, 121], [508, 121], [460, 80], [420, 20], [400, -20], [400, -70], [420, -100], [450, -112], [500, -112], [520, -92]]
corrections["Ę"] = {"strokes": [s_E_stem, s_E_top, s_E_mid, s_Ę_bot], "medians": [m_E_stem, m_E_top, m_E_mid, m_Ę_bot]}

# ª (superscript a)
cnts_ª = load_svg_glyph(open(os.path.join(FONT_DIR, "ª.svg"), encoding="utf-8").read())
s_ª = segs_to_path(cnts_ª[0]) + " " + segs_to_path(cnts_ª[1])
m_ª = [[200, 780], [100, 700], [100, 560], [200, 520], [250, 600], [250, 780], [250, 500]]
corrections["ª"] = {"strokes": [s_ª], "medians": [m_ª]}

# =========================================================================
# Ligatures æ and œ
# =========================================================================
cnts_æ = load_svg_glyph(open(os.path.join(FONT_DIR, "æ.svg"), encoding="utf-8").read())
s_æ = " ".join(segs_to_path(cnt) for cnt in cnts_æ)
m_æ = [[200, 350], [150, 180], [280, 110], [380, 200], [380, 480], [300, 600], [150, 600], [200, 400], [500, 400], [700, 400], [750, 520], [650, 620], [520, 550], [520, 200], [650, 110], [750, 130]]
corrections["æ"] = {"strokes": [s_æ], "medians": [m_æ]}

cnts_œ = load_svg_glyph(open(os.path.join(FONT_DIR, "œ.svg"), encoding="utf-8").read())
s_œ = " ".join(segs_to_path(cnt) for cnt in cnts_œ)
m_œ = [[300, 600], [150, 500], [120, 350], [180, 150], [300, 110], [420, 200], [450, 350], [550, 380], [750, 380], [800, 520], [700, 620], [580, 550], [580, 200], [700, 110], [800, 130]]
corrections["œ"] = {"strokes": [s_œ], "medians": [m_œ]}

# =========================================================================
# Ɓ, Ƃ, ƥ, ð
# =========================================================================
cnts_Ɓ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ɓ.svg"), encoding="utf-8").read())
s_Ɓ = " ".join(segs_to_path(cnt) for cnt in cnts_Ɓ)
m_Ɓ = [[10, 812], [273, 812], [225, 812], [225, 81], [273, 81], [450, 100], [580, 250], [500, 430], [273, 440], [450, 470], [560, 600], [480, 730], [273, 733]]
corrections["Ɓ"] = {"strokes": [s_Ɓ], "medians": [m_Ɓ]}

cnts_Ƃ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ƃ.svg"), encoding="utf-8").read())
s_Ƃ = " ".join(segs_to_path(cnt) for cnt in cnts_Ƃ)
m_Ƃ = [[100, 812], [500, 812], [145, 812], [145, 81], [191, 81], [350, 100], [500, 250], [450, 420], [191, 423]]
corrections["Ƃ"] = {"strokes": [s_Ƃ], "medians": [m_Ƃ]}

cnts_ƥ = load_svg_glyph(open(os.path.join(FONT_DIR, "ƥ.svg"), encoding="utf-8").read())
s_ƥ = " ".join(segs_to_path(cnt) for cnt in cnts_ƥ)
m_ƥ = [[87, 864], [132, 864], [132, -165], [177, 145], [350, 110], [500, 250], [500, 450], [350, 564], [177, 564]]
corrections["ƥ"] = {"strokes": [s_ƥ], "medians": [m_ƥ]}

cnts_ð = load_svg_glyph(open(os.path.join(FONT_DIR, "ð.svg"), encoding="utf-8").read())
s_ð = " ".join(segs_to_path(cnt) for cnt in cnts_ð)
m_ð = [[500, 864], [350, 800], [200, 700], [500, 550], [480, 145], [350, 100], [150, 200], [150, 450], [350, 550], [480, 500]]
corrections["ð"] = {"strokes": [s_ð], "medians": [m_ð]}

# =========================================================================
# Ů, ů (Circular Ring Diacritics)
# =========================================================================
for c in ["Ů", "ů"]:
    cnts_ring = load_svg_glyph(open(os.path.join(FONT_DIR, f"{c}.svg"), encoding="utf-8").read())
    # Ring contours are cnts_ring[0] and cnts_ring[1]
    poly_out = [s.p0 for s in cnts_ring[0]]
    poly_in = [s.p0 for s in cnts_ring[1]]
    center_x = (np.mean([p[0] for p in poly_out]) + np.mean([p[0] for p in poly_in])) / 2.0
    center_y = (np.mean([p[1] for p in poly_out]) + np.mean([p[1] for p in poly_in])) / 2.0
    s_ring = segs_to_path(cnts_ring[0]) + " " + segs_to_path(cnts_ring[1])
    m_ring = circle_median(center_x, center_y, radius=85.2, n_points=24)
    
    # Body stroke
    body_segs = cnts_ring[2:]
    s_body = " ".join(segs_to_path(cnt) for cnt in body_segs)
    # Get body median from queue
    q_item = next(it for it in queue if it["char"] == c)
    m_body = q_item["pred"]["medians"][-1]
    corrections[c] = {"strokes": [s_ring, s_body], "medians": [m_ring, m_body]}

# =========================================================================
# Ų, ų (Continuous Ogonek on u / U)
# =========================================================================
for c in ["Ų", "ų"]:
    cnts_u = load_svg_glyph(open(os.path.join(FONT_DIR, f"{c}.svg"), encoding="utf-8").read())
    s_u = " ".join(segs_to_path(cnt) for cnt in cnts_u)
    q_item = next(it for it in queue if it["char"] == c)
    m_u = q_item["pred"]["medians"][0]
    # append ogonek path points to the median if not ending at ogonek
    if m_u[-1][1] > 0:
        ogonek_pts = [[450, 70], [410, 20], [390, -25], [390, -65], [410, -96], [440, -112], [490, -112], [510, -92]]
        m_u = m_u + ogonek_pts
    corrections[c] = {"strokes": [s_u], "medians": [m_u]}

# =========================================================================
# Ń, Ň, Ñ, Ņ (N variants)
# =========================================================================
m_N_base = [[145, 95], [145, 780], [636, 120], [636, 780]]
for c, diac_name in [("Ń", "acute"), ("Ň", "caron"), ("Ñ", "tilde")]:
    cnts_N = load_svg_glyph(open(os.path.join(FONT_DIR, f"{c}.svg"), encoding="utf-8").read())
    s_diac = segs_to_path(cnts_N[0])
    s_body = segs_to_path(cnts_N[1])
    q_item = next(it for it in queue if it["char"] == c)
    m_diac = q_item["pred"]["medians"][0]
    if diac_name == "acute":
        m_diac = fix_acute_median(m_diac)
    corrections[c] = {"strokes": [s_diac, s_body], "medians": [m_diac, m_N_base]}

cnts_Ņ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ņ.svg"), encoding="utf-8").read())
s_Ņ_body = segs_to_path(cnts_Ņ[1])
s_Ņ_cedilla = segs_to_path(cnts_Ņ[0])
m_Ņ_cedilla = [[390, 80], [390, -95]]
corrections["Ņ"] = {"strokes": [s_Ņ_body, s_Ņ_cedilla], "medians": [m_N_base, m_Ņ_cedilla]}

print(f"Total manual corrections prepared: {len(corrections)}")

# =========================================================================
# Process and Validate All 100 Characters
# =========================================================================
final_dataset = {}
validation_results = []
bad_iou_count = 0

for item in queue:
    c = item["char"]
    font = item["font"]
    font_path = os.path.join(FONT_DIR, f"{c}.svg")
    cnts_font = load_svg_glyph(open(font_path, encoding="utf-8").read())
    
    if c in corrections:
        strokes = corrections[c]["strokes"]
        medians = corrections[c]["medians"]
    else:
        # From queue
        pred = item.get("pred", {})
        strokes = list(pred.get("strokes", []))
        medians = list(pred.get("medians", []))
        
        # Check acute accents in medians
        for mi in range(len(medians)):
            # If stroke is small diacritic (e.g. acute on ú, ý, etc.)
            m = medians[mi]
            if len(m) >= 2 and m[0][1] >= 650:
                medians[mi] = fix_acute_median(m)
                
    # Calculate IoU
    iou_score = check_iou(cnts_font, strokes)
    if iou_score < 0.98:
        bad_iou_count += 1
        print(f"WARNING: Low IoU for '{c}': {iou_score:.4f}")
    else:
        # print(f"Valid '{c}': {len(strokes)} strokes, IoU = {iou_score:.4f}")
        pass
        
    final_dataset[c] = {
        "strokes": strokes,
        "medians": medians
    }
    validation_results.append({
        "char": c,
        "strokes": len(strokes),
        "iou": float(iou_score)
    })

print(f"\nCompleted validation. Total chars: {len(final_dataset)}, Bad IoU (<0.98): {bad_iou_count}")

# Save to hanzi-writer-data-youyin/data/
for c, data in final_dataset.items():
    p = os.path.join(DATA_DIR, f"{c}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

print(f"Saved {len(final_dataset)} JSON files to {DATA_DIR}")

# Update manifest.json
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)

changed_manifest = False
for c in final_dataset.keys():
    e = manifest["characters"].setdefault(c, {"data_files": [], "fonts": []})
    fn = f"{c}.json"
    if fn not in e["data_files"]:
        e["data_files"] = sorted(e["data_files"] + [fn])
        changed_manifest = True
    if "noto-sans" not in e["fonts"]:
        e["fonts"] = sorted(e["fonts"] + ["noto-sans"])
        changed_manifest = True
    if c in manifest.get("missing", []):
        manifest["missing"].remove(c)
        changed_manifest = True

if changed_manifest:
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, MANIFEST_PATH)
    print("Updated manifest.json with all characters.")

# Commit to dashboard API on port 9000
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
committed_count = 0
for c, data in final_dataset.items():
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:9000/api/commit",
            data=json.dumps({"char": c, "variant": "", "data": data, "overwrite": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        opener.open(req, timeout=3)
        committed_count += 1
    except Exception as e:
        print(f"Error committing '{c}': {e}")

print(f"Successfully committed {committed_count} / {len(final_dataset)} to dashboard API.")
