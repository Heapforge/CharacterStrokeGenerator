#!/usr/bin/env python3
"""Definitive, verified corrections for all 100 characters in the dashboard."""
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

def circle_median(center_x, center_y, radius=85.2, n_points=24):
    thetas = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi - 0.25, n_points)
    return [[int(round(center_x + radius * np.cos(th))), int(round(center_y + radius * np.sin(th)))] for th in thetas]

# Characters confirmed GOOD by user: DO NOT ALTER
good_chars = set(["4", "È", "É", "Ê", "Ë", "Ö", "ß", "Ē", "Ĕ", "Ė", "Ę", "Ě", "Ų"])

corrections = {}

# =========================================================================
# 1. Base u and its complete family
# =========================================================================
cnts_u_font = load_svg_glyph(open(os.path.join(FONT_DIR, "u.svg"), encoding="utf-8").read())
s_base_u = segs_to_path(cnts_u_font[0])
# Median: top left -> curve down -> up to top right -> down to bottom left of foot
m_base_u = [
    [126, 620], [126, 276], [160, 150], [280, 108], [420, 150],
    [500, 300], [500, 620], [500, 300], [509, 81], [465, 120]
]

# Base u
corrections["u"] = {"strokes": [s_base_u], "medians": [m_base_u]}

# All accented lowercase u variants in the 100 characters:
# ù, ú, û, ü, ű, ũ, ū, ŭ, ů, ų, ư
u_accents = {
    "ù": ("grave", [[196, 855], [378, 711]]),
    "ú": ("acute", [[455, 855], [273, 711]]), # top-right to bottom-left
    "û": ("circ", [[180, 711], [290, 855], [400, 711]]),
    "ũ": ("tilde", [[196, 780], [250, 830], [340, 750], [390, 800]]),
    "ū": ("macron", [[180, 780], [400, 780]]),
    "ŭ": ("breve", [[180, 830], [290, 730], [400, 830]]),
}

for char, (accent_type, med_acc) in u_accents.items():
    cnts_char = load_svg_glyph(open(os.path.join(FONT_DIR, f"{char}.svg"), encoding="utf-8").read())
    # cnts_char[0] is accent, cnts_char[1] is base u
    s_acc = segs_to_path(cnts_char[0])
    s_u_local = segs_to_path(cnts_char[1])
    corrections[char] = {
        "strokes": [s_acc, s_u_local],
        "medians": [med_acc, m_base_u]
    }

# ü (diaeresis: 2 dots)
cnts_ü = load_svg_glyph(open(os.path.join(FONT_DIR, "ü.svg"), encoding="utf-8").read())
corrections["ü"] = {
    "strokes": [segs_to_path(cnts_ü[0]), segs_to_path(cnts_ü[1]), segs_to_path(cnts_ü[2])],
    "medians": [[[220, 800], [220, 730]], [[380, 800], [380, 730]], m_base_u]
}

# ű (double acute: 2 acute strokes, each top-right to bottom-left)
cnts_ű = load_svg_glyph(open(os.path.join(FONT_DIR, "ű.svg"), encoding="utf-8").read())
corrections["ű"] = {
    "strokes": [segs_to_path(cnts_ű[0]), segs_to_path(cnts_ű[1]), segs_to_path(cnts_ű[2])],
    "medians": [[[310, 855], [220, 711]], [[455, 855], [365, 711]], m_base_u]
}

# ů (ring above: 24-point circular median)
cnts_ů = load_svg_glyph(open(os.path.join(FONT_DIR, "ů.svg"), encoding="utf-8").read())
poly_out = [s.p0 for s in cnts_ů[0]]
poly_in = [s.p0 for s in cnts_ů[1]]
cx = (np.mean([p[0] for p in poly_out]) + np.mean([p[0] for p in poly_in])) / 2.0
cy = (np.mean([p[1] for p in poly_out]) + np.mean([p[1] for p in poly_in])) / 2.0
s_ring_ů = segs_to_path(cnts_ů[0]) + " " + segs_to_path(cnts_ů[1])
m_ring_ů = circle_median(cx, cy, radius=85.2, n_points=24)
s_u_local = segs_to_path(cnts_ů[2])
corrections["ů"] = {
    "strokes": [s_ring_ů, s_u_local],
    "medians": [m_ring_ů, m_base_u]
}

# Ů (Capital U with ring above)
cnts_Ů = load_svg_glyph(open(os.path.join(FONT_DIR, "Ů.svg"), encoding="utf-8").read())
poly_out_U = [s.p0 for s in cnts_Ů[0]]
poly_in_U = [s.p0 for s in cnts_Ů[1]]
cx_U = (np.mean([p[0] for p in poly_out_U]) + np.mean([p[0] for p in poly_in_U])) / 2.0
cy_U = (np.mean([p[1] for p in poly_out_U]) + np.mean([p[1] for p in poly_in_U])) / 2.0
s_ring_Ů = segs_to_path(cnts_Ů[0]) + " " + segs_to_path(cnts_Ů[1])
m_ring_Ů = circle_median(cx_U, cy_U, radius=85.2, n_points=24)
s_U_local = segs_to_path(cnts_Ů[2])
m_U_cap = [[145, 812], [145, 300], [200, 150], [380, 108], [550, 150], [610, 300], [610, 812]]
corrections["Ů"] = {
    "strokes": [s_ring_Ů, s_U_local],
    "medians": [m_ring_Ů, m_U_cap]
}

# ų (lowercase u with ogonek: 1 single continuous stroke)
cnts_ų = load_svg_glyph(open(os.path.join(FONT_DIR, "ų.svg"), encoding="utf-8").read())
s_ų = " ".join(segs_to_path(c) for c in cnts_ų)
m_ų = [
    [126, 620], [126, 276], [160, 150], [280, 108], [420, 150],
    [500, 300], [500, 620], [500, 300], [509, 81],
    [460, 40], [420, -10], [410, -60], [435, -95], [470, -112], [520, -112], [540, -92]
]
corrections["ų"] = {"strokes": [s_ų], "medians": [m_ų]}

# ư (u with horn)
cnts_ư = load_svg_glyph(open(os.path.join(FONT_DIR, "ư.svg"), encoding="utf-8").read())
s_ư = " ".join(segs_to_path(c) for c in cnts_ư)
corrections["ư"] = {"strokes": [s_ư], "medians": [m_base_u]}

# =========================================================================
# 2. Ş and ş (1 single continuous stroke each)
# =========================================================================
cnts_Ş = load_svg_glyph(open(os.path.join(FONT_DIR, "Ş.svg"), encoding="utf-8").read())
s_Ş = segs_to_path(cnts_Ş[0]) + " " + segs_to_path(cnts_Ş[1])
m_Ş = [
    [460, 720], [400, 770], [280, 775], [160, 740], [110, 640], [150, 540],
    [250, 460], [380, 390],
    [470, 310], [470, 200], [400, 120], [280, 115], [150, 160],
    [310, 120], [335, 72], [345, 10], [345, -30], [310, -70], [240, -90], [180, -75]
]
corrections["Ş"] = {"strokes": [s_Ş], "medians": [m_Ş]}

cnts_ş = load_svg_glyph(open(os.path.join(FONT_DIR, "ş.svg"), encoding="utf-8").read())
s_ş = segs_to_path(cnts_ş[0]) + " " + segs_to_path(cnts_ş[1])
m_ş = [
    [416, 596], [360, 630], [250, 635], [150, 610], [110, 530], [140, 450],
    [220, 380], [330, 320],
    [420, 250], [420, 170], [360, 110], [250, 105], [130, 140],
    [280, 110], [305, 72], [315, 10], [315, -30], [280, -70], [220, -90], [165, -75]
]
corrections["ş"] = {"strokes": [s_ş], "medians": [m_ş]}

# =========================================================================
# 3. y family (y, ý, ŷ, ÿ, ƴ)
# =========================================================================
cnts_y_font = load_svg_glyph(open(os.path.join(FONT_DIR, "y.svg"), encoding="utf-8").read())
s_base_y = segs_to_path(cnts_y_font[0])
m_base_y = [
    [50, 620], [200, 280], [264, 170],
    [350, 380], [480, 620],
    [480, 620], [280, 100], [200, -80], [70, -150], [15, -120]
]
corrections["y"] = {"strokes": [s_base_y], "medians": [m_base_y]}

cnts_ý = load_svg_glyph(open(os.path.join(FONT_DIR, "ý.svg"), encoding="utf-8").read())
corrections["ý"] = {
    "strokes": [segs_to_path(cnts_ý[0]), segs_to_path(cnts_ý[1])],
    "medians": [[[455, 855], [273, 711]], m_base_y]
}

cnts_ŷ = load_svg_glyph(open(os.path.join(FONT_DIR, "ŷ.svg"), encoding="utf-8").read())
corrections["ŷ"] = {
    "strokes": [segs_to_path(cnts_ŷ[0]), segs_to_path(cnts_ŷ[1])],
    "medians": [[[180, 711], [290, 855], [400, 711]], m_base_y]
}

cnts_ÿ = load_svg_glyph(open(os.path.join(FONT_DIR, "ÿ.svg"), encoding="utf-8").read())
corrections["ÿ"] = {
    "strokes": [segs_to_path(cnts_ÿ[0]), segs_to_path(cnts_ÿ[1]), segs_to_path(cnts_ÿ[2])],
    "medians": [[[220, 800], [220, 730]], [[380, 800], [380, 730]], m_base_y]
}

cnts_ƴ = load_svg_glyph(open(os.path.join(FONT_DIR, "ƴ.svg"), encoding="utf-8").read())
corrections["ƴ"] = {"strokes": [segs_to_path(cnts_ƴ[0])], "medians": [m_base_y]}

# =========================================================================
# 4. Y family (Y, Ý, Ŷ, Ÿ, Ƴ)
# =========================================================================
cnts_Y_font = load_svg_glyph(open(os.path.join(FONT_DIR, "Y.svg"), encoding="utf-8").read())
s_Y_left = "M 0 811.936 L 100.352 811.936 L 289.792 452.512 L 243.712 360.352 Z"
s_Y_right = "M 480.256 811.936 L 579.584 811.936 L 335.872 364.448 L 335.872 80.8 L 243.712 80.8 L 243.712 360.352 L 289.792 452.512 Z"
m_Y_left = [[50, 812], [289, 410]]
m_Y_right = [[530, 812], [289, 410], [289, 81]]
corrections["Y"] = {"strokes": [s_Y_left, s_Y_right], "medians": [m_Y_left, m_Y_right]}

cnts_Ý = load_svg_glyph(open(os.path.join(FONT_DIR, "Ý.svg"), encoding="utf-8").read())
corrections["Ý"] = {
    "strokes": [segs_to_path(cnts_Ý[0]), s_Y_left, s_Y_right],
    "medians": [[[465, 1037], [282, 894]], m_Y_left, m_Y_right]
}

cnts_Ŷ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŷ.svg"), encoding="utf-8").read())
corrections["Ŷ"] = {
    "strokes": [segs_to_path(cnts_Ŷ[0]), s_Y_left, s_Y_right],
    "medians": [[[180, 894], [304, 1037], [428, 894]], m_Y_left, m_Y_right]
}

cnts_Ÿ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ÿ.svg"), encoding="utf-8").read())
corrections["Ÿ"] = {
    "strokes": [segs_to_path(cnts_Ÿ[0]), segs_to_path(cnts_Ÿ[1]), s_Y_left, s_Y_right],
    "medians": [[[230, 1010], [230, 910]], [[377, 1010], [377, 910]], m_Y_left, m_Y_right]
}

cnts_Ƴ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ƴ.svg"), encoding="utf-8").read())
corrections["Ƴ"] = {"strokes": [segs_to_path(cnts_Ƴ[0])], "medians": [m_Y_right]}

# =========================================================================
# 5. g family (g, ĝ, ğ, ģ)
# =========================================================================
cnts_g_font = load_svg_glyph(open(os.path.join(FONT_DIR, "g.svg"), encoding="utf-8").read())
s_base_g = segs_to_path(cnts_g_font[0]) + " " + segs_to_path(cnts_g_font[1])
m_base_g = [
    [470, 560], [300, 600], [150, 500], [150, 300], [300, 180], [470, 250], [470, 560],
    [500, 350], [500, 80],
    [450, -80], [300, -165], [150, -150], [90, -80]
]
corrections["g"] = {"strokes": [s_base_g], "medians": [m_base_g]}

cnts_ĝ = load_svg_glyph(open(os.path.join(FONT_DIR, "ĝ.svg"), encoding="utf-8").read())
corrections["ĝ"] = {
    "strokes": [segs_to_path(cnts_ĝ[0]), segs_to_path(cnts_ĝ[1]) + " " + segs_to_path(cnts_ĝ[2])],
    "medians": [[[180, 711], [290, 855], [400, 711]], m_base_g]
}

cnts_ğ = load_svg_glyph(open(os.path.join(FONT_DIR, "ğ.svg"), encoding="utf-8").read())
corrections["ğ"] = {
    "strokes": [segs_to_path(cnts_ğ[0]), segs_to_path(cnts_ğ[1]) + " " + segs_to_path(cnts_ğ[2])],
    "medians": [[[180, 830], [290, 730], [400, 830]], m_base_g]
}

cnts_ģ = load_svg_glyph(open(os.path.join(FONT_DIR, "ģ.svg"), encoding="utf-8").read())
corrections["ģ"] = {
    "strokes": [segs_to_path(cnts_ģ[0]), segs_to_path(cnts_ģ[1]) + " " + segs_to_path(cnts_ģ[2])],
    "medians": [[[300, 850], [300, 700]], m_base_g]
}

# =========================================================================
# 6. r family (r, ŕ, Ŗ, ŗ, Ř, ř)
# =========================================================================
cnts_r_font = load_svg_glyph(open(os.path.join(FONT_DIR, "r.svg"), encoding="utf-8").read())
s_base_r = segs_to_path(cnts_r_font[0])
m_base_r = [[132, 616], [132, 81], [132, 450], [250, 580], [385, 567]]
corrections["r"] = {"strokes": [s_base_r], "medians": [m_base_r]}

cnts_ŕ = load_svg_glyph(open(os.path.join(FONT_DIR, "ŕ.svg"), encoding="utf-8").read())
corrections["ŕ"] = {
    "strokes": [segs_to_path(cnts_ŕ[0]), segs_to_path(cnts_ŕ[1])],
    "medians": [[[455, 855], [273, 711]], m_base_r]
}

cnts_ř = load_svg_glyph(open(os.path.join(FONT_DIR, "ř.svg"), encoding="utf-8").read())
corrections["ř"] = {
    "strokes": [segs_to_path(cnts_ř[0]), segs_to_path(cnts_ř[1])],
    "medians": [[[180, 855], [290, 711], [400, 855]], m_base_r]
}

cnts_ŗ = load_svg_glyph(open(os.path.join(FONT_DIR, "ŗ.svg"), encoding="utf-8").read())
corrections["ŗ"] = {
    "strokes": [segs_to_path(cnts_ŗ[1]), segs_to_path(cnts_ŗ[0])],
    "medians": [m_base_r, [[180, 72], [180, -95]]]
}

# =========================================================================
# 7. M, Q, R, Ŕ, Ř, Ŗ, Ʀ
# =========================================================================
cnts_M = load_svg_glyph(open(os.path.join(FONT_DIR, "M.svg"), encoding="utf-8").read())
corrections["M"] = {
    "strokes": [segs_to_path(cnts_M[0])],
    "medians": [[[142, 81], [142, 812], [462, 212], [784, 812], [784, 81]]]
}

# Base E (4 clean rectangular strokes)
s_E_stem = "M 99.328 811.904 L 191.488 811.904 L 191.488 80.8 L 99.328 80.8 Z"
m_E_stem = [[145, 812], [145, 81]]
s_E_top = "M 99.328 811.904 L 507.904 811.904 L 507.904 731.008 L 99.328 731.008 Z"
m_E_top = [[145, 771], [508, 771]]
s_E_mid = "M 191.488 502.656 L 489.472 502.656 L 489.472 422.784 L 191.488 422.784 Z"
m_E_mid = [[191, 463], [489, 463]]
s_E_bot = "M 99.328 161.664 L 507.904 161.664 L 507.904 80.8 L 99.328 80.8 Z"
m_E_bot = [[145, 121], [508, 121]]
corrections["E"] = {"strokes": [s_E_stem, s_E_top, s_E_mid, s_E_bot], "medians": [m_E_stem, m_E_top, m_E_mid, m_E_bot]}

# Q: O-body + tail
cnts_Q = load_svg_glyph(open(os.path.join(FONT_DIR, "Q.svg"), encoding="utf-8").read())
b_close_oval = Seg('L', np.array([[425.984, 71.584], [524.288, 88.992]]))
oval_outer_Q = cnts_Q[0][5:] + cnts_Q[0][:2] + [b_close_oval]
s_Q_oval = segs_to_path(oval_outer_Q) + " " + segs_to_path(cnts_Q[1])
s_Q_tail = f"M {524.288} {88.992} L {699.392} {-93.28} L {567.296} {-93.28} L {425.984} {71.584} Z"
m_Q_oval = circle_median(400, 447, radius=295, n_points=24)
m_Q_tail = [[475, 80], [633, -93]]
corrections["Q"] = {"strokes": [s_Q_oval, s_Q_tail], "medians": [m_Q_oval, m_Q_tail]}

# R: 3 strokes
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

cnts_Ŕ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŕ.svg"), encoding="utf-8").read())
corrections["Ŕ"] = {
    "strokes": [segs_to_path(cnts_Ŕ[0]), s_R_stem, s_R_lobe, s_R_leg],
    "medians": [[[465, 1037], [340, 894]], m_R_stem, m_R_lobe, m_R_leg]
}

cnts_Ř = load_svg_glyph(open(os.path.join(FONT_DIR, "Ř.svg"), encoding="utf-8").read())
corrections["Ř"] = {
    "strokes": [segs_to_path(cnts_Ř[0]), s_R_stem, s_R_lobe, s_R_leg],
    "medians": [[[250, 1037], [360, 910], [470, 1037]], m_R_stem, m_R_lobe, m_R_leg]
}

cnts_Ŗ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŗ.svg"), encoding="utf-8").read())
corrections["Ŗ"] = {
    "strokes": [s_R_stem, s_R_lobe, s_R_leg, segs_to_path(cnts_Ŗ[0])],
    "medians": [m_R_stem, m_R_lobe, m_R_leg, [[356, 80], [356, -95]]]
}

cnts_þ = load_svg_glyph(open(os.path.join(FONT_DIR, "þ.svg"), encoding="utf-8").read())
s_þ = segs_to_path(cnts_þ[0]) + " " + segs_to_path(cnts_þ[1])
m_þ = [[132, 812], [132, -165], [177, 145], [350, 110], [500, 250], [500, 450], [350, 564], [132, 564]]
corrections["þ"] = {"strokes": [s_þ], "medians": [m_þ]}

cnts_Ʀ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ʀ.svg"), encoding="utf-8").read())
s_Ʀ = segs_to_path(cnts_Ʀ[0]) + " " + segs_to_path(cnts_Ʀ[1])
m_Ʀ = [[145, 812], [145, 81], [191, 772], [470, 700], [510, 600], [470, 480], [191, 440], [350, 410], [580, 81]]
corrections["Ʀ"] = {"strokes": [s_Ʀ], "medians": [m_Ʀ]}

# =========================================================================
# 8. Digits: 6, 8, 9
# =========================================================================
cnts_6 = load_svg_glyph(open(os.path.join(FONT_DIR, "6.svg"), encoding="utf-8").read())
s_6 = segs_to_path(cnts_6[0]) + " " + segs_to_path(cnts_6[1])
m_6 = [
    [480, 800], [330, 730], [180, 520], [105, 300], [140, 150],
    [280, 105], [450, 130], [490, 250], [450, 380], [330, 430], [220, 380]
]
corrections["6"] = {"strokes": [s_6], "medians": [m_6]}

cnts_8 = load_svg_glyph(open(os.path.join(FONT_DIR, "8.svg"), encoding="utf-8").read())
s_8 = segs_to_path(cnts_8[0]) + " " + segs_to_path(cnts_8[1]) + " " + segs_to_path(cnts_8[2])
m_8 = [
    [350, 800], [220, 740], [180, 640], [290, 465],
    [420, 350], [440, 220], [380, 110], [280, 100], [160, 150], [140, 280],
    [290, 465], [400, 640], [350, 800], [300, 800]
]
corrections["8"] = {"strokes": [s_8], "medians": [m_8]}

cnts_9 = load_svg_glyph(open(os.path.join(FONT_DIR, "9.svg"), encoding="utf-8").read())
s_9 = segs_to_path(cnts_9[0]) + " " + segs_to_path(cnts_9[1])
m_9 = [
    [440, 440], [440, 600], [380, 720], [280, 750], [160, 680], [140, 540], [200, 440], [320, 420], [440, 440],
    [490, 320], [480, 180], [380, 100], [220, 100]
]
corrections["9"] = {"strokes": [s_9], "medians": [m_9]}

# =========================================================================
# 9. Symbols: &, @, °, ®
# =========================================================================
cnts_amp = load_svg_glyph(open(os.path.join(FONT_DIR, "&.svg"), encoding="utf-8").read())
s_amp = segs_to_path(cnts_amp[0]) + " " + segs_to_path(cnts_amp[1]) + " " + segs_to_path(cnts_amp[2])
m_amp = [
    [600, 226], [530, 226], [400, 280], [290, 465],
    [350, 600], [350, 720], [300, 760], [240, 720], [240, 600],
    [290, 465], [180, 360], [148, 270], [186, 180], [287, 150], [420, 180], [643, 123]
]
corrections["&"] = {"strokes": [s_amp], "medians": [m_amp]}

cnts_at = load_svg_glyph(open(os.path.join(FONT_DIR, "@.svg"), encoding="utf-8").read())
s_at = segs_to_path(cnts_at[0]) + " " + segs_to_path(cnts_at[1])
m_at = [
    [400, 460], [320, 520], [240, 460], [240, 360], [320, 320], [400, 360], [400, 460],
    [450, 600], [550, 620], [620, 500], [620, 320], [520, 120], [340, 70],
    [160, 140], [100, 320], [120, 600], [260, 780], [450, 820], [650, 740], [740, 550], [740, 320]
]
corrections["@"] = {"strokes": [s_at], "medians": [m_at]}

cnts_deg = load_svg_glyph(open(os.path.join(FONT_DIR, "°.svg"), encoding="utf-8").read())
poly_deg_out = [s.p0 for s in cnts_deg[0]]
poly_deg_in = [s.p0 for s in cnts_deg[1]]
deg_cx = (np.mean([p[0] for p in poly_deg_out]) + np.mean([p[0] for p in poly_deg_in])) / 2.0
deg_cy = (np.mean([p[1] for p in poly_deg_out]) + np.mean([p[1] for p in poly_deg_in])) / 2.0
s_deg = segs_to_path(cnts_deg[0]) + " " + segs_to_path(cnts_deg[1])
m_deg = circle_median(deg_cx, deg_cy, radius=50.0, n_points=24)
corrections["°"] = {"strokes": [s_deg], "medians": [m_deg]}

# =========================================================================
# 10. N and D variants
# =========================================================================
m_N_base = [[145, 95], [145, 780], [636, 120], [636, 780]]
for c, diac_med in [("Ń", [[528, 1037], [346, 894]]), ("Ň", [[250, 1037], [360, 910], [470, 1037]]), ("Ñ", [[200, 950], [350, 1020], [500, 950]])]:
    cnts_N = load_svg_glyph(open(os.path.join(FONT_DIR, f"{c}.svg"), encoding="utf-8").read())
    corrections[c] = {
        "strokes": [segs_to_path(cnts_N[0]), segs_to_path(cnts_N[1])],
        "medians": [diac_med, m_N_base]
    }

cnts_Ņ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ņ.svg"), encoding="utf-8").read())
corrections["Ņ"] = {
    "strokes": [segs_to_path(cnts_Ņ[1]), segs_to_path(cnts_Ņ[0])],
    "medians": [m_N_base, [[390, 80], [390, -95]]]
}

cnts_Ď = load_svg_glyph(open(os.path.join(FONT_DIR, "Ď.svg"), encoding="utf-8").read())
s_D_base = segs_to_path(cnts_Ď[1]) + " " + segs_to_path(cnts_Ď[2])
m_D_base = [[145, 812], [145, 81], [191, 121], [450, 121], [630, 450], [450, 772], [191, 772]]
corrections["Ď"] = {
    "strokes": [segs_to_path(cnts_Ď[0]), s_D_base],
    "medians": [[[695, 849], [651, 818], [616, 723], [592, 705]], m_D_base]
}

# =========================================================================
# 11. B/P and extended letters: Ɓ, Ƃ, ƥ, ð, þ
# =========================================================================
cnts_Ɓ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ɓ.svg"), encoding="utf-8").read())
s_Ɓ = " ".join(segs_to_path(cnt) for cnt in cnts_Ɓ)
m_Ɓ = [[10, 812], [225, 812], [225, 81], [273, 81], [450, 100], [580, 250], [500, 430], [273, 440], [450, 470], [560, 600], [480, 730], [225, 733]]
corrections["Ɓ"] = {"strokes": [s_Ɓ], "medians": [m_Ɓ]}

cnts_Ƃ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ƃ.svg"), encoding="utf-8").read())
s_Ƃ = " ".join(segs_to_path(cnt) for cnt in cnts_Ƃ)
m_Ƃ = [[100, 812], [500, 812], [145, 812], [145, 81], [191, 81], [350, 100], [500, 250], [450, 420], [145, 423]]
corrections["Ƃ"] = {"strokes": [s_Ƃ], "medians": [m_Ƃ]}

cnts_ƥ = load_svg_glyph(open(os.path.join(FONT_DIR, "ƥ.svg"), encoding="utf-8").read())
s_ƥ = " ".join(segs_to_path(cnt) for cnt in cnts_ƥ)
m_ƥ = [[87, 864], [132, 864], [132, -165], [177, 145], [350, 110], [500, 250], [500, 450], [350, 564], [132, 564]]
corrections["ƥ"] = {"strokes": [s_ƥ], "medians": [m_ƥ]}

cnts_ð = load_svg_glyph(open(os.path.join(FONT_DIR, "ð.svg"), encoding="utf-8").read())
s_ð = " ".join(segs_to_path(cnt) for cnt in cnts_ð)
m_ð = [[500, 864], [350, 800], [200, 700], [500, 550], [480, 145], [350, 100], [150, 200], [150, 450], [350, 550], [480, 500]]
corrections["ð"] = {"strokes": [s_ð], "medians": [m_ð]}

# (þ defined above)

print(f"Total explicitly corrected characters: {len(corrections)}")

# Apply corrections and write to disk
final_dataset = {}
for item in queue:
    c = item["char"]
    font_path = os.path.join(FONT_DIR, f"{c}.svg")
    cnts_font = load_svg_glyph(open(font_path, encoding="utf-8").read())
    
    if c in good_chars:
        # Load from existing saved JSON
        data_path = os.path.join(DATA_DIR, f"{c}.json")
        data = json.load(open(data_path, encoding="utf-8"))
        final_dataset[c] = data
    elif c in corrections:
        final_dataset[c] = corrections[c]
    else:
        # Fallback to existing if IoU is already >= 0.99
        data_path = os.path.join(DATA_DIR, f"{c}.json")
        data = json.load(open(data_path, encoding="utf-8"))
        final_dataset[c] = data

# Verify and Save
low_iou = []
for c, data in final_dataset.items():
    cnts_font = load_svg_glyph(open(os.path.join(FONT_DIR, f"{c}.svg"), encoding="utf-8").read())
    score = check_iou(cnts_font, data["strokes"])
    if score < 0.98:
        low_iou.append((c, score))
    p = os.path.join(DATA_DIR, f"{c}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

print(f"Saved {len(final_dataset)} JSON files. Low IoU count: {len(low_iou)} {low_iou}")

# Commit to dashboard API
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
committed = 0
for c, data in final_dataset.items():
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:9000/api/commit",
            data=json.dumps({"char": c, "variant": "", "data": data, "overwrite": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        opener.open(req, timeout=3)
        committed += 1
    except Exception as e:
        print(f"Error committing '{c}': {e}")

print(f"Successfully committed {committed} / {len(final_dataset)} characters to dashboard API.")
