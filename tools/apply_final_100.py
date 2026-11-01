#!/usr/bin/env python3
"""Master correction and verification script for the first 100 characters in the dashboard."""
import json
import os
import sys
import numpy as np
import cairo

sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path, Seg, parse_hw_path
from strokegen.raster import render, iou, SPAN, X0, Y_TOP, _trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")
DATA_DIR = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")
MANIFEST_PATH = os.path.join(ROOT, "training_data", "font_glyphs", "manifest.json")

# The 13 confirmed GOOD characters: NEVER MODIFY
GOOD_CHARS = set(["4", "È", "É", "Ê", "Ë", "Ö", "ß", "Ē", "Ĕ", "Ė", "Ę", "Ě", "Ų"])

def circle_median(cx, cy, radius, n_points=24):
    thetas = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi - 0.25, n_points)
    return [[int(round(cx + radius * np.cos(th))), int(round(cy + radius * np.sin(th)))] for th in thetas]

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

def build_corrections():
    corrections = {}
    
    # ---------------------------------------------------------------------
    # 1. Ş and ş (1 single continuous stroke, no terminal shortening)
    # ---------------------------------------------------------------------
    cnts_Ş = load_svg_glyph(open(os.path.join(FONT_DIR, "Ş.svg")).read())
    s_Ş = segs_to_path(cnts_Ş[0]) + " " + segs_to_path(cnts_Ş[1])
    m_Ş = [
        [485, 750], [420, 780], [280, 780], [160, 740], [100, 640], [150, 540],
        [250, 460], [380, 390], [470, 320], [490, 260], [460, 180], [380, 110],
        [250, 100], [140, 115], [58, 145], [140, 115], [260, 100], [260, 70],
        [250, 20], [310, -10], [345, -50], [310, -110], [210, -145], [162, -135]
    ]
    corrections["Ş"] = {"strokes": [s_Ş], "medians": [m_Ş]}

    cnts_ş = load_svg_glyph(open(os.path.join(FONT_DIR, "ş.svg")).read())
    s_ş = segs_to_path(cnts_ş[0]) + " " + segs_to_path(cnts_ş[1])
    m_ş = [
        [418, 567], [350, 595], [255, 603], [150, 580], [95, 510], [130, 440],
        [200, 400], [300, 350], [380, 290], [400, 230], [380, 160], [320, 115],
        [225, 105], [130, 115], [58, 145], [130, 115], [240, 95], [240, 70],
        [240, 20], [290, -10], [325, -50], [290, -110], [195, -145], [146, -135]
    ]
    corrections["ş"] = {"strokes": [s_ş], "medians": [m_ş]}

    # ---------------------------------------------------------------------
    # 2. Lowercase y family: y, ý, ŷ, ÿ, ƴ (2 quiz strokes for base y)
    # ---------------------------------------------------------------------
    cnts_y = load_svg_glyph(open(os.path.join(FONT_DIR, "y.svg")).read())
    c_y = cnts_y[0]
    p_junc = np.array([186.444, 167.84])
    s20_bot = Seg('L', np.array([c_y[20].p0, p_junc]))
    s20_top = Seg('L', np.array([p_junc, c_y[20].p1]))
    b_cut = Seg('L', np.array([c_y[4].p0, p_junc]))
    b_cut_rev = Seg('L', np.array([p_junc, c_y[4].p0]))
    s_y0 = segs_to_path([s20_top, c_y[0], c_y[1], c_y[2], c_y[3], b_cut])
    s_y1 = segs_to_path([b_cut_rev] + c_y[5:20] + [s20_bot])
    m_y0 = [[50, 629], [224, 168]]
    m_y1 = [[475, 629], [285, 168], [210, -30], [130, -115], [30, -125]]
    corrections["y"] = {"strokes": [s_y0, s_y1], "medians": [m_y0, m_y1]}

    cnts_ý = load_svg_glyph(open(os.path.join(FONT_DIR, "ý.svg")).read())
    corrections["ý"] = {
        "strokes": [segs_to_path(cnts_ý[0]), s_y0, s_y1],
        "medians": [[[360, 865], [236, 701]], m_y0, m_y1]
    }

    cnts_ŷ = load_svg_glyph(open(os.path.join(FONT_DIR, "ŷ.svg")).read())
    corrections["ŷ"] = {
        "strokes": [segs_to_path(cnts_ŷ[0]), s_y0, s_y1],
        "medians": [[[180, 711], [290, 855], [400, 711]], m_y0, m_y1]
    }

    cnts_ÿ = load_svg_glyph(open(os.path.join(FONT_DIR, "ÿ.svg")).read())
    corrections["ÿ"] = {
        "strokes": [segs_to_path(cnts_ÿ[0]), segs_to_path(cnts_ÿ[1]), s_y0, s_y1],
        "medians": [[[166, 828], [166, 727]], [[358, 828], [358, 727]], m_y0, m_y1]
    }

    cnts_ƴ = load_svg_glyph(open(os.path.join(FONT_DIR, "ƴ.svg")).read())
    c_ƴ = cnts_ƴ[0]
    s18_bot = Seg('L', np.array([c_ƴ[18].p0, p_junc]))
    s18_top = Seg('L', np.array([p_junc, c_ƴ[18].p1]))
    b_cut_ƴ = Seg('L', np.array([c_ƴ[22].p1, p_junc]))
    b_cut_rev_ƴ = Seg('L', np.array([p_junc, c_ƴ[22].p1]))
    s_ƴ0 = segs_to_path([s18_top, c_ƴ[19], c_ƴ[20], c_ƴ[21], c_ƴ[22], b_cut_ƴ])
    s_ƴ1_hook = segs_to_path([b_cut_rev_ƴ] + c_ƴ[24:] + c_ƴ[:18] + [s18_bot])
    m_y1_hook = [[520, 600], [450, 560], [285, 168], [210, -30], [130, -115], [30, -125]]
    corrections["ƴ"] = {"strokes": [s_ƴ0, s_ƴ1_hook], "medians": [m_y0, m_y1_hook]}

    # ---------------------------------------------------------------------
    # 3. Uppercase R family: R, Ŕ, Ŗ, Ř, Ʀ (exactly 2 strokes for R body)
    # ---------------------------------------------------------------------
    cnts_R = load_svg_glyph(open(os.path.join(FONT_DIR, "R.svg")).read())
    s_R_stem = "M 99.328 811.936 L 191.488 811.936 L 191.488 80.8 L 99.328 80.8 Z"
    m_R_stem = [[145, 812], [145, 81]]
    b_top = Seg('L', np.array([[191.488, 811.936], [301.056, 811.936]]))
    b_stem_up = Seg('L', np.array([[191.488, 382.88], [191.488, 811.936]]))
    s_R_lobe_leg = segs_to_path([b_top] + cnts_R[0][:9] + [b_stem_up]) + " " + segs_to_path(cnts_R[1])
    m_R_lobe_leg = [
        [191, 772], [300, 772], [470, 720], [520, 600], [470, 470], [300, 420],
        [210, 420], [350, 400], [480, 240], [570, 81]
    ]
    corrections["R"] = {"strokes": [s_R_stem, s_R_lobe_leg], "medians": [m_R_stem, m_R_lobe_leg]}

    cnts_Ŕ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŕ.svg")).read())
    corrections["Ŕ"] = {
        "strokes": [segs_to_path(cnts_Ŕ[0]), s_R_stem, s_R_lobe_leg],
        "medians": [[[465, 1037], [340, 894]], m_R_stem, m_R_lobe_leg]
    }

    cnts_Ř = load_svg_glyph(open(os.path.join(FONT_DIR, "Ř.svg")).read())
    corrections["Ř"] = {
        "strokes": [segs_to_path(cnts_Ř[0]), s_R_stem, s_R_lobe_leg],
        "medians": [[[250, 1037], [360, 910], [470, 1037]], m_R_stem, m_R_lobe_leg]
    }

    cnts_Ŗ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŗ.svg")).read())
    corrections["Ŗ"] = {
        "strokes": [s_R_stem, s_R_lobe_leg, segs_to_path(cnts_Ŗ[0])],
        "medians": [m_R_stem, m_R_lobe_leg, [[356, 80], [356, -95]]]
    }

    # Ʀ (Latin letter YR)
    cnts_Ʀ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ʀ.svg")).read())
    b_stem_up_Ʀ = Seg('L', np.array([[191.488, 280.48], [191.488, 709.536]]))
    s_Ʀ_lobe_leg = segs_to_path([b_stem_up_Ʀ] + cnts_Ʀ[0][7:] + [cnts_Ʀ[0][0], cnts_Ʀ[0][1]]) + " " + segs_to_path(cnts_Ʀ[1])
    m_Ʀ_lobe_leg = [
        [191, 670], [300, 670], [470, 620], [520, 500], [470, 370], [300, 320],
        [210, 320], [350, 300], [480, 140], [570, -21.6]
    ]
    corrections["Ʀ"] = {"strokes": [s_R_stem, s_Ʀ_lobe_leg], "medians": [m_R_stem, m_Ʀ_lobe_leg]}

    # ---------------------------------------------------------------------
    # 4. Lowercase r family: r, ŕ, ŗ, ř
    # ---------------------------------------------------------------------
    cnts_r = load_svg_glyph(open(os.path.join(FONT_DIR, "r.svg")).read())
    s_r = segs_to_path(cnts_r[0])
    m_r = [[132, 616], [132, 81], [132, 450], [240, 595], [330, 610], [385, 570]]
    corrections["r"] = {"strokes": [s_r], "medians": [m_r]}

    cnts_ŕ = load_svg_glyph(open(os.path.join(FONT_DIR, "ŕ.svg")).read())
    corrections["ŕ"] = {
        "strokes": [segs_to_path(cnts_ŕ[0]), s_r],
        "medians": [[[360, 865], [220, 701]], m_r]
    }

    cnts_ř = load_svg_glyph(open(os.path.join(FONT_DIR, "ř.svg")).read())
    corrections["ř"] = {
        "strokes": [segs_to_path(cnts_ř[0]), s_r],
        "medians": [[[95, 855], [245, 715], [395, 855]], m_r]
    }

    cnts_ŗ = load_svg_glyph(open(os.path.join(FONT_DIR, "ŗ.svg")).read())
    corrections["ŗ"] = {
        "strokes": [s_r, segs_to_path(cnts_ŗ[0])],
        "medians": [m_r, [[132, 20], [132, -140]]]
    }

    # ---------------------------------------------------------------------
    # 5. Lowercase g family: g, ĝ, ğ, ģ
    # ---------------------------------------------------------------------
    cnts_g = load_svg_glyph(open(os.path.join(FONT_DIR, "g.svg")).read())
    s_g = segs_to_path(cnts_g[0]) + " " + segs_to_path(cnts_g[1])
    m_g = [
        [480, 620], [380, 602], [280, 602], [180, 580], [103, 490], [103, 350],
        [103, 220], [180, 120], [280, 107], [400, 120], [470, 200], [498, 350],
        [480, 620], [498, 350], [498, 80], [460, -60], [350, -145], [200, -145], [105, -80]
    ]
    corrections["g"] = {"strokes": [s_g], "medians": [m_g]}

    cnts_ĝ = load_svg_glyph(open(os.path.join(FONT_DIR, "ĝ.svg")).read())
    corrections["ĝ"] = {
        "strokes": [segs_to_path(cnts_ĝ[0]), s_g],
        "medians": [[[180, 711], [290, 855], [400, 711]], m_g]
    }

    cnts_ğ = load_svg_glyph(open(os.path.join(FONT_DIR, "ğ.svg")).read())
    corrections["ğ"] = {
        "strokes": [segs_to_path(cnts_ğ[0]), s_g],
        "medians": [[[180, 830], [290, 730], [400, 830]], m_g]
    }

    cnts_ģ = load_svg_glyph(open(os.path.join(FONT_DIR, "ģ.svg")).read())
    corrections["ģ"] = {
        "strokes": [segs_to_path(cnts_ģ[0]), s_g],
        "medians": [[[300, 850], [300, 700]], m_g]
    }

    # ---------------------------------------------------------------------
    # 6. Lowercase u family: ù, ú, û, ü, ũ, ū, ŭ, ů, ű, ų, ư, and Capital Ů
    # ---------------------------------------------------------------------
    cnts_u = load_svg_glyph(open(os.path.join(FONT_DIR, "u.svg")).read())
    s_u = segs_to_path(cnts_u[0])
    m_base_u = [
        [126, 620], [126, 276], [160, 150], [280, 108], [420, 150],
        [500, 300], [500, 620], [500, 300], [509, 81], [465, 120]
    ]

    cnts_ù = load_svg_glyph(open(os.path.join(FONT_DIR, "ù.svg")).read())
    corrections["ù"] = {"strokes": [segs_to_path(cnts_ù[0]), segs_to_path(cnts_ù[1])], "medians": [[[196, 855], [378, 711]], m_base_u]}

    cnts_ú = load_svg_glyph(open(os.path.join(FONT_DIR, "ú.svg")).read())
    corrections["ú"] = {"strokes": [segs_to_path(cnts_ú[0]), segs_to_path(cnts_ú[1])], "medians": [[[455, 855], [273, 711]], m_base_u]}

    cnts_û = load_svg_glyph(open(os.path.join(FONT_DIR, "û.svg")).read())
    corrections["û"] = {"strokes": [segs_to_path(cnts_û[0]), segs_to_path(cnts_û[1])], "medians": [[[180, 711], [290, 855], [400, 711]], m_base_u]}

    cnts_ü = load_svg_glyph(open(os.path.join(FONT_DIR, "ü.svg")).read())
    corrections["ü"] = {"strokes": [segs_to_path(cnts_ü[0]), segs_to_path(cnts_ü[1]), segs_to_path(cnts_ü[2])], "medians": [[[220, 800], [220, 730]], [[380, 800], [380, 730]], m_base_u]}

    cnts_ũ = load_svg_glyph(open(os.path.join(FONT_DIR, "ũ.svg")).read())
    corrections["ũ"] = {"strokes": [segs_to_path(cnts_ũ[0]), segs_to_path(cnts_ũ[1])], "medians": [[[150, 760], [240, 825], [380, 725], [485, 780]], m_base_u]}

    cnts_ū = load_svg_glyph(open(os.path.join(FONT_DIR, "ū.svg")).read())
    corrections["ū"] = {"strokes": [segs_to_path(cnts_ū[0]), segs_to_path(cnts_ū[1])], "medians": [[[180, 737], [450, 737]], m_base_u]}

    cnts_ŭ = load_svg_glyph(open(os.path.join(FONT_DIR, "ŭ.svg")).read())
    corrections["ŭ"] = {"strokes": [segs_to_path(cnts_ŭ[0]), segs_to_path(cnts_ŭ[1])], "medians": [[[175, 830], [315, 715], [455, 830]], m_base_u]}

    cnts_ů = load_svg_glyph(open(os.path.join(FONT_DIR, "ů.svg")).read())
    s_ring_ů = segs_to_path(cnts_ů[0]) + " " + segs_to_path(cnts_ů[1])
    m_ring_ů = circle_median(315.0, 750.0, radius=85.2, n_points=24)
    corrections["ů"] = {"strokes": [s_ring_ů, segs_to_path(cnts_ů[2])], "medians": [m_ring_ů, m_base_u]}

    cnts_Ů = load_svg_glyph(open(os.path.join(FONT_DIR, "Ů.svg")).read())
    s_ring_Ů = segs_to_path(cnts_Ů[0]) + " " + segs_to_path(cnts_Ů[1])
    m_ring_Ů = circle_median(377.0, 930.0, radius=85.2, n_points=24)
    m_U_cap = [[145, 812], [145, 300], [200, 150], [380, 108], [550, 150], [610, 300], [610, 812]]
    corrections["Ů"] = {"strokes": [s_ring_Ů, segs_to_path(cnts_Ů[2])], "medians": [m_ring_Ů, m_U_cap]}

    cnts_ű = load_svg_glyph(open(os.path.join(FONT_DIR, "ű.svg")).read())
    corrections["ű"] = {
        "strokes": [segs_to_path(cnts_ű[1]), segs_to_path(cnts_ű[0]), segs_to_path(cnts_ű[2])],
        "medians": [[[295, 855], [160, 711]], [[475, 855], [340, 711]], m_base_u]
    }

    cnts_ų = load_svg_glyph(open(os.path.join(FONT_DIR, "ų.svg")).read())
    s_ų = " ".join(segs_to_path(c) for c in cnts_ų)
    m_ų = [
        [126, 620], [126, 276], [160, 150], [280, 108], [420, 150],
        [500, 300], [500, 620], [500, 300], [509, 81],
        [460, 40], [420, -10], [410, -60], [435, -95], [470, -112], [520, -112], [540, -92]
    ]
    corrections["ų"] = {"strokes": [s_ų], "medians": [m_ų]}

    cnts_ư = load_svg_glyph(open(os.path.join(FONT_DIR, "ư.svg")).read())
    c_ư = cnts_ư[0]
    b_close_horn = Seg('L', np.array([c_ư[2].p1, c_ư[20].p0]))
    b_close_u = Seg('L', np.array([c_ư[19].p1, c_ư[3].p0]))
    s_horn = segs_to_path([c_ư[20], c_ư[21], c_ư[22], c_ư[23], c_ư[0], c_ư[1], c_ư[2], b_close_horn])
    s_u_body = segs_to_path(c_ư[3:20] + [b_close_u])
    m_horn = [[670, 715], [710, 660], [670, 580], [550, 536]]
    corrections["ư"] = {"strokes": [s_horn, s_u_body], "medians": [m_horn, m_base_u]}

    # ---------------------------------------------------------------------
    # 7. Digits and Symbols: 6, 8, 9, &, @, °, þ, ð
    # ---------------------------------------------------------------------
    cnts_6 = load_svg_glyph(open(os.path.join(FONT_DIR, "6.svg")).read())
    s_6 = segs_to_path(cnts_6[0]) + " " + segs_to_path(cnts_6[1])
    m_6 = [
        [481, 775], [420, 785], [330, 785], [230, 750], [150, 680], [105, 570],
        [90, 430], [90, 280], [120, 160], [200, 110], [305, 107], [420, 140],
        [490, 230], [490, 350], [440, 460], [320, 503], [220, 480], [150, 400], [150, 330]
    ]
    corrections["6"] = {"strokes": [s_6], "medians": [m_6]}

    cnts_8 = load_svg_glyph(open(os.path.join(FONT_DIR, "8.svg")).read())
    s_8 = segs_to_path(cnts_8[0]) + " " + segs_to_path(cnts_8[1]) + " " + segs_to_path(cnts_8[2])
    m_8 = [
        [290, 785], [190, 760], [115, 650], [170, 540], [290, 465], [410, 390],
        [490, 270], [430, 150], [295, 107], [160, 150], [94, 270], [180, 390],
        [290, 465], [410, 540], [465, 650], [390, 760], [290, 785]
    ]
    corrections["8"] = {"strokes": [s_8], "medians": [m_8]}

    cnts_9 = load_svg_glyph(open(os.path.join(FONT_DIR, "9.svg")).read())
    s_9 = segs_to_path(cnts_9[0]) + " " + segs_to_path(cnts_9[1])
    m_9 = [
        [460, 680], [380, 770], [285, 784], [180, 750], [96, 600], [140, 460],
        [270, 390], [380, 420], [475, 520], [475, 680], [480, 450], [460, 300],
        [410, 200], [315, 140], [200, 110], [104, 118]
    ]
    corrections["9"] = {"strokes": [s_9], "medians": [m_9]}

    cnts_amp = load_svg_glyph(open(os.path.join(FONT_DIR, "&.svg")).read())
    s_amp = segs_to_path(cnts_amp[0]) + " " + segs_to_path(cnts_amp[1]) + " " + segs_to_path(cnts_amp[2])
    m_amp = [
        [660, 440], [600, 340], [540, 240], [420, 140], [285, 110], [130, 150],
        [101, 270], [150, 380], [260, 480], [350, 560], [440, 660], [400, 780],
        [310, 820], [210, 760], [170, 650], [220, 540], [340, 450], [420, 300],
        [530, 230], [630, 140], [730, 80]
    ]
    corrections["&"] = {"strokes": [s_amp], "medians": [m_amp]}

    cnts_at = load_svg_glyph(open(os.path.join(FONT_DIR, "@.svg")).read())
    s_at = segs_to_path(cnts_at[0]) + " " + segs_to_path(cnts_at[1])
    thetas_a = np.linspace(np.pi/2, -3*np.pi/2, 12)
    pts_a = [[int(450 + 130 * np.cos(th)), int(400 + 170 * np.sin(th))] for th in thetas_a]
    pts_spiral = [
        [595, 250], [595, 450], [620, 560],
        [640, 320], [670, 260], [740, 270], [806, 380], [820, 500], [770, 640], [670, 730],
        [500, 778], [330, 760], [200, 680], [120, 530], [96, 370], [120, 220], [180, 120],
        [300, 45], [443, 23], [580, 40], [663, 66]
    ]
    m_at = pts_a + pts_spiral
    corrections["@"] = {"strokes": [s_at], "medians": [m_at]}

    cnts_deg = load_svg_glyph(open(os.path.join(FONT_DIR, "°.svg")).read())
    s_deg = segs_to_path(cnts_deg[0]) + " " + segs_to_path(cnts_deg[1])
    m_deg = circle_median(219.15, 665.0, radius=130.5, n_points=24)
    corrections["°"] = {"strokes": [s_deg], "medians": [m_deg]}

    cnts_þ = load_svg_glyph(open(os.path.join(FONT_DIR, "þ.svg")).read())
    s_þ = segs_to_path(cnts_þ[0]) + " " + segs_to_path(cnts_þ[1])
    m_þ = [
        [132, 850], [132, -155], [132, 400], [200, 570], [330, 600], [460, 560],
        [525, 450], [525, 260], [460, 150], [330, 107], [200, 130], [132, 200]
    ]
    corrections["þ"] = {"strokes": [s_þ], "medians": [m_þ]}

    cnts_eth = load_svg_glyph(open(os.path.join(FONT_DIR, "ð.svg")).read())
    s_eth = segs_to_path(cnts_eth[0]) + " " + segs_to_path(cnts_eth[1])
    m_eth = [
        [485, 830], [180, 655], [205, 835], [340, 750], [480, 600], [525, 360],
        [470, 160], [310, 107], [180, 140], [95, 250], [95, 400], [200, 520],
        [340, 550], [455, 500]
    ]
    corrections["ð"] = {"strokes": [s_eth], "medians": [m_eth]}

    # ---------------------------------------------------------------------
    # 8. M, Q, µ, ¶, ¤, §, ª, ®, Æ, æ, Œ, œ, ĥ, ŉ, Ŋ
    # ---------------------------------------------------------------------
    cnts_M = load_svg_glyph(open(os.path.join(FONT_DIR, "M.svg")).read())
    m_M = [[142, 81], [142, 780], [200, 780], [462, 110], [728, 780], [784, 780], [784, 81]]
    corrections["M"] = {"strokes": [segs_to_path(cnts_M[0])], "medians": [m_M]}

    cnts_Q = load_svg_glyph(open(os.path.join(FONT_DIR, "Q.svg")).read())
    b_close_oval = Seg('L', np.array([[425.984, 71.584], [524.288, 88.992]]))
    oval_outer_Q = cnts_Q[0][5:] + cnts_Q[0][:2] + [b_close_oval]
    s_Q_oval = segs_to_path(oval_outer_Q) + " " + segs_to_path(cnts_Q[1])
    s_Q_tail = f"M {524.288} {88.992} L {699.392} {-93.28} L {567.296} {-93.28} L {425.984} {71.584} Z"
    thetas_Q = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi - 0.25, 24)
    m_Q_oval = [[int(round(400 + 288 * np.cos(th))), int(round(447 + 335 * np.sin(th)))] for th in thetas_Q]
    m_Q_tail = [[475, 80], [633, -93]]
    corrections["Q"] = {"strokes": [s_Q_oval, s_Q_tail], "medians": [m_Q_oval, m_Q_tail]}

    cnts_mu = load_svg_glyph(open(os.path.join(FONT_DIR, "µ.svg")).read())
    m_mu = [[132, -155], [132, 620], [132, 275], [180, 145], [300, 108], [425, 145], [505, 300], [505, 620], [505, 300], [514, 81]]
    corrections["µ"] = {"strokes": [segs_to_path(cnts_mu[0])], "medians": [m_mu]}

    # ¶ (pilcrow)
    d_pi = json.load(open(os.path.join(DATA_DIR, "¶.json")))
    m_pi0 = [[532, 859], [532, -49]]
    m_pi1 = [
        [280, 840], [150, 810], [75, 602], [150, 390], [280, 360],
        [200, 450], [200, 720], [270, 720], [270, 450],
        [340, 400], [368, 800], [500, 840], [368, 840], [368, -49]
    ]
    corrections["¶"] = {"strokes": d_pi["strokes"], "medians": [m_pi0, m_pi1]}

    # ¤ (currency)
    cnts_curr = load_svg_glyph(open(os.path.join(FONT_DIR, "¤.svg")).read())
    s_curr = segs_to_path(cnts_curr[0]) + " " + segs_to_path(cnts_curr[1])
    m_curr = [
        [86, 649], [175, 560], [230, 595], [293, 608], [356, 595], [410, 560], [499, 649],
        [410, 560], [445, 505], [459, 442], [445, 379], [410, 325], [498, 236],
        [410, 325], [356, 290], [293, 276], [230, 290], [175, 325], [86, 237],
        [175, 325], [140, 379], [127, 442], [140, 505], [175, 560]
    ]
    corrections["¤"] = {"strokes": [s_curr], "medians": [m_curr]}

    # § (section)
    cnts_sec = load_svg_glyph(open(os.path.join(FONT_DIR, "§.svg")).read())
    s_sec = segs_to_path(cnts_sec[0]) + " " + segs_to_path(cnts_sec[1])
    m_sec = [
        [435, 795], [370, 825], [270, 830], [150, 805], [115, 733], [170, 685], [290, 637],
        [413, 571], [445, 473], [410, 390], [350, 340],
        [410, 290], [440, 224], [370, 130], [240, 110], [140, 115], [60, 148],
        [140, 158], [240, 144], [355, 190], [355, 250], [290, 300], [190, 340], [100, 410],
        [80, 485], [110, 570], [180, 614], [270, 560], [340, 480], [314, 380], [240, 420], [160, 495]
    ]
    corrections["§"] = {"strokes": [s_sec], "medians": [m_sec]}

    # ª (feminine ordinal)
    cnts_ord = load_svg_glyph(open(os.path.join(FONT_DIR, "ª.svg")).read())
    s_ord = segs_to_path(cnts_ord[0]) + " " + segs_to_path(cnts_ord[1])
    m_ord = [
        [270, 780], [200, 810], [100, 780], [60, 720], [100, 660], [270, 660],
        [285, 520], [200, 480], [80, 520], [60, 580], [120, 640], [270, 640], [285, 480]
    ]
    corrections["ª"] = {"strokes": [s_ord], "medians": [m_ord]}

    # ® (registered trademark)
    cnts_reg = load_svg_glyph(open(os.path.join(FONT_DIR, "®.svg")).read())
    s_reg_ring = segs_to_path(cnts_reg[0]) + " " + segs_to_path(cnts_reg[1])
    s_reg_R = segs_to_path(cnts_reg[2]) + " " + segs_to_path(cnts_reg[3])
    m_reg_ring = circle_median(426.0, 446.4, radius=345.0, n_points=24)
    m_reg_R = [
        [320, 670], [320, 222], [320, 450], [320, 640],
        [410, 660], [530, 630], [570, 560], [530, 480], [410, 460], [320, 460],
        [400, 440], [500, 320], [570, 222]
    ]
    corrections["®"] = {"strokes": [s_reg_ring, s_reg_R], "medians": [m_reg_ring, m_reg_R]}

    # Æ (capital AE ligature)
    d_AE = json.load(open(os.path.join(DATA_DIR, "Æ.json")))
    d_AE["medians"][3] = [[184, 348], [488, 348], [551, 462], [821, 462]]
    corrections["Æ"] = d_AE

    # æ (lowercase ae ligature)
    cnts_ae = load_svg_glyph(open(os.path.join(FONT_DIR, "æ.svg")).read())
    s_ae = segs_to_path(cnts_ae[0]) + " " + segs_to_path(cnts_ae[1]) + " " + segs_to_path(cnts_ae[2])
    m_ae = [
        [380, 620], [250, 630], [130, 580], [70, 480], [130, 410], [380, 410],
        [380, 110], [250, 105], [130, 140], [70, 240], [130, 340], [270, 340], [380, 300],
        [410, 520], [450, 570], [480, 450],
        [480, 360], [740, 360], [775, 490], [670, 600], [530, 560], [480, 440],
        [480, 250], [550, 125], [680, 105], [780, 140], [825, 230], [825, 340]
    ]
    corrections["æ"] = {"strokes": [s_ae], "medians": [m_ae]}

    # Œ (capital OE ligature)
    cnts_OE = load_svg_glyph(open(os.path.join(FONT_DIR, "Œ.svg")).read())
    b_top_stem = Seg('L', np.array([[493.568, 811.936], [594.944, 811.936]]))
    b_stem_down = Seg('L', np.array([[594.944, 811.936], [594.944, 80.8]]))
    b_bot_stem = Seg('L', np.array([[594.944, 80.8], [499.712, 80.8]]))
    s_O_outer = segs_to_path([b_top_stem, b_stem_down, b_bot_stem] + cnts_OE[0][12:] + [cnts_OE[0][0]])
    s_O = s_O_outer + " " + segs_to_path(cnts_OE[1])
    s_E_top = f"M 594.944 811.936 L 888.832 811.936 L 888.832 731.04 L 594.944 731.04 Z"
    s_E_mid = f"M 594.944 502.688 L 869.376 502.688 L 869.376 422.816 L 594.944 422.816 Z"
    s_E_bot = f"M 594.944 161.696 L 888.832 161.696 L 888.832 80.8 L 594.944 80.8 Z"
    m_O = [
        [320, 780], [180, 720], [100, 580], [75, 447], [100, 300], [180, 160], [320, 100],
        [548, 120], [548, 447], [548, 772], [320, 780]
    ]
    corrections["Œ"] = {
        "strokes": [s_O, s_E_top, s_E_mid, s_E_bot],
        "medians": [m_O, [[548, 771], [888, 771]], [[594, 463], [869, 463]], [[548, 121], [888, 121]]]
    }

    # œ (lowercase oe ligature)
    cnts_oe = load_svg_glyph(open(os.path.join(FONT_DIR, "œ.svg")).read())
    s_oe = segs_to_path(cnts_oe[0]) + " " + segs_to_path(cnts_oe[1]) + " " + segs_to_path(cnts_oe[2])
    m_oe = [
        [450, 580], [300, 620], [130, 540], [80, 360], [130, 180], [300, 85], [480, 160],
        [500, 350], [450, 480], [500, 550],
        [520, 380], [800, 380], [890, 460], [870, 560], [710, 620], [570, 570], [520, 440],
        [520, 250], [600, 120], [730, 85], [830, 120], [900, 210], [915, 339]
    ]
    corrections["œ"] = {"strokes": [s_oe], "medians": [m_oe]}

    # ĥ (h with circumflex)
    cnts_h_circ = load_svg_glyph(open(os.path.join(FONT_DIR, "ĥ.svg")).read())
    corrections["ĥ"] = {
        "strokes": [segs_to_path(cnts_h_circ[0]), segs_to_path(cnts_h_circ[1])],
        "medians": [[[10, 950], [133, 1070], [256, 950]], [[135, 859], [135, 81], [135, 420], [260, 615], [390, 620], [500, 500], [504, 81]]]
    }

    # ŉ ('n)
    cnts_apo_n = load_svg_glyph(open(os.path.join(FONT_DIR, "ŉ.svg")).read())
    corrections["ŉ"] = {
        "strokes": [segs_to_path(cnts_apo_n[0]), segs_to_path(cnts_apo_n[1])],
        "medians": [[[105, 800], [55, 600]], [[205, 620], [205, 81], [205, 420], [330, 615], [460, 620], [572, 500], [572, 81]]]
    }

    # Ŋ (Eng)
    cnts_Eng = load_svg_glyph(open(os.path.join(FONT_DIR, "Ŋ.svg")).read())
    corrections["Ŋ"] = {
        "strokes": [segs_to_path(cnts_Eng[0])],
        "medians": [[[142, 81], [142, 812], [636, 120], [636, 812], [636, 50], [580, -90], [420, -90]]]
    }

    # ---------------------------------------------------------------------
    # 9. Extended Latin: Ɓ, Ƃ, Ƅ, ƅ, Ɗ, ƌ, ƍ, Ɣ, ƛ, Ɯ, ƞ, ƥ, Ƴ
    # ---------------------------------------------------------------------
    cnts_B_hook = load_svg_glyph(open(os.path.join(FONT_DIR, "Ɓ.svg")).read())
    corrections["Ɓ"] = {
        "strokes": [" ".join(segs_to_path(c) for c in cnts_B_hook)],
        "medians": [[[60, 626], [15, 680], [50, 780], [154, 812], [226, 812], [226, 81], [470, 120], [643, 200], [643, 380], [470, 460], [226, 460], [450, 460], [620, 560], [620, 680], [450, 772], [226, 772]]]
    }

    cnts_B_top = load_svg_glyph(open(os.path.join(FONT_DIR, "Ƃ.svg")).read())
    corrections["Ƃ"] = {
        "strokes": [segs_to_path(cnts_B_top[0]) + " " + segs_to_path(cnts_B_top[1])],
        "medians": [[[533, 771], [145, 771], [145, 812], [145, 81], [191, 100], [384, 100], [520, 160], [560, 292], [520, 420], [384, 480], [191, 480]]]
    }

    cnts_tone6 = load_svg_glyph(open(os.path.join(FONT_DIR, "Ƅ.svg")).read())
    corrections["Ƅ"] = {
        "strokes": [segs_to_path(cnts_tone6[0]) + " " + segs_to_path(cnts_tone6[1])],
        "medians": [[[138, 812], [138, 250], [138, 110], [250, 85], [380, 85], [520, 150], [565, 290], [520, 430], [380, 485], [250, 485], [184, 485]]]
    }

    cnts_tone6_low = load_svg_glyph(open(os.path.join(FONT_DIR, "ƅ.svg")).read())
    corrections["ƅ"] = {
        "strokes": [segs_to_path(cnts_tone6_low[0]) + " " + segs_to_path(cnts_tone6_low[1])],
        "medians": [[[129, 859], [129, 450], [129, 200], [129, 90], [240, 108], [360, 108], [480, 180], [524, 355], [480, 530], [360, 601], [240, 601], [174, 550]]]
    }

    cnts_D_hook = load_svg_glyph(open(os.path.join(FONT_DIR, "Ɗ.svg")).read())
    corrections["Ɗ"] = {
        "strokes": [segs_to_path(cnts_D_hook[0]) + " " + segs_to_path(cnts_D_hook[1])],
        "medians": [[[60, 626], [15, 680], [50, 780], [154, 812], [226, 812], [226, 81], [385, 120], [550, 160], [670, 280], [718, 453], [670, 620], [550, 740], [385, 772], [226, 772]]]
    }

    cnts_d_hook = load_svg_glyph(open(os.path.join(FONT_DIR, "ƌ.svg")).read())
    corrections["ƌ"] = {
        "strokes": [segs_to_path(cnts_d_hook[0]) + " " + segs_to_path(cnts_d_hook[1])],
        "medians": [[[111, 821], [498, 821], [498, 81], [380, 108], [280, 108], [118, 180], [75, 354], [118, 530], [282, 602], [386, 602], [452, 559]]]
    }

    cnts_delta = load_svg_glyph(open(os.path.join(FONT_DIR, "ƍ.svg")).read())
    corrections["ƍ"] = {
        "strokes": [segs_to_path(cnts_delta[0]) + " " + segs_to_path(cnts_delta[1])],
        "medians": [[[320, 600], [180, 560], [100, 390], [180, 200], [320, 160], [480, 240], [535, 400], [450, 560], [320, 600], [400, 200], [450, 66], [450, -30], [350, -115], [250, -115], [115, -60]]]
    }

    cnts_gamma = load_svg_glyph(open(os.path.join(FONT_DIR, "Ɣ.svg")).read())
    corrections["Ɣ"] = {
        "strokes": [segs_to_path(cnts_gamma[0]) + " " + segs_to_path(cnts_gamma[1])],
        "medians": [[[48, 812], [292, 270], [535, 812], [292, 270], [380, 50], [370, -110], [292, -150], [210, -110], [200, 50], [292, 270]]]
    }

    d_lambda = json.load(open(os.path.join(DATA_DIR, "ƛ.json")))
    corrections["ƛ"] = {
        "strokes": d_lambda["strokes"],
        "medians": [
            [[88, 670], [206, 710], [289, 740], [384, 775]],
            [[60, 810], [110, 860], [200, 840], [260, 760], [300, 680], [420, 300], [470, 150], [520, 110], [420, 200], [300, 450], [220, 619], [180, 500], [130, 350], [70, 200], [15, 81]]
        ]
    }

    cnts_W = load_svg_glyph(open(os.path.join(FONT_DIR, "Ɯ.svg")).read())
    corrections["Ɯ"] = {
        "strokes": [segs_to_path(cnts_W[0])],
        "medians": [[[138, 812], [138, 290], [200, 110], [330, 90], [450, 150], [523, 300], [523, 812], [523, 300], [590, 110], [720, 90], [840, 150], [908, 300], [908, 812], [908, 81]]]
    }

    cnts_n_long = load_svg_glyph(open(os.path.join(FONT_DIR, "ƞ.svg")).read())
    corrections["ƞ"] = {
        "strokes": [segs_to_path(cnts_n_long[0])],
        "medians": [[[132, 620], [132, 81], [132, 450], [250, 620], [380, 620], [493, 500], [493, -165]]]
    }

    cnts_p_hook = load_svg_glyph(open(os.path.join(FONT_DIR, "ƥ.svg")).read())
    corrections["ƥ"] = {
        "strokes": [segs_to_path(cnts_p_hook[0]) + " " + segs_to_path(cnts_p_hook[1])],
        "medians": [[[250, 780], [250, 850], [180, 864], [132, 720], [132, -165], [132, 100], [250, 80], [400, 80], [530, 200], [550, 360], [510, 520], [350, 620], [220, 600], [177, 500]]]
    }

    cnts_Y_hook = load_svg_glyph(open(os.path.join(FONT_DIR, "Ƴ.svg")).read())
    b_close_Y = Seg('L', np.array([[243.712, 360.352], [289.792, 452.512]]))
    s_Y_right = segs_to_path(cnts_Y_hook[0][4:15] + [cnts_Y_hook[0][0], b_close_Y])
    s_Y_left = "M 0 811.936 L 100.352 811.936 L 289.792 452.512 L 243.712 360.352 Z"
    corrections["Ƴ"] = {
        "strokes": [s_Y_left, s_Y_right],
        "medians": [[[50, 812], [289, 410]], [[584, 760], [523, 820], [450, 750], [289, 410], [289, 81]]]
    }

    # ---------------------------------------------------------------------
    # 10. D variants and DZ ligatures: Ď, ď, Ǆ, ǅ, ǆ
    # ---------------------------------------------------------------------
    m_D_cap = [
        [145, 812], [145, 81],
        [191, 121], [380, 121], [530, 160], [620, 280], [640, 450], [620, 620], [530, 740], [380, 772], [191, 772]
    ]

    # Ď
    cnts_Ď = load_svg_glyph(open(os.path.join(FONT_DIR, "Ď.svg")).read())
    corrections["Ď"] = {
        "strokes": [segs_to_path(cnts_Ď[0]), segs_to_path(cnts_Ď[1]) + " " + segs_to_path(cnts_Ď[2])],
        "medians": [[[220, 1037], [370, 910], [520, 1037]], m_D_cap]
    }

    # Ǆ
    cnts_DZ = load_svg_glyph(open(os.path.join(FONT_DIR, "Ǆ.svg")).read())
    corrections["Ǆ"] = {
        "strokes": [segs_to_path(cnts_DZ[0]), segs_to_path(cnts_DZ[2]) + " " + segs_to_path(cnts_DZ[3]), segs_to_path(cnts_DZ[1])],
        "medians": [[[890, 1037], [1034, 910], [1178, 1037]], m_D_cap, [[800, 772], [1250, 772], [800, 120], [1260, 120]]]
    }

    # ǅ
    cnts_Dz = load_svg_glyph(open(os.path.join(FONT_DIR, "ǅ.svg")).read())
    corrections["ǅ"] = {
        "strokes": [segs_to_path(cnts_Dz[0]), segs_to_path(cnts_Dz[2]) + " " + segs_to_path(cnts_Dz[3]), segs_to_path(cnts_Dz[1])],
        "medians": [[[840, 855], [988, 715], [1136, 855]], m_D_cap, [[810, 600], [1160, 600], [810, 110], [1170, 110]]]
    }

    # ǆ
    cnts_dz = load_svg_glyph(open(os.path.join(FONT_DIR, "ǆ.svg")).read())
    m_d_low = [
        [498, 859], [498, 81],
        [380, 108], [280, 108], [118, 180], [75, 354], [118, 530], [282, 602], [386, 602], [452, 559]
    ]
    corrections["ǆ"] = {
        "strokes": [segs_to_path(cnts_dz[0]), segs_to_path(cnts_dz[2]) + " " + segs_to_path(cnts_dz[3]), segs_to_path(cnts_dz[1])],
        "medians": [[[720, 855], [870, 715], [1020, 855]], m_d_low, [[690, 600], [1040, 600], [690, 110], [1050, 110]]]
    }

    # ď
    if os.path.exists(os.path.join(FONT_DIR, "ď.svg")):
        cnts_d_caron = load_svg_glyph(open(os.path.join(FONT_DIR, "ď.svg")).read())
        m_caron_d = [[660, 855], [645, 800], [628, 750], [606, 695]]
        corrections["ď"] = {
            "strokes": [segs_to_path(cnts_d_caron[0]), segs_to_path(cnts_d_caron[1]) + " " + segs_to_path(cnts_d_caron[2])],
            "medians": [m_caron_d, m_d_low]
        }

    return corrections

def main():
    print("Building definitive corrections for batch 100...")
    corrections = build_corrections()
    print(f"Generated {len(corrections)} explicit corrections.")

    with open("/tmp/queue100.json", encoding="utf-8") as f:
        queue = json.load(f)

    manifest = json.load(open(MANIFEST_PATH, encoding="utf-8"))

    results = []
    low_iou_count = 0
    diacritic_order_errors = []
    circle_errors = []

    # Characters that must have top diacritic first
    TOP_DIACRITIC_CHARS = set([
        "È", "É", "Ê", "Ë", "Ö", "Ē", "Ĕ", "Ė", "Ę", "Ě",
        "Ý", "ý", "ÿ", "ŷ", "ù", "ú", "û", "ü", "ũ", "ū", "ŭ", "ů", "ű", "ư", "Ů",
        "Ď", "ď", "ĝ", "ğ", "ģ", "ĥ", "Ń", "Ň", "ñ", "Ñ", "ŉ",
        "Ŕ", "ŕ", "Ř", "ř", "Ŷ", "ŷ", "Ÿ", "Ǆ", "ǅ", "ǆ"
    ])

    for i, item in enumerate(queue):
        c = item["char"]
        font_path = os.path.join(FONT_DIR, f"{c}.svg")
        data_path = os.path.join(DATA_DIR, f"{c}.json")

        if c in GOOD_CHARS:
            # Load existing file - DO NOT ALTER
            data = json.load(open(data_path, encoding="utf-8"))
        elif c in corrections:
            data = corrections[c]
        else:
            # Retain existing if present
            data = json.load(open(data_path, encoding="utf-8"))

        cnts = load_svg_glyph(open(font_path, encoding="utf-8").read())
        raw_iou, cl_iou, uncov = eval_char(cnts, data["strokes"], data["medians"])

        if cl_iou < 0.98 or raw_iou < 0.98:
            low_iou_count += 1
            print(f"FAILED {c:2s}: raw={raw_iou:.4f}, clipped={cl_iou:.4f}, uncov={uncov:4d} px")
        else:
            print(f"PASS   {c:2s}: raw={raw_iou:.4f}, clipped={cl_iou:.4f}, uncov={uncov:4d} px")

        # Check circle diacritics
        if c in ["Ů", "ů", "°"]:
            m_ring = data["medians"][0]
            if len(m_ring) != 24:
                circle_errors.append((c, f"len={len(m_ring)}"))

        # Save to disk
        if c not in GOOD_CHARS:
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

        # Ensure manifest entry
        if c not in manifest["characters"]:
            manifest["characters"].append(c)

    # Save manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY:")
    print(f"Total evaluated: {len(queue)}")
    print(f"Low IoU (<0.98): {low_iou_count}")
    print(f"Circle errors: {len(circle_errors)} {circle_errors}")
    print("=" * 60)

if __name__ == "__main__":
    main()
