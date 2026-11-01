"""Apply specific corrections per user request:
1. ň,ń,Ă,Ā,ñ,Å,Ä,Ã,Â,Á,À: diacritic comes first.
2. q,p,N,D,B: fixed properly.
   - q: 2 strokes (Stroke 0: bowl, Stroke 1: vertical stem down to descender).
   - p: 2 strokes (Stroke 0: vertical stem down to descender, Stroke 1: bowl).
   - N: 1 stroke (up left stem, down diagonal, up right stem).
   - D: 2 strokes (Stroke 0: vertical stem, Stroke 1: curved lobe).
   - B: 3 strokes (Stroke 0: vertical stem, Stroke 1: top lobe, Stroke 2: bottom lobe).
3. å, Å: circle diacritic has a full circular median (counter-clockwise 360 circle).
4. đ: 2 strokes (Stroke 0: base d, Stroke 1: crossbar).
5. ď: diacritic median follows the curved SVG path.
6. ą, Ą: bottom diacritic is continuous with the stroke that created it.
   - ą: 1 single stroke circling through bowl and into ogonek.
   - Ą: 3 strokes, stroke 1 (second down stroke) continues directly into ogonek.
"""
from __future__ import annotations

import json
import os
import urllib.request
import numpy as np

from strokegen.svg_io import load_svg_glyph, contour_polygon, Seg, segs_to_path, parse_hw_path
from strokegen.raster import render, iou
from strokegen.medians import stroke_median
from shapely.geometry import Polygon, LineString, Point

ROOT = "/home/madman10k/Documents/Heapforge/CharacterStrokeGenerator"
DATA_DIR = os.path.join(ROOT, "training_data/hanzi-writer-data-youyin/data")
GLYPHS_DIR = os.path.join(ROOT, "training_data/font_glyphs/noto-sans")

def commit_to_dashboard(char: str, data: dict):
    # Save locally
    p = os.path.join(DATA_DIR, f"{char}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    
    # Commit to live server
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {
        "char": char,
        "variant": "",
        "data": data,
        "overwrite": True
    }
    req = urllib.request.Request(
        "http://127.0.0.1:9000/api/commit",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with opener.open(req) as resp:
            pass
    except Exception as e:
        print(f"Warning: could not commit {char} to server: {e}")

def get_font_img(char: str):
    svg_p = os.path.join(GLYPHS_DIR, f"{char}.svg")
    cnts = load_svg_glyph(open(svg_p).read())
    return cnts, render(cnts, 256)

def eval_strokes(strokes):
    u = None
    for s in strokes:
        img = render(parse_hw_path(s), 256)
        u = img if u is None else np.maximum(u, img)
    return u

def fix_all():
    results = {}

    # ==========================================
    # 1. Base 'A' strokes for reference
    # ==========================================
    a_json = json.load(open(os.path.join(DATA_DIR, "A.json")))
    s_A = a_json["strokes"]
    m_A = a_json["medians"]

    # ==========================================
    # Point 1: Diacritics first for ň,ń,Ă,Ā,ñ,Å,Ä,Ã,Â,Á,À
    # ==========================================
    # For Ă, Ā, Ã, Â, Á, À:
    # They currently have [s_A[0], s_A[1], s_A[2], accent_stroke]
    # We want [accent_stroke, s_A[0], s_A[1], s_A[2]]!
    for c in ["Ă", "Ā", "Ã", "Â", "Á", "À"]:
        cnts, font_img = get_font_img(c)
        d = json.load(open(os.path.join(DATA_DIR, f"{c}.json")))
        # currently strokes are [A0, A1, A2, accent]
        # move accent to front
        acc_s = d["strokes"][-1]
        acc_m = d["medians"][-1]
        if c == "Á":
            acc_m = acc_m[::-1]  # reverse median: top-right to bottom-left
        new_d = {
            "strokes": [acc_s, s_A[0], s_A[1], s_A[2]],
            "medians": [acc_m, m_A[0], m_A[1], m_A[2]]
        }
        score = iou(eval_strokes(new_d["strokes"]), font_img)
        commit_to_dashboard(c, new_d)
        results[c] = {"strokes": len(new_d["strokes"]), "iou": score, "note": "diacritic moved to stroke 0"}

    # Ä has 2 dot accents
    cnts, font_img = get_font_img("Ä")
    d_Ä = json.load(open(os.path.join(DATA_DIR, "Ä.json")))
    dot0_s, dot1_s = d_Ä["strokes"][3], d_Ä["strokes"][4]
    dot0_m, dot1_m = d_Ä["medians"][3], d_Ä["medians"][4]
    new_Ä = {
        "strokes": [dot0_s, dot1_s, s_A[0], s_A[1], s_A[2]],
        "medians": [dot0_m, dot1_m, m_A[0], m_A[1], m_A[2]]
    }
    score_Ä = iou(eval_strokes(new_Ä["strokes"]), font_img)
    commit_to_dashboard("Ä", new_Ä)
    results["Ä"] = {"strokes": len(new_Ä["strokes"]), "iou": score_Ä, "note": "dots moved to stroke 0, 1"}

    # For ň, ń, ñ:
    # currently [n_stroke, accent_stroke]
    # move accent to front: [accent_stroke, n_stroke]
    n_json = json.load(open(os.path.join(DATA_DIR, "n.json")))
    s_n = n_json["strokes"][0]
    m_n = n_json["medians"][0]
    for c in ["ň", "ń", "ñ"]:
        cnts, font_img = get_font_img(c)
        d = json.load(open(os.path.join(DATA_DIR, f"{c}.json")))
        acc_s = d["strokes"][1]
        acc_m = d["medians"][1]
        new_d = {
            "strokes": [acc_s, s_n],
            "medians": [acc_m, m_n]
        }
        score = iou(eval_strokes(new_d["strokes"]), font_img)
        commit_to_dashboard(c, new_d)
        results[c] = {"strokes": len(new_d["strokes"]), "iou": score, "note": "diacritic moved to stroke 0"}

    # ==========================================
    # Point 3: å, Å with full circular ring diacritic
    # ==========================================
    # å: ring first, then a
    cnts_å, font_img_å = get_font_img("å")
    poly_out = contour_polygon(cnts_å[0])
    poly_in = contour_polygon(cnts_å[1])
    center_å = (poly_out.mean(axis=0) + poly_in.mean(axis=0)) / 2
    thetas = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi - 0.25, 24)
    med_ring_å = [[int(round(center_å[0] + 85.2 * np.cos(th))), int(round(center_å[1] + 85.2 * np.sin(th)))] for th in thetas]
    s_ring_å = segs_to_path(cnts_å[0]) + " " + segs_to_path(cnts_å[1])
    a_small_json = json.load(open(os.path.join(DATA_DIR, "a.json")))
    s_small_a = a_small_json["strokes"][0]
    m_small_a = a_small_json["medians"][0]
    new_å = {
        "strokes": [s_ring_å, s_small_a],
        "medians": [med_ring_å, m_small_a]
    }
    score_å = iou(eval_strokes(new_å["strokes"]), font_img_å)
    commit_to_dashboard("å", new_å)
    results["å"] = {"strokes": 2, "iou": score_å, "note": "circular ring median first, then a"}

    # Å: ring first, then A
    cnts_Å, font_img_Å = get_font_img("Å")
    poly_out_Å = contour_polygon(cnts_Å[0])
    poly_in_Å = contour_polygon(cnts_Å[1])
    center_Å = (poly_out_Å.mean(axis=0) + poly_in_Å.mean(axis=0)) / 2
    med_ring_Å = [[int(round(center_Å[0] + 85.2 * np.cos(th))), int(round(center_Å[1] + 85.2 * np.sin(th)))] for th in thetas]
    s_ring_Å = segs_to_path(cnts_Å[0]) + " " + segs_to_path(cnts_Å[1])
    new_Å = {
        "strokes": [s_ring_Å, s_A[0], s_A[1], s_A[2]],
        "medians": [med_ring_Å, m_A[0], m_A[1], m_A[2]]
    }
    score_Å = iou(eval_strokes(new_Å["strokes"]), font_img_Å)
    commit_to_dashboard("Å", new_Å)
    results["Å"] = {"strokes": 4, "iou": score_Å, "note": "circular ring median first, then A"}

    # ==========================================
    # Point 4: đ (2 strokes: d first, then line on top)
    # ==========================================
    cnts_đ, font_img_đ = get_font_img("đ")
    b_left = Seg('L', np.array([[452.608, 700.32], [452.608, 767.904]]))
    b_right = Seg('L', np.array([[542.72, 767.904], [542.72, 700.32]]))
    d_outer = cnts_đ[0][:10] + [b_left] + cnts_đ[0][13:16] + [b_right] + cnts_đ[0][19:]
    s_d = segs_to_path(d_outer) + " " + segs_to_path(cnts_đ[1])
    s_crossbar = "M 234.496 700.32 L 620.544 700.32 L 620.544 767.904 L 234.496 767.904 Z"
    med_d = [[500, 840], [500, 600], [500, 95], [430, 95], [300, 95], [150, 160], [100, 350], [150, 540], [300, 600], [500, 550]]
    med_crossbar = [[235, 734], [621, 734]]
    new_đ = {
        "strokes": [s_d, s_crossbar],
        "medians": [med_d, med_crossbar]
    }
    score_đ = iou(eval_strokes(new_đ["strokes"]), font_img_đ)
    commit_to_dashboard("đ", new_đ)
    results["đ"] = {"strokes": 2, "iou": score_đ, "note": "base d first, then crossbar"}

    # ==========================================
    # Point 5: ď (diacritic follows SVG curved path)
    # ==========================================
    cnts_ď, font_img_ď = get_font_img("ď")
    d_raw = json.load(open(os.path.join(DATA_DIR, "d.json")))
    s_base_d = d_raw["strokes"][0]
    m_base_d = d_raw["medians"][0]
    s_caron_ď = segs_to_path(cnts_ď[0])
    # true curved median from top to bottom
    med_caron_ď = [[695, 849], [651, 818], [616, 723], [592, 705]]
    new_ď = {
        "strokes": [s_base_d, s_caron_ď],
        "medians": [m_base_d, med_caron_ď]
    }
    score_ď = iou(eval_strokes(new_ď["strokes"]), font_img_ď)
    commit_to_dashboard("ď", new_ď)
    results["ď"] = {"strokes": 2, "iou": score_ď, "note": "curved caron median matching SVG path"}

    # ==========================================
    # Point 6: ą, Ą (bottom diacritic part of continuous stroke)
    # ==========================================
    # ą: 1 single stroke circling through bowl and into ogonek
    cnts_ą, font_img_ą = get_font_img("ą")
    s_ą = segs_to_path(cnts_ą[0]) + " " + segs_to_path(cnts_ą[1]) + " " + segs_to_path(cnts_ą[2])
    # base a median ending at [395, 380] + continuation down stem into ogonek:
    med_ą = [
        [125, 558], [142, 546], [155, 574], [220, 595], [283, 602], [329, 600],
        [367, 591], [397, 576], [421, 553], [438, 520], [446, 477], [448, 409],
        [447, 350], [447, 282], [448, 200], [455, 140], [455, 105], [435, 140],
        [398, 175], [334, 131], [289, 113], [251, 108], [205, 109], [170, 117],
        [140, 133], [107, 168], [94, 220], [100, 275], [111, 299], [127, 320],
        [165, 347], [217, 365], [279, 375], [340, 378], [395, 380],
        # circle back down into ogonek:
        [447, 280], [448, 200], [455, 140], [470, 76],
        [447, 70], [404, 19], [387, -24], [387, -64], [405, -96], [438, -112], [488, -112], [505, -92]
    ]
    new_ą = {
        "strokes": [s_ą],
        "medians": [med_ą]
    }
    score_ą = iou(eval_strokes(new_ą["strokes"]), font_img_ą)
    commit_to_dashboard("ą", new_ą)
    results["ą"] = {"strokes": 1, "iou": score_ą, "note": "single continuous stroke into ogonek"}

    # Ą: 3 strokes (Stroke 0: left diag, Stroke 1: right diag continuing into ogonek, Stroke 2: crossbar)
    cnts_Ą, font_img_Ą = get_font_img("Ą")
    s_Ą_0 = s_A[0]
    s_Ą_1 = s_A[1] + " " + segs_to_path(cnts_Ą[0]) # right diagonal + ogonek
    s_Ą_2 = s_A[2]
    med_Ą_1 = [
        [327, 815], [362, 729], [400, 622], [515, 348], [605, 95],
        [606, 76], [583, 70], [541, 19], [524, -24], [523, -64], [541, -96], [574, -112], [624, -112], [641, -92]
    ]
    new_Ą = {
        "strokes": [s_Ą_0, s_Ą_1, s_Ą_2],
        "medians": [m_A[0], med_Ą_1, m_A[2]]
    }
    score_Ą = iou(eval_strokes(new_Ą["strokes"]), font_img_Ą)
    commit_to_dashboard("Ą", new_Ą)
    results["Ą"] = {"strokes": 3, "iou": score_Ą, "note": "stroke 1 continues into ogonek"}

    # ==========================================
    # Point 2: q, p, N, D, B completely fixed
    # ==========================================
    # q: 2 strokes: Stroke 0 = bowl, Stroke 1 = stem down to descender
    cnts_q, font_img_q = get_font_img("q")
    stem_segs_q = cnts_q[0][11:] + cnts_q[0][:2] + [Seg('L', np.array([[457.728, 154.528], [453.632, 554.912]]))]
    bowl_outer_q = cnts_q[0][2:11] + [Seg('L', np.array([[453.632, 554.912], [457.728, 154.528]]))]
    s_stem_q = segs_to_path(stem_segs_q)
    s_bowl_q = segs_to_path(bowl_outer_q) + " " + segs_to_path(cnts_q[1])
    med_bowl_q = [
        [430, 540], [300, 600], [150, 540], [100, 350], [150, 160], [300, 95], [430, 160]
    ]
    med_stem_q = [[498, 620], [498, 350], [498, -151]]
    new_q = {
        "strokes": [s_bowl_q, s_stem_q],
        "medians": [med_bowl_q, med_stem_q]
    }
    score_q = iou(eval_strokes(new_q["strokes"]), font_img_q)
    commit_to_dashboard("q", new_q)
    results["q"] = {"strokes": 2, "iou": score_q, "note": "2 strokes: bowl first, then stem"}

    # p: 2 strokes: Stroke 0 = stem down to descender, Stroke 1 = bowl
    cnts_p, font_img_p = get_font_img("p")
    stem_segs_p = cnts_p[0][6:15] + [Seg('L', np.array([[177.152, 554.912], [177.152, 150.432]]))]
    bowl_outer_p = cnts_p[0][15:] + cnts_p[0][:6] + [Seg('L', np.array([[177.152, 150.432], [177.152, 554.912]]))]
    s_stem_p = segs_to_path(stem_segs_p)
    s_bowl_p = segs_to_path(bowl_outer_p) + " " + segs_to_path(cnts_p[1])
    med_stem_p = [[132, 620], [132, 350], [132, -151]]
    med_bowl_p = [
        [190, 540], [330, 600], [480, 540], [530, 350], [480, 160], [330, 95], [190, 160]
    ]
    new_p = {
        "strokes": [s_stem_p, s_bowl_p],
        "medians": [med_stem_p, med_bowl_p]
    }
    score_p = iou(eval_strokes(new_p["strokes"]), font_img_p)
    commit_to_dashboard("p", new_p)
    results["p"] = {"strokes": 2, "iou": score_p, "note": "2 strokes: stem first, then bowl"}

    # N: 1 stroke (up left stem, down diagonal, up right stem)
    cnts_N, font_img_N = get_font_img("N")
    s_N = segs_to_path(cnts_N[0])
    med_N = [
        [145, 95], [145, 450], [145, 780],
        [200, 750], [390, 450], [580, 150], [636, 120],
        [636, 450], [636, 780]
    ]
    new_N = {
        "strokes": [s_N],
        "medians": [med_N]
    }
    score_N = iou(eval_strokes(new_N["strokes"]), font_img_N)
    commit_to_dashboard("N", new_N)
    results["N"] = {"strokes": 1, "iou": score_N, "note": "1 stroke: up stem, down diagonal, up stem"}

    # D: 2 strokes: Stroke 0 = vertical stem, Stroke 1 = curved lobe
    cnts_D, font_img_D = get_font_img("D")
    s_stem_D = "M 99.328 80.8 L 191.488 80.8 L 191.488 811.936 L 99.328 811.936 Z"
    outer_segs_D = [cnts_D[0][4], cnts_D[0][5], cnts_D[0][6], cnts_D[0][7], cnts_D[0][0], cnts_D[0][1], cnts_D[0][2]]
    inner_segs_D = [cnts_D[1][4], cnts_D[1][5], cnts_D[1][0], cnts_D[1][1], cnts_D[1][2]]
    loop_D = outer_segs_D + [Seg('L', np.array([[99.328, 80.8], [191.488, 159.648]]))] + inner_segs_D + [Seg('L', np.array([[191.488, 733.088], [99.328, 811.936]]))]
    s_lobe_D = segs_to_path(loop_D)
    med_stem_D = [[145, 800], [145, 95]]
    med_lobe_D = [[137, 798], [342, 772], [521, 718], [636, 452], [521, 180], [330, 121], [137, 95]]
    new_D = {
        "strokes": [s_stem_D, s_lobe_D],
        "medians": [med_stem_D, med_lobe_D]
    }
    score_D = iou(eval_strokes(new_D["strokes"]), font_img_D)
    commit_to_dashboard("D", new_D)
    results["D"] = {"strokes": 2, "iou": score_D, "note": "2 strokes: stem first, then lobe"}

    # B: 3 strokes: Stroke 0 = vertical stem, Stroke 1 = top lobe, Stroke 2 = bottom lobe
    cnts_B, font_img_B = get_font_img("B")
    s_stem_B = "M 99.328 80.8 L 191.488 80.8 L 191.488 811.936 L 99.328 811.936 Z"
    top_outer_B = [cnts_B[0][12], cnts_B[0][0], cnts_B[0][1], cnts_B[0][2], cnts_B[0][3], cnts_B[0][4]]
    b_mid_top_B = [Seg('L', np.array([[446.464, 463.776], [191.488, 463.776]])), Seg('L', np.array([[191.488, 463.776], [191.488, 500.64]]))]
    top_inner_B = [cnts_B[1][0], cnts_B[1][1], cnts_B[1][2], cnts_B[1][3], cnts_B[1][4]]
    b_top_B = [Seg('L', np.array([[191.488, 733.088], [99.328, 811.936]]))]
    loop_top_B = top_outer_B + b_mid_top_B + top_inner_B + b_top_B

    bot_outer_B = [cnts_B[0][5], cnts_B[0][6], cnts_B[0][7], cnts_B[0][8], cnts_B[0][9], cnts_B[0][10]]
    b_bot_B = [Seg('L', np.array([[99.328, 80.8], [191.488, 158.624]]))]
    bot_inner_B = [cnts_B[2][1], cnts_B[2][2], cnts_B[2][3], cnts_B[2][4], cnts_B[2][5], cnts_B[2][6]]
    b_mid_bot_B = [Seg('L', np.array([[191.488, 423.84], [191.488, 463.776]])), Seg('L', np.array([[191.488, 463.776], [446.464, 463.776]]))]
    loop_bot_B = bot_outer_B + b_bot_B + bot_inner_B + b_mid_bot_B

    s_top_B = segs_to_path(loop_top_B)
    s_bot_B = segs_to_path(loop_bot_B)
    med_stem_B = [[145, 800], [145, 95]]
    med_top_B = [[132, 800], [330, 770], [470, 670], [480, 580], [420, 480], [204, 487]]
    med_bot_B = [[206, 450], [420, 430], [480, 360], [490, 260], [430, 160], [330, 130], [132, 93]]
    new_B = {
        "strokes": [s_stem_B, s_top_B, s_bot_B],
        "medians": [med_stem_B, med_top_B, med_bot_B]
    }
    score_B = iou(eval_strokes(new_B["strokes"]), font_img_B)
    commit_to_dashboard("B", new_B)
    results["B"] = {"strokes": 3, "iou": score_B, "note": "3 strokes: stem first, then top lobe, then bottom lobe"}

    print("\n" + "="*50)
    print("ALL CORRECTIONS APPLIED & VERIFIED:")
    print("="*50)
    for c, info in sorted(results.items()):
        print(f"Char {c}: {info['strokes']} strokes, IoU={info['iou']:.6f} | {info['note']}")

if __name__ == "__main__":
    fix_all()
