import json, os, sys
import numpy as np

sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path, Seg
from tools.fix_svgs import check_iou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")

# 1. Test u median
cnts_u = load_svg_glyph(open(os.path.join(FONT_DIR, "u.svg"), encoding="utf-8").read())
s_u = segs_to_path(cnts_u[0])
med_u = [
    [126, 620], [126, 276], [160, 150], [280, 108], [420, 150],
    [500, 300], [500, 620], [500, 300], [509, 81], [465, 120]
]
iou_u = check_iou(cnts_u, [s_u])
print(f"u: IoU = {iou_u:.4f}, med start={med_u[0]}, end={med_u[-1]}")

# 2. Test Ş single stroke
cnts_Ş = load_svg_glyph(open(os.path.join(FONT_DIR, "Ş.svg"), encoding="utf-8").read())
s_Ş = segs_to_path(cnts_Ş[0]) + " " + segs_to_path(cnts_Ş[1])
med_Ş = [
    [460, 720], [400, 770], [280, 775], [160, 740], [110, 640], [150, 540],
    [250, 460], [380, 390],
    [470, 310], [470, 200], [400, 120], [280, 115], [150, 160],
    [310, 120], [335, 72], [345, 10], [345, -30], [310, -70], [240, -90], [180, -75]
]
iou_Ş = check_iou(cnts_Ş, [s_Ş])
print(f"Ş: IoU = {iou_Ş:.4f}, strokes={1}, med start={med_Ş[0]}, end={med_Ş[-1]}")

# 3. Test ş single stroke
cnts_ş = load_svg_glyph(open(os.path.join(FONT_DIR, "ş.svg"), encoding="utf-8").read())
s_ş = segs_to_path(cnts_ş[0]) + " " + segs_to_path(cnts_ş[1])
med_ş = [
    [416, 596], [360, 630], [250, 635], [150, 610], [110, 530], [140, 450],
    [220, 380], [330, 320],
    [420, 250], [420, 170], [360, 110], [250, 105], [130, 140],
    [280, 110], [305, 72], [315, 10], [315, -30], [280, -70], [220, -90], [165, -75]
]
iou_ş = check_iou(cnts_ş, [s_ş])
print(f"ş: IoU = {iou_ş:.4f}, strokes={1}, med start={med_ş[0]}, end={med_ş[-1]}")

# 4. Test g single stroke
cnts_g = load_svg_glyph(open(os.path.join(FONT_DIR, "g.svg"), encoding="utf-8").read())
s_g = segs_to_path(cnts_g[0]) + " " + segs_to_path(cnts_g[1])
med_g = [
    [470, 560], [300, 600], [150, 500], [150, 300], [300, 180], [470, 250], [470, 560],
    [500, 350], [500, 80],
    [450, -80], [300, -165], [150, -150], [90, -80]
]
iou_g = check_iou(cnts_g, [s_g])
print(f"g: IoU = {iou_g:.4f}, med start={med_g[0]}, end={med_g[-1]}")

# 5. Test y single stroke
cnts_y = load_svg_glyph(open(os.path.join(FONT_DIR, "y.svg"), encoding="utf-8").read())
s_y = segs_to_path(cnts_y[0])
med_y = [
    [50, 620], [200, 280], [264, 170],
    [350, 380], [480, 620],
    [480, 620], [280, 100], [200, -80], [70, -150], [15, -120]
]
iou_y = check_iou(cnts_y, [s_y])
print(f"y: IoU = {iou_y:.4f}, med start={med_y[0]}, end={med_y[-1]}")

# 6. Test r
cnts_r = load_svg_glyph(open(os.path.join(FONT_DIR, "r.svg"), encoding="utf-8").read())
s_r = segs_to_path(cnts_r[0])
med_r = [
    [132, 616], [132, 81], [132, 450], [250, 580], [385, 567]
]
iou_r = check_iou(cnts_r, [s_r])
print(f"r: IoU = {iou_r:.4f}, med start={med_r[0]}, end={med_r[-1]}")
