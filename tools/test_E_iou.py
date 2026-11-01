import sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph
from tools.fix_svgs import check_iou

cnts = load_svg_glyph(open("training_data/font_glyphs/noto-sans/E.svg").read())

# 4 overlapping strokes for E:
# Stem: [99.328, 80.8] to [191.488, 811.904]
s_stem = "M 99.328 811.904 L 191.488 811.904 L 191.488 80.8 L 99.328 80.8 Z"
med_stem = [[145, 812], [145, 81]]

# Top bar:
s_top = "M 99.328 811.904 L 507.904 811.904 L 507.904 731.008 L 99.328 731.008 Z"
med_top = [[145, 771], [508, 771]]

# Middle bar:
s_mid = "M 191.488 502.656 L 489.472 502.656 L 489.472 422.784 L 191.488 422.784 Z"
med_mid = [[191, 463], [489, 463]]

# Bottom bar:
s_bot = "M 99.328 161.664 L 507.904 161.664 L 507.904 80.8 L 99.328 80.8 Z"
med_bot = [[145, 121], [508, 121]]

score = check_iou(cnts, [s_stem, s_top, s_mid, s_bot])
print(f"E 4-stroke IoU: {score:.6f}")
