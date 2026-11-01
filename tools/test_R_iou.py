import sys
sys.path.insert(0, ".")
import numpy as np
from strokegen.svg_io import load_svg_glyph, segs_to_path, Seg
from tools.fix_svgs import check_iou

cnts = load_svg_glyph(open("training_data/font_glyphs/noto-sans/R.svg").read())

# Stroke 0: Vertical stem
s_stem = "M 99.328 811.936 L 191.488 811.936 L 191.488 80.8 L 99.328 80.8 Z"

# Stroke 1: Top lobe with hollow counter
# Outer arc: Seg 12 (from 191.488, 811.936), Seg 0, 1, 2, 3, 4 (to 420.864, 408.48)
# Bridge into stem at [191.488, 460.704] and up to [191.488, 811.936]
b_lobe_in = Seg('L', np.array([[420.864, 408.48], [191.488, 460.704]]))
b_lobe_up = Seg('L', np.array([[191.488, 460.704], [191.488, 811.936]]))
b_lobe_top = Seg('L', np.array([[191.488, 811.936], [301.056, 811.936]]))

s_lobe_outer = segs_to_path([b_lobe_top] + cnts[0][:5] + [b_lobe_in, b_lobe_up])
s_lobe = s_lobe_outer + " " + segs_to_path(cnts[1])

# Stroke 2: Diagonal leg
# Seg 5, 6, 7, 8: [336.896, 382.88] -> [191.488, 382.88]
# Bridge up stem: [191.488, 382.88] -> [191.488, 460.704]
# Bridge across to [420.864, 408.48]
b_leg_stem = Seg('L', np.array([[191.488, 382.88], [191.488, 460.704]]))
b_leg_across = Seg('L', np.array([[191.488, 460.704], [420.864, 408.48]]))

s_leg = segs_to_path([cnts[0][5], cnts[0][6], cnts[0][7], cnts[0][8], b_leg_stem, b_leg_across])

score = check_iou(cnts, [s_stem, s_lobe, s_leg])
print(f"R 3-stroke IoU: {score:.6f}")
