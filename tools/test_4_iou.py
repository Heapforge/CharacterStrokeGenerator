import sys
sys.path.insert(0, ".")
import numpy as np
from strokegen.svg_io import load_svg_glyph, segs_to_path, Seg
from tools.fix_svgs import check_iou

cnts = load_svg_glyph(open("training_data/font_glyphs/noto-sans/4.svg").read())

# Stroke 1: Vertical stem
# x from 371.712 to 458.752, y from 80.8 to 816.032
s_stem = "M 371.712 816.032 L 458.752 816.032 L 458.752 80.8 L 371.712 80.8 Z"
med_stem = [[415, 816], [415, 81]]

# Stroke 0: L-stroke (diagonal down, then horizontal crossbar)
# Contour 0 segs:
# Seg 4: [371.712, 246.688] -> [21.504, 246.688]
# Seg 5: [21.504, 246.688] -> [21.504, 323.488]
# Seg 6: [21.504, 323.488] -> [365.568, 816.032]
# Top bridge into inner diagonal: [365.568, 816.032] -> [376.832, 719.776] (or Seg 2 of cnts[1])
# cnts[1] Seg 2..6: inner diagonal down to [109.568, 327.584] and cross to [371.712, 327.584]
# Crossbar continuation: [371.712, 327.584] -> [565.248, 327.584]
# Seg 10: [565.248, 327.584] -> [565.248, 246.688]
# Seg 0: [565.248, 246.688] -> [458.752, 246.688]
# Bridge: [458.752, 246.688] -> [371.712, 246.688]

b_top = Seg('L', np.array([[365.568, 816.032], [376.832, 719.776]]))
b_across_top = Seg('L', np.array([[371.712, 327.584], [458.752, 327.584]]))
b_across_bot = Seg('L', np.array([[458.752, 246.688], [371.712, 246.688]]))

s_L = (
    segs_to_path([cnts[0][4], cnts[0][5], cnts[0][6], b_top] + 
                 cnts[1][2:7] + 
                 [b_across_top, cnts[0][9], cnts[0][10], cnts[0][0], b_across_bot])
)

med_L = [[365, 800], [210, 560], [35, 285], [300, 285], [565, 285]]

score = check_iou(cnts, [s_L, s_stem])
print(f"Digit 4 IoU: {score:.6f}")
