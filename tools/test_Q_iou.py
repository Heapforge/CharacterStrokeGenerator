import sys
sys.path.insert(0, ".")
import numpy as np
from strokegen.svg_io import load_svg_glyph, segs_to_path, Seg
from tools.fix_svgs import check_iou

cnts = load_svg_glyph(open("training_data/font_glyphs/noto-sans/Q.svg").read())
# Contour 0 has 16 segments.
# Seg 0, 1: oval down to [524.288, 88.992]
# Seg 2, 3, 4: tail
# Seg 5..15: rest of oval up from [425.984, 71.584]

b_close_oval = Seg('L', np.array([[425.984, 71.584], [524.288, 88.992]]))
b_close_tail = Seg('L', np.array([[425.984, 71.584], [524.288, 88.992]])) # from 425 to 524

oval_outer = cnts[0][5:] + cnts[0][:2] + [b_close_oval]
s_oval = segs_to_path(oval_outer) + " " + segs_to_path(cnts[1])
s_tail = f"M {524.288} {88.992} L {699.392} {-93.28} L {567.296} {-93.28} L {425.984} {71.584} Z"

score = check_iou(cnts, [s_oval, s_tail])
print(f"Q test IoU: {score:.6f}")
