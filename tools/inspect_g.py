import sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path
from tools.fix_svgs import check_iou

cnts = load_svg_glyph(open("training_data/font_glyphs/noto-sans/g.svg").read())
print(f"g contours: {len(cnts)}")
for ci, cnt in enumerate(cnts):
    pts = [s.p0 for s in cnt]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    print(f"Contour {ci}: {len(cnt)} segs, bbox=[{min(xs):.1f}, {min(ys):.1f}, {max(xs):.1f}, {max(ys):.1f}]")
