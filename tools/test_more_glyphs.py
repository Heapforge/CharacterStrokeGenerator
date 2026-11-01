import sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path
from tools.fix_svgs import check_iou

chars = ["6", "8", "9", "@", "&", "æ", "œ", "Ɓ", "Ƃ", "ƥ", "ð", "ª", "®"]
for c in chars:
    cnts = load_svg_glyph(open(f"training_data/font_glyphs/noto-sans/{c}.svg").read())
    # Full multi-contour path
    full_path = " ".join(segs_to_path(cnt) for cnt in cnts)
    score = check_iou(cnts, [full_path])
    print(f"Char '{c}': {len(cnts)} contours, full hollow path IoU = {score:.6f}")
