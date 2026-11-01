import sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path
from tools.fix_svgs import check_iou

for c in ["þ", "Ʀ"]:
    cnts = load_svg_glyph(open(f"training_data/font_glyphs/noto-sans/{c}.svg").read())
    print(f"Char {c}: {len(cnts)} contours")
    for ci, cnt in enumerate(cnts):
        print(f"  Contour {ci}: {len(cnt)} segs")
    full_path = " ".join(segs_to_path(cnt) for cnt in cnts)
    print(f"  Full path IoU: {check_iou(cnts, [full_path]):.6f}")
