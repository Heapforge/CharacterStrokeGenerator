import sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path
from tools.fix_svgs import check_iou

for c in ["ª", "Ǝ"]:
    cnts = load_svg_glyph(open(f"training_data/font_glyphs/noto-sans/{c}.svg").read())
    print(f"Char {c}: {len(cnts)} contours")
    for ci, cnt in enumerate(cnts):
        pts = [s.p0 for s in cnt]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print(f"  Contour {ci}: {len(cnt)} segs, bbox=[{min(xs):.1f}, {min(ys):.1f}, {max(xs):.1f}, {max(ys):.1f}]")
    full_path = " ".join(segs_to_path(cnt) for cnt in cnts)
    print(f"  Full path IoU: {check_iou(cnts, [full_path]):.6f}")
