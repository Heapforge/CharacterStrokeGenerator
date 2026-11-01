import json, sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path, HW_BASE
from strokegen.raster import render, iou
from tools.fix_svgs import check_iou

chars = ["Q", "Ö", "6", "8", "9", "4", "R", "E", "æ", "œ", "Ɓ", "Ƃ", "ƥ", "ð"]
for c in chars:
    font_path = f"training_data/font_glyphs/noto-sans/{c}.svg"
    contours = load_svg_glyph(open(font_path, encoding="utf-8").read())
    print(f"Char {c}: {len(contours)} contours")
    for i, cnt in enumerate(contours):
        pts = [s.p0 for s in cnt]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        area = sum(s.p0[0] * s.p1[1] - s.p1[0] * s.p0[1] for s in cnt)
        print(f"  Contour {i}: len={len(cnt)}, bbox=[{min(xs):.1f}, {min(ys):.1f}, {max(xs):.1f}, {max(ys):.1f}], area={area:.1f}")
