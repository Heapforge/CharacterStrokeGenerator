import sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path

cnts = load_svg_glyph(open("training_data/font_glyphs/noto-sans/Q.svg").read())
print(f"Q contours: {len(cnts)}")
for ci, cnt in enumerate(cnts):
    print(f"Contour {ci}: {len(cnt)} segs")
    for si, s in enumerate(cnt):
        print(f"  Seg {si}: {s.kind} {s.p0} -> {s.p1}")
