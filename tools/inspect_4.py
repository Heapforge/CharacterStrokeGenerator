import sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph

cnts = load_svg_glyph(open("training_data/font_glyphs/noto-sans/4.svg").read())
print(f"4 contours: {len(cnts)}")
for ci, cnt in enumerate(cnts):
    print(f"Contour {ci}: {len(cnt)} segs")
    for si, s in enumerate(cnt):
        print(f"  Seg {si}: {s.kind} {s.p0} -> {s.p1}")
