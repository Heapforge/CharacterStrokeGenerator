import json, sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, segs_to_path, HW_BASE

with open("/tmp/queue100.json") as f:
    queue = json.load(f)

print(f"Total: {len(queue)}")
for item in queue:
    c = item["char"]
    font = item["font"]
    font_path = f"training_data/font_glyphs/{font}/{c}.svg"
    contours = load_svg_glyph(open(font_path, encoding="utf-8").read())
    
    # Check if multiple contours where some are elevated above 650 (top diacritics)
    elevated = []
    main_body = []
    for ci, cnt in enumerate(contours):
        pts = [s.p0 for s in cnt]
        ys = [p[1] for p in pts]
        min_y = min(ys)
        max_y = max(ys)
        if min_y >= 650:
            elevated.append((ci, min_y, max_y))
        else:
            main_body.append((ci, min_y, max_y))
            
    if elevated and main_body:
        pred_strokes = item.get("pred", {}).get("strokes", [])
        pred_medians = item.get("pred", {}).get("medians", [])
        print(f"Char '{c}': {len(contours)} contours ({len(elevated)} elevated diacritics, {len(main_body)} body contours). Pred strokes: {len(pred_strokes)}")
