import json, os
from strokegen.svg_io import load_svg_glyph, segs_to_path, HW_BASE
from strokegen.raster import render, iou

with open("/tmp/queue100.json") as f:
    queue = json.load(f)

print(f"Loaded {len(queue)} items.")

report = []
for i, item in enumerate(queue):
    c = item["char"]
    font = item["font"]
    font_path = f"training_data/font_glyphs/{font}/{c}.svg"
    contours = load_svg_glyph(open(font_path, encoding="utf-8").read())
    
    # Analyze contours
    contour_info = []
    for ci, cnt in enumerate(contours):
        pts = []
        for s in cnt:
            pts.append([s.p0[0], s.p0[1]])
            pts.append([s.p1[0], s.p1[1]])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        
        # calculate signed area to determine winding
        area = 0.0
        for s in cnt:
            area += (s.p0[0] * s.p1[1] - s.p1[0] * s.p0[1])
        contour_info.append({
            "id": ci,
            "num_segs": len(cnt),
            "bbox": [round(v, 1) for v in bbox],
            "area": round(float(area), 1),
            "is_hole": bool(area < 0)
        })
        
    pred = item.get("pred", {})
    strokes = pred.get("strokes", [])
    medians = pred.get("medians", [])
    
    report.append({
        "idx": i,
        "char": c,
        "num_contours": len(contours),
        "contours": contour_info,
        "num_pred_strokes": len(strokes),
        "num_pred_medians": len(medians)
    })

with open("/tmp/analyze_100_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("Saved /tmp/analyze_100_report.json")
# Print characters that have multiple contours
multi = [r for r in report if r["num_contours"] > 1]
print(f"Characters with >1 contours: {len(multi)}")
for m in multi[:20]:
    print(f"Idx {m['idx']:2d} Char '{m['char']}': {m['num_contours']} contours")
