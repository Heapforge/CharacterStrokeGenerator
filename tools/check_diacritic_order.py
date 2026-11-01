import json, sys
sys.path.insert(0, ".")
from strokegen.svg_io import parse_hw_path

with open("/tmp/queue100.json") as f:
    queue = json.load(f)

for item in queue:
    c = item["char"]
    strokes = item.get("pred", {}).get("strokes", [])
    medians = item.get("pred", {}).get("medians", [])
    if not strokes or len(strokes) < 2:
        continue
    
    # Calculate bbox of each stroke
    stroke_info = []
    for si, s in enumerate(strokes):
        loops = parse_hw_path(s)
        pts = []
        for l in loops:
            for seg in l:
                pts.append(seg.p0)
                pts.append(seg.p1)
        ys = [p[1] for p in pts]
        min_y, max_y = min(ys), max(ys)
        stroke_info.append((si, min_y, max_y))
        
    # Check if a stroke with min_y >= 650 exists, and if it is not at the beginning
    has_diacritic = any(min_y >= 650 for _, min_y, _ in stroke_info)
    first_is_diacritic = stroke_info[0][1] >= 650
    if has_diacritic:
        diacritic_indices = [si for si, min_y, _ in stroke_info if min_y >= 650]
        print(f"Char '{c}': total {len(strokes)} strokes, diacritic strokes at indices: {diacritic_indices}, first is diacritic: {first_is_diacritic}")
