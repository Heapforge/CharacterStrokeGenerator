import json, os, sys
sys.path.insert(0, ".")
from strokegen.svg_io import load_svg_glyph, parse_hw_path
from tools.fix_svgs import check_iou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "training_data", "font_glyphs", "noto-sans")
DATA_DIR = os.path.join(ROOT, "training_data", "hanzi-writer-data-youyin", "data")

with open("/tmp/queue100.json") as f:
    queue = json.load(f)

good_chars = set(["4", "È", "É", "Ê", "Ë", "Ö", "ß", "Ē", "Ĕ", "Ė", "Ę", "Ě", "Ų"])

diagnostics = []
for item in queue:
    c = item["char"]
    if c in good_chars:
        continue
    data_path = os.path.join(DATA_DIR, f"{c}.json")
    d = json.load(open(data_path, encoding="utf-8"))
    strokes = d["strokes"]
    medians = d["medians"]
    
    cnts_font = load_svg_glyph(open(os.path.join(FONT_DIR, f"{c}.svg"), encoding="utf-8").read())
    score = check_iou(cnts_font, strokes)
    
    # Check median coverage: min/max of medians vs bbox of font
    all_med_pts = [p for m in medians for p in m]
    med_xs = [p[0] for p in all_med_pts]
    med_ys = [p[1] for p in all_med_pts]
    
    font_pts = [s.p0 for cnt in cnts_font for s in cnt]
    font_xs = [p[0] for p in font_pts]
    font_ys = [p[1] for p in font_pts]
    
    y_cov = (max(med_ys) - min(med_ys)) / max(1, (max(font_ys) - min(font_ys)))
    x_cov = (max(med_xs) - min(med_xs)) / max(1, (max(font_xs) - min(font_xs)))
    
    diagnostics.append({
        "char": c,
        "strokes": len(strokes),
        "iou": round(score, 4),
        "x_cov": round(x_cov, 3),
        "y_cov": round(y_cov, 3),
        "first_med": medians[0][:2],
        "last_med": medians[-1][-2:],
    })

print(f"{'Char':<4} {'Strokes':<8} {'IoU':<8} {'X-Cov':<8} {'Y-Cov':<8} {'Note'}")
for diag in diagnostics:
    note = []
    if diag["iou"] < 0.99:
        note.append(f"IoU={diag['iou']}")
    if diag["y_cov"] < 0.75:
        note.append(f"LowYCov={diag['y_cov']}")
    if diag["x_cov"] < 0.75:
        note.append(f"LowXCov={diag['x_cov']}")
    print(f"{diag['char']:<4} {diag['strokes']:<8} {diag['iou']:<8} {diag['x_cov']:<8} {diag['y_cov']:<8} {', '.join(note)}")
