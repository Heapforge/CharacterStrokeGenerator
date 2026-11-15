import json, os, sys
import cairo
import numpy as np
from strokegen.svg_io import load_svg_glyph, parse_hw_path, HW_BASE
from strokegen.raster import render, iou, SPAN, X0, Y_TOP, _trace

def check_iou(font_contours, stroke_strs):
    r_font = render(font_contours)
    surf = cairo.ImageSurface(cairo.FORMAT_A8, 256, 256)
    ctx = cairo.Context(surf)
    s = 256.0 / SPAN
    ctx.scale(s, -s)
    ctx.translate(-X0, -Y_TOP)
    for st_str in stroke_strs:
        loops = parse_hw_path(st_str)
        for loop in loops:
            _trace(ctx, loop)
        ctx.fill()
    surf.flush()
    stride = surf.get_stride()
    buf = np.frombuffer(surf.get_data(), np.uint8).reshape(256, stride)[:, :256].astype(np.float32) / 255.0
    return float(iou(r_font, buf))

with open("/tmp/queue100.json") as f:
    queue = json.load(f)

print(f"Total queue items: {len(queue)}")
results = []
for i, item in enumerate(queue):
    c = item["char"]
    font = item["font"]
    font_svg_path = f"training_data/font_glyphs/{font}/{c}.svg"
    hw_path = f"training_data/hanzi-writer-data-youyin/data/{c}.json"
    
    font_exists = os.path.exists(font_svg_path)
    hw_exists = os.path.exists(hw_path)
    
    pred_strokes = item["pred"]["strokes"] if item.get("pred") else []
    pred_medians = item["pred"]["medians"] if item.get("pred") else []
    
    pred_iou = None
    hw_iou = None
    if font_exists and pred_strokes:
        font_contours = load_svg_glyph(open(font_svg_path, encoding="utf-8").read())
        pred_iou = check_iou(font_contours, pred_strokes)
        
        if hw_exists:
            hw_data = json.load(open(hw_path, encoding="utf-8"))
            hw_iou = check_iou(font_contours, hw_data["strokes"])
            
    results.append({
        "idx": i,
        "char": c,
        "font": font,
        "font_exists": font_exists,
        "hw_exists": hw_exists,
        "num_pred_strokes": len(pred_strokes),
        "pred_iou": pred_iou,
        "hw_iou": hw_iou,
    })

print(f"{'Idx':<4} {'Char':<4} {'Font':<10} {'HW?':<6} {'Strokes':<8} {'Pred IoU':<10} {'HW IoU':<10}")
bad_pred_count = 0
for r in results:
    piou_str = f"{r['pred_iou']:.4f}" if r['pred_iou'] is not None else "None"
    hiou_str = f"{r['hw_iou']:.4f}" if r['hw_iou'] is not None else "None"
    if r['pred_iou'] is not None and r['pred_iou'] < 0.98:
        bad_pred_count += 1
    print(f"{r['idx']:<4} {r['char']:<4} {r['font']:<10} {str(r['hw_exists']):<6} {r['num_pred_strokes']:<8} {piou_str:<10} {hiou_str:<10}")

print(f"\nTotal bad pred IoU (<0.98): {bad_pred_count} / {len(results)}")
with open("/tmp/check_100_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
