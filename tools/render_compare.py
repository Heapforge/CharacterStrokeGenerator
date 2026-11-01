#!/usr/bin/env python3
"""Render a glyph's hanzi-writer data next to its font SVG, for eyeballing corrections.

    uv run python tools/render_compare.py data.json out.png --char Ḕ [--font noto-sans]

Left: the font glyph. Middle: every stroke filled in its own colour with its median (dot =
start, arrow = end) and its index. Right: red = stroke area outside the font, blue = font area
no stroke covers. The title line gives the union IoU. data.json is {"strokes": [...], "medians": [...]}.
The view covers x -300..1400, y -400..1300 in hw units so accents above the em box show too.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys

import cairo
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from strokegen.raster import _trace, iou  # noqa: E402
from strokegen.svg_io import load_svg_glyph, parse_hw_path  # noqa: E402

PANEL = 512
X_MIN, X_MAX, Y_MIN, Y_MAX = -300.0, 1400.0, -400.0, 1300.0


def _ctx(surf, ox):
    ctx = cairo.Context(surf)
    s = PANEL / (X_MAX - X_MIN)
    ctx.translate(ox, 24)
    ctx.scale(s, -s)
    ctx.translate(-X_MIN, -Y_MAX)
    return ctx


def _fill(loops, size=PANEL):
    surf = cairo.ImageSurface(cairo.FORMAT_A8, size, size)
    ctx = cairo.Context(surf)
    s = size / (X_MAX - X_MIN)
    ctx.scale(s, -s)
    ctx.translate(-X_MIN, -Y_MAX)
    for loop in loops:
        _trace(ctx, loop)
    ctx.fill()
    surf.flush()
    a = np.frombuffer(surf.get_data(), np.uint8).reshape(size, surf.get_stride())[:, :size]
    return a > 127


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--char", required=True)
    ap.add_argument("--font", default="noto-sans")
    a = ap.parse_args()
    d = json.load(open(a.data, encoding="utf-8"))
    font = load_svg_glyph(open(os.path.join(ROOT, "training_data", "font_glyphs", a.font, a.char + ".svg"),
                               encoding="utf-8").read())
    strokes = [parse_hw_path(s) for s in d["strokes"]]

    F = _fill(font)
    U = np.zeros_like(F)
    for st in strokes:
        U |= _fill(st)
    score = iou(F.astype(np.float32), U.astype(np.float32))

    W = PANEL * 3 + 40
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, PANEL + 30)
    c = cairo.Context(surf)
    c.set_source_rgb(1, 1, 1)
    c.paint()
    c.set_source_rgb(0, 0, 0)
    c.set_font_size(16)
    c.move_to(8, 18)
    c.show_text(f"{a.char}  strokes={len(strokes)}  union IoU={score:.4f}   [font | strokes+medians | red=extra blue=missing]")

    # font + em box
    ctx = _ctx(surf, 0)
    for loop in font:
        _trace(ctx, loop)
    ctx.set_source_rgb(0.2, 0.2, 0.2)
    ctx.fill()
    for ox in (0, PANEL + 20, 2 * PANEL + 40):
        ctx = _ctx(surf, ox)
        ctx.rectangle(0, -124, 1024, 1024)
        ctx.set_source_rgb(0.75, 0.75, 0.75)
        ctx.set_line_width(3)
        ctx.stroke()

    # strokes + medians
    ctx = _ctx(surf, PANEL + 20)
    n = max(1, len(strokes))
    for i, (st, med) in enumerate(zip(strokes, d["medians"])):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.65, 0.95)
        for loop in st:
            _trace(ctx, loop)
        ctx.set_source_rgba(r, g, b, 0.45)
        ctx.fill_preserve()
        ctx.set_source_rgb(r * 0.6, g * 0.6, b * 0.6)
        ctx.set_line_width(3)
        ctx.stroke()
    for i, med in enumerate(d["medians"]):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.9, 0.55)
        m = np.asarray(med, float)
        ctx.set_source_rgb(r, g, b)
        ctx.set_line_width(7)
        ctx.move_to(*m[0])
        for p in m[1:]:
            ctx.line_to(*p)
        ctx.stroke()
        ctx.arc(*m[0], 16, 0, 2 * np.pi)
        ctx.fill()
        if len(m) > 1:
            v = m[-1] - m[-2]
            v = v / (np.linalg.norm(v) + 1e-9) * 40
            nrm = np.array([-v[1], v[0]]) * 0.6
            ctx.move_to(*(m[-1] + v))
            ctx.line_to(*(m[-1] - v * 0.3 + nrm))
            ctx.line_to(*(m[-1] - v * 0.3 - nrm))
            ctx.close_path()
            ctx.fill()
        ctx.save()
        ctx.translate(m[0][0] + 20, m[0][1] + 20)
        ctx.scale(1, -1)
        ctx.set_font_size(60)
        ctx.move_to(0, 0)
        ctx.show_text(str(i))
        ctx.restore()

    # diff panel
    ox = 2 * PANEL + 40
    img = np.full((PANEL, PANEL, 4), 255, np.uint8)
    img[F & U] = (200, 200, 200, 255)
    img[U & ~F] = (40, 40, 230, 255)   # BGRA: red
    img[F & ~U] = (230, 120, 30, 255)  # BGRA: blue
    diff = cairo.ImageSurface.create_for_data(memoryview(np.ascontiguousarray(img)), cairo.FORMAT_RGB24, PANEL, PANEL)
    c.set_source_surface(diff, ox, 24)
    c.paint()
    surf.write_to_png(a.out)
    print(f"{a.char}: {len(strokes)} strokes, union IoU {score:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
