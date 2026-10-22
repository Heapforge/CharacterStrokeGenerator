"""Rasterize hw-space outlines with cairo."""
from __future__ import annotations

import cairo
import numpy as np

from .svg_io import Seg

# hw frame: x in [0, 1024], y in [-124, 900] (y up)
X0, Y_TOP, SPAN = 0.0, 900.0, 1024.0


def _trace(ctx: cairo.Context, loop: list[Seg]) -> None:
    ctx.move_to(*loop[0].p0)
    for s in loop:
        c = s.cubic()
        if s.kind == "L":
            ctx.line_to(*c[3])
        else:
            ctx.curve_to(*c[1], *c[2], *c[3])
    ctx.close_path()


def render(loops: list[list[Seg]], size: int = 256, even_odd: bool = False) -> np.ndarray:
    """Fill the given closed loops together (one fill), return float32 coverage in [0, 1], shape (size, size)."""
    surf = cairo.ImageSurface(cairo.FORMAT_A8, size, size)
    ctx = cairo.Context(surf)
    s = size / SPAN
    ctx.set_matrix(cairo.Matrix(s, 0, 0, -s, -X0 * s, Y_TOP * s))
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD if even_odd else cairo.FILL_RULE_WINDING)
    for loop in loops:
        if loop:
            _trace(ctx, loop)
    ctx.fill()
    surf.flush()
    stride = surf.get_stride()
    buf = np.frombuffer(surf.get_data(), np.uint8).reshape(size, stride)[:, :size]
    return buf.astype(np.float32) / 255.0


def render_union(strokes: list[list[Seg]], size: int = 256) -> np.ndarray:
    """Union of separately filled strokes (each stroke filled with nonzero winding)."""
    out = np.zeros((size, size), np.float32)
    for st in strokes:
        out = np.maximum(out, render([st], size))
    return out


def iou(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    A, B = a > thr, b > thr
    u = np.logical_or(A, B).sum()
    return float(np.logical_and(A, B).sum() / u) if u else 1.0


def hw_to_pixel(p: np.ndarray, size: int) -> np.ndarray:
    """hw coords -> pixel coords (x right, y down), continuous."""
    s = size / SPAN
    return np.stack([(p[..., 0] - X0) * s, (Y_TOP - p[..., 1]) * s], -1)
