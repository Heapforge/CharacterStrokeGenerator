"""SVG path parsing, segment geometry and hanzi-writer serialization.

All geometry inside strokegen lives in hanzi-writer space ("hw"): a 1024-unit em
with y pointing up, where y_hw = 900 - y_svg for the Arphic font exports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_TOKEN = re.compile(r"[MmLlHhVvQqTtCcSsZzAa]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARGC = {"M": 2, "L": 2, "H": 1, "V": 1, "Q": 4, "T": 2, "C": 6, "S": 4, "Z": 0}

HW_EM = 1024.0
HW_BASE = 900.0  # y_hw = HW_BASE - y_svg (for a 1024-unit viewBox starting at 0)


@dataclass
class Seg:
    """One outline segment. kind is 'L', 'Q' or 'C'; pts holds all control points incl. both ends."""

    kind: str
    pts: np.ndarray  # (2|3|4, 2) float

    @property
    def p0(self) -> np.ndarray:
        return self.pts[0]

    @property
    def p1(self) -> np.ndarray:
        return self.pts[-1]

    def reversed(self) -> "Seg":
        return Seg(self.kind, self.pts[::-1].copy())

    def transformed(self, A: np.ndarray, b: np.ndarray) -> "Seg":
        return Seg(self.kind, self.pts @ A.T + b)

    def cubic(self) -> np.ndarray:
        """Control points as an equivalent cubic (4, 2)."""
        p = self.pts
        if self.kind == "L":
            return np.stack([p[0], p[0] + (p[1] - p[0]) / 3, p[0] + 2 * (p[1] - p[0]) / 3, p[1]])
        if self.kind == "Q":
            return np.stack([p[0], p[0] + 2 / 3 * (p[1] - p[0]), p[2] + 2 / 3 * (p[1] - p[2]), p[2]])
        return p.copy()

    def sample(self, n: int) -> np.ndarray:
        """n points at uniform t in [0, 1] (inclusive)."""
        t = np.linspace(0.0, 1.0, n)[:, None]
        p = self.pts
        if self.kind == "L":
            return p[0] * (1 - t) + p[1] * t
        if self.kind == "Q":
            return (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t**2 * p[2]
        return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t**2 * p[2] + t**3 * p[3]

    def flatten(self, tol: float = 2.0) -> np.ndarray:
        """Polyline approximation; returns points incl. both ends."""
        if self.kind == "L":
            return self.pts.copy()
        est = np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum()
        n = int(np.clip(np.ceil(est / (tol * 4)), 2, 64)) + 1
        return self.sample(n)

    def length(self) -> float:
        if self.kind == "L":
            return float(np.linalg.norm(self.pts[1] - self.pts[0]))
        s = self.sample(17)
        return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum())

    def split(self, t: float = 0.5) -> tuple["Seg", "Seg"]:
        """De Casteljau split at t."""
        p = self.pts
        levels = [p]
        while len(levels[-1]) > 1:
            q = levels[-1]
            levels.append(q[:-1] * (1 - t) + q[1:] * t)
        left = np.stack([lv[0] for lv in levels])
        right = np.stack([lv[-1] for lv in levels[::-1]])
        return Seg(self.kind, left), Seg(self.kind, right)

    def tangent_start(self) -> np.ndarray:
        for q in self.pts[1:]:
            d = q - self.pts[0]
            if np.hypot(*d) > 1e-6:
                return d / np.hypot(*d)
        return np.array([1.0, 0.0])

    def tangent_end(self) -> np.ndarray:
        for q in self.pts[-2::-1]:
            d = self.pts[-1] - q
            if np.hypot(*d) > 1e-6:
                return d / np.hypot(*d)
        return np.array([1.0, 0.0])


Contour = list  # list[Seg], closed: contour[i].p1 == contour[i+1].p0, last.p1 == first.p0


def parse_path(d: str) -> list[Contour]:
    """Parse an SVG path 'd' string into closed contours of absolute segments (SVG coordinates)."""
    toks = _TOKEN.findall(d)
    i, cmd = 0, None
    x = y = sx = sy = 0.0
    last_ctrl = None  # for T / S reflection
    last_kind = None
    contours: list[Contour] = []
    cur: Contour | None = None

    def close():
        nonlocal cur
        if cur is not None:
            if cur and np.hypot(x - sx, y - sy) > 1e-9:
                cur.append(Seg("L", np.array([[x, y], [sx, sy]], float)))
            if cur:
                contours.append(cur)
        cur = None

    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
        if cmd is None:
            raise ValueError("path data must start with a command")
        U, rel = cmd.upper(), cmd.islower()
        if U == "A":
            raise ValueError("arc commands are not supported")
        if U == "Z":
            close()
            x, y = sx, sy
            last_kind = None
            continue
        k = _ARGC[U]
        a = [float(v) for v in toks[i:i + k]]
        if len(a) < k:
            raise ValueError("truncated path data")
        i += k
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if U == "M":
            close()
            x, y = a[0] + ox, a[1] + oy
            sx, sy = x, y
            cur = []
            cmd = "l" if rel else "L"
            last_kind = None
            continue
        if cur is None:  # drawing without a moveto: start at current point
            cur = []
            sx, sy = x, y
        p0 = np.array([x, y])
        if U in "LHV":
            if U == "H":
                nx, ny = a[0] + (x if rel else 0.0), y
            elif U == "V":
                nx, ny = x, a[0] + (y if rel else 0.0)
            else:
                nx, ny = a[0] + ox, a[1] + oy
            seg = Seg("L", np.array([p0, [nx, ny]]))
            last_kind, last_ctrl = "L", None
        elif U in "QT":
            if U == "Q":
                c = np.array([a[0] + ox, a[1] + oy])
                nx, ny = a[2] + ox, a[3] + oy
            else:
                c = 2 * p0 - last_ctrl if last_kind == "Q" else p0.copy()
                nx, ny = a[0] + ox, a[1] + oy
            seg = Seg("Q", np.array([p0, c, [nx, ny]]))
            last_kind, last_ctrl = "Q", c
        else:  # C / S
            if U == "C":
                c1 = np.array([a[0] + ox, a[1] + oy])
                c2 = np.array([a[2] + ox, a[3] + oy])
                nx, ny = a[4] + ox, a[5] + oy
            else:
                c1 = 2 * p0 - last_ctrl if last_kind == "C" else p0.copy()
                c2 = np.array([a[0] + ox, a[1] + oy])
                nx, ny = a[2] + ox, a[3] + oy
            seg = Seg("C", np.array([p0, c1, c2, [nx, ny]]))
            last_kind, last_ctrl = "C", c2
        if np.hypot(nx - x, ny - y) > 1e-9 or seg.kind != "L":
            if np.abs(seg.pts - seg.pts[0]).max() > 1e-9:  # drop fully degenerate segments
                cur.append(seg)
        x, y = nx, ny
    close()
    return contours


_VIEWBOX = re.compile(r'viewBox\s*=\s*"([^"]*)"')
_PATH_D = re.compile(r"<path\b[^>]*?\bd\s*=\s*\"([^\"]*)\"", re.S)


def svg_to_hw_transform(svg_text: str) -> tuple[np.ndarray, np.ndarray]:
    """Affine (A, b) mapping SVG coords to hw coords: p_hw = A @ p + b.

    For the Arphic exports (viewBox y range 0..1024) this is exactly y_hw = 900 - y.
    Other viewBoxes are scaled so their height maps to the 1024 em.
    """
    m = _VIEWBOX.search(svg_text)
    vy, vh = 0.0, HW_EM
    if m:
        vals = [float(v) for v in m.group(1).replace(",", " ").split()]
        if len(vals) == 4 and vals[3] > 0:
            vy, vh = vals[1], vals[3]
    s = HW_EM / vh
    A = np.array([[s, 0.0], [0.0, -s]])
    b = np.array([0.0, HW_BASE + s * vy])
    return A, b


def load_svg_glyph(svg_text: str) -> list[Contour]:
    """All path outlines of an SVG file, as contours in hw coordinates."""
    A, b = svg_to_hw_transform(svg_text)
    out = []
    for d in _PATH_D.findall(svg_text):
        for c in parse_path(d):
            out.append([s.transformed(A, b) for s in c])
    return out


def parse_hw_path(d: str) -> list[Contour]:
    """Parse a hanzi-writer stroke path (already in hw coordinates)."""
    return parse_path(d)


def _fmt(v: float) -> str:
    r = round(float(v), 1)
    return str(int(r)) if r == int(r) else f"{r:.1f}"


def segs_to_path(segs: list[Seg]) -> str:
    """Serialize one closed loop of segments to a hanzi-writer style absolute path string."""
    if not segs:
        return ""
    parts = [f"M {_fmt(segs[0].p0[0])} {_fmt(segs[0].p0[1])}"]
    for s in segs:
        parts.append(s.kind + " " + " ".join(f"{_fmt(px)} {_fmt(py)}" for px, py in s.pts[1:]))
    parts.append("Z")
    return " ".join(parts)


def contour_polygon(segs: list[Seg], tol: float = 2.0) -> np.ndarray:
    """Closed polyline (without repeated last point) for a loop of segments."""
    pts = [segs[0].p0[None]]
    for s in segs:
        pts.append(s.flatten(tol)[1:])
    poly = np.concatenate(pts)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    return poly


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
