"""Stroke medians from stroke outlines (no learning).

Voronoi skeleton of the densely sampled outline -> longest path through the skeleton ->
ends extended toward the stroke tips -> Ramer-Douglas-Peucker simplification.
The result is unoriented; the order model decides which end is the start.
"""
from __future__ import annotations

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import Voronoi
from shapely.geometry import LineString, Polygon
from shapely.validation import make_valid

from .svg_io import Seg, contour_polygon

SAMPLE_STEP = 6.0
RDP_EPS = 3.0
SPUR_FRAC = 0.3  # trim path ends whose clearance < SPUR_FRAC * median clearance (corner spurs)
TIP_FRAC = 0.0  # extension toward the tip (fraction of gap); GT medians stop at the skeleton end


def stroke_polygon(loops: list[list[Seg]]):
    geoms = []
    for loop in loops:
        pts = contour_polygon(loop, 2.0)
        if len(pts) >= 3:
            geoms.append(make_valid(Polygon(pts)))
    if not geoms:
        return None
    g = shapely.union_all(geoms)
    if g.geom_type == "MultiPolygon" or g.geom_type == "GeometryCollection":
        polys = [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon"]
        if not polys:
            return None
        g = max(polys, key=lambda p: p.area)
    return g if g.geom_type == "Polygon" and g.area > 1 else None


def _boundary_samples(poly: Polygon, step: float) -> np.ndarray:
    pts = []
    for ring in [poly.exterior, *poly.interiors]:
        seg = shapely.segmentize(ring, step)
        pts.append(np.asarray(seg.coords)[:-1])
    return np.concatenate(pts)


def skeleton_path(poly: Polygon, step: float = SAMPLE_STEP) -> np.ndarray | None:
    pts = _boundary_samples(poly, step)
    if len(pts) < 4:
        return None
    vor = Voronoi(pts + np.random.default_rng(0).normal(0, 1e-3, pts.shape))
    V = vor.vertices
    inside = shapely.contains_xy(poly, V[:, 0], V[:, 1])
    E = np.array([rv for rv in vor.ridge_vertices if rv[0] >= 0 and rv[1] >= 0 and inside[rv[0]] and inside[rv[1]]])
    if len(E) == 0:
        return None
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) + 1e-6
    n = len(V)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    _, comp = connected_components(G, directed=False)
    used = np.unique(E)
    big = np.bincount(comp[used]).argmax()
    nodes = used[comp[used] == big]
    # tree diameter by double sweep
    d0 = dijkstra(G, indices=nodes[0])
    a = nodes[np.argmax(np.where(np.isfinite(d0[nodes]), d0[nodes], -1))]
    da, pred = dijkstra(G, indices=a, return_predecessors=True)
    b = nodes[np.argmax(np.where(np.isfinite(da[nodes]), da[nodes], -1))]
    path = [b]
    while path[-1] != a and pred[path[-1]] >= 0:
        path.append(pred[path[-1]])
    return V[path[::-1]]


def _extend(path: np.ndarray, poly: Polygon, frac: float) -> np.ndarray:
    """Push both ends toward the outline along the local end direction."""
    out = path.copy()
    for end in (0, -1):
        seq = path if end == 0 else path[::-1]  # seq[0] is the end being extended
        p = seq[0]
        # end direction: from a point ~25 units along the path toward the end
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seq, axis=0), axis=1))]
        q_idx = min(int(np.searchsorted(dist, min(25.0, dist[-1] * 0.3))), len(seq) - 1)
        d = p - seq[q_idx]
        L = np.hypot(*d)
        if L < 1e-6:
            continue
        d /= L
        ray = LineString([p, p + d * 400])
        hit = ray.intersection(poly.exterior)
        if hit.is_empty:
            continue
        hits = shapely.get_coordinates(hit)
        gap = float(np.linalg.norm(hits - p, axis=1).min())
        out[end] = p + d * gap * frac
    return out


def trim_spurs(path: np.ndarray, poly: Polygon, frac: float = SPUR_FRAC) -> np.ndarray:
    """Drop the skeleton's end branches into sharp outline corners (low clearance)."""
    c = shapely.distance(shapely.points(path), poly.boundary)
    keep = np.where(c >= frac * np.median(c))[0]
    if len(keep) < 2:
        return path
    return path[keep[0]:keep[-1] + 1]


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.hypot(*ab)
    if L < 1e-9:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / L
    i = int(np.argmax(d))
    if d[i] > eps:
        left = rdp(pts[: i + 1], eps)
        return np.vstack([left[:-1], rdp(pts[i:], eps)])
    return np.vstack([a, b])


def stroke_median(loops: list[list[Seg]], eps: float = RDP_EPS, tip_frac: float = TIP_FRAC) -> np.ndarray:
    """Unoriented median polyline (k, 2) in hw coords, rounded to integers."""
    poly = stroke_polygon(loops)
    if poly is None:
        pts = np.concatenate([s.pts for loop in loops for s in loop])
        c = pts.mean(0)
        return np.round(np.stack([c - 1, c + 1]))
    path = skeleton_path(poly)
    if path is None or len(path) < 2:
        c = np.array(poly.centroid.coords[0])
        return np.round(np.stack([c - 1, c + 1]))
    path = trim_spurs(path, poly)
    if tip_frac > 0:
        path = _extend(path, poly, tip_frac)
    return np.round(rdp(path, eps))


def resample(poly: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    if cum[-1] < 1e-9:
        return np.repeat(poly[:1], n, 0)
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])], 1)


def median_distance(a: np.ndarray, b: np.ndarray, n: int = 32) -> float:
    """Mean distance between two polylines resampled by arclength (same orientation)."""
    return float(np.linalg.norm(resample(a, n) - resample(b, n), axis=1).mean())
