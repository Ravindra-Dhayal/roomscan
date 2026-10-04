from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .ingest import Capture

# ----------------------------------------------------------------------------- fusion


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """One point (the voxel mean) per occupied voxel; makes density ~ surface area."""
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=pts[:, k]) / cnt
    return out


def fuse(cap: Capture, max_frames: int = 250, pixel_stride: int = 3,
         min_conf: int = 1, voxel: float = 0.02) -> np.ndarray:
    """Fuse depth frames (evenly spaced) into one voxel-downsampled world point cloud."""
    idx = np.unique(np.linspace(0, cap.n_frames - 1, min(max_frames, cap.n_frames)).astype(int))
    chunks = []
    for i in idx:
        p = cap.points_world(int(i), min_conf)
        chunks.append(p[:: pixel_stride])
    return voxel_downsample(np.concatenate(chunks), voxel)


# --------------------------------------------------------------- floor and ceiling


@dataclass
class Plane1D:
    y: float            # height of the plane (world y)
    sigma: float        # robust thickness (std) of the points on the plane, m
    n: int              # points supporting it


@dataclass
class VerticalResult:
    floor: Plane1D
    ceiling: Plane1D | None
    ceiling_height: float | None
    ceiling_height_sigma: float | None   # combined plane thickness, NOT yet a calibrated interval
    ceiling_status: str                  # "found" | "abstain: <reason>"


def _refine(y: np.ndarray, centre: float, half: float = 0.04) -> Plane1D:
    sel = y[np.abs(y - centre) < half]
    if len(sel) < 20:
        return Plane1D(centre, half, len(sel))
    med = float(np.median(sel))
    sigma = float(1.4826 * np.median(np.abs(sel - med)))
    return Plane1D(med, max(sigma, 0.002), len(sel))


def find_floor_ceiling(pts: np.ndarray, min_ceiling_h: float = 1.9, max_ceiling_h: float = 5.0,
                       bin_w: float = 0.01) -> VerticalResult:
    """Floor = strongest horizontal surface. Ceiling = a *sharp* horizontal surface between
    min_ceiling_h and max_ceiling_h above it. If there is no such surface (floor-only scan,
    dark or mirrored ceiling) the ceiling is reported as an abstention, never guessed."""
    y = pts[:, 1]
    edges = np.arange(y.min(), y.max() + bin_w, bin_w)
    h, _ = np.histogram(y, bins=edges)
    hs = np.convolve(h, np.ones(3), mode="same")          # +-1 cm smoothing
    centres = edges[:-1] + bin_w / 2
    fi = int(np.argmax(hs))
    floor = _refine(y, float(centres[fi]))

    lo, hi = floor.y + min_ceiling_h, floor.y + max_ceiling_h
    band = (centres >= lo) & (centres <= hi)
    if band.sum() < 5:
        return VerticalResult(floor, None, None, None, "abstain: no data above minimum ceiling height")
    ci = int(np.flatnonzero(band)[np.argmax(hs[band])])
    # sharpness: peak mass vs the typical mass of bins in +-30 cm around it
    win = (np.abs(centres - centres[ci]) < 0.30) & (np.abs(centres - centres[ci]) > 0.05)
    background = max(float(np.median(hs[win])), 1.0)
    sharp = hs[ci] / background
    mass = hs[ci] / max(float(hs[fi]), 1.0)
    if sharp < 6.0 or mass < 0.02:
        return VerticalResult(floor, None, None, None,
                              f"abstain: no sharp ceiling plane (sharpness {sharp:.1f}, mass {mass:.3f})")
    ceil = _refine(y, float(centres[ci]))
    height = ceil.y - floor.y
    sigma = float(np.hypot(floor.sigma, ceil.sigma))
    return VerticalResult(floor, ceil, height, sigma, "found")


# ------------------------------------------------------------ orientation and walls


def manhattan_angle(pts_xz: np.ndarray, k: int = 15, max_pts: int = 40000, seed: int = 0) -> float:
    """Dominant wall direction (radians, modulo 90 degrees) from local 2-D normals."""
    rng = np.random.default_rng(seed)
    # a vertical wall projects to many stacked points; collapse to unique 2 cm cells first,
    # otherwise the k nearest neighbours are all at the same spot and the normals are noise
    cells = np.unique(np.floor(pts_xz / 0.02).astype(np.int64), axis=0)
    pts_xz = (cells + 0.5) * 0.02
    if len(pts_xz) > max_pts:
        pts_xz = pts_xz[rng.choice(len(pts_xz), max_pts, replace=False)]
    _, nn = cKDTree(pts_xz).query(pts_xz, k=k)
    nb = pts_xz[nn] - pts_xz[nn].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    w, v = np.linalg.eigh(cov)
    linear = w[:, 0] < 0.15 * w[:, 1]                       # keep clearly line-like neighbourhoods
    n = v[linear, :, 0]
    theta = np.arctan2(n[:, 1], n[:, 0])
    z = np.exp(4j * theta).sum()                            # 4-fold symmetry -> mod 90 deg
    return float(np.angle(z) / 4.0)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def full_height_mask(pts: np.ndarray, floor_y: float, top_y: float, cell: float = 0.05,
                     hbin: float = 0.10, min_frac: float = 0.6, lo: float = 0.30):
    """Cells whose points cover most of the height range floor+lo .. top_y. Walls do; furniture,
    clutter and clipped glimpses do not. Returns (cell_index_per_point, wall_cell_flag_per_point)."""
    sel = (pts[:, 1] > floor_y + lo) & (pts[:, 1] < top_y)
    p = pts[sel]
    nb = max(int(np.floor((top_y - floor_y - lo) / hbin)), 1)
    ij = np.floor(p[:, [0, 2]] / cell).astype(np.int64)
    hb = np.clip(((p[:, 1] - floor_y - lo) / hbin).astype(np.int64), 0, nb - 1)
    key = (ij[:, 0] - ij[:, 0].min()) * (ij[:, 1].max() - ij[:, 1].min() + 1) + (ij[:, 1] - ij[:, 1].min())
    uk, inv = np.unique(key, return_inverse=True)
    occupied = np.zeros((len(uk), nb), bool)
    occupied[inv, hb] = True
    wall_cell = occupied.sum(axis=1) >= min_frac * nb
    return p, wall_cell[inv]


@dataclass
class Wall:
    p0: tuple[float, float]       # world (x, z)
    p1: tuple[float, float]
    length: float
    support: int


def wall_segments(pts_xz: np.ndarray, theta: float, bin_w: float = 0.02, min_len: float = 0.5,
                  gap: float = 0.30, line_tol: float = 0.04, min_support: int = 150) -> list[Wall]:
    """Axis-aligned (in the Manhattan frame) wall lines: histogram peaks along each axis, then
    the occupied runs of points lying on each peak line."""
    R = _rot(theta)
    q = pts_xz @ R                                           # coordinates in the Manhattan frame
    walls: list[Wall] = []
    for axis in (0, 1):
        a, b = q[:, axis], q[:, 1 - axis]
        edges = np.arange(a.min(), a.max() + bin_w, bin_w)
        h, _ = np.histogram(a, bins=edges)
        hs = np.convolve(h, np.ones(3), mode="same").astype(float)
        used = np.zeros(len(hs), bool)
        for i in np.argsort(hs)[::-1]:
            if hs[i] < min_support:
                break
            if used[max(0, i - 8): i + 9].any():            # suppress within 16 cm of a stronger line
                continue
            used[i] = True
            c = edges[i] + bin_w / 2
            on = np.abs(a - c) < line_tol
            if on.sum() < min_support:
                continue
            c = float(np.median(a[on]))
            t = np.sort(b[on])
            breaks = np.flatnonzero(np.diff(t) > gap)
            for s, e in zip(np.r_[0, breaks + 1], np.r_[breaks, len(t) - 1]):
                if t[e] - t[s] >= min_len and e - s + 1 >= min_support // 3:
                    ends = np.array([[c, t[s]], [c, t[e]]]) if axis == 0 else np.array([[t[s], c], [t[e], c]])
                    w = ends @ R.T
                    walls.append(Wall(tuple(w[0]), tuple(w[1]), float(t[e] - t[s]), int(e - s + 1)))
    return walls


# ------------------------------------------------------------------------ footprint


@dataclass
class Footprint:
    polygons: list[np.ndarray]    # simplified outlines, world (x, z), metres
    area: float                   # floor area from the raster, m^2
    origin: tuple[float, float]
    cell: float
    mask: np.ndarray = field(repr=False, default=None)


def footprint(pts: np.ndarray, floor_y: float, wall_xz: np.ndarray | None = None, cell: float = 0.05,
              tol: float = 0.05, min_area: float = 1.0, eps: float = 0.12) -> Footprint:
    """Occupied floor cells united with wall cells, closed and filled; outlines simplified."""
    f = pts[np.abs(pts[:, 1] - floor_y) < tol][:, [0, 2]]
    allxy = f if wall_xz is None or len(wall_xz) == 0 else np.vstack([f, wall_xz])
    ox, oz = allxy.min(axis=0) - 0.5
    W = int((allxy[:, 0].max() - ox) / cell) + 2
    H = int((allxy[:, 1].max() - oz) / cell) + 2
    ij = np.floor((f - [ox, oz]) / cell).astype(int)
    cnt = np.zeros((H, W), np.int32)
    np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1)
    m = (cnt >= 2).astype(np.uint8)
    if wall_xz is not None and len(wall_xz):
        wi = np.floor((wall_xz - [ox, oz]) / cell).astype(int)
        m[wi[:, 1], wi[:, 0]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys, filled = [], np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) * cell * cell < min_area:
            continue
        cv2.drawContours(filled, [c], -1, 1, cv2.FILLED)
        ap = cv2.approxPolyDP(c, eps / cell, True)[:, 0, :].astype(float)
        polys.append((ap + 0.5) * cell + [ox, oz])
    return Footprint(polys, float(filled.sum() * cell * cell), (ox, oz), cell, filled)


# ------------------------------------------------------------------------ top level


@dataclass
class SpaceGeometry:
    vertical: VerticalResult
    theta: float
    walls: list[Wall]
    footprint: Footprint
    wall_xz: np.ndarray = field(repr=False, default=None)   # points on full-height surfaces


def analyze(pts: np.ndarray) -> SpaceGeometry:
    v = find_floor_ceiling(pts)
    top = v.ceiling.y - 0.25 if v.ceiling else v.floor.y + 1.8
    p, is_wall = full_height_mask(pts, v.floor.y, top)
    wall_xz = p[is_wall][:, [0, 2]]
    theta = manhattan_angle(wall_xz)
    walls = wall_segments(wall_xz, theta, min_support=300)
    return SpaceGeometry(v, theta, walls, footprint(pts, v.floor.y, wall_xz), wall_xz)