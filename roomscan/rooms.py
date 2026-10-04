"""Rooms, doorways and adjacency from a fused LiDAR point cloud.

Method: free floor space (floor cells minus wall cells) -> distance transform -> watershed
gives candidate regions. Two regions that touch along a *wide* boundary are one room; a
boundary of door size (0.55-1.35 m) is a doorway and makes the rooms adjacent.

Known limits (reported, not hidden): doors must be open in the capture (a closed door looks
like a wall); doorway width is refined from the wall points (refine_opening) but is unvalidated
against tape measurements, so no accuracy claim is made yet; windows are not detected yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from .lidar import SpaceGeometry, _rot, find_floor_ceiling


@dataclass
class Opening:
    kind: str                       # "door" (windows: not implemented)
    p0: tuple[float, float]         # world (x, z) of the two ends of the doorway
    p1: tuple[float, float]
    width: float
    rooms: tuple[int, int] | None = None


@dataclass
class Room:
    id: int
    polygon: np.ndarray              # (n,2) world (x, z), Manhattan-snapped when possible
    area: float                     
    wall_lengths: list[float]
    ceiling_height: float | None
    ceiling_status: str
    mask: np.ndarray = field(repr=False, default=None)


@dataclass
class Layout:
    rooms: list[Room]
    openings: list[Opening]
    adjacency: list[tuple[int, int]]


def _snap_manhattan(poly: np.ndarray, theta: float):
    """Snap a simplified outline to axis-aligned edges in the Manhattan frame (or None)."""
    R = _rot(theta)
    q = poly @ R
    n = len(q)
    edges = []
    for i in range(n):
        a, b = q[i], q[(i + 1) % n]
        d = b - a
        edges.append(["V" if abs(d[0]) < abs(d[1]) else "H", a, b])
    merged: list[list] = []
    for e in edges:
        if merged and merged[-1][0] == e[0]:
            merged[-1][2] = e[2]
        else:
            merged.append(e[:])
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        merged[0][1] = merged[-1][1]
        merged.pop()
    if len(merged) < 4 or len(merged) % 2:
        return None
    coords = [float((a[0] + b[0]) / 2) if t == "V" else float((a[1] + b[1]) / 2) for t, a, b in merged]
    verts = []
    for i in range(len(merged)):
        c0, c1 = coords[i - 1], coords[i]
        verts.append([c0, c1] if merged[i - 1][0] == "V" else [c1, c0])
    return np.array(verts) @ R.T


def refine_opening(o: Opening, wall_xz: np.ndarray, band=0.15, reach=1.6) -> Opening:
    """Replace the raster width by the gap between the nearest full-height wall points on either
    side of the doorway centre, measured along the doorway axis. The raster neck is biased
    narrow (wall cells are dilated and the neck is eroded), the points are not."""
    a, b = np.array(o.p0), np.array(o.p1)
    m, u = (a + b) / 2, (b - a) / (np.linalg.norm(b - a) + 1e-9)
    n = np.array([-u[1], u[0]])
    rel = wall_xz - m
    t, s = rel @ u, rel @ n
    sel = (np.abs(s) < band) & (np.abs(t) < reach)
    ts = t[sel]
    neg, pos = ts[ts <= 0], ts[ts > 0]
    if len(neg) == 0 or len(pos) == 0:
        return o
    t0, t1 = float(neg.max()), float(pos.min())
    if not (0.5 < t1 - t0 < 1.8):
        return o
    return Opening(o.kind, tuple(map(float, m + u * t0)), tuple(map(float, m + u * t1)),
                   t1 - t0, o.rooms)


def segment_rooms(pts: np.ndarray, geom: SpaceGeometry, door_w=(0.55, 1.35),
                  min_room_area=1.5, eps=0.10) -> Layout:
    fp = geom.footprint
    cell, (ox, oz) = fp.cell, fp.origin
    H, W = fp.mask.shape

    def to_ij(xy):
        return np.floor((np.asarray(xy) - [ox, oz]) / cell).astype(int)

    wallm = np.zeros((H, W), np.uint8)
    wi = to_ij(geom.wall_xz)
    ok = (wi[:, 0] >= 0) & (wi[:, 0] < W) & (wi[:, 1] >= 0) & (wi[:, 1] < H)
    wallm[wi[ok, 1], wi[ok, 0]] = 1
    wallm = cv2.dilate(wallm, np.ones((3, 3), np.uint8))
    free = ((fp.mask > 0) & (wallm == 0)).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if free.sum() == 0:
        return Layout([], [], [])
    sm = ndi.gaussian_filter(ndi.distance_transform_edt(free) * cell, 1.5)
    pk = peak_local_max(sm, min_distance=int(0.5 / cell), threshold_abs=0.45, labels=free)
    mk = np.zeros(free.shape, np.int32)
    for i, (r, c) in enumerate(pk):
        mk[r, c] = i + 1
    lab = watershed(-sm, mk, mask=free > 0)

    # boundary pixels between touching regions, per pair
    bpix: dict[tuple[int, int], list] = {}
    for dr, dc in ((0, 1), (1, 0)):
        a, b = lab[: H - dr, : W - dc], lab[dr:, dc:]
        rr, cc = np.nonzero((a > 0) & (b > 0) & (a != b))
        for r, c in zip(rr, cc):
            x, y = int(a[r, c]), int(b[r, c])
            bpix.setdefault((min(x, y), max(x, y)), []).append((r, c))

    def extent(px):
        p = np.array(px, float)[:, ::-1] * cell
        d = p - p.mean(0)
        u = np.linalg.svd(d, full_matrices=False)[2][0]
        t = d @ u
        return float(t.max() - t.min() + cell), p.mean(0) + u * t.max(), p.mean(0) + u * t.min()

    parent = list(range(len(pk) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    info = {k: extent(v) for k, v in bpix.items()}
    for (x, y), (w, _, _) in info.items():
        if w > door_w[1]:
            parent[find(x)] = find(y)
    merged = np.zeros_like(lab)
    for l in range(1, len(pk) + 1):
        merged[lab == l] = find(l)

    rooms: list[Room] = []
    root_to_id: dict[int, int] = {}
    ij_all = to_ij(pts[:, [0, 2]])
    inb = (ij_all[:, 0] >= 0) & (ij_all[:, 0] < W) & (ij_all[:, 1] >= 0) & (ij_all[:, 1] < H)
    for l in np.unique(merged):
        if l == 0:
            continue
        m = (merged == l).astype(np.uint8)
        if m.sum() * cell * cell < min_room_area:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        md = cv2.dilate(m, np.ones((3, 3), np.uint8))          
        cnts, _ = cv2.findContours(md, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, eps / cell, True)[:, 0, :].astype(float)
        poly = (ap + 0.5) * cell + [ox, oz]
        snapped = _snap_manhattan(poly, geom.theta)
        poly = snapped if snapped is not None else poly
        edges = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
        inroom = np.zeros(len(pts), bool)
        inroom[inb] = md[ij_all[inb, 1], ij_all[inb, 0]] > 0
        ch, st = None, "abstain: too few points"
        if inroom.sum() > 5000:
            v = find_floor_ceiling(pts[inroom])
            ch, st = v.ceiling_height, v.ceiling_status
        root_to_id[int(l)] = len(rooms)
        rooms.append(Room(len(rooms), poly, float(md.sum() * cell * cell), [float(x) for x in edges], ch, st, md))

    openings, adj = [], set()
    for (x, y), (w, pa, pb) in info.items():
        rx, ry = find(x), find(y)
        if rx == ry or not (door_w[0] <= w <= door_w[1]):
            continue
        if rx in root_to_id and ry in root_to_id:
            pair = tuple(sorted((root_to_id[rx], root_to_id[ry])))
            a = pa + [ox, oz]
            b = pb + [ox, oz]
            openings.append(refine_opening(
                Opening("door", tuple(map(float, a)), tuple(map(float, b)), w, pair), geom.wall_xz))
            adj.add(pair)
    return Layout(rooms, openings, sorted(adj))