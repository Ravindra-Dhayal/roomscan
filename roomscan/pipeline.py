from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import intervals, lidar, rooms
from .ingest import load_capture
from .schema import Opening, Plan, Room, Wall


def detect_tier(path: Path) -> str:
    if (path / "odometry.csv").exists() or any((p / "odometry.csv").exists() for p in path.iterdir() if p.is_dir()):
        return "lidar"
    if any(path.glob("*.mp4")) or any(path.glob("*.mov")):
        return "video"
    if any(p.is_dir() for p in path.iterdir()):
        return "photo"
    raise ValueError(f"cannot tell the input tier of {path}")


def _ccw(poly: np.ndarray) -> np.ndarray:
    x, z = poly[:, 0], poly[:, 1]
    return poly if 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z) > 0 else poly[::-1]


def run_lidar(path: Path) -> Plan:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    cap = load_capture(path)
    t["load"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pts = lidar.fuse(cap)
    t["fuse"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    geom = lidar.analyze(pts)
    layout = rooms.segment_rooms(pts, geom)
    t["geometry"] = time.perf_counter() - t0

    out_rooms = []
    for r in layout.rooms:
        poly = _ccw(r.polygon)
        walls = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            walls.append(Wall(id=f"r{r.id}w{k}", room_id=r.id, p0=tuple(map(float, a)), p1=tuple(map(float, b)),
                              length=intervals.make(float(np.linalg.norm(b - a)), "length")))
        ch = None if r.ceiling_height is None else intervals.make(r.ceiling_height, "ceiling")
        out_rooms.append(Room(id=r.id, polygon=[tuple(map(float, p)) for p in poly],
                              area=intervals.make(r.area, "area", unit="m2"), walls=walls,
                              ceiling_height=ch, ceiling_status=r.ceiling_status))
    out_open = [Opening(id=f"o{i}", kind=o.kind, room_ids=o.rooms, p0=o.p0, p1=o.p1,
                        width=intervals.make(o.width, "opening")) for i, o in enumerate(layout.openings)]
    warnings = ["Confidence intervals are provisional (calibrated=false) until calibrate.py is fitted.",
                "No drift correction applied yet."]
    if not out_rooms:
        warnings.append("No rooms found.")
    status = {"rooms": "implemented", "doors": "implemented", "windows": "not_implemented",
              "drift": "not_implemented", "damage": "not_implemented", "scope": "not_implemented",
              "calibration": "provisional"}
    return Plan(tier="lidar", source=str(path), rooms=out_rooms, openings=out_open,
                adjacency=layout.adjacency, module_status=status, warnings=warnings,
                timing_s={k: round(v, 2) for k, v in t.items()})


def run(path: str | Path) -> Plan:
    path = Path(path)
    tier = detect_tier(path)
    if tier == "lidar":
        return run_lidar(path)
    raise NotImplementedError(f"tier '{tier}' is not implemented yet (only 'lidar' is)")