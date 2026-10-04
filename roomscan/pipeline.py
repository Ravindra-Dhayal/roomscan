"""One capture in, one Plan out."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import dataclasses

from . import damage, drift, intervals, lidar, rooms, visual
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


def _video_root(path: Path) -> Path:
    if any(path.glob("*.mp4")) or any(path.glob("*.mov")):
        return path
    nested = [p for p in path.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".m4v", ".avi"}]
    if len(nested) == 1:
        return nested[0].parent
    raise ValueError(f"cannot find exactly one video under {path}")


def _ccw(poly: np.ndarray) -> np.ndarray:
    x, z = poly[:, 0], poly[:, 1]
    return poly if 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z) > 0 else poly[::-1]


def run_lidar(path: Path, drift_on: bool = True) -> Plan:
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    cap = load_capture(path)
    t["load"] = time.perf_counter() - t0
    rep = None
    if drift_on:
        t0 = time.perf_counter()
        cap, rep = drift.correct(cap)
        t["drift"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pts = lidar.fuse(cap)
    t["fuse"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    geom = lidar.analyze(pts)
    layout = rooms.segment_rooms(pts, geom)
    damage_result = damage.analyze(pts, geom)
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
    warnings = []
    if any(not r.area.calibrated for r in out_rooms):
        warnings.append("Confidence intervals are provisional (calibrated=false) until calibrate.py is fitted.")
    if rep is None:
        warnings.append("Drift handling disabled (--no-drift): poses used as-is.")
    else:
        warnings += rep.notes
    status = {"rooms": "implemented", "doors": "implemented", "windows": "not_implemented",
              "drift": "implemented" if drift_on else "disabled", "damage": damage_result.status,
              "concealed_damage": "implemented", "scope": "implemented",
              "calibration": "provisional"}
    if not out_rooms:
        warnings.append("No rooms found.")
    return Plan(tier="lidar", source=str(path), rooms=out_rooms, openings=out_open,
                adjacency=layout.adjacency, module_status=status, warnings=warnings,
                                damage_regions=damage_result.regions, concealed_damage_flags=damage_result.concealed,
                                scope_items=damage_result.scope,
                diagnostics={"drift": dataclasses.asdict(rep) if rep else None},
                timing_s={k: round(v, 2) for k, v in t.items()})


def run(path: str | Path, drift_on: bool = True, tier: str | None = None) -> Plan:
    path = Path(path)
    selected = tier or detect_tier(path)
    if selected == "lidar":
        return run_lidar(path, drift_on)
    if selected == "photo":
        return visual.run_photo(path)
    if selected == "video":
        return visual.run_video(_video_root(path))
    raise ValueError(f"unsupported input tier: {selected}")