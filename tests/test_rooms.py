from pathlib import Path

import numpy as np
import pytest

from roomscan import lidar, rooms
from roomscan.ingest import load_capture

DATA = Path(__file__).resolve().parents[1] / "data"


def two_room_flat(door_w=0.9, H=2.5, yaw_deg=15.0, seed=0):
    """Two 4 x 3.5 m rooms side by side, shared wall at x=4 with a door of door_w."""
    rng = np.random.default_rng(seed)
    y0 = -1.5
    def plane(n, f):
        return np.array([f(*rng.random(2)) for _ in range(n)])
    parts = [plane(120000, lambda a, b: (a * 8.0, y0, b * 3.5)),
             plane(60000, lambda a, b: (a * 8.0, y0 + H, b * 3.5)),
             plane(60000, lambda a, b: (a * 8.0, y0 + b * H, 0.0)),
             plane(60000, lambda a, b: (a * 8.0, y0 + b * H, 3.5)),
             plane(40000, lambda a, b: (0.0, y0 + b * H, a * 3.5)),
             plane(40000, lambda a, b: (8.0, y0 + b * H, a * 3.5))]
    # shared wall with a doorway centred at z = 1.75
    for _ in range(1):
        z = rng.random(70000) * 3.5
        keep = np.abs(z - 1.75) > door_w / 2
        yy = y0 + rng.random(70000) * H
        parts.append(np.stack([np.full(keep.sum(), 4.0), yy[keep], z[keep]], 1))
    p = np.concatenate(parts) + rng.normal(0, 0.005, (sum(len(x) for x in parts), 3))
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return lidar.voxel_downsample(p @ R.T + [2.0, 0.0, 1.0], 0.02)


def test_two_rooms_one_door():
    pts = two_room_flat(door_w=0.9)
    L = rooms.segment_rooms(pts, lidar.analyze(pts))
    assert len(L.rooms) == 2
    assert L.adjacency == [(0, 1)]
    assert len(L.openings) == 1 and L.openings[0].kind == "door"
    assert L.openings[0].width == pytest.approx(0.9, abs=0.05)
    for r in L.rooms:
        assert r.area == pytest.approx(14.0, rel=0.12)
        assert r.ceiling_height == pytest.approx(2.5, abs=0.02)


def test_no_door_means_not_adjacent():
    pts = two_room_flat(door_w=0.05)
    L = rooms.segment_rooms(pts, lidar.analyze(pts))
    assert L.adjacency == []


def test_real_multiroom_smoke():
    d = DATA / "single_scan_with_ceiling"
    if not d.exists():
        pytest.skip("data not present")
    pts = lidar.fuse(load_capture(d))
    L = rooms.segment_rooms(pts, lidar.analyze(pts))
    assert 4 <= len(L.rooms) <= 14
    assert len(L.adjacency) >= 3
    ids = {i for pair in L.adjacency for i in pair}
    assert ids <= {r.id for r in L.rooms}