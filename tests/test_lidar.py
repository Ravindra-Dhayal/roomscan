from pathlib import Path

import numpy as np
import pytest

from roomscan import lidar
from roomscan.ingest import load_capture

DATA = Path(__file__).resolve().parents[1] / "data"


def synthetic_room(L=4.0, Wd=3.0, H=2.5, ceiling=True, yaw_deg=20.0, noise=0.005, seed=0):
    """Box room with floor at y=-1.5, a table block (clutter) and optional ceiling."""
    rng = np.random.default_rng(seed)
    def plane(n, f):
        return np.array([f(*rng.random(2)) for _ in range(n)])
    y0 = -1.5
    parts = [plane(60000, lambda a, b: (a * L, y0, b * Wd))]                         # floor
    if ceiling:
        parts.append(plane(30000, lambda a, b: (a * L, y0 + H, b * Wd)))             # ceiling
    parts += [plane(40000, lambda a, b: (a * L, y0 + b * H, 0.0)),                   # walls
              plane(40000, lambda a, b: (a * L, y0 + b * H, Wd)),
              plane(30000, lambda a, b: (0.0, y0 + b * H, a * Wd)),
              plane(30000, lambda a, b: (L, y0 + b * H, a * Wd))]
    parts.append(plane(8000, lambda a, b: (1.0 + a * 0.8, y0 + 0.75, 1.0 + b * 0.8)))  # table top
    p = np.concatenate(parts) + rng.normal(0, noise, (sum(len(x) for x in parts), 3))
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return lidar.voxel_downsample(p @ R.T + [3.0, 0.0, -2.0], 0.02)


def test_ceiling_height_within_gate():
    g = lidar.analyze(synthetic_room(H=2.5))
    v = g.vertical
    assert v.ceiling_status == "found"
    assert v.ceiling_height == pytest.approx(2.5, abs=0.01)
    assert v.ceiling_height_sigma < 0.02


def test_ceiling_abstains_when_absent():
    v = lidar.analyze(synthetic_room(ceiling=False)).vertical
    assert v.ceiling is None and v.ceiling_height is None
    assert v.ceiling_status.startswith("abstain")


def test_orientation_and_walls():
    g = lidar.analyze(synthetic_room(L=4.0, Wd=3.0, yaw_deg=20.0))
    # the synthetic yaw rotates (x, z) by -20 degrees; wall angle is defined modulo 90
    d = (np.degrees(g.theta) + 20.0 + 45) % 90 - 45
    assert abs(d) < 1.0
    lengths = sorted(w.length for w in g.walls)
    assert len(g.walls) <= 6                                  # table must not become walls
    for target in (4.0, 4.0, 3.0, 3.0):
        j = int(np.argmin([abs(x - target) for x in lengths]))
        assert abs(lengths[j] - target) < 0.1


def test_footprint_area():
    g = lidar.analyze(synthetic_room(L=4.0, Wd=3.0))
    assert g.footprint.area == pytest.approx(12.0, rel=0.08)


@pytest.mark.parametrize("name,sub,has_ceiling", [
    ("single_scan_with_ceiling", "c7d28f72c6", True),
    ("single_scan_floor_only", "1a8384c3f6", False),
])
def test_real_ceiling_behaviour(name, sub, has_ceiling):
    d = DATA / name
    if not d.exists():
        pytest.skip("data not present")
    v = lidar.analyze(lidar.fuse(load_capture(d))).vertical
    assert (v.ceiling is not None) == has_ceiling
    if has_ceiling:
        assert 2.4 < v.ceiling_height < 3.4