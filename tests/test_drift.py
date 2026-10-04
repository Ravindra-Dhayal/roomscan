from pathlib import Path

import numpy as np
import pytest

from roomscan import drift
from roomscan.drift import _compose, _inv, edge_residual, icp2d, solve_graph
from roomscan.ingest import load_capture

DATA = Path(__file__).resolve().parents[1] / "data"


def test_se2_inverse_roundtrip():
    a = np.array([[0.3, 1.0, -2.0], [-0.1, 0.5, 0.2]])
    out = _compose(a, _inv(a))
    assert np.allclose(out, 0, atol=1e-12)


def _l_room(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.random(n)
    walls = [np.c_[t * 5, np.zeros(n)], np.c_[np.zeros(n), t * 3], np.c_[t * 5, np.full(n, 3.0)], np.c_[np.full(n, 5.0), t * 3]]
    return np.concatenate(walls) + rng.normal(0, 0.004, (4 * n, 2))


def test_icp_recovers_known_transform():
    dst = _l_room()
    th, t = np.radians(2.0), np.array([0.12, -0.08])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    src = (dst - t) @ R                     # src @ R.T + t == dst  (R orthogonal)
    T, frac, rmse = icp2d(src, dst)
    assert abs(T[0] - th) < np.radians(0.2)
    assert np.hypot(T[1] - t[0], T[2] - t[1]) < 0.01
    assert frac > 0.9 and rmse < 0.01


def test_pose_graph_removes_injected_drift():
    n = 30
    k = np.arange(n)
    D_true = np.stack([0.002 * k, 0.01 * k, -0.006 * k], 1)          # slowly accumulating drift
    edges = []
    for i in range(n):
        for j in range(i + 8, n, 3):
            edges.append((i, j, _compose(_inv(D_true[i]), D_true[j]), 1.0))
    D = solve_graph(n, edges)
    before = edge_residual(np.zeros((n, 3)), edges)
    after = edge_residual(D, edges)
    assert before > 0.1 and after < 0.3 * before
    assert np.hypot(D[-1, 1] - D_true[-1, 1], D[-1, 2] - D_true[-1, 2]) < 0.5 * np.hypot(*D_true[-1, 1:])


def test_real_scan_ablation_does_no_harm():
    d = DATA / "single_room"
    if not d.exists():
        pytest.skip("data not present")
    r = drift.ablation(load_capture(d))
    assert r["on"]["floor_sigma_m"] <= r["off"]["floor_sigma_m"] * 1.05
    assert r["on"]["wall_thickness_m"] <= r["off"]["wall_thickness_m"] * 1.05
    assert r["report"].notes or r["report"].applied       # a rejection must always be explained