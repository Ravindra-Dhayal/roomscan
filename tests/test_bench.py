import json

import numpy as np
import pytest

from roomscan import bench, calibrate, cli, intervals
from roomscan.schema import Interval, Opening, Plan, Room, Wall


def iv(v, hw=0.03, unit="m"):
    return Interval(value=v, lo=v - hw, hi=v + hw, unit=unit)


def room(i, walls, ceil, area=None, hw=0.03):
    n = len(walls)
    ws = [Wall(id=f"r{i}w{k}", room_id=i, p0=(0, 0), p1=(1, 0), length=iv(w, hw)) for k, w in enumerate(walls)]
    a = area or walls[0] * walls[1]
    return Room(id=i, polygon=[(0, 0), (1, 0), (1, 1)], area=iv(a, 0.5, "m2"), walls=ws,
                ceiling_height=None if ceil is None else iv(ceil, 0.02), ceiling_status="found" if ceil else "abstain")


def plan(rooms, openings=(), adj=(), tier="lidar"):
    ops = [Opening(id=f"o{i}", kind=k, room_ids=None, p0=(0, 0), p1=(1, 0), width=iv(w, 0.05))
           for i, (k, w) in enumerate(openings)]
    return Plan(tier=tier, source="x", rooms=rooms, openings=ops, adjacency=list(adj))


GT = {"rooms": [{"name": "bed", "ceiling_height": 2.70, "walls": [3.5, 3.0, 3.5, 3.0]},
                {"name": "hall", "ceiling_height": 2.70, "walls": [4.2, 1.1, 4.2, 1.1]}],
      "openings": [{"kind": "door", "width": 0.82}, {"kind": "door", "width": 0.90}],
      "adjacency": [["bed", "hall"]]}


def test_matching_is_order_independent_and_errors_signed():
    p = plan([room(0, [4.21, 1.10, 4.19, 1.11], 2.705), room(1, [3.51, 3.0, 3.49, 3.01], 2.695)],
             [("door", 0.83), ("door", 0.90)], [(0, 1)])
    r = bench.compare(p, GT, "c")
    assert {n for n, _, _ in r.room_match} == {"bed", "hall"}
    assert max(abs(w["err"]) for w in r.walls) < 0.02
    assert [round(o["err"], 3) for o in r.openings] == [0.01, 0.0] or [round(o["err"], 3) for o in r.openings] == [0.0, 0.01]
    assert r.adjacency["recall"] == 1.0 and r.adjacency["precision"] == 1.0


def test_missed_and_phantom_openings_both_count_as_misses():
    p = plan([room(0, [3.5, 3.0, 3.5, 3.0], 2.70), room(1, [4.2, 1.1, 4.2, 1.1], 2.70)],
             [("door", 0.82), ("window", 1.2)])               # 2nd door missed, window is a phantom
    r = bench.compare(p, GT)
    g = bench.gate_report([r])["lidar"]["opening_width_hit_rate"]
    assert g["value"] == pytest.approx(1 / 3) and g["passed"] is False
    assert "missed=1" in g["detail"] and "phantom=1" in g["detail"]


def test_ceiling_abstain_is_a_miss_when_gt_has_ceiling():
    p = plan([room(0, [3.5, 3.0, 3.5, 3.0], None), room(1, [4.2, 1.1, 4.2, 1.1], 2.70)])
    g = bench.gate_report([bench.compare(p, GT)])["lidar"]["ceiling_within_1.5cm"]
    assert g["value"] == 0.5 and g["passed"] is False and "abstained=1" in g["detail"]


def test_repeatable_but_biased_vs_unrepeatable():
    base = [room(0, [3.5, 3.0, 3.5, 3.0], 2.75), room(1, [4.2, 1.1, 4.2, 1.1], 2.75)]
    again = [room(0, [3.505, 3.0, 3.5, 3.0], 2.752), room(1, [4.2, 1.1, 4.2, 1.1], 2.752)]
    rep = bench.repeatability(bench.compare(plan(base), GT), bench.compare(plan(again), GT))
    assert rep["ceiling_diagnosis"] == "repeatable-but-biased" and rep["walls_passed"] is True
    noisy = [room(0, [3.5, 3.0, 3.5, 3.0], 2.80), room(1, [4.2, 1.1, 4.2, 1.1], 2.80)]
    rep2 = bench.repeatability(bench.compare(plan(base), GT), bench.compare(plan(noisy), GT))
    assert rep2["ceiling_diagnosis"] == "unrepeatable"


def test_video_wall_tolerance_gate():
    p = plan([room(0, [3.5 * 1.05, 3.0, 3.5, 3.0], 2.7), room(1, [4.2, 1.1, 4.2, 1.1], 2.7)], tier="video")
    assert bench.gate_report([bench.compare(p, GT)])["video"]["wall_length_within_tol"]["passed"] is False


def test_conformal_quantile_and_calibrated_intervals(tmp_path, monkeypatch):
    assert calibrate.conformal_quantile([0.01] * 19 + [0.05]) == 0.05           # n=20 -> ceil(21*.95)=20 -> max
    assert calibrate.conformal_quantile(list(np.linspace(0, 1, 100))) == pytest.approx(0.96, abs=0.01)   # 96th of 100
    p = plan([room(0, [3.5 + 0.02, 3.0, 3.5, 3.0], 2.70), room(1, [4.2, 1.1, 4.2, 1.1], 2.70)])
    cal = calibrate.fit([bench.compare(p, GT)])
    assert cal["lidar"]["length"]["small_sample"] is True
    f = tmp_path / "cal.json"
    calibrate.save(cal, f)
    intervals.load_calibration(f)
    x = intervals.make(3.0, "length")
    assert x.calibrated and x.method.startswith("split_conformal")
    intervals.load_calibration(tmp_path / "missing.json")
    assert intervals.make(3.0, "length").calibrated is False


def test_manifest_end_to_end(tmp_path):
    (tmp_path / "gt.json").write_text(json.dumps(GT))
    for n in ("a", "b"):
        (tmp_path / n).mkdir()
    (tmp_path / "m.json").write_text(json.dumps({"captures": [
        {"id": "a", "path": "a", "ground_truth": "gt.json"},
        {"id": "b", "path": "b", "ground_truth": "gt.json", "repeat_of": "a"}]}))
    fake = lambda p: plan([room(0, [3.5, 3.0, 3.5, 3.0], 2.70), room(1, [4.2, 1.1, 4.2, 1.1], 2.70)],
                          [("door", 0.82), ("door", 0.90)], [(0, 1)])
    rep = bench.run_manifest(tmp_path / "m.json", fake)
    md = bench.to_markdown(rep)
    assert "opening_width_hit_rate" in md and "a~b" in rep["repeatability"]
    assert rep["gates"]["lidar"]["opening_width_hit_rate"]["passed"]