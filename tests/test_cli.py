import json
from pathlib import Path

import pytest

from roomscan import cli, pipeline
from roomscan.schema import Plan

DATA = Path(__file__).resolve().parents[1] / "data"


def test_schema_command_writes_valid_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["schema"]) == 0
    s = json.loads((tmp_path / "schema" / "plan.schema.json").read_text())
    assert s["title"] == "Plan" and "rooms" in s["properties"]


def test_tier_detection(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "odometry.csv").write_text("x")
    assert pipeline.detect_tier(tmp_path / "a") == "lidar"
    (tmp_path / "v").mkdir()
    (tmp_path / "v" / "walk.mp4").write_text("x")
    assert pipeline.detect_tier(tmp_path / "v") == "video"
    (tmp_path / "p" / "kitchen").mkdir(parents=True)
    assert pipeline.detect_tier(tmp_path / "p") == "photo"


def test_unimplemented_tier_fails_cleanly(tmp_path, capsys):
    (tmp_path / "v").mkdir()
    (tmp_path / "v" / "walk.mp4").write_text("x")
    assert cli.main(["run", str(tmp_path / "v"), "--out", str(tmp_path / "o")]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_bad_path_exit_code(tmp_path):
    (tmp_path / "empty").mkdir()
    assert cli.main(["run", str(tmp_path / "empty"), "--out", str(tmp_path / "o")]) != 0


def test_real_run_outputs_valid_plan(tmp_path):
    d = DATA / "single_scan_with_ceiling"
    if not d.exists():
        pytest.skip("data not present")
    assert cli.main(["run", str(d), "--out", str(tmp_path)]) == 0
    plan = Plan.model_validate_json((tmp_path / "plan.json").read_text())
    assert len(plan.rooms) >= 4 and plan.tier == "lidar"
    assert (tmp_path / "plan.png").stat().st_size > 10_000
    for r in plan.rooms:
        assert r.area.lo < r.area.value < r.area.hi
        for w in r.walls:
            assert w.length.lo < w.length.value < w.length.hi
    assert plan.module_status["windows"] == "not_implemented"      # no silent claims