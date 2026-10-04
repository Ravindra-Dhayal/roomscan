import json

import cv2
import numpy as np

from roomscan import pipeline


def _photo(path):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (290, 210), (255, 255, 255), 3)
    cv2.imwrite(str(path), image)


def test_photo_folders_stitch_and_use_sidecar_dimensions(tmp_path):
    for name in ("kitchen", "hall"):
        room = tmp_path / name
        room.mkdir()
        _photo(room / "001.jpg")
        _photo(room / "002.jpg")
    (tmp_path / "scale.json").write_text(json.dumps({"rooms": {
        "kitchen": {"width_m": 4.0, "depth_m": 3.0, "ceiling_height_m": 2.5},
        "hall": {"width_m": 2.0, "depth_m": 5.0, "ceiling_height_m": 2.5},
    }}))

    plan = pipeline.run(tmp_path)

    assert plan.tier == "photo"
    assert len(plan.rooms) == 2
    assert plan.adjacency == [(0, 1)]
    assert {room.area.value for room in plan.rooms} == {10.0, 12.0}
    assert plan.rooms[1].walls[0].length.calibrated is False
    assert all(room.ceiling_height.value == 2.5 for room in plan.rooms)
    assert plan.module_status["rooms"] == "implemented"


def test_photo_without_scale_is_explicitly_provisional(tmp_path):
    room = tmp_path / "room"
    room.mkdir()
    _photo(room / "001.jpg")

    plan = pipeline.run(tmp_path)

    assert plan.module_status["rooms"] == "implemented_relative"
    assert any("scale" in warning for warning in plan.warnings)
    assert plan.concealed_damage_flags[0].rule == "metric_scale_not_observed"


def test_short_video_produces_plan(tmp_path):
    video = tmp_path / "walk.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
    for index in range(8):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (30 + index, 30), (290, 210), (255, 255, 255), 3)
        writer.write(image)
    writer.release()

    plan = pipeline.run(tmp_path, tier="video")

    assert plan.tier == "video"
    assert len(plan.rooms) == 1
    assert plan.rooms[0].area.unit == "m2"