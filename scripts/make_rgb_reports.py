"""Generate reproducible video and derived-photo reports from the checked-in RGB clips.

Derived photos are useful for exercising the photo adapter, but they are not a substitute for
independent iPhone still captures in the assignment benchmark.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roomscan import pipeline  # noqa: E402
from roomscan.render import render_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DERIVED = ROOT / "capture" / "derived_photo"


def video_for(capture: Path) -> Path:
    videos = list(capture.rglob("rgb.mp4"))
    if len(videos) != 1:
        raise FileNotFoundError(f"expected one rgb.mp4 under {capture}")
    return videos[0]


def extract_stills(video: Path, destination: Path, count: int = 8) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    import numpy as np
    targets = set(np.unique(np.linspace(0, total - 1, count).astype(int)).tolist())
    index = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in targets:
            cv2.imwrite(str(destination / f"{saved:02d}.jpg"), frame)
            saved += 1
        index += 1
        if index >= total:
            break
    cap.release()


def write_plan(name: str, plan, prefix: str) -> None:
    out = REPORTS / f"{prefix}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, str(out / "plan.png"))


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    for capture in sorted(p for p in (ROOT / "data").iterdir() if p.is_dir()):
        name = capture.name
        video = video_for(capture)
        write_plan(name, pipeline.run(capture, tier="video"), "video")
        room = DERIVED / name / "room"
        extract_stills(video, room)
        photo_plan = pipeline.run(DERIVED / name, tier="photo")
        photo_plan.warnings.append("Derived from video frames; not an independent photo-tier capture.")
        write_plan(name, photo_plan, "photo_derived")
        print(json.dumps({"capture": name, "video": str(video), "photo_frames": len(list(room.glob("*.jpg"))) }))


if __name__ == "__main__":
    main()