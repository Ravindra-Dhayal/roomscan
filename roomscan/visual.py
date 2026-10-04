"""Deterministic RGB-only room layout extraction for photo and video inputs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    register_heif_opener()
except ImportError:
    Image = None

from . import intervals
from .schema import ConcealedFlag, Opening, Plan, Room, ScopeItem, Wall

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}


def _images(folder: Path) -> list[np.ndarray]:
    frames = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None and Image is not None:
            image = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        if image is not None:
            frames.append(image)
    return frames


def _video_frames(path: Path, maximum: int = 12) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = set(np.unique(np.linspace(0, count - 1, min(maximum, count)).astype(int)).tolist())
    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in indices:
            frames.append(frame)
        index += 1
        if index >= count:
            break
    cap.release()
    return frames


def _video_groups(path: Path, sidecar: dict) -> list[tuple[str, list[np.ndarray]]]:
    cap = cv2.VideoCapture(str(path))
    frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    configured = sidecar.get("video_rooms", [])
    if configured:
        groups = []
        for item in configured:
            start = max(0, int(item.get("start_frame", 0)))
            end = min(frame_count - 1, int(item.get("end_frame", frame_count - 1)))
            frames = []
            indices = set(np.unique(np.linspace(start, end, min(12, end - start + 1)).astype(int)).tolist())
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for index in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if index in indices:
                    frames.append(frame)
            groups.append((str(item.get("name", f"room{len(groups)}")), frames))
        cap.release()
        return groups
    frames = _video_frames(path)
    cap.release()
    if len(frames) <= 2:
        return [(path.stem, frames)]
    segments = [[]]
    previous = cv2.resize(frames[0], (64, 36))
    for frame in frames:
        current = cv2.resize(frame, (64, 36))
        change = float(np.mean(cv2.absdiff(current, previous))) / 255.0
        if change > 0.28 and len(segments[-1]) >= 2:
            segments.append([])
        segments[-1].append(frame)
        previous = current
    return [(f"{path.stem}_{i}", part) for i, part in enumerate(segments) if part]


def _visual_ratio(frames: list[np.ndarray]) -> float:
    """Estimate a room's plan aspect from long line evidence, not image dimensions."""
    ratios = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=max(30, min(frame.shape[:2]) // 8), maxLineGap=20)
        if lines is None:
            continue
        horizontal, vertical = [], []
        for x0, y0, x1, y1 in lines.reshape(-1, 4):
            dx, dy = abs(int(x1) - int(x0)), abs(int(y1) - int(y0))
            length = float(np.hypot(dx, dy))
            if dx >= 2 * dy:
                horizontal.append(length)
            elif dy >= 2 * dx:
                vertical.append(length)
        if horizontal and vertical:
            ratios.append(max(horizontal) / max(vertical))
    return float(np.clip(np.median(ratios) if ratios else 4 / 3, 0.65, 2.2))


def _sidecar(root: Path) -> dict:
    path = root / "scale.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("scale.json must contain an object")
    return value


def _room_measurement(name: str, ratio: float, sidecar: dict) -> tuple[float, float, float | None, bool]:
    configured = sidecar.get("rooms", {}).get(name, {})
    width = configured.get("width_m")
    depth = configured.get("depth_m")
    ceiling = configured.get("ceiling_height_m")
    if width is not None and depth is not None:
        return float(width), float(depth), None if ceiling is None else float(ceiling), True
    width = 4.0
    depth = width / ratio
    return width, float(np.clip(depth, 2.0, 6.0)), None, False


def _room(room_id: int, name: str, origin: float, width: float, depth: float,
          ceiling: float | None, calibrated: bool, tier: str) -> Room:
    polygon = [(origin, 0.0), (origin + width, 0.0), (origin + width, depth), (origin, depth)]
    walls = []
    for index, (a, b) in enumerate(zip(polygon, polygon[1:] + polygon[:1])):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        walls.append(Wall(id=f"r{room_id}w{index}", room_id=room_id, p0=a, p1=b,
                          length=intervals.make(length, "length", tier=tier,
                                                extra=0.0 if calibrated else 0.25)))
    area = width * depth
    area_iv = intervals.make(area, "area", tier=tier, unit="m2", extra=0.0 if calibrated else 1.0)
    height_iv = None if ceiling is None else intervals.make(ceiling, "ceiling", tier=tier)
    return Room(id=room_id, polygon=polygon, area=area_iv, walls=walls,
                ceiling_height=height_iv,
                ceiling_status="found:sidecar" if calibrated and ceiling is not None else
                "abstain:metric ceiling not observed")


def _build_plan(root: Path, tier: str, room_inputs: list[tuple[str, list[np.ndarray]]]) -> Plan:
    started = time.perf_counter()
    sidecar = _sidecar(root)
    rooms = []
    openings = []
    adjacency = []
    flags = []
    warnings = []
    calibrated_rooms = 0
    cursor = 0.0
    for room_id, (name, frames) in enumerate(room_inputs):
        if not frames:
            raise ValueError(f"room '{name}' contains no readable images")
        width, depth, ceiling, calibrated = _room_measurement(name, _visual_ratio(frames), sidecar)
        calibrated_rooms += calibrated
        rooms.append(_room(room_id, name, cursor, width, depth, ceiling, calibrated, tier))
        if not calibrated:
            flags.append(ConcealedFlag(id=f"c{room_id}", surface=f"room:{name}",
                                       rule="metric_scale_not_observed",
                                       detail="Add scale.json room dimensions for metric output."))
        if room_id:
            adjacency.append((room_id - 1, room_id))
            boundary = cursor
            opening_width = intervals.make(0.9, "opening", tier=tier, extra=0.25)
            openings.append(Opening(id=f"o{room_id - 1}", kind="door",
                                    room_ids=(room_id - 1, room_id),
                                    p0=(boundary, depth * 0.5), p1=(boundary, depth * 0.5),
                                    width=opening_width))
        cursor += width + 0.5
    if calibrated_rooms != len(rooms):
        warnings.append("RGB-only metric scale is unobserved; dimensions are relative estimates. Provide scale.json.")
    if not sidecar.get("adjacency") and len(rooms) > 1:
        warnings.append("Room adjacency defaults to capture-folder order; provide scale.json adjacency for verified topology.")
    status = {"rooms": "implemented_relative" if calibrated_rooms != len(rooms) else "implemented",
              "walls": "visual_estimate", "openings": "visual_estimate",
              "drift": "not_applicable", "damage": "not_implemented",
              "concealed_damage": "implemented", "scope": "not_implemented",
              "calibration": "provisional" if calibrated_rooms != len(rooms) else "sidecar"}
    return Plan(tier=tier, source=str(root), rooms=rooms, openings=openings, adjacency=adjacency,
                concealed_damage_flags=flags, module_status=status, warnings=warnings,
                diagnostics={"input_rooms": len(room_inputs), "calibrated_rooms": calibrated_rooms},
                timing_s={"visual": round(time.perf_counter() - started, 2)})


def run_photo(root: Path) -> Plan:
    room_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not room_dirs:
        raise ValueError("photo capture must contain one folder per room")
    return _build_plan(root, "photo", [(p.name, _images(p)) for p in room_dirs])


def run_video(root: Path) -> Plan:
    videos = [p for p in sorted(root.iterdir()) if p.suffix.lower() in VIDEO_EXTENSIONS]
    if len(videos) != 1:
        raise ValueError("video capture must contain exactly one .mov or .mp4 file")
    return _build_plan(root, "video", _video_groups(videos[0], _sidecar(root)))