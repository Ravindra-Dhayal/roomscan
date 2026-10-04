# roomscan

LiDAR, photo, and video captures to a stitched floor plan with explicit intervals and drift
diagnostics. LiDAR is metric from sensor depth; RGB tiers require an explicit metric sidecar.

## Quickstart

On a clean machine with Python 3.11 or newer:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m roomscan run data\single_scan_with_ceiling --out out
```

The command writes `out/plan.json` and `out/plan.png`. Drift can be ablated with:

```powershell
.venv\Scripts\python.exe -m roomscan ablate data\single_scan_with_ceiling --out out
```

Photo input is a folder containing one subfolder per room, with 2-8 readable images in each
folder. Video input is a folder containing exactly one `.mov` or `.mp4`. Both use the same run
command. For metric RGB output, add a `scale.json` beside the room folders:

```json
{"rooms": {"kitchen": {"width_m": 4.0, "depth_m": 3.0, "ceiling_height_m": 2.5}}, "video_rooms": [{"name": "kitchen", "start_frame": 0, "end_frame": 180}]}
```

Without that sidecar, the visual tiers still produce a stitched plan, but dimensions are marked
relative and uncalibrated.

Run the tests with `.venv\Scripts\python.exe -m pytest -q`. The benchmark and calibration
commands consume the JSON format shown in `bench/manifest.example.json`.

If the required Apple hardware is unavailable, read `NO_IPHONE_SUBMISSION_PLAN.md` and package
the current truthful prototype with `.venv\Scripts\python.exe scripts\package_submission.py`.

To regenerate video reports and exercise the photo adapter using representative frames from
the checked-in videos, run `.venv\Scripts\python.exe scripts\make_rgb_reports.py`. The derived
photo reports are diagnostic only and do not replace independent still-photo captures for the
assignment benchmark.

## Current limitations

The checked-in captures are LiDAR-only and are not a complete case-study submission. The
repository still needs a controlled three-tier capture set, laser ground truth, calibrated
intervals, consumer-app exports, and a shipped fix-loop before/after. Monocular photos alone do
not determine metric scale without an additional known-size cue; claiming otherwise would make
the photo gate unverifiable.

## Disclosures

No pretrained model or remote API is used by the current LiDAR path. Geometry uses NumPy,
SciPy, OpenCV, scikit-image, and Pydantic.
