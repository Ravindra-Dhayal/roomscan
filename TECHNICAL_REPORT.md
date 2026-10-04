# Roomscan Technical Report

## 1. Scope and architecture

Roomscan accepts one capture directory and emits one Pydantic `Plan`, JSON, and a rendered
floor plan. The pipeline has three input adapters: LiDAR depth plus poses, per-room RGB photo
folders, and one RGB video file. LiDAR fuses depth in world coordinates, estimates floor,
ceiling, Manhattan orientation, walls, footprint, rooms, and openings. RGB inputs use line
features for relative aspect estimates and can consume `scale.json` room dimensions.

No remote service is called. The LiDAR path uses NumPy, SciPy, OpenCV, scikit-image, and
Pydantic. HEIC photo decoding is provided by Pillow and pillow-heif.

## 2. Tier design and device matrix

LiDAR requires a Pro-class iPhone/iPad export containing depth, confidence, poses, intrinsics,
and IMU. Photos require 2-8 images in one folder per room. Video requires one `.mov` or `.mp4`.
The RGB path is deliberately marked provisional without metric scale because monocular imagery
has a scale ambiguity. The exact hardware and claims are recorded in `DEVICE_MATRIX.md`.

## 3. Drift handling

LiDAR trajectories are divided into approximately one-metre nodes. Candidate loop closures are
found by temporal separation and spatial proximity, aligned with trimmed 2-D ICP, and accepted
only when held-out closure residual improves by at least 15%. A floor-height anchor corrects
small vertical drift but abstains on multi-level scans. `roomscan ablate` renders the footprint
with correction disabled and enabled. A rejected correction is reported instead of silently
using an unvalidated pose graph.

## 4. Error budget and calibration

Every dimension is represented by an `Interval`. Before benchmark fitting, widths are explicit
provisional heuristics and `calibrated=false`. `roomscan calibrate` fits split-conformal widths
per tier and measurement type from held-out ground truth. The benchmark harness measures wall,
ceiling, opening, footprint, adjacency, interval coverage, and repeatability gates from the
assignment. It does not manufacture ground truth; laser/tape measurements must be added to the
manifest.

## 5. Damage and scope

The current LiDAR damage module reports only dense geometric wall irregularities and emits an
inspection scope item. Missing ceiling or surface evidence produces a concealed-damage flag with
the rule that fired. RGB damage classification, staged two-class damage, and thermal/texture
reasoning remain unsupported and must not be claimed as benchmark results.

## 6. Fix loop and comparison

`roomscan fixloop BEFORE AFTER --out fixloop/results` regenerates both manifests and writes JSON
and Markdown gate diffs. `roomscan head2head OURS INCUMBENT --ground-truth TRUTH` compares errors
against shared truth and writes the required table. `roomscan audit` reports whether the external
evidence bundle is complete. The repository still needs raw repeat captures, laser measurements,
and a consumer-app export before a truthful before/after prediction can be filled in.

## Known failure modes

Mirrors, glass, low light, closed doors, sparse ceilings, furniture occlusion, and unobserved
RGB scale can cause abstention or wide intervals. The output warning and module status fields
are part of the contract; callers must not treat provisional intervals as accuracy guarantees.
