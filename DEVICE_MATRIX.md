# Device Matrix

| Input tier | Hardware | Capture artifact | Current status | Honest accuracy claim |
|---|---|---|---|---|
| LiDAR | iPhone Pro/iPad Pro with LiDAR, Stray Scanner export | depth, confidence, poses, intrinsics, IMU | Runs end to end | Benchmark values are reported by `roomscan bench`; checked-in intervals are provisional until calibration |
| Video | iPhone 15 or newer | handheld `.mov`/`.mp4` | Runs with visual estimates | Metric scale requires `scale.json`; fallback is provisional and uncalibrated |
| Photos | iPhone 15 or newer, 2-8 stills per room | one folder per room | Runs with visual estimates and folder-order stitching | Metric scale requires `scale.json`; fallback is provisional and uncalibrated |

The walk-in accuracy gates are not yet demonstrated for the visual tiers. This matrix prevents
the LiDAR result from being generalized to thinner sensor input.