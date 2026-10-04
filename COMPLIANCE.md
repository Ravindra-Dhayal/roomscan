| Requirement | File path | Artifact | Status |
|---|---|---|---|
| One command per capture | roomscan/cli.py | `python -m roomscan run <dir>` | DONE (all detected tiers; RGB metric output requires sidecar) |
| Capture route protocol | CAPTURE_PROTOCOL.md | Stock Stray Scanner and RGB hand-off protocol | PARTIAL (metric RGB calibration still required) |
| Device matrix | DEVICE_MATRIX.md | tier/hardware/accuracy table | PARTIAL (visual gates not benchmarked) |
| JSON to published schema | roomscan/schema.py, schema/plan.schema.json | out/plan.json | DONE (validated in tests) |
| Rendered plan | roomscan/render.py | out/plan.png | DONE |
| Per-room walls, ceiling, area, openings | roomscan/lidar.py, roomscan/rooms.py | plan.json rooms/openings | PARTIAL (doors yes, windows no) |
| Stitched multi-room plan, adjacency | roomscan/rooms.py | plan.json adjacency | PARTIAL (rooms placed from one scan) |
| Confidence interval on every measurement | roomscan/intervals.py | Interval objects | PROVISIONAL (calibrated=false) |
| Damage regions, concealed flags, scope | roomscan/damage.py, roomscan/pipeline.py | plan.json damage_regions/concealed_damage_flags/scope_items | PARTIAL (geometric LiDAR anomalies; no RGB/thermal class model) |
| Drift handling + ablation | roomscan/drift.py, `python -m roomscan ablate <dir>` | reports/drift_ablation_*.png | DONE (pose graph gated by held-out validation + floor-height anchor) |
| Video tier | roomscan/visual.py, roomscan/pipeline.py | `plan.json` from `.mov`/`.mp4` | PARTIAL (visual estimate and scene grouping; metric calibration required) |
| Photo tier | roomscan/visual.py, roomscan/pipeline.py | `plan.json` from per-room image folders | PARTIAL (folder stitching; metric calibration required) |
| Benchmark gates, repeatability table | roomscan/bench.py, `python -m roomscan bench` | bench/results.md | HARNESS DONE, no real ground truth yet |
| Calibration (interval widths) | roomscan/calibrate.py, `python -m roomscan calibrate` | calibration.json | HARNESS DONE, not fitted on real data yet |
| Head-to-head vs consumer app | roomscan/head_to_head.py, roomscan/cli.py | `reports/head_to_head.md/.json` | TOOLING DONE; consumer export required |
| Fix loop | roomscan/fixloop.py, roomscan/cli.py, fixloop/DECLARATION.md | `fixloop/results.md/.json` | TOOLING DONE; raw before/after manifests and shipped fix required |
| Technical report | TECHNICAL_REPORT.md | architecture, tiers, drift, calibration, limitations | DONE (evidence-dependent claims marked) |
| Reproduction bundle instructions | bench/README.md | raw data/GT/manifest procedure | DONE; raw benchmark bundle not supplied |
| Submission readiness audit | roomscan/audit.py, roomscan/cli.py | `reports/submission_audit.md/.json` | DONE (reports external blockers) |
| No-hardware submission route | NO_IPHONE_SUBMISSION_PLAN.md, scripts/package_submission.py | truthful prototype archive and hardware acquisition plan | DONE (full walk-in evidence still unavailable) |