| Requirement | File path | Artifact | Status |
|---|---|---|---|
| One command per capture | roomscan/cli.py | `python -m roomscan run <dir>` | DONE (lidar tier only) |
| JSON to published schema | roomscan/schema.py, schema/plan.schema.json | out/plan.json | DONE (validated in tests) |
| Rendered plan | roomscan/render.py | out/plan.png | DONE |
| Per-room walls, ceiling, area, openings | roomscan/lidar.py, roomscan/rooms.py | plan.json rooms/openings | PARTIAL (doors yes, windows no) |
| Stitched multi-room plan, adjacency | roomscan/rooms.py | plan.json adjacency | PARTIAL (rooms placed from one scan) |
| Confidence interval on every measurement | roomscan/intervals.py | Interval objects | PROVISIONAL (calibrated=false) |
| Damage regions, concealed flags, scope | - | - | TODO |
| Drift handling + ablation | - | - | TODO |
| Video tier | - | - | TODO |
| Photo tier | - | - | TODO |
| Benchmark, repeatability, head-to-head | bench/ | - | TODO |
| Fix loop | fixloop/ | - | TODO |