# Benchmark Bundle

Add one manifest entry per capture. Every entry must retain its raw capture directory and a
laser/tape ground-truth JSON. Include at minimum:

- one three-room-plus-connector capture;
- one furnished room with two staged damage classes;
- the same rooms at photo, video, and LiDAR tiers;
- one repeated room at the same tier;
- raw measurements for every wall, opening, ceiling, and footprint;
- the incumbent app export for two LiDAR rooms.

Use `ground_truth_template.json` as the starting shape. Measure inside wall faces, record the
measurement method and operator in the ground-truth file, and do not derive truth from roomscan
outputs.

Run and fit only after the bundle is complete:

```powershell
.venv\Scripts\python.exe -m roomscan bench bench\manifest.json --out bench\results
.venv\Scripts\python.exe -m roomscan calibrate bench\manifest.json --out calibration.json
```

The checked-in example manifest is a schema example, not a valid scored benchmark, because its
capture paths are intentionally absent.
