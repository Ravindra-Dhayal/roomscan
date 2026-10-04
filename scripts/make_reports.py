import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roomscan import drift, pipeline  # noqa: E402
from roomscan.ingest import load_capture  # noqa: E402
from roomscan.render import render_plan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
names = sys.argv[1:] or sorted(p.name for p in (ROOT / "data").iterdir() if p.is_dir())
(ROOT / "reports").mkdir(exist_ok=True)
for n in names:
    cap_dir = ROOT / "data" / n
    print(f"== {n}")
    r = drift.ablation(load_capture(cap_dir), str(ROOT / "reports" / f"drift_ablation_{n}.png"))
    r["report"] = dataclasses.asdict(r["report"])
    (ROOT / "reports" / f"drift_ablation_{n}.json").write_text(json.dumps(r, indent=2, default=float))
    plan = pipeline.run(cap_dir)
    (ROOT / "reports" / f"plan_{n}.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, str(ROOT / "reports" / f"plan_{n}.png"))
    print(f"   {len(plan.rooms)} rooms, drift notes: {r['report']['notes']}")