from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline
from .render import render_plan
from .schema import Plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roomscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="capture folder -> plan.json + plan.png")
    r.add_argument("capture")
    r.add_argument("--out", default="out")
    r.add_argument("--no-drift", action="store_true", help="use poses as-is (for the ablation)")
    r.add_argument("--tier", choices=("photo", "video", "lidar"), help="override automatic tier detection")
    ab = sub.add_parser("ablate", help="drift handling on vs off -> drift_ablation.png/.json")
    ab.add_argument("capture")
    ab.add_argument("--out", default="out")
    sub.add_parser("schema", help="write schema/plan.schema.json")
    b = sub.add_parser("bench", help="run a benchmark manifest -> bench/results.md + .json")
    b.add_argument("manifest")
    b.add_argument("--out", default="bench/results")
    c = sub.add_parser("calibrate", help="fit interval widths from a benchmark manifest -> calibration.json")
    c.add_argument("manifest")
    c.add_argument("--out", default="calibration.json")
    f = sub.add_parser("fixloop", help="regenerate and compare before/after benchmark manifests")
    f.add_argument("before")
    f.add_argument("after")
    f.add_argument("--out", default="fixloop/results")
    h = sub.add_parser("head2head", help="compare Plan JSON dimensions with an incumbent export")
    h.add_argument("ours")
    h.add_argument("incumbent")
    h.add_argument("--ground-truth", help="JSON dimension map or Plan JSON used to score errors")
    s = sub.add_parser("audit", help="check submission artifacts and external evidence")
    s.add_argument("--root", default=".")
    s.add_argument("--out", default="reports/submission_audit")
    h.add_argument("--out", default="reports/head_to_head")
    a = ap.parse_args(argv)

    if a.cmd == "schema":
        p = Path("schema/plan.schema.json")
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(Plan.model_json_schema(), indent=2))
        print(f"wrote {p}")
        return 0

    if a.cmd in ("bench", "calibrate"):
        from . import bench, calibrate, intervals
        intervals._CAL = {}          # benchmark with provisional intervals, never with a previous fit
        rep = bench.run_manifest(a.manifest, pipeline.run)
        if a.cmd == "calibrate":
            cal = calibrate.fit(list(rep["results"].values()))
            calibrate.save(cal, a.out)
            print(f"wrote {a.out}: " + ", ".join(f"{t}:{sorted(k)}" for t, k in cal.items()))
            return 0
        base = Path(a.out)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".md").write_text(bench.to_markdown(rep))
        import dataclasses
        base.with_suffix(".json").write_text(json.dumps(
            {"gates": rep["gates"], "repeatability": rep["repeatability"], "timing_s": rep["timing_s"],
             "results": {k: dataclasses.asdict(v) for k, v in rep["results"].items()}}, indent=2, default=float))
        print(bench.to_markdown(rep))
        return 0

    if a.cmd == "fixloop":
        from . import fixloop
        diff = fixloop.write(a.before, a.after, a.out)
        print(fixloop.to_markdown(diff))
        return 0

    if a.cmd == "head2head":
        from . import head_to_head
        result = head_to_head.compare(a.ours, a.incumbent, a.ground_truth)
        base = Path(a.out)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".json").write_text(json.dumps(result, indent=2))
        base.with_suffix(".md").write_text(head_to_head.to_markdown(result))
        print(head_to_head.to_markdown(result))
        return 0

    if a.cmd == "audit":
        from . import audit
        result = audit.run(a.root)
        base = Path(a.out)
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".json").write_text(json.dumps(result, indent=2))
        base.with_suffix(".md").write_text(audit.to_markdown(result))
        print(audit.to_markdown(result))
        return 0 if result["ready"] else 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "ablate":
        import dataclasses
        from . import drift
        from .ingest import load_capture
        r = drift.ablation(load_capture(a.capture), str(out / "drift_ablation.png"))
        r["report"] = dataclasses.asdict(r["report"])
        (out / "drift_ablation.json").write_text(json.dumps(r, indent=2, default=float))
        print(f"drift ablation -> {out}/drift_ablation.png, .json")
        for k in r["off"]:
            print(f"  {k:20s} off={r['off'][k]:<10.4g} on={r['on'][k]:<10.4g}")
        print("  notes:", r["report"]["notes"] or "none")
        return 0
    try:
        plan = pipeline.run(a.capture, drift_on=not a.no_drift, tier=getattr(a, "tier", None))
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, str(out / "plan.png"))
    print(f"{plan.tier}: {len(plan.rooms)} rooms, {len(plan.openings)} openings -> {out}/plan.json, plan.png "
          f"({sum(plan.timing_s.values()):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())