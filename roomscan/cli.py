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
    sub.add_parser("schema", help="write schema/plan.schema.json")
    a = ap.parse_args(argv)

    if a.cmd == "schema":
        p = Path("schema/plan.schema.json")
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps(Plan.model_json_schema(), indent=2))
        print(f"wrote {p}")
        return 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        plan = pipeline.run(a.capture)
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