"""Regenerable before/after benchmark comparison for the fix loop."""
from __future__ import annotations

import json
from pathlib import Path

from . import bench, pipeline


def _flatten_gates(report: dict) -> dict[str, dict]:
    return {f"{tier}.{name}": value for tier, gates in report.get("gates", {}).items()
            for name, value in gates.items()}


def compare_manifests(before: str | Path, after: str | Path) -> dict:
    before_report = bench.run_manifest(before, pipeline.run)
    after_report = bench.run_manifest(after, pipeline.run)
    b, a = _flatten_gates(before_report), _flatten_gates(after_report)
    changes = []
    for key in sorted(set(b) | set(a)):
        old, new = b.get(key), a.get(key)
        changes.append({"gate": key, "before": old, "after": new,
                        "passed_change": None if old is None or new is None else
                        new.get("passed") != old.get("passed")})
    return {"before": before_report, "after": after_report, "changes": changes}


def to_markdown(diff: dict) -> str:
    lines = ["# Fix loop comparison", "", "| Gate | Before | After | Status change |",
             "|---|---|---|---|"]
    for row in diff["changes"]:
        old = "missing" if row["before"] is None else ("PASS" if row["before"].get("passed") else "FAIL")
        new = "missing" if row["after"] is None else ("PASS" if row["after"].get("passed") else "FAIL")
        change = "-" if row["passed_change"] is None else ("changed" if row["passed_change"] else "unchanged")
        lines.append(f"| {row['gate']} | {old} | {new} | {change} |")
    return "\n".join(lines) + "\n"


def write(before: str | Path, after: str | Path, out: str | Path) -> dict:
    diff = compare_manifests(before, after)
    base = Path(out)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(diff, indent=2, default=float))
    base.with_suffix(".md").write_text(to_markdown(diff))
    return diff