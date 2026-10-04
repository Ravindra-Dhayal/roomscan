"""Dimension-by-dimension comparison with an incumbent export."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import Plan


def _dimensions(plan: Plan) -> dict[str, float]:
    values: dict[str, float] = {}
    for room in plan.rooms:
        for index, wall in enumerate(room.walls):
            values[f"room:{room.id}:wall:{index}"] = wall.length.value
        if room.ceiling_height is not None:
            values[f"room:{room.id}:ceiling"] = room.ceiling_height.value
    for opening in plan.openings:
        values[f"opening:{opening.id}:width"] = opening.width.value
    return values


def load_dimensions(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text())
    if "rooms" in raw and "openings" in raw and "schema_version" in raw:
        return _dimensions(Plan.model_validate(raw))
    if not isinstance(raw, dict) or not all(isinstance(v, (float, int)) for v in raw.values()):
        raise ValueError("incumbent export must be a Plan JSON or {dimension_id: metres}")
    return {str(k): float(v) for k, v in raw.items()}


def compare(ours: str | Path, incumbent: str | Path, ground_truth: str | Path | None = None) -> dict:
    left, right = load_dimensions(ours), load_dimensions(incumbent)
    truth = load_dimensions(ground_truth) if ground_truth is not None else None
    rows = []
    keys = set(left) & set(right) if truth is None else set(left) & set(right) & set(truth)
    for key in sorted(keys):
        row = {"dimension": key, "ours_m": left[key], "incumbent_m": right[key]}
        if truth is not None:
            row["truth_m"] = truth[key]
            row["ours_error_m"] = abs(left[key] - truth[key])
            row["incumbent_error_m"] = abs(right[key] - truth[key])
            row["ours_better_or_tied"] = row["ours_error_m"] <= row["incumbent_error_m"]
        else:
            row["ours_better_or_tied"] = None
        rows.append(row)
    scored = [r for r in rows if r["ours_better_or_tied"] is not None]
    return {"shared_dimensions": len(rows), "ours_better_or_tied": sum(r["ours_better_or_tied"] for r in scored),
            "share": (sum(r["ours_better_or_tied"] for r in scored) / len(scored)) if scored else None,
            "rows": rows}


def to_markdown(result: dict) -> str:
    lines = ["# Head-to-head comparison", "", f"Shared dimensions: {result['shared_dimensions']}",
             f"Ours better or tied: {result['ours_better_or_tied']}", "",
             "| Dimension | Ours (m) | Incumbent (m) | Ours error (m) | Incumbent error (m) | Ours better/tied |",
             "|---|---:|---:|---:|---:|---|"]
    for row in result["rows"]:
        ours_error = "-" if "ours_error_m" not in row else f"{row['ours_error_m']:.3f}"
        incumbent_error = "-" if "incumbent_error_m" not in row else f"{row['incumbent_error_m']:.3f}"
        lines.append(f"| {row['dimension']} | {row['ours_m']:.3f} | {row['incumbent_m']:.3f} | "
                     f"{ours_error} | {incumbent_error} | {row['ours_better_or_tied']} |")
    return "\n".join(lines) + "\n"