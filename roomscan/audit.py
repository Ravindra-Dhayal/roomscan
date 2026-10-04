"""Check repository artifacts required for a defensible case-study submission."""
from __future__ import annotations

import json
from pathlib import Path


REQUIRED_FILES = (
    "README.md", "COMPLIANCE.md", "CAPTURE_PROTOCOL.md", "DEVICE_MATRIX.md",
    "TECHNICAL_REPORT.md", "bench/README.md", "bench/manifest.example.json",
    "fixloop/DECLARATION.md",
)


def run(root: str | Path = ".") -> dict:
    root = Path(root)
    files = {path: (root / path).exists() for path in REQUIRED_FILES}
    captures = [p for p in (root / "data").iterdir() if p.is_dir()] if (root / "data").exists() else []
    videos = list((root / "data").rglob("*.mp4")) + list((root / "data").rglob("*.mov")) if (root / "data").exists() else []
    stills = [p for p in (root / "data").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".heic", ".webp"}] if (root / "data").exists() else []
    manifest = root / "bench" / "manifest.json"
    calibration = root / "calibration.json"
    incumbent = list((root / "bench").glob("**/*incumbent*"))
    raw_complete = False
    if manifest.exists():
        entries = json.loads(manifest.read_text()).get("captures", [])
        raw_complete = bool(entries) and all(
            (manifest.parent / c["path"]).exists() and (manifest.parent / c["ground_truth"]).exists()
            for c in entries
        )
    checks = {
        "repository_documents": all(files.values()),
        "checked_in_raw_capture_folders": bool(captures),
        "video_capture_evidence": bool(videos),
        "independent_photo_capture_evidence": bool(stills),
        "scored_benchmark_manifest_and_ground_truth": raw_complete,
        "calibration_fit": calibration.exists(),
        "consumer_app_export": bool(incumbent),
        "fixloop_results": (root / "fixloop" / "results.json").exists(),
    }
    return {"checks": checks, "files": files, "capture_folders": [p.name for p in captures],
            "video_files": [str(p.relative_to(root)) for p in videos],
            "photo_files": [str(p.relative_to(root)) for p in stills],
            "ready": all(checks.values()),
            "blocked_by_external_evidence": [name for name, ok in checks.items() if not ok]}


def to_markdown(report: dict) -> str:
    lines = ["# Submission audit", "", "| Check | Status |", "|---|---|"]
    for name, passed in report["checks"].items():
        lines.append(f"| {name} | {'READY' if passed else 'BLOCKED'} |")
    lines += ["", f"Overall: **{'READY' if report['ready'] else 'BLOCKED'}**"]
    if report["blocked_by_external_evidence"]:
        lines += ["", "Missing evidence:"]
        lines.extend(f"- {name}" for name in report["blocked_by_external_evidence"])
    return "\n".join(lines) + "\n"
