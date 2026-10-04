"""Create a truthful source-and-evidence submission archive."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", ".tmp-validation", ".tmp-video-check"}
EXCLUDED_NAMES = {"roomscan_submission.zip"}


def should_include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in EXCLUDED_PARTS for part in relative.parts) and path.name not in EXCLUDED_NAMES


def package(root: Path, output: Path) -> int:
    output = output.resolve()
    files = [p for p in root.rglob("*") if p.is_file() and p.resolve() != output and should_include(p, root)]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="roomscan_submission.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.out)
    if not output.is_absolute():
        output = root / output
    count = package(root, output)
    print(f"wrote {output} ({count} files)")


if __name__ == "__main__":
    main()