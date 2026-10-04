from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import Interval

# half-width = max(floor, rel * value) at 95%. Placeholders, widest for thinnest input.
PROVISIONAL = {
    "lidar": {"length": (0.03, 0.01), "area": (0.3, 0.05), "opening": (0.05, 0.03), "ceiling": (0.02, 0.005)},
    "video": {"length": (0.08, 0.03), "area": (0.6, 0.10), "opening": (0.10, 0.06), "ceiling": (0.06, 0.02)},
    "photo": {"length": (0.20, 0.08), "area": (1.0, 0.16), "opening": (0.20, 0.12), "ceiling": (0.15, 0.06)},
}

_CAL: dict | None = None


def load_calibration(path: str | Path | None = None) -> dict:
    """Read calibration.json (path, $ROOMSCAN_CALIBRATION, or ./calibration.json). {} if absent."""
    global _CAL
    p = Path(path or os.environ.get("ROOMSCAN_CALIBRATION", "calibration.json"))
    _CAL = json.loads(p.read_text()) if p.exists() else {}
    return _CAL


def make(value: float, kind: str, tier: str = "lidar", unit: str = "m", extra: float = 0.0) -> Interval:
    global _CAL
    if _CAL is None:
        load_calibration()
    c = (_CAL or {}).get(tier, {}).get(kind)
    if c:
        hw = c["abs"] if "abs" in c else c["rel"] * abs(value)
        method = f"split_conformal(n={c['n']}{', small-sample' if c.get('small_sample') else ''})"
        calibrated = True
    else:
        floor, rel = PROVISIONAL[tier][kind]
        hw, method, calibrated = max(floor, rel * abs(value)), "provisional_heuristic", False
    hw += extra
    return Interval(value=round(value, 4), lo=round(value - hw, 4), hi=round(value + hw, 4), unit=unit,
                    calibrated=calibrated, method=method)