from __future__ import annotations

from .schema import Interval

# half-width = max(floor, rel * value) at 95%, per tier. Placeholders to be fitted.
PROVISIONAL = {
    "lidar": {"length": (0.03, 0.01), "area": (0.3, 0.05), "opening": (0.05, 0.03), "ceiling": (0.02, 0.005)},
}


def make(value: float, kind: str, tier: str = "lidar", unit: str = "m", extra: float = 0.0) -> Interval:
    floor, rel = PROVISIONAL[tier][kind]
    hw = max(floor, rel * abs(value)) + extra
    return Interval(value=round(value, 4), lo=round(value - hw, 4), hi=round(value + hw, 4), unit=unit)