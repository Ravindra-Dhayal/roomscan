from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .bench import CaptureResult

KIND_MODE = {"length": "rel", "area": "rel", "opening": "abs", "ceiling": "abs"}


def conformal_quantile(res: list[float], level: float = 0.95) -> float:
    r = sorted(abs(x) for x in res)
    n = len(r)
    k = min(n, math.ceil((n + 1) * level))
    return float(r[k - 1])


def fit(results: list[CaptureResult], level: float = 0.95) -> dict:
    cal: dict = {}
    for tier in sorted({r.tier for r in results}):
        rs = [r for r in results if r.tier == tier]
        resid = {
            "length": [abs(w["err"]) / w["truth"] for r in rs for w in r.walls],
            "area": [abs(e) for r in rs for (_, _, e) in r.room_match if not np.isnan(e)],
            "opening": [abs(o["err"]) for r in rs for o in r.openings if o["err"] is not None],
            "ceiling": [abs(c["err"]) for r in rs for c in r.ceilings if c["err"] is not None],
        }
        cal[tier] = {}
        for kind, v in resid.items():
            if v:
                cal[tier][kind] = {KIND_MODE[kind]: round(conformal_quantile(v, level), 5), "n": len(v),
                                   "small_sample": len(v) < 20, "level": level}
    return cal


def save(cal: dict, path: str | Path = "calibration.json") -> None:
    Path(path).write_text(json.dumps(cal, indent=2))