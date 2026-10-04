"""Conservative surface anomaly and scope extraction from a fused point cloud."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import lidar
from .schema import ConcealedFlag, DamageRegion, ScopeItem


@dataclass
class DamageResult:
    regions: list[DamageRegion]
    concealed: list[ConcealedFlag]
    scope: list[ScopeItem]
    status: str


def _interval(value: float, floor: float = 0.05):
    from .intervals import make
    return make(max(value, 0.01), "length", tier="lidar", extra=floor)


def analyze(pts: np.ndarray, geom: lidar.SpaceGeometry) -> DamageResult:
    """Report only repeatable, geometric wall irregularities as damage candidates.

    A candidate is not a damage claim: it requires dense support and a robust offset from
    an observed wall line. Texture, colour, and concealed damage need RGB or thermal data
    and are therefore represented as explicit abstentions.
    """
    regions: list[DamageRegion] = []
    scope: list[ScopeItem] = []
    if len(geom.wall_xz):
        R = lidar._rot(geom.theta)
        wall = geom.wall_xz @ R
        xy = pts[(pts[:, 1] > geom.vertical.floor.y + 0.25) & (pts[:, 1] < geom.vertical.floor.y + 1.8)][:, [0, 2]] @ R
        for i, w in enumerate(geom.walls):
            ends = np.array([w.p0, w.p1]) @ R
            axis = 0 if abs(ends[0, 0] - ends[1, 0]) < abs(ends[0, 1] - ends[1, 1]) else 1
            fixed, lo, hi = ends[:, axis].mean(), *np.sort(ends[:, 1 - axis])
            near = xy[(np.abs(xy[:, axis] - fixed) < 0.08) & (xy[:, 1 - axis] >= lo) & (xy[:, 1 - axis] <= hi)]
            if len(near) < 80:
                continue
            residual = np.abs(near[:, axis] - fixed)
            anomalous = residual > 0.08
            if anomalous.sum() < 40 or anomalous.mean() < 0.08:
                continue
            extent = float(np.ptp(near[anomalous, 1 - axis]))
            if extent < 0.15:
                continue
            surface = f"wall:{i}"
            regions.append(DamageRegion(id=f"d{i}", surface=surface,
                                        damage_class="surface_irregularity", extent=_interval(extent),
                                        confidence=float(min(0.95, 0.5 + anomalous.mean()))) )
            scope.append(ScopeItem(id=f"s{i}", surface=surface,
                                   description="Inspect and repair observed wall irregularity",
                                   quantity=_interval(extent)))

    concealed: list[ConcealedFlag] = []
    if geom.vertical.ceiling is None:
        concealed.append(ConcealedFlag(id="c0", surface="ceiling",
                                       rule="ceiling_plane_not_observed",
                                       detail=geom.vertical.ceiling_status))
    if not len(geom.wall_xz):
        concealed.append(ConcealedFlag(id="c1", surface="walls",
                                       rule="insufficient_full_height_surface_support",
                                       detail="No full-height wall evidence was observed."))
    status = "implemented" if regions else "no_geometric_anomaly_detected"
    return DamageResult(regions, concealed, scope, status)