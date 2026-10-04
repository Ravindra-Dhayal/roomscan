from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1.0"


class Interval(BaseModel):
    """A measurement with a confidence interval. `calibrated` is False until the interval width
    has been fitted on benchmark data; consumers must not treat an uncalibrated interval as a
    coverage guarantee."""
    value: float
    lo: float
    hi: float
    unit: Literal["m", "m2"] = "m"
    level: float = 0.95
    calibrated: bool = False
    method: str = "provisional_heuristic"


class Wall(BaseModel):
    id: str
    room_id: int
    p0: tuple[float, float]
    p1: tuple[float, float]
    length: Interval


class Room(BaseModel):
    id: int
    polygon: list[tuple[float, float]] = Field(description="Outline in plan metres (x, z), counter-clockwise")
    area: Interval
    walls: list[Wall]
    ceiling_height: Interval | None = Field(None, description="null when not observed; see ceiling_status")
    ceiling_status: str


class Opening(BaseModel):
    id: str
    kind: Literal["door", "window"]
    room_ids: tuple[int, int] | None = None
    p0: tuple[float, float]
    p1: tuple[float, float]
    width: Interval


class DamageRegion(BaseModel):
    id: str
    surface: str
    damage_class: str
    extent: Interval
    confidence: float


class ConcealedFlag(BaseModel):
    id: str
    surface: str
    rule: str
    detail: str


class ScopeItem(BaseModel):
    id: str
    surface: str
    description: str
    quantity: Interval


class Plan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    tier: Literal["photo", "video", "lidar"]
    source: str
    rooms: list[Room]
    openings: list[Opening]
    adjacency: list[tuple[int, int]]
    damage_regions: list[DamageRegion] = []
    concealed_damage_flags: list[ConcealedFlag] = []
    scope_items: list[ScopeItem] = []
    module_status: dict[str, str] = Field(default_factory=dict, description="implemented / not_implemented per module")
    warnings: list[str] = []
    diagnostics: dict = Field(default_factory=dict, description="per-module run diagnostics (e.g. drift)")
    timing_s: dict[str, float] = {}