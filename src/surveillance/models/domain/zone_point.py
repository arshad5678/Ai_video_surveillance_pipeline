"""ZonePoint domain model — a single polygon vertex, in image pixel coordinates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZonePoint:
    x: float
    y: float
