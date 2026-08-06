"""Zone domain model — a configured polygon region of interest.

Framework-free: `polygon` is a plain tuple of ZonePoint, not a Shapely
geometry — the geometry library is an implementation detail private to
ZoneManager, never exposed on this model.
"""

from dataclasses import dataclass
from typing import Tuple

from .zone_point import ZonePoint


@dataclass(frozen=True)
class Zone:
    zone_id: str
    zone_name: str
    zone_type: str
    polygon: Tuple[ZonePoint, ...]
    enabled: bool
