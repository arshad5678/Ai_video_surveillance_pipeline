"""IntrusionEvent domain model — the interchange type between IntrusionDetector and future consumers."""

from dataclasses import dataclass
from enum import Enum


class IntrusionEventType(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"


@dataclass(frozen=True)
class IntrusionEvent:
    """A single zone-occupancy transition for one (track_id, zone_id) pair.

    Emitted only on state change (ENTER: outside->inside, EXIT: inside->
    outside) — never once per frame while a track remains in the same
    state. Framework-free: no ZoneManager/Shapely types leak through.
    """

    event_id: str
    track_id: int
    zone_id: str
    zone_name: str
    event_type: IntrusionEventType
    timestamp: float
    frame_index: int
    source_id: str
