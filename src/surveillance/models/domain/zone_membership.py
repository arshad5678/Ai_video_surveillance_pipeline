"""ZoneMembership domain model — the interchange type between ZoneManager and future consumers.

Purely descriptive spatial state for one (Track, Zone) pair at one frame —
deliberately NOT an event. Whether a sequence of these constitutes an
"intrusion" or "loitering" is a future module's decision, not this one's.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneMembership:
    track_id: int
    zone_id: str
    inside: bool
    timestamp: float
    frame_index: int
    source_id: str
