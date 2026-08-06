"""IntrusionState domain model — a read-only snapshot of one (track_id, zone_id) pair's state.

Framework-free and immutable: IntrusionDetector hands out snapshots of
this via its `get_state()` accessor, never a reference into its own
mutable internal dict, so a caller (e.g. a future Loitering Detection
module) can safely hold onto one without risk of it changing underfoot.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class IntrusionState:
    track_id: int
    zone_id: str
    currently_inside: bool
    first_seen_inside_timestamp: Optional[float]
    last_seen_timestamp: float
