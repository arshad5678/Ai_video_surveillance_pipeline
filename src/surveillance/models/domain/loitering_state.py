"""LoiteringState domain model — internal bookkeeping for one (track_id, zone_id) continuous stay.

Framework-free and immutable. Unlike IntrusionState, this module has no
requirement to expose it via a public accessor, so it stays purely
internal to LoiteringDetector's private state dict.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoiteringState:
    track_id: int
    zone_id: str
    entered_timestamp: float
    current_dwell_time: float
    event_emitted: bool
