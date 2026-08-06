"""LoiteringEvent domain model — the interchange type between LoiteringDetector and future consumers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoiteringEvent:
    """One person exceeding a dwell-time threshold in one zone, for one continuous stay.

    Emitted exactly once per continuous inside-period, the moment
    dwell_time first reaches threshold_seconds — never repeated while the
    same stay continues. Framework-free: no IntrusionDetector/tracking
    types leak through.
    """

    event_id: str
    track_id: int
    zone_id: str
    zone_name: str
    dwell_time_seconds: float
    threshold_seconds: float
    timestamp: float
    frame_index: int
    source_id: str
