"""Types describing tracking configuration.

Scoped to this module only — future consumers (zone/event logic, etc.)
never see these; they only ever receive `Track` objects.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingConfig:
    """Fully-resolved parameters controlling MultiObjectTracker's behavior.

    Mirrors the standard ByteTrack parameter set. `history_length` is the
    one addition beyond that set — it isn't a ByteTrack concept, but the
    module needs a configurable cap on how many trajectory points a Track
    retains (requirement: "Support configurable history length").
    """

    tracker_type: str = "bytetrack"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30
    minimum_box_area: float = 10.0
    history_length: int = 30
    verbose: bool = False
