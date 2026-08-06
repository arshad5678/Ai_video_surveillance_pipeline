"""Track domain model — the interchange type between MultiObjectTracker and future consumers."""

from dataclasses import dataclass
from typing import Tuple

from .bounding_box import BoundingBox


@dataclass(frozen=True)
class Track:
    """A single tracked person, with identity stable across frames.

    Framework-free: no ByteTrack/Kalman-filter types leak through this
    class, so a future Zone Manager can consume it without knowing a
    tracking algorithm exists at all.

    `time_since_update == 0` means matched this frame; `> 0` means the
    track is coasting on its last known motion (occluded/missed) but is
    still within `track_buffer` and therefore not yet removed. There is
    no separate "lost" flag — that state is exactly `time_since_update > 0`.

    `history` is a snapshot tuple of past (center_x, center_y) points,
    oldest first, capped at the configured history_length — immutable so
    a stored Track can never be corrupted by a later mutation elsewhere.
    """

    track_id: int
    bounding_box: BoundingBox
    confidence: float
    class_name: str
    class_id: int
    timestamp: float
    frame_index: int
    source_id: str
    is_confirmed: bool
    age: int
    hits: int
    time_since_update: int
    history: Tuple[Tuple[float, float], ...]
