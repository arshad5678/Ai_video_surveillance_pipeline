"""Detection domain model — the interchange type between PersonDetector and future consumers."""

from dataclasses import dataclass
from typing import Optional

from .bounding_box import BoundingBox


@dataclass(frozen=True)
class Detection:
    """A single person detection in a single frame.

    Framework-free: no Ultralytics/YOLO types leak through this class, so
    a future tracker or event stage can consume it without knowing YOLO
    exists. `track_id` is None here — the future tracker assigns identity
    across frames by producing a new Detection via `dataclasses.replace`
    (this class is immutable).
    """

    track_id: Optional[int]
    class_name: str
    class_id: int
    confidence: float
    bounding_box: BoundingBox
    timestamp: float
    frame_index: int
    source_id: str
