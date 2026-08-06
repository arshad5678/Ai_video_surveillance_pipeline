"""Unit tests for the Detection/BoundingBox domain models — pure dataclasses, no I/O."""

import dataclasses

from src.surveillance.models.domain.bounding_box import BoundingBox
from src.surveillance.models.domain.detection import Detection


def test_bounding_box_derives_width_and_height() -> None:
    box = BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0)

    assert box.width == 40.0
    assert box.height == 60.0


def test_bounding_box_derives_center() -> None:
    box = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=50.0)

    assert box.center_x == 50.0
    assert box.center_y == 25.0


def test_detection_track_id_defaults_to_none_and_is_replaceable() -> None:
    detection = Detection(
        track_id=None,
        class_name="person",
        class_id=0,
        confidence=0.91,
        bounding_box=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),
        timestamp=123.0,
        frame_index=4,
        source_id="cam-1",
    )

    assert detection.track_id is None

    # Immutable — a future tracker assigns identity via dataclasses.replace.
    tracked = dataclasses.replace(detection, track_id=7)
    assert tracked.track_id == 7
    assert detection.track_id is None  # original untouched
