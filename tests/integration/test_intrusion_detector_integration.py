"""Integration test: full chain VideoInput -> ... -> ZoneManager -> IntrusionDetector.

A statically-looped demo image (as used in the Prompt 5-7 integration
tests) never changes zone membership, so it can prove ENTER but never
EXIT. This test instead pans ultralytics' bundled demo photo (zidane.jpg,
two real people) horizontally across frames using a real affine warp, so
a real detected person's box genuinely sweeps from outside a zone, into
it, and back outside — producing a real ENTER then a real EXIT for the
same track.
"""

from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
import ultralytics

from src.surveillance.models.domain.intrusion_event import IntrusionEvent, IntrusionEventType
from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
from src.surveillance.pipelines.frame_processing import FrameProcessor, PreprocessConfig
from src.surveillance.pipelines.intrusion import IntrusionConfig, IntrusionDetector
from src.surveillance.pipelines.tracking import MultiObjectTracker, TrackingConfig
from src.surveillance.pipelines.video_input import VideoInput, VideoSourceConfig, VideoSourceType
from src.surveillance.pipelines.zones import ZoneManager

pytestmark = pytest.mark.integration

WEIGHTS_PATH = Path("weights/yolov8n.pt")
DEMO_IMAGE_PATH = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
FRAME_COUNT = 24

# The left person in zidane.jpg sits at approximately x=625 (confirmed
# empirically in Prompt 5-7 runs). The pan sweeps that center from ~475
# (outside, left of the zone) through ~625 (inside) to ~775 (outside,
# right of the zone) — a real outside -> inside -> outside crossing.
ZONES_YAML = """
zones:
  - id: sweep_zone
    name: Sweep Zone
    type: intrusion
    enabled: true
    polygon:
      - [550, 350]
      - [700, 350]
      - [700, 550]
      - [550, 550]
"""


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    image = cv2.imread(str(DEMO_IMAGE_PATH))
    assert image is not None, f"Could not read bundled demo image: {DEMO_IMAGE_PATH}"
    height, width = image.shape[:2]

    video_path = tmp_path_factory.mktemp("integration") / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))

    for i in range(FRAME_COUNT):
        tx = -150.0 + i * (300.0 / (FRAME_COUNT - 1))  # sweeps -150 -> +150
        translation = np.float32([[1, 0, tx], [0, 1, 0]])
        shifted = cv2.warpAffine(image, translation, (width, height), borderMode=cv2.BORDER_REPLICATE)
        writer.write(shifted)

    writer.release()
    return video_path


@pytest.fixture(scope="module")
def zones_config(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("zones") / "zones.yaml"
    path.write_text(ZONES_YAML)
    return path


def _ensure_weights_available() -> None:
    if WEIGHTS_PATH.exists():
        return
    try:
        from ultralytics import YOLO

        YOLO(str(WEIGHTS_PATH))
    except Exception:
        pytest.skip(f"YOLO weights unavailable at {WEIGHTS_PATH} and could not be downloaded (no network?).")


def test_enter_and_exit_events_produced_across_a_real_sweep(sample_video: Path, zones_config: Path) -> None:
    _ensure_weights_available()

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(sample_video))
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))
    tracker = MultiObjectTracker(TrackingConfig(frame_rate=8, track_buffer=60))
    zone_manager = ZoneManager(zones_config)
    intrusion_detector = IntrusionDetector(IntrusionConfig(), list(zone_manager.zones))

    all_events: List[IntrusionEvent] = []
    with VideoInput(video_config) as video_input:
        processed_frames = processor.process_stream(video_input.frames())
        detections_stream = detector.detect_stream(processed_frames)
        tracks_stream = tracker.track_stream(detections_stream)
        memberships_stream = zone_manager.evaluate_stream(tracks_stream)
        for events in intrusion_detector.evaluate_stream(memberships_stream):
            all_events.extend(events)

    enter_events = [e for e in all_events if e.event_type is IntrusionEventType.ENTER]
    exit_events = [e for e in all_events if e.event_type is IntrusionEventType.EXIT]

    assert len(enter_events) >= 1, "Expected at least one ENTER as a track swept into the zone."
    assert len(exit_events) >= 1, "Expected at least one EXIT as the same track swept back out."

    for event in all_events:
        assert isinstance(event, IntrusionEvent)
        assert event.zone_id == "sweep_zone"
        assert event.zone_name == "Sweep Zone"
        assert event.event_id  # non-empty

    # Same track_id should own both an ENTER and a later EXIT (proving a
    # genuine sweep-through, not two different tracks each firing once).
    enter_track_ids = {e.track_id for e in enter_events}
    exit_track_ids = {e.track_id for e in exit_events}
    common_track_ids = enter_track_ids & exit_track_ids
    assert common_track_ids, "Expected the same track to produce both an ENTER and an EXIT."

    track_id = next(iter(common_track_ids))
    first_enter = min(e.frame_index for e in enter_events if e.track_id == track_id)
    first_exit_after_enter = min(
        (e.frame_index for e in exit_events if e.track_id == track_id and e.frame_index > first_enter),
        default=None,
    )
    assert first_exit_after_enter is not None, "EXIT should occur strictly after ENTER for the same track."

    # No duplicate consecutive ENTERs for the same (track, zone) without an EXIT between them.
    events_for_track = sorted(
        (e for e in all_events if e.track_id == track_id), key=lambda e: e.frame_index
    )
    event_type_sequence = [e.event_type for e in events_for_track]
    for previous, current in zip(event_type_sequence, event_type_sequence[1:]):
        assert previous != current, "Consecutive events for the same track must alternate ENTER/EXIT."
