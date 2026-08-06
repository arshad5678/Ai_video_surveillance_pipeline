"""Integration test: full chain VideoInput -> ... -> LoiteringDetector -> EventEngine.

Reuses the static-image-loop technique from the Prompt 7/9 integration
tests (a person sits continuously inside a zone) with a low
loitering.threshold_seconds, so the same real run naturally produces both
an intrusion ENTER event (early) and a loitering event (once real
processing time crosses the low threshold) — giving EventEngine a
genuinely mixed stream to unify, not a single-source one.
"""

from pathlib import Path
from typing import List

import cv2
import pytest
import ultralytics

from src.surveillance.models.domain.surveillance_event import EventSeverity, EventType, SurveillanceEvent
from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
from src.surveillance.pipelines.events import EventEngine, EventEngineConfig
from src.surveillance.pipelines.frame_processing import FrameProcessor, PreprocessConfig
from src.surveillance.pipelines.intrusion import IntrusionConfig, IntrusionDetector
from src.surveillance.pipelines.loitering import LoiteringConfig, LoiteringDetector
from src.surveillance.pipelines.tracking import MultiObjectTracker, TrackingConfig
from src.surveillance.pipelines.video_input import VideoInput, VideoSourceConfig, VideoSourceType
from src.surveillance.pipelines.zones import ZoneManager

pytestmark = pytest.mark.integration

WEIGHTS_PATH = Path("weights/yolov8n.pt")
DEMO_IMAGE_PATH = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
FRAME_COUNT = 20

ZONES_YAML = """
zones:
  - id: watch_zone
    name: Watch Zone
    type: intrusion
    enabled: true
    polygon:
      - [850, 200]
      - [1200, 200]
      - [1200, 500]
      - [850, 500]
"""

DEFAULT_SEVERITY_MAPPING = {
    "intrusion_enter": EventSeverity.HIGH,
    "intrusion_exit": EventSeverity.LOW,
    "loitering": EventSeverity.MEDIUM,
}


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    image = cv2.imread(str(DEMO_IMAGE_PATH))
    assert image is not None, f"Could not read bundled demo image: {DEMO_IMAGE_PATH}"
    height, width = image.shape[:2]

    video_path = tmp_path_factory.mktemp("integration") / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for _ in range(FRAME_COUNT):
        writer.write(image)
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


def test_event_engine_produces_one_unified_stream(sample_video: Path, zones_config: Path) -> None:
    _ensure_weights_available()

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(sample_video))
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))
    tracker = MultiObjectTracker(TrackingConfig(frame_rate=5))
    zone_manager = ZoneManager(zones_config)
    intrusion_detector = IntrusionDetector(IntrusionConfig(), list(zone_manager.zones))
    loitering_detector = LoiteringDetector(
        LoiteringConfig(threshold_seconds=0.05), list(zone_manager.zones), intrusion_detector
    )
    event_engine = EventEngine(EventEngineConfig(severity_mapping=dict(DEFAULT_SEVERITY_MAPPING)))

    all_events: List[SurveillanceEvent] = []

    with VideoInput(video_config) as video_input:
        processed_frames = processor.process_stream(video_input.frames())
        detections_stream = detector.detect_stream(processed_frames)

        for tracks in tracker.track_stream(detections_stream):
            memberships = zone_manager.evaluate(tracks)
            intrusion_events = intrusion_detector.evaluate(memberships)  # must run before loitering
            loitering_events = loitering_detector.evaluate(tracks, memberships)

            frame_events = event_engine.evaluate(intrusion_events, loitering_events)
            all_events.extend(frame_events)

    assert len(all_events) >= 2, "Expected at least one intrusion and one loitering event."

    event_types = {e.event_type for e in all_events}
    assert EventType.INTRUSION_ENTER in event_types
    assert EventType.LOITERING in event_types

    # Every event is the unified type -- no leakage of IntrusionEvent/LoiteringEvent.
    for event in all_events:
        assert isinstance(event, SurveillanceEvent)
        assert event.zone_id == "watch_zone"
        assert event.zone_name == "Watch Zone"
        assert event.event_id

    # Global ordering: since each frame's output is individually sorted and
    # frames are processed in increasing order, the concatenated stream
    # must also be non-decreasing in (timestamp, frame_index).
    ordering_keys = [(e.timestamp, e.frame_index) for e in all_events]
    assert ordering_keys == sorted(ordering_keys)

    # No duplicates across the whole run.
    dedup_keys = [(e.track_id, e.zone_id, e.timestamp, e.event_type) for e in all_events]
    assert len(dedup_keys) == len(set(dedup_keys))

    # Severity mapping applied correctly for both sources.
    enter_events = [e for e in all_events if e.event_type is EventType.INTRUSION_ENTER]
    loitering_events_out = [e for e in all_events if e.event_type is EventType.LOITERING]
    assert all(e.severity is EventSeverity.HIGH for e in enter_events)
    assert all(e.severity is EventSeverity.MEDIUM for e in loitering_events_out)
