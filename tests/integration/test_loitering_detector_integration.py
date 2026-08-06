"""Integration test: full chain VideoInput -> ... -> IntrusionDetector -> LoiteringDetector.

Unlike Prompt 8's intrusion test, loitering needs sustained presence, not
movement — so this reuses the simpler static-image-loop technique from
the Prompt 5-7 integration tests (ultralytics' bundled zidane.jpg, looped
unchanged). Frame.timestamp is real wall-clock time (VideoInput uses
time.time() per frame), and reading a file is effectively instant, so a
very low threshold_seconds is used to make the dwell time cross the
threshold within real (fast) processing time across a handful of frames,
without the test needing to run for many seconds of wall-clock time.

This proves the actual wiring requirement documented on LoiteringDetector:
IntrusionDetector.evaluate() must run before LoiteringDetector.evaluate()
for the same frame, since the latter reads the former's freshly-updated
first_seen_inside_timestamp via get_state().
"""

from pathlib import Path
from typing import List

import cv2
import pytest
import ultralytics

from src.surveillance.models.domain.loitering_event import LoiteringEvent
from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
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

# The right person in zidane.jpg sits at approximately (945, 378)
# (confirmed empirically in Prompt 5-7 runs) — this zone contains them on
# every frame, since the image never changes.
ZONES_YAML = """
zones:
  - id: loiter_zone
    name: Loiter Zone
    type: intrusion
    enabled: true
    polygon:
      - [850, 200]
      - [1200, 200]
      - [1200, 500]
      - [850, 500]
"""


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


def test_loitering_event_fires_only_after_dwell_threshold(sample_video: Path, zones_config: Path) -> None:
    _ensure_weights_available()

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(sample_video))
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))
    tracker = MultiObjectTracker(TrackingConfig(frame_rate=5))
    zone_manager = ZoneManager(zones_config)
    intrusion_detector = IntrusionDetector(IntrusionConfig(), list(zone_manager.zones))
    # Real per-frame CPU inference easily takes tens of milliseconds, so a
    # 0.05s threshold reliably crosses within a few frames of real
    # wall-clock time without an artificially long-running test.
    loitering_detector = LoiteringDetector(
        LoiteringConfig(threshold_seconds=0.05), list(zone_manager.zones), intrusion_detector
    )

    all_events: List[LoiteringEvent] = []
    first_event_frame_index = None

    with VideoInput(video_config) as video_input:
        processed_frames = processor.process_stream(video_input.frames())
        detections_stream = detector.detect_stream(processed_frames)

        for tracks in tracker.track_stream(detections_stream):
            memberships = zone_manager.evaluate(tracks)
            intrusion_detector.evaluate(memberships)  # must run first: refreshes first_seen_inside_timestamp
            loitering_events = loitering_detector.evaluate(tracks, memberships)

            all_events.extend(loitering_events)
            if loitering_events and first_event_frame_index is None:
                first_event_frame_index = loitering_events[0].frame_index

    assert len(all_events) >= 1, "Expected at least one loitering event once the dwell threshold was crossed."
    assert first_event_frame_index is not None
    assert first_event_frame_index > 0, "Loitering must not fire on the very first observation (dwell ~= 0)."

    for event in all_events:
        assert isinstance(event, LoiteringEvent)
        assert event.zone_id == "loiter_zone"
        assert event.zone_name == "Loiter Zone"
        assert event.dwell_time_seconds >= event.threshold_seconds
        assert event.event_id

    # Exactly one event per (track_id, zone_id) continuous stay — no duplicates
    # while the person remains inside for the rest of the clip.
    keys = [(e.track_id, e.zone_id) for e in all_events]
    assert len(keys) == len(set(keys)), "Expected no duplicate loitering events for the same continuous stay."
