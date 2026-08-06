"""Integration test: full chain VideoInput -> ... -> EventEngine -> OutputGenerator.

Reuses the static-image-loop technique from prior integration tests (a
person sits continuously inside a zone) with a low loitering threshold,
so intrusion + loitering events fire during the run and OutputGenerator
gets a real mixed stream of frames/tracks/events to render.

Uses the per-frame (non-stream) methods -- process()/detect()/update() --
rather than the chained generators, since OutputGenerator needs the
actual frame image alongside that same frame's tracks/events, and the
image doesn't flow through the tracker/zone/event stages.
"""

from pathlib import Path

import cv2
import pytest
import ultralytics

from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
from src.surveillance.models.domain.surveillance_event import EventSeverity
from src.surveillance.pipelines.events import EventEngine, EventEngineConfig
from src.surveillance.pipelines.frame_processing import FrameProcessor, PreprocessConfig
from src.surveillance.pipelines.intrusion import IntrusionConfig, IntrusionDetector
from src.surveillance.pipelines.loitering import LoiteringConfig, LoiteringDetector
from src.surveillance.pipelines.output import OutputConfig, OutputGenerator
from src.surveillance.pipelines.tracking import MultiObjectTracker, TrackingConfig
from src.surveillance.pipelines.video_input import VideoInput, VideoSourceConfig, VideoSourceType
from src.surveillance.pipelines.zones import ZoneManager

pytestmark = pytest.mark.integration

WEIGHTS_PATH = Path("weights/yolov8n.pt")
DEMO_IMAGE_PATH = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
FRAME_COUNT = 20
FRAME_RATE = 5.0

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


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    image = cv2.imread(str(DEMO_IMAGE_PATH))
    assert image is not None, f"Could not read bundled demo image: {DEMO_IMAGE_PATH}"
    height, width = image.shape[:2]

    video_path = tmp_path_factory.mktemp("integration") / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, (width, height))
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


def test_output_generator_produces_all_artifacts(sample_video: Path, zones_config: Path, tmp_path: Path) -> None:
    _ensure_weights_available()

    output_dir = tmp_path / "output"

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(sample_video))
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))
    tracker = MultiObjectTracker(TrackingConfig(frame_rate=FRAME_RATE))
    zone_manager = ZoneManager(zones_config)
    intrusion_detector = IntrusionDetector(IntrusionConfig(), list(zone_manager.zones))
    loitering_detector = LoiteringDetector(
        LoiteringConfig(threshold_seconds=0.05), list(zone_manager.zones), intrusion_detector
    )
    event_engine = EventEngine(
        EventEngineConfig(
            severity_mapping={
                "intrusion_enter": EventSeverity.HIGH,
                "intrusion_exit": EventSeverity.LOW,
                "loitering": EventSeverity.MEDIUM,
            }
        )
    )
    output_generator = OutputGenerator(
        OutputConfig(
            clip_pre_seconds=1.0,
            clip_post_seconds=1.0,
            output_directory=str(output_dir),
            frame_rate=FRAME_RATE,
        )
    )

    total_events = 0

    with VideoInput(video_config) as video_input:
        for frame in video_input.frames():
            processed = processor.process(frame)
            if processed is None:
                continue
            detections = detector.detect(processed)
            tracks = tracker.update(detections)
            memberships = zone_manager.evaluate(tracks)
            intrusion_events = intrusion_detector.evaluate(memberships)  # must run before loitering
            loitering_events = loitering_detector.evaluate(tracks, memberships)
            events = event_engine.evaluate(intrusion_events, loitering_events)

            output_generator.write_frame(processed, tracks, list(zone_manager.zones), events)
            total_events += len(events)

    output_generator.release()

    assert total_events >= 2, "Expected at least one intrusion and one loitering event to drive output generation."

    # Annotated video exists.
    video_path = output_generator.latest_video()
    assert video_path is not None and video_path.exists() and video_path.stat().st_size > 0

    # Snapshots exist -- one per event, sequentially numbered.
    snapshot_files = sorted((output_dir / "snapshots").glob("event_*.jpg"))
    assert len(snapshot_files) == total_events
    assert snapshot_files[0].name == "event_001.jpg"

    # Video clips exist -- one per event.
    clip_files = sorted((output_dir / "clips").glob("event_*.mp4"))
    assert len(clip_files) == total_events
    for clip_file in clip_files:
        assert clip_file.stat().st_size > 0

    # JSON log exists and has one record per event.
    log_paths = output_generator.latest_event_log()
    assert log_paths.json_path.exists()
    import json

    records = json.loads(log_paths.json_path.read_text())
    assert len(records) == total_events

    # CSV log exists with a header plus one row per event.
    assert log_paths.csv_path.exists()
    csv_lines = log_paths.csv_path.read_text().strip().splitlines()
    assert len(csv_lines) == total_events + 1
