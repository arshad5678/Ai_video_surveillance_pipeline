"""Integration test: run the real pipeline end-to-end, then hit the real FastAPI app.

Reuses the same static-image-loop technique as the Prompt 10/11
integration tests to produce real intrusion + loitering events, runs
them through the real OutputGenerator to produce real files on disk,
then verifies the FastAPI Backend (Prompt 12) correctly exposes exactly
those files/events over HTTP -- JSON responses, file downloads, and
config reload -- with no mocking anywhere in this test.
"""

import json
import time
from pathlib import Path

import cv2
import pytest
import ultralytics
from fastapi.testclient import TestClient

from src.surveillance.api.app import create_app
from src.surveillance.api.dependencies.container import ServiceContainer
from src.surveillance.api.dependencies.providers import get_container
from src.surveillance.core.settings import get_settings
from src.surveillance.models.domain.surveillance_event import EventSeverity
from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
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


def _ensure_weights_available() -> None:
    if WEIGHTS_PATH.exists():
        return
    try:
        from ultralytics import YOLO

        YOLO(str(WEIGHTS_PATH))
    except Exception:
        pytest.skip(f"YOLO weights unavailable at {WEIGHTS_PATH} and could not be downloaded (no network?).")


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Runs the full pipeline once and returns (container, output_dir) for every test in this module."""
    _ensure_weights_available()

    workdir = tmp_path_factory.mktemp("api_integration")
    output_dir = workdir / "output"
    zones_path = workdir / "zones.yaml"
    zones_path.write_text(ZONES_YAML)

    image = cv2.imread(str(DEMO_IMAGE_PATH))
    assert image is not None
    height, width = image.shape[:2]
    video_path = workdir / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FRAME_RATE, (width, height))
    for _ in range(FRAME_COUNT):
        writer.write(image)
    writer.release()

    zone_manager = ZoneManager(zones_path)
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))
    tracker = MultiObjectTracker(TrackingConfig(frame_rate=FRAME_RATE))
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
        OutputConfig(clip_pre_seconds=1.0, clip_post_seconds=1.0, output_directory=str(output_dir), frame_rate=FRAME_RATE)
    )

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(video_path))
    with VideoInput(video_config) as video_input:
        for frame in video_input.frames():
            processed = processor.process(frame)
            if processed is None:
                continue
            detections = detector.detect(processed)
            tracks = tracker.update(detections)
            memberships = zone_manager.evaluate(tracks)
            intrusion_events = intrusion_detector.evaluate(memberships)
            loitering_events = loitering_detector.evaluate(tracks, memberships)
            events = event_engine.evaluate(intrusion_events, loitering_events)
            output_generator.write_frame(processed, tracks, list(zone_manager.zones), events)
    output_generator.release()

    container = ServiceContainer(
        settings=get_settings(),
        config_path=str(workdir / "config.yaml"),
        zones_path=str(zones_path),
        yaml_config={"app": {"name": "AI Video Surveillance Pipeline"}},
        zone_manager=zone_manager,
        event_engine=event_engine,
        output_generator=output_generator,
        video_source_config=VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(video_path)),
        started_at=time.monotonic() - 1.0,
    )
    return container, output_dir


@pytest.fixture()
def client(pipeline_run) -> TestClient:
    container, _ = pipeline_run
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_reports_up(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_events_endpoint_returns_real_pipeline_events(client: TestClient, pipeline_run) -> None:
    container, _ = pipeline_run

    response = client.get("/api/v1/events")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] >= 2
    event_types = {event["event_type"] for event in body["events"]}
    assert "intrusion_enter" in event_types
    assert "loitering" in event_types

    # Cross-check against the real JSON log OutputGenerator wrote to disk.
    raw_records = json.loads(container.output_generator.latest_event_log().json_path.read_text())
    assert body["count"] == len(raw_records)


def test_get_single_event_matches_log(client: TestClient) -> None:
    events = client.get("/api/v1/events").json()["events"]
    target = events[0]

    response = client.get(f"/api/v1/events/{target['event_id']}")

    assert response.status_code == 200
    assert response.json()["event_id"] == target["event_id"]


def test_outputs_latest_video_downloads_real_mp4(client: TestClient) -> None:
    response = client.get("/api/v1/outputs/latest/video")

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 0
    assert b"ftyp" in response.content[:16]  # MP4 'ftyp' box always appears near the start


def test_outputs_latest_snapshot_downloads_real_jpeg(client: TestClient) -> None:
    response = client.get("/api/v1/outputs/latest/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_outputs_latest_json_matches_disk_file(client: TestClient, pipeline_run) -> None:
    container, _ = pipeline_run
    on_disk = container.output_generator.latest_event_log().json_path.read_text()

    response = client.get("/api/v1/outputs/latest/json")

    assert response.status_code == 200
    assert response.json() == json.loads(on_disk)


def test_outputs_latest_csv_matches_disk_file(client: TestClient, pipeline_run) -> None:
    container, _ = pipeline_run
    on_disk = container.output_generator.latest_event_log().csv_path.read_bytes()

    response = client.get("/api/v1/outputs/latest/csv")

    assert response.status_code == 200
    assert response.content == on_disk


def test_system_status_reflects_real_run(client: TestClient) -> None:
    response = client.get("/api/v1/system")

    assert response.status_code == 200
    body = response.json()
    assert body["frame_count"] == FRAME_COUNT
    assert body["event_count"] >= 2
    assert body["track_count"] >= 1


def test_config_reload_against_real_zones_file(client: TestClient, pipeline_run) -> None:
    container, _ = pipeline_run

    response = client.post("/api/v1/config/reload")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reloaded"
    assert body["zone_count"] == len(container.zone_manager.zones)
