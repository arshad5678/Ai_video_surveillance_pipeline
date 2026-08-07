"""Unit tests for the FastAPI Backend module.

Uses a real (but tmp_path-isolated) ServiceContainer -- ZoneManager,
EventEngine, and OutputGenerator are all cheap to construct (no YOLO/
tracking involved) -- injected via `app.dependency_overrides[get_container]`
rather than the real lifespan-built one, so no repo-root files are
touched and no network/hardware is required.
"""

import json
import time
from pathlib import Path
from types import MappingProxyType
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from src.surveillance.api.app import create_app
from src.surveillance.api.dependencies.container import ServiceContainer
from src.surveillance.api.dependencies.providers import get_container
from src.surveillance.core.settings import get_settings
from src.surveillance.models.domain.surveillance_event import EventSeverity
from src.surveillance.pipelines.events import EventEngine, EventEngineConfig
from src.surveillance.pipelines.output import OutputConfig, OutputGenerator
from src.surveillance.pipelines.video_input import VideoSourceConfig, VideoSourceType
from src.surveillance.pipelines.zones import ZoneManager

ZONES_YAML = """
zones:
  - id: watch_zone
    name: Watch Zone
    type: intrusion
    enabled: true
    polygon:
      - [0, 0]
      - [100, 0]
      - [100, 100]
      - [0, 100]
"""

_SEVERITY_MAPPING = {
    "intrusion_enter": EventSeverity.HIGH,
    "intrusion_exit": EventSeverity.LOW,
    "loitering": EventSeverity.MEDIUM,
}


def make_container(tmp_path: Path) -> ServiceContainer:
    zones_path = tmp_path / "zones.yaml"
    zones_path.write_text(ZONES_YAML)

    return ServiceContainer(
        settings=get_settings(),
        config_path=str(tmp_path / "config.yaml"),  # doesn't need to exist; yaml_config is set directly below
        zones_path=str(zones_path),
        yaml_config={"app": {"name": "Test Pipeline", "version": "0.1.0"}},
        zone_manager=ZoneManager(zones_path),
        event_engine=EventEngine(EventEngineConfig(severity_mapping=dict(_SEVERITY_MAPPING))),
        output_generator=OutputGenerator(OutputConfig(output_directory=str(tmp_path / "output"))),
        video_source_config=VideoSourceConfig(source_type=VideoSourceType.FILE, uri="does/not/exist.mp4"),
        started_at=time.monotonic() - 5.0,  # pretend the app has been up 5s
    )


def write_event_records(container: ServiceContainer, records) -> None:
    log_paths = container.output_generator.latest_event_log()
    log_paths.json_path.parent.mkdir(parents=True, exist_ok=True)
    log_paths.json_path.write_text(json.dumps(records))


@pytest.fixture()
def container(tmp_path: Path) -> ServiceContainer:
    return make_container(tmp_path)


@pytest.fixture()
def client(container: ServiceContainer) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --- health -----------------------------------------------------------------


def test_health_returns_status_version_uptime(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["version"] == "0.1.0"
    assert body["uptime"] >= 5.0


# --- config -------------------------------------------------------------


def test_get_config_returns_loaded_yaml_and_zone_count(client: TestClient) -> None:
    response = client.get("/api/v1/config")

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["app"]["name"] == "Test Pipeline"
    assert body["zone_count"] == 1


def test_config_reload_updates_zone_count(client: TestClient, container: ServiceContainer, tmp_path: Path) -> None:
    zones_path = Path(container.zones_path)
    zones_path.write_text(ZONES_YAML.replace("watch_zone", "second_zone") + "\n" + ZONES_YAML)
    # Two zones now share duplicate ids across the concatenated blocks, so instead
    # append a second distinct zone properly:
    zones_path.write_text(
        ZONES_YAML
        + """
  - id: second_zone
    name: Second Zone
    type: intrusion
    enabled: true
    polygon:
      - [200, 200]
      - [300, 200]
      - [300, 300]
      - [200, 300]
"""
    )

    response = client.post("/api/v1/config/reload")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "reloaded"
    assert body["zone_count"] == 2


def test_config_reload_failure_returns_400_and_leaves_state_untouched(
    client: TestClient, container: ServiceContainer
) -> None:
    Path(container.zones_path).write_text("not: [valid, zones")  # malformed YAML

    response = client.post("/api/v1/config/reload")

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "ConfigurationReloadError"
    assert len(container.zone_manager.zones) == 1  # unchanged


def test_build_container_and_reload_do_not_wipe_existing_pipeline_output(tmp_path: Path) -> None:
    """Regression test: OutputGenerator defaults to wiping output/ on construction
    (clean_previous_outputs=True, added so a real pipeline run starts fresh), but
    this API's OutputGenerator is read-only and gets rebuilt on every startup and
    every /config/reload -- pointed at the *same* directory a real pipeline run
    already populated. build_container()/reload_config() must build it with
    clean_previous_outputs=False, or restarting/reloading the API would silently
    delete real snapshots/clips/logs a pipeline run already produced.
    """
    from src.surveillance.api.dependencies.container import build_container, reload_config

    output_dir = tmp_path / "output"
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True)
    real_snapshot = snapshots_dir / "event_001.jpg"
    real_snapshot.write_bytes(b"real pipeline snapshot bytes")

    # build_container() always reads zones from DEFAULT_ZONES_CONFIG_PATH (config/zones.yaml,
    # the real repo file) -- only output_directory needs to be redirected to tmp_path here.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f'output:\n  output_directory: "{output_dir.as_posix()}"\n')

    settings = get_settings().model_copy(update={"config_path": str(config_path)})

    built_container = build_container(settings)
    assert real_snapshot.exists(), "build_container() must not wipe pre-existing output artifacts"

    reload_config(built_container)
    assert real_snapshot.exists(), "reload_config() must not wipe pre-existing output artifacts"


# --- camera ---------------------------------------------------------------


def test_camera_status_reports_disconnected_for_missing_file(client: TestClient) -> None:
    response = client.get("/api/v1/camera/status")

    assert response.status_code == 200
    body = response.json()
    assert body["source_type"] == "file"
    assert body["connected"] is False
    assert body["fps"] is None
    assert body["resolution"] is None


# --- events -----------------------------------------------------------------


def _record(event_id, event_type="intrusion_enter", severity="HIGH", track_id=1, zone_id="watch_zone", timestamp=1.0, frame_index=0):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "severity": severity,
        "track_id": track_id,
        "zone_id": zone_id,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "payload": {},
    }


def test_list_events_empty_when_no_log_exists(client: TestClient) -> None:
    response = client.get("/api/v1/events")

    assert response.status_code == 200
    assert response.json() == {"events": [], "count": 0}


def test_list_events_returns_newest_first(client: TestClient, container: ServiceContainer) -> None:
    write_event_records(
        container,
        [_record("e1", timestamp=1.0, frame_index=0), _record("e2", timestamp=2.0, frame_index=0)],
    )

    response = client.get("/api/v1/events")

    assert response.status_code == 200
    body = response.json()
    assert [e["event_id"] for e in body["events"]] == ["e2", "e1"]
    assert body["count"] == 2


def test_list_events_filters_by_severity_and_zone(client: TestClient, container: ServiceContainer) -> None:
    write_event_records(
        container,
        [
            _record("e1", severity="HIGH", zone_id="watch_zone"),
            _record("e2", severity="LOW", zone_id="watch_zone", timestamp=2.0),
            _record("e3", severity="HIGH", zone_id="other_zone", timestamp=3.0),
        ],
    )

    response = client.get("/api/v1/events", params={"severity": "HIGH", "zone_id": "watch_zone"})

    assert response.status_code == 200
    body = response.json()
    assert [e["event_id"] for e in body["events"]] == ["e1"]


def test_list_events_respects_limit(client: TestClient, container: ServiceContainer) -> None:
    write_event_records(container, [_record(f"e{i}", timestamp=float(i)) for i in range(5)])

    response = client.get("/api/v1/events", params={"limit": 2})

    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_list_events_invalid_track_id_returns_422(client: TestClient) -> None:
    response = client.get("/api/v1/events", params={"track_id": "not-a-number"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ValidationError"


def test_get_event_by_id(client: TestClient, container: ServiceContainer) -> None:
    write_event_records(container, [_record("e1"), _record("e2")])

    response = client.get("/api/v1/events/e2")

    assert response.status_code == 200
    assert response.json()["event_id"] == "e2"


def test_get_event_missing_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/events/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "ResourceNotFoundError"


# --- outputs ----------------------------------------------------------------


def test_latest_video_404_when_not_generated(client: TestClient) -> None:
    response = client.get("/api/v1/outputs/latest/video")
    assert response.status_code == 404


def test_latest_snapshot_streams_real_file(client: TestClient, container: ServiceContainer) -> None:
    # Write directly via the real OutputGenerator write path so the file is genuine.
    from src.surveillance.models.domain.bounding_box import BoundingBox
    from src.surveillance.models.domain.frame import Frame
    from src.surveillance.models.domain.processed_frame import ProcessedFrame
    from src.surveillance.models.domain.surveillance_event import EventSource, EventType, SurveillanceEvent
    from src.surveillance.models.domain.track import Track
    import numpy as np

    image = np.zeros((40, 40, 3), dtype="uint8")
    frame = Frame(index=0, image=image, timestamp=1.0, source_id="cam-1")
    processed = ProcessedFrame(original=frame, image=image)
    track = Track(
        track_id=1,
        bounding_box=BoundingBox(x1=1, y1=1, x2=10, y2=10),
        confidence=0.9,
        class_name="person",
        class_id=0,
        timestamp=1.0,
        frame_index=0,
        source_id="cam-1",
        is_confirmed=True,
        age=1,
        hits=1,
        time_since_update=0,
        history=(),
    )
    event = SurveillanceEvent(
        event_id="evt-1",
        event_type=EventType.INTRUSION_ENTER,
        severity=EventSeverity.HIGH,
        source=EventSource.INTRUSION,
        track_id=1,
        zone_id="watch_zone",
        zone_name="Watch Zone",
        timestamp=1.0,
        frame_index=0,
        source_id="cam-1",
        payload=MappingProxyType({}),
    )
    container.output_generator.write_frame(processed, [track], [], [event])

    response = client.get("/api/v1/outputs/latest/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0


def test_latest_json_and_csv_stream_real_files(client: TestClient, container: ServiceContainer) -> None:
    write_event_records(container, [_record("e1")])
    csv_path = container.output_generator.latest_event_log().csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("event_id,event_type,severity,track_id,zone_id,timestamp,frame_index\ne1,intrusion_enter,HIGH,1,watch_zone,1.0,0\n")

    json_response = client.get("/api/v1/outputs/latest/json")
    csv_response = client.get("/api/v1/outputs/latest/csv")

    assert json_response.status_code == 200
    assert json_response.headers["content-type"] == "application/json"
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"] == "text/csv; charset=utf-8"


# --- system -----------------------------------------------------------------


def test_system_status_reports_counts_from_event_log(client: TestClient, container: ServiceContainer) -> None:
    write_event_records(
        container,
        [_record("e1", track_id=1), _record("e2", track_id=2), _record("e3", track_id=1, timestamp=2.0)],
    )

    response = client.get("/api/v1/system")

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline_status"] == "ready"
    assert body["event_count"] == 3
    assert body["track_count"] == 2  # distinct track_ids: {1, 2}
    assert body["memory_usage_mb"] > 0
    assert "zone_manager" in body["modules_initialized"]


# --- root / docs --------------------------------------------------------


def test_root_and_docs_are_reachable(client: TestClient) -> None:
    assert client.get("/").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
