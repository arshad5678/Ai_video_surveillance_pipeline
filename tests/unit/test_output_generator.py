"""Unit tests for OutputGenerator — real (tiny, tmp_path-scoped) file I/O throughout,
except where a failure path needs cv2 itself to be forced into erroring."""

from pathlib import Path
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.surveillance.models.domain.bounding_box import BoundingBox
from src.surveillance.models.domain.frame import Frame
from src.surveillance.models.domain.processed_frame import ProcessedFrame
from src.surveillance.models.domain.surveillance_event import (
    EventSeverity,
    EventSource,
    EventType,
    SurveillanceEvent,
)
from src.surveillance.models.domain.track import Track
from src.surveillance.models.domain.zone import Zone
from src.surveillance.models.domain.zone_point import ZonePoint
from src.surveillance.pipelines.output import (
    ClipGenerationError,
    OutputConfig,
    OutputGenerator,
    SnapshotError,
    VideoWriterError,
)


def make_config(output_directory: Path, **overrides) -> OutputConfig:
    defaults = dict(
        annotated_video=True,
        snapshots=True,
        clips=True,
        json_log=True,
        csv_log=True,
        clip_pre_seconds=0.2,
        clip_post_seconds=0.2,
        output_directory=str(output_directory),
        video_codec="mp4v",
        jpeg_quality=90,
        frame_rate=10.0,
    )
    defaults.update(overrides)
    return OutputConfig(**defaults)


def make_processed_frame(index: int = 0, timestamp: float = 1.0) -> ProcessedFrame:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    frame = Frame(index=index, image=image, timestamp=timestamp, source_id="cam-1")
    return ProcessedFrame(original=frame, image=image)


def make_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        bounding_box=BoundingBox(x1=10, y1=10, x2=40, y2=50),
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
        history=((25.0, 30.0),),
    )


def make_zone() -> Zone:
    return Zone(
        zone_id="zone_a",
        zone_name="Zone A",
        zone_type="intrusion",
        polygon=(ZonePoint(x=0, y=0), ZonePoint(x=80, y=0), ZonePoint(x=80, y=60), ZonePoint(x=0, y=60)),
        enabled=True,
    )


def make_event(event_id: str = "evt-1", track_id: int = 1, frame_index: int = 0) -> SurveillanceEvent:
    return SurveillanceEvent(
        event_id=event_id,
        event_type=EventType.INTRUSION_ENTER,
        severity=EventSeverity.HIGH,
        source=EventSource.INTRUSION,
        track_id=track_id,
        zone_id="zone_a",
        zone_name="Zone A",
        timestamp=1.0,
        frame_index=frame_index,
        source_id="cam-1",
        payload=MappingProxyType({"event_type": "ENTER"}),
    )


# --- directory creation -----------------------------------------------------


def test_creates_output_directories_on_init(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    OutputGenerator(make_config(output_dir))

    assert (output_dir / "annotated_video").is_dir()
    assert (output_dir / "snapshots").is_dir()
    assert (output_dir / "clips").is_dir()
    assert (output_dir / "logs").is_dir()


# --- annotated video ---------------------------------------------------------


def test_write_frame_creates_annotated_video(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))

    generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [])
    generator.release()

    video_path = generator.latest_video()
    assert video_path is not None
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_video_writer_error_raised_when_writer_fails_to_open(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))

    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = False
    with patch("src.surveillance.pipelines.output._video_writer.cv2.VideoWriter", return_value=mock_writer):
        with pytest.raises(VideoWriterError):
            generator.write_frame(make_processed_frame(), [], [], [])


# --- snapshots ----------------------------------------------------------------


def test_write_frame_with_event_creates_snapshot(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))

    generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [make_event()])

    snapshot_path = generator.latest_snapshot()
    assert snapshot_path is not None
    assert snapshot_path.name == "event_001.jpg"
    assert snapshot_path.exists()


def test_snapshot_error_raised_when_imwrite_fails(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))

    with patch("src.surveillance.pipelines.output.output_generator.cv2.imwrite", return_value=False):
        with pytest.raises(SnapshotError):
            generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [make_event()])


def test_no_snapshot_when_disabled(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out", snapshots=False))

    generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [make_event()])

    assert generator.latest_snapshot() is None
    assert list((tmp_path / "out" / "snapshots").iterdir()) == []


# --- clips ----------------------------------------------------------------


def test_clip_file_written_after_post_event_frames_collected(tmp_path: Path) -> None:
    # frame_rate=10, clip_post_seconds=0.2 -> 2 post-frames needed to complete a clip.
    generator = OutputGenerator(make_config(tmp_path / "out"))

    generator.write_frame(make_processed_frame(index=0), [make_track()], [make_zone()], [make_event(frame_index=0)])
    clip_path = tmp_path / "out" / "clips" / "event_001.mp4"
    assert not clip_path.exists()  # not yet complete: needs 2 post-event frames

    generator.write_frame(make_processed_frame(index=1), [make_track()], [make_zone()], [])
    generator.write_frame(make_processed_frame(index=2), [make_track()], [make_zone()], [])

    assert clip_path.exists()
    assert clip_path.stat().st_size > 0


def test_incomplete_clip_finalized_on_release(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))

    generator.write_frame(make_processed_frame(index=0), [make_track()], [make_zone()], [make_event(frame_index=0)])
    clip_path = tmp_path / "out" / "clips" / "event_001.mp4"
    assert not clip_path.exists()

    generator.release()

    assert clip_path.exists()  # finalized with whatever frames were available


def test_clip_generation_error_wraps_writer_failure(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))
    generator.write_frame(make_processed_frame(index=0), [make_track()], [make_zone()], [make_event(frame_index=0)])

    mock_writer = MagicMock()
    mock_writer.isOpened.return_value = False
    with patch("src.surveillance.pipelines.output._video_writer.cv2.VideoWriter", return_value=mock_writer):
        generator.write_frame(make_processed_frame(index=1), [], [], [])  # 1st post-frame: not yet complete
        with pytest.raises(ClipGenerationError):
            generator.write_frame(make_processed_frame(index=2), [], [], [])  # 2nd post-frame: completes -> writes


def test_no_clip_when_disabled(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out", clips=False))

    for index in range(4):
        generator.write_frame(make_processed_frame(index=index), [make_track()], [make_zone()], [make_event(frame_index=0)] if index == 0 else [])
    generator.release()

    assert list((tmp_path / "out" / "clips").iterdir()) == []


# --- JSON / CSV logs -----------------------------------------------------


def test_json_and_csv_logs_written_on_event(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))

    generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [make_event(event_id="evt-1")])

    log_paths = generator.latest_event_log()
    assert log_paths.json_path.exists()
    assert log_paths.csv_path.exists()

    json_content = log_paths.json_path.read_text()
    assert "evt-1" in json_content
    assert "intrusion_enter" in json_content

    csv_lines = log_paths.csv_path.read_text().strip().splitlines()
    assert csv_lines[0] == "event_id,event_type,severity,track_id,zone_id,timestamp,frame_index"
    assert csv_lines[1].startswith("evt-1,intrusion_enter,HIGH,1,zone_a,")


def test_json_log_accumulates_multiple_events(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out"))

    generator.write_frame(make_processed_frame(index=0), [make_track()], [make_zone()], [make_event(event_id="evt-1", frame_index=0)])
    generator.write_frame(make_processed_frame(index=1), [make_track()], [make_zone()], [make_event(event_id="evt-2", frame_index=1)])

    import json

    records = json.loads(generator.latest_event_log().json_path.read_text())
    assert [r["event_id"] for r in records] == ["evt-1", "evt-2"]


def test_no_logs_when_disabled(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out", json_log=False, csv_log=False))

    generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [make_event()])

    log_paths = generator.latest_event_log()
    assert log_paths.json_path is None
    assert log_paths.csv_path is None
    assert list((tmp_path / "out" / "logs").iterdir()) == []


# --- getters / misc -----------------------------------------------------


def test_latest_snapshot_falls_back_to_disk_for_a_fresh_instance(tmp_path: Path) -> None:
    # Simulates a separate reader process (e.g. the Prompt 12 API) that never
    # called write_frame() itself, but the snapshot files already exist on disk.
    writer = OutputGenerator(make_config(tmp_path / "out"))
    writer.write_frame(make_processed_frame(), [make_track()], [make_zone()], [make_event(event_id="evt-1")])

    reader = OutputGenerator(make_config(tmp_path / "out"))

    assert reader.latest_snapshot() == writer.latest_snapshot() == tmp_path / "out" / "snapshots" / "event_001.jpg"


def test_latest_video_is_none_when_annotated_video_disabled(tmp_path: Path) -> None:
    generator = OutputGenerator(make_config(tmp_path / "out", annotated_video=False))

    generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [])

    assert generator.latest_video() is None


def test_context_manager_releases_resources(tmp_path: Path) -> None:
    with OutputGenerator(make_config(tmp_path / "out")) as generator:
        generator.write_frame(make_processed_frame(), [make_track()], [make_zone()], [])

    assert generator.latest_video().exists()
