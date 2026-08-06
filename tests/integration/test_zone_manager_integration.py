"""Integration test: VideoInput -> FrameProcessor -> PersonDetector -> MultiObjectTracker -> ZoneManager.

Reuses ultralytics' bundled demo image (zidane.jpg, two real people at
known approximate positions in a 1280x720 frame) looped into a short
clip, same approach as the Prompt 5/6 integration tests. Zones are sized
around each person's known position so the test can assert meaningful
inside/outside results, not just "it ran without crashing".
"""

from pathlib import Path
from typing import List

import cv2
import pytest
import ultralytics

from src.surveillance.models.domain.zone_membership import ZoneMembership
from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
from src.surveillance.pipelines.frame_processing import FrameProcessor, PreprocessConfig
from src.surveillance.pipelines.tracking import MultiObjectTracker, TrackingConfig
from src.surveillance.pipelines.video_input import VideoInput, VideoSourceConfig, VideoSourceType
from src.surveillance.pipelines.zones import ZoneManager

pytestmark = pytest.mark.integration

WEIGHTS_PATH = Path("weights/yolov8n.pt")
DEMO_IMAGE_PATH = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
FRAME_COUNT = 10

# zidane.jpg is 1280x720; the two people sit roughly around (625, 455) and
# (945, 378) (confirmed empirically in Prompt 5/6 runs). One zone is sized
# to contain the left person only, the other to contain the right person only.
ZONES_YAML = """
zones:
  - id: left_zone
    name: Left Zone
    type: monitoring
    enabled: true
    polygon:
      - [400, 300]
      - [800, 300]
      - [800, 600]
      - [400, 600]
  - id: right_zone
    name: Right Zone
    type: monitoring
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


def test_zone_memberships_produced_correctly_end_to_end(sample_video: Path, zones_config: Path) -> None:
    _ensure_weights_available()

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(sample_video))
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))
    tracker = MultiObjectTracker(TrackingConfig(frame_rate=5))
    zone_manager = ZoneManager(zones_config)

    per_frame_memberships: List[List[ZoneMembership]] = []
    with VideoInput(video_config) as video_input:
        processed_frames = processor.process_stream(video_input.frames())
        detections_stream = detector.detect_stream(processed_frames)
        tracks_stream = tracker.track_stream(detections_stream)
        for memberships in zone_manager.evaluate_stream(tracks_stream):
            per_frame_memberships.append(memberships)

    assert len(per_frame_memberships) == FRAME_COUNT

    # Every frame with 2 confirmed tracks x 2 zones = 4 memberships.
    stable_frames = per_frame_memberships[2:]
    assert all(len(m) == 4 for m in stable_frames), "Expected 2 tracks x 2 zones = 4 memberships per frame."

    for memberships in stable_frames:
        for membership in memberships:
            assert isinstance(membership, ZoneMembership)
            assert membership.zone_id in {"left_zone", "right_zone"}

    last_frame = stable_frames[-1]
    inside_zone_ids = {m.zone_id for m in last_frame if m.inside}
    assert inside_zone_ids == {"left_zone", "right_zone"}, (
        "Expected exactly one person inside each zone by the last frame."
    )

    # Each zone should have exactly one track inside it and one outside it
    # (two people, two zones, each person in a different zone).
    for zone_id in ("left_zone", "right_zone"):
        zone_memberships = [m for m in last_frame if m.zone_id == zone_id]
        assert sum(1 for m in zone_memberships if m.inside) == 1
        assert sum(1 for m in zone_memberships if not m.inside) == 1
