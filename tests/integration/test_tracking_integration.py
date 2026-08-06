"""Integration test: VideoInput -> FrameProcessor -> PersonDetector -> MultiObjectTracker.

Reuses ultralytics' bundled demo image (zidane.jpg, two real people) looped
into a short clip, same approach as the Prompt 5 detection integration
test. Since the same two people appear in (almost) the same position every
frame, this verifies that track IDs assigned early on remain stable through
to the last frame — the core promise of this module.
"""

from pathlib import Path
from typing import List

import cv2
import pytest
import ultralytics

from src.surveillance.models.domain.track import Track
from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
from src.surveillance.pipelines.frame_processing import FrameProcessor, PreprocessConfig
from src.surveillance.pipelines.tracking import MultiObjectTracker, TrackingConfig
from src.surveillance.pipelines.video_input import VideoInput, VideoSourceConfig, VideoSourceType

pytestmark = pytest.mark.integration

WEIGHTS_PATH = Path("weights/yolov8n.pt")
DEMO_IMAGE_PATH = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
FRAME_COUNT = 15


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


def _ensure_weights_available() -> None:
    if WEIGHTS_PATH.exists():
        return
    try:
        from ultralytics import YOLO

        YOLO(str(WEIGHTS_PATH))
    except Exception:
        pytest.skip(f"YOLO weights unavailable at {WEIGHTS_PATH} and could not be downloaded (no network?).")


def test_track_ids_remain_stable_across_frames(sample_video: Path) -> None:
    _ensure_weights_available()

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(sample_video))
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))
    tracker = MultiObjectTracker(TrackingConfig(frame_rate=5))

    per_frame_tracks: List[List[Track]] = []
    with VideoInput(video_config) as video_input:
        processed_frames = processor.process_stream(video_input.frames())
        detections_stream = detector.detect_stream(processed_frames)
        for tracks in tracker.track_stream(detections_stream):
            per_frame_tracks.append(tracks)

    assert len(per_frame_tracks) == FRAME_COUNT

    per_frame_ids = [{t.track_id for t in tracks} for tracks in per_frame_tracks]

    # First frame or two may still be settling (a track needs 2 hits to be
    # "confirmed"); compare stability from frame index 2 onward.
    stable_frames = per_frame_ids[2:]
    assert all(len(ids) == 2 for ids in stable_frames), "Expected 2 stable person tracks per frame."

    reference_ids = stable_frames[0]
    assert all(ids == reference_ids for ids in stable_frames), "Track IDs should not change once stabilized."

    last_frame_tracks = per_frame_tracks[-1]
    assert all(t.is_confirmed for t in last_frame_tracks)
    assert all(t.hits >= FRAME_COUNT - 2 for t in last_frame_tracks)
    assert all(len(t.history) > 1 for t in last_frame_tracks)
    for track in last_frame_tracks:
        assert track.class_name == "person"
        assert track.time_since_update == 0
