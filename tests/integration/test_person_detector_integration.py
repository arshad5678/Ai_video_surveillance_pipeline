"""Integration test: VideoInput -> FrameProcessor -> PersonDetector, real YOLO inference.

Builds its sample video from ultralytics' own bundled demo image
(zidane.jpg, ships with the ultralytics package — no network or external
dataset needed) so the assertions are meaningful: this image reliably
contains people that a COCO-pretrained YOLO model detects.

Marked `integration`: loads real weights and runs real inference, so it's
slower than the mocked unit tests. If weights/yolov8n.pt is missing, this
attempts a one-time download and skips cleanly if that's not possible
(e.g. no network).
"""

from pathlib import Path
from typing import List

import cv2
import pytest
import ultralytics

from src.surveillance.models.domain.detection import Detection
from src.surveillance.pipelines.detection import DetectorConfig, PersonDetector
from src.surveillance.pipelines.frame_processing import FrameProcessor, PreprocessConfig
from src.surveillance.pipelines.video_input import VideoInput, VideoSourceConfig, VideoSourceType

pytestmark = pytest.mark.integration

WEIGHTS_PATH = Path("weights/yolov8n.pt")
DEMO_IMAGE_PATH = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> Path:
    """A short synthetic clip built by looping ultralytics' bundled demo photo (real people)."""
    image = cv2.imread(str(DEMO_IMAGE_PATH))
    assert image is not None, f"Could not read bundled demo image: {DEMO_IMAGE_PATH}"

    height, width = image.shape[:2]
    video_path = tmp_path_factory.mktemp("integration") / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for _ in range(10):
        writer.write(image)
    writer.release()
    return video_path


def _ensure_weights_available() -> None:
    if WEIGHTS_PATH.exists():
        return
    try:
        from ultralytics import YOLO

        YOLO(str(WEIGHTS_PATH))  # triggers a one-time download to weights/
    except Exception:
        pytest.skip(f"YOLO weights unavailable at {WEIGHTS_PATH} and could not be downloaded (no network?).")


def test_end_to_end_pipeline_detects_people_in_sample_video(sample_video: Path) -> None:
    _ensure_weights_available()

    video_config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri=str(sample_video))
    processor = FrameProcessor(PreprocessConfig())
    detector = PersonDetector(DetectorConfig(model_path=str(WEIGHTS_PATH), device="cpu"))

    all_detections: List[Detection] = []
    with VideoInput(video_config) as video_input:
        processed_frames = processor.process_stream(video_input.frames())
        for detections in detector.detect_stream(processed_frames):
            all_detections.extend(detections)

    assert len(all_detections) > 0, "Expected at least one person detection in the demo image."
    for detection in all_detections:
        assert isinstance(detection, Detection)
        assert detection.class_name == "person"
        assert detection.track_id is None
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.bounding_box.width > 0
        assert detection.bounding_box.height > 0
