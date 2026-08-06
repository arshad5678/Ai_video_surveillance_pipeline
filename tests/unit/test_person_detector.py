"""Unit tests for PersonDetector — ultralytics.YOLO is mocked throughout, so
these tests need no GPU, no network access, and no real model weights."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.surveillance.models.domain.frame import Frame
from src.surveillance.models.domain.processed_frame import ProcessedFrame
from src.surveillance.pipelines.detection import (
    DetectorConfig,
    DeviceUnavailableError,
    InferenceError,
    InvalidDetectorConfigError,
    ModelLoadError,
    PersonDetector,
)

PATCH_TARGET = "src.surveillance.pipelines.detection.person_detector.YOLO"

COCO_NAMES = {0: "person", 1: "bicycle", 2: "car"}


class FakeBox:
    """Mimics a single ultralytics Boxes row closely enough for _to_detections()."""

    def __init__(self, cls_id: int, conf: float, xyxy: tuple) -> None:
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes


def make_processed_frame(index: int = 0) -> ProcessedFrame:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    frame = Frame(index=index, image=image, timestamp=42.0, source_id="cam-1")
    return ProcessedFrame(original=frame, image=image)


def make_dummy_model_file(tmp_path: Path) -> str:
    model_file = tmp_path / "yolov8n.pt"
    model_file.write_bytes(b"not a real model, just needs to exist")
    return str(model_file)


def make_mock_yolo(names: dict = COCO_NAMES) -> MagicMock:
    mock_model = MagicMock()
    mock_model.names = names
    return mock_model


def test_missing_model_file_raises_model_load_error() -> None:
    config = DetectorConfig(model_path="does/not/exist.pt")

    with pytest.raises(ModelLoadError):
        PersonDetector(config)


def test_invalid_confidence_threshold_raises_before_loading_model(tmp_path: Path) -> None:
    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path), confidence_threshold=1.5)

    with pytest.raises(InvalidDetectorConfigError):
        PersonDetector(config)


def test_invalid_iou_threshold_raises(tmp_path: Path) -> None:
    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path), iou_threshold=-0.1)

    with pytest.raises(InvalidDetectorConfigError):
        PersonDetector(config)


def test_invalid_image_size_raises(tmp_path: Path) -> None:
    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path), image_size=0)

    with pytest.raises(InvalidDetectorConfigError):
        PersonDetector(config)


def test_cuda_requested_but_unavailable_raises_device_unavailable_error(tmp_path: Path) -> None:
    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path), device="cuda")

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(DeviceUnavailableError):
            PersonDetector(config)


@patch(PATCH_TARGET)
def test_model_without_person_class_raises_model_load_error(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
    mock_yolo_cls.return_value = make_mock_yolo(names={0: "cat", 1: "dog"})
    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path))

    with pytest.raises(ModelLoadError):
        PersonDetector(config)


@patch(PATCH_TARGET)
def test_model_is_loaded_exactly_once(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
    mock_model = make_mock_yolo()
    mock_model.predict.return_value = [FakeResult(boxes=[])]
    mock_yolo_cls.return_value = mock_model

    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path))
    detector = PersonDetector(config)

    for i in range(5):
        detector.detect(make_processed_frame(index=i))

    mock_yolo_cls.assert_called_once()
    assert mock_model.predict.call_count == 5


@patch(PATCH_TARGET)
def test_detect_returns_person_detections_with_correct_fields(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
    mock_model = make_mock_yolo()
    mock_model.predict.return_value = [
        FakeResult(boxes=[FakeBox(cls_id=0, conf=0.87, xyxy=(10.0, 20.0, 60.0, 120.0))])
    ]
    mock_yolo_cls.return_value = mock_model

    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path))
    detector = PersonDetector(config)
    processed_frame = make_processed_frame(index=3)

    detections = detector.detect(processed_frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.track_id is None
    assert detection.class_name == "person"
    assert detection.class_id == 0
    assert detection.confidence == pytest.approx(0.87, abs=1e-4)
    assert detection.bounding_box.x1 == 10.0
    assert detection.bounding_box.y2 == 120.0
    assert detection.frame_index == 3
    assert detection.timestamp == processed_frame.timestamp
    assert detection.source_id == processed_frame.source_id


@patch(PATCH_TARGET)
def test_detect_filters_out_non_person_boxes_as_safety_net(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
    mock_model = make_mock_yolo()
    mock_model.predict.return_value = [
        FakeResult(
            boxes=[
                FakeBox(cls_id=2, conf=0.9, xyxy=(0.0, 0.0, 10.0, 10.0)),  # car — must be dropped
                FakeBox(cls_id=0, conf=0.8, xyxy=(0.0, 0.0, 5.0, 5.0)),  # person — kept
            ]
        )
    ]
    mock_yolo_cls.return_value = mock_model

    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path))
    detector = PersonDetector(config)

    detections = detector.detect(make_processed_frame())

    assert len(detections) == 1
    assert detections[0].class_name == "person"


@patch(PATCH_TARGET)
def test_detect_raises_inference_error_on_predict_failure(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
    mock_model = make_mock_yolo()
    mock_model.predict.side_effect = RuntimeError("boom")
    mock_yolo_cls.return_value = mock_model

    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path))
    detector = PersonDetector(config)

    with pytest.raises(InferenceError):
        detector.detect(make_processed_frame())


@patch(PATCH_TARGET)
def test_original_processed_frame_image_is_not_mutated(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
    mock_model = make_mock_yolo()
    mock_model.predict.return_value = [FakeResult(boxes=[])]
    mock_yolo_cls.return_value = mock_model

    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path))
    detector = PersonDetector(config)
    processed_frame = make_processed_frame()
    original_bytes = processed_frame.image.copy()

    detector.detect(processed_frame)

    assert np.array_equal(processed_frame.image, original_bytes)


@patch(PATCH_TARGET)
def test_detect_stream_yields_one_detection_list_per_frame(mock_yolo_cls: MagicMock, tmp_path: Path) -> None:
    mock_model = make_mock_yolo()
    mock_model.predict.return_value = [FakeResult(boxes=[])]
    mock_yolo_cls.return_value = mock_model

    config = DetectorConfig(model_path=make_dummy_model_file(tmp_path))
    detector = PersonDetector(config)
    frames = [make_processed_frame(index=i) for i in range(3)]

    results = list(detector.detect_stream(frames))

    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)
