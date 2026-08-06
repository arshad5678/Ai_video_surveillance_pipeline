"""PersonDetector: detects people in ProcessedFrame images using Ultralytics YOLO.

Sole responsibility: run YOLO inference restricted to the "person" class and
translate raw results into framework-free Detection objects. No tracking,
event, API, or persistence logic lives here or is imported by this module.
"""

from pathlib import Path
from typing import Any, Iterable, Iterator, List

import torch
from loguru import logger
from ultralytics import YOLO

from ...models.domain.bounding_box import BoundingBox
from ...models.domain.detection import Detection
from ...models.domain.processed_frame import ProcessedFrame
from .exceptions import (
    DeviceUnavailableError,
    InferenceError,
    InvalidDetectorConfigError,
    ModelLoadError,
)
from .types import DetectorConfig


class PersonDetector:
    """Detects people in ProcessedFrame images using a YOLO model.

    The model is loaded exactly once, at construction time, and reused for
    every subsequent detect() call — never reloaded per frame.

    Usage:
        detector = PersonDetector(config)
        for detections in detector.detect_stream(processed_frames):
            ...  # hand `detections` to a future tracker
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._validate_config(config)
        self._config = config
        self._model = self._load_model()
        self._person_class_id = self._resolve_person_class_id()
        self._frame_count = 0

    def detect(self, processed_frame: ProcessedFrame) -> List[Detection]:
        """Run person detection on a single ProcessedFrame.

        Only `processed_frame.image` is read — the ProcessedFrame (and the
        original Frame it wraps) is never modified.

        Raises:
            InferenceError: the underlying YOLO call failed.
        """
        logger.debug("Running inference on frame index={}", processed_frame.index)

        try:
            results = self._model.predict(
                source=processed_frame.image,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.image_size,
                device=self._config.device,
                classes=[self._person_class_id],
                verbose=self._config.verbose,
            )
        except Exception as exc:
            logger.error("Inference failed on frame index={}: {}", processed_frame.index, exc)
            raise InferenceError(
                f"YOLO inference failed on frame index={processed_frame.index}: {exc}"
            ) from exc

        detections = self._to_detections(results, processed_frame)

        self._frame_count += 1
        logger.debug(
            "Frame index={} produced {} person detection(s).", processed_frame.index, len(detections)
        )
        if self._frame_count % 100 == 0:
            logger.info("Ran inference on {} frames so far.", self._frame_count)

        return detections

    def detect_stream(self, processed_frames: Iterable[ProcessedFrame]) -> Iterator[List[Detection]]:
        """Generator: run detect() over a stream of ProcessedFrame objects.

        Chains directly onto FrameProcessor.process_stream():
            detector.detect_stream(processor.process_stream(video_input.frames()))
        """
        logger.info("Detection started.")
        count = 0
        for processed_frame in processed_frames:
            yield self.detect(processed_frame)
            count += 1
        logger.info("Detection completed: {} frames processed.", count)

    def _load_model(self) -> YOLO:
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            logger.error("YOLO model file not found: {}", model_path)
            raise ModelLoadError(f"Model file not found: {model_path}")

        self._check_device_available(self._config.device)

        logger.info("Loading YOLO model from {} (device={})", model_path, self._config.device)
        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            logger.error("Failed to load YOLO model from {}: {}", model_path, exc)
            raise ModelLoadError(f"Failed to load YOLO model from {model_path}: {exc}") from exc

        logger.info("YOLO model loaded successfully. device={}", self._config.device)
        return model

    @staticmethod
    def _check_device_available(device: str) -> None:
        if device == "cpu":
            return

        if device == "cuda":
            if not torch.cuda.is_available():
                logger.error("Requested device=cuda but CUDA is not available on this machine.")
                raise DeviceUnavailableError("CUDA device requested but not available on this machine.")
            return

        if device == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                logger.error("Requested device=mps but MPS is not available on this machine.")
                raise DeviceUnavailableError("MPS device requested but not available on this machine.")
            return

        raise DeviceUnavailableError(f"Unsupported device: {device!r} (expected cpu, cuda, or mps).")

    def _resolve_person_class_id(self) -> int:
        for class_id, name in self._model.names.items():
            if str(name).lower() == "person":
                return int(class_id)
        raise ModelLoadError("Loaded YOLO model does not define a 'person' class.")

    def _to_detections(self, results: Any, processed_frame: ProcessedFrame) -> List[Detection]:
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls[0])
            if class_id != self._person_class_id:
                # Safety net — `classes=[...]` above should already exclude these.
                continue

            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            confidence = float(box.conf[0])

            detections.append(
                Detection(
                    track_id=None,
                    class_name="person",
                    class_id=class_id,
                    confidence=confidence,
                    bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    timestamp=processed_frame.timestamp,
                    frame_index=processed_frame.index,
                    source_id=processed_frame.source_id,
                )
            )

        return detections

    @staticmethod
    def _validate_config(config: DetectorConfig) -> None:
        if not 0.0 <= config.confidence_threshold <= 1.0:
            raise InvalidDetectorConfigError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= config.iou_threshold <= 1.0:
            raise InvalidDetectorConfigError("iou_threshold must be between 0 and 1.")
        if config.image_size <= 0:
            raise InvalidDetectorConfigError("image_size must be positive.")
        if config.device not in ("cpu", "cuda", "mps"):
            raise InvalidDetectorConfigError(
                f"device must be one of 'cpu', 'cuda', 'mps' — got {config.device!r}."
            )
