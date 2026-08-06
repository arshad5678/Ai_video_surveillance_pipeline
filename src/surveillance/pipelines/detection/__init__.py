"""Person Detection module — detects people in ProcessedFrame images via YOLO.

Public API:
    PersonDetector          — loads a YOLO model once, detects people per frame
    build_detector_config   — resolves DetectorConfig from config.yaml
    DetectorConfig
    PersonDetectionError, InvalidDetectorConfigError, ModelLoadError,
    DeviceUnavailableError, InferenceError
"""

from .config import build_detector_config
from .exceptions import (
    DeviceUnavailableError,
    InferenceError,
    InvalidDetectorConfigError,
    ModelLoadError,
    PersonDetectionError,
)
from .person_detector import PersonDetector
from .types import DetectorConfig

__all__ = [
    "PersonDetector",
    "build_detector_config",
    "DetectorConfig",
    "PersonDetectionError",
    "InvalidDetectorConfigError",
    "ModelLoadError",
    "DeviceUnavailableError",
    "InferenceError",
]
