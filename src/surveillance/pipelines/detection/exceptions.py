"""Exception hierarchy for the Person Detection module."""


class PersonDetectionError(Exception):
    """Base exception for all person detection failures."""


class InvalidDetectorConfigError(PersonDetectionError):
    """Raised when detector configuration values are invalid (e.g. thresholds out of range)."""


class ModelLoadError(PersonDetectionError):
    """Raised when the YOLO model file is missing, invalid, or fails to load."""


class DeviceUnavailableError(PersonDetectionError):
    """Raised when the configured device (cuda/mps) is not available on this machine."""


class InferenceError(PersonDetectionError):
    """Raised when YOLO inference fails on a given frame."""
