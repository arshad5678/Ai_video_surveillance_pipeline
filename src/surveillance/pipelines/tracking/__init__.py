"""Multi-Object Tracking module — assigns stable IDs to per-frame Detection lists.

Self-contained ByteTrack-style implementation (Kalman motion prediction +
two-stage IoU/Hungarian association) — no Ultralytics dependency anywhere
in this module.

Public API:
    MultiObjectTracker      — loads the tracking backend once, tracks people per frame
    build_tracking_config   — resolves TrackingConfig from config.yaml
    TrackingConfig
    TrackingError, TrackerInitializationError, TrackingInferenceError, InvalidTrackingConfigError
"""

from .config import build_tracking_config
from .exceptions import (
    InvalidTrackingConfigError,
    TrackerInitializationError,
    TrackingError,
    TrackingInferenceError,
)
from .multi_object_tracker import MultiObjectTracker
from .types import TrackingConfig

__all__ = [
    "MultiObjectTracker",
    "build_tracking_config",
    "TrackingConfig",
    "TrackingError",
    "TrackerInitializationError",
    "TrackingInferenceError",
    "InvalidTrackingConfigError",
]
