"""Exception hierarchy for the Multi-Object Tracking module."""


class TrackingError(Exception):
    """Base exception for all multi-object tracking failures."""


class TrackerInitializationError(TrackingError):
    """Raised when the tracking backend fails to initialize."""


class TrackingInferenceError(TrackingError):
    """Raised when a per-frame tracking update fails."""


class InvalidTrackingConfigError(TrackingError):
    """Raised when tracking configuration values are invalid."""
