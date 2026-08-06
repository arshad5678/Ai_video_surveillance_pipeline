"""Exception hierarchy for the Frame Processing module."""


class FrameProcessingError(Exception):
    """Base exception for all frame processing failures."""


class InvalidPreprocessConfigError(FrameProcessingError):
    """Raised when preprocessing configuration values are invalid (e.g. non-positive dimensions)."""
