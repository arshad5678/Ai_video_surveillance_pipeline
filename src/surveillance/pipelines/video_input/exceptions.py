"""Exception hierarchy for the Video Input module."""


class VideoSourceError(Exception):
    """Base exception for all video input failures."""


class VideoSourceNotFoundError(VideoSourceError):
    """Raised when a source fails pre-open validation (bad path, malformed URL, invalid index)."""


class VideoSourceConnectionError(VideoSourceError):
    """Raised when a source cannot be opened, or reconnection attempts are exhausted mid-stream."""
