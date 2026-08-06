"""Exception hierarchy for the Output Generation module."""


class OutputGenerationError(Exception):
    """Base exception for all Output Generation failures."""


class VideoWriterError(OutputGenerationError):
    """Raised when the annotated-video (or clip) writer fails to open or write."""


class SnapshotError(OutputGenerationError):
    """Raised when saving an event snapshot JPEG fails."""


class ClipGenerationError(OutputGenerationError):
    """Raised when assembling or writing an event video clip fails."""


class LogExportError(OutputGenerationError):
    """Raised when writing the JSON or CSV event log fails."""
