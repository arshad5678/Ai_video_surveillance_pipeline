"""Output Generation module — visualization and export only.

Consumes Track, Zone, and SurveillanceEvent objects (plus the frame
image itself, via ProcessedFrame) and produces the annotated video,
per-event snapshots, per-event clips, and JSON/CSV event logs. Does not
implement REST APIs, dashboards, notifications, or databases — those are
future modules built on top of the files this module writes.

Public API:
    OutputGenerator        — writes annotated video/snapshots/clips/logs per frame
    build_output_config    — resolves OutputConfig from config.yaml
    OutputConfig
    EventLogPaths           — return type of OutputGenerator.latest_event_log()
    OutputGenerationError, VideoWriterError, SnapshotError, ClipGenerationError, LogExportError
"""

from .config import build_output_config
from .exceptions import (
    ClipGenerationError,
    LogExportError,
    OutputGenerationError,
    SnapshotError,
    VideoWriterError,
)
from .output_generator import OutputGenerator
from .types import EventLogPaths, OutputConfig

__all__ = [
    "OutputGenerator",
    "build_output_config",
    "OutputConfig",
    "EventLogPaths",
    "OutputGenerationError",
    "VideoWriterError",
    "SnapshotError",
    "ClipGenerationError",
    "LogExportError",
]
