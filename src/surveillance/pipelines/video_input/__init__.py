"""Video Input module — acquires frames from a webcam, local file, or RTSP stream.

Public API:
    VideoInput                  — opens a source and yields Frame objects
    build_video_source_config   — resolves VideoSourceConfig from .env + config.yaml
    VideoSourceConfig, VideoSourceType
    VideoSourceError, VideoSourceNotFoundError, VideoSourceConnectionError
"""

from .config import build_video_source_config
from .exceptions import VideoSourceConnectionError, VideoSourceError, VideoSourceNotFoundError
from .types import VideoSourceConfig, VideoSourceType
from .video_input import VideoInput

__all__ = [
    "VideoInput",
    "build_video_source_config",
    "VideoSourceConfig",
    "VideoSourceType",
    "VideoSourceError",
    "VideoSourceNotFoundError",
    "VideoSourceConnectionError",
]
