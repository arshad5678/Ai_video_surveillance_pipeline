"""Types describing a video source and how VideoInput should handle it.

Scoped to this module only — future consumers (detection, tracking, etc.)
never see these; they only ever receive `Frame` objects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Union


class VideoSourceType(str, Enum):
    """The exactly three source kinds this module supports."""

    WEBCAM = "webcam"
    FILE = "file"
    RTSP = "rtsp"


@dataclass(frozen=True)
class VideoSourceConfig:
    """Fully-resolved parameters needed to open a video source.

    `uri` is an int (device index) for webcams, or a str (file path /
    RTSP URL) for files and network streams.
    """

    source_type: VideoSourceType
    uri: Union[int, str]
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 2.0
    read_timeout_seconds: float = 5.0
