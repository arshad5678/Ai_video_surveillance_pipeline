"""Types describing a video source and how VideoInput should handle it.

Scoped to this module only — future consumers (detection, tracking, etc.)
never see these; they only ever receive `Frame` objects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union


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


@dataclass(frozen=True)
class VideoSourceStatus:
    """Point-in-time connectivity snapshot, returned by VideoInput.probe_status().

    Not used by the streaming path (frames()) at all — exists solely so a
    future consumer (Prompt 12's Camera Router) can report real fps/
    resolution/connectivity without duplicating VideoInput's own
    open/validate/close logic.
    """

    source_type: VideoSourceType
    uri: Union[int, str]
    connected: bool
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
