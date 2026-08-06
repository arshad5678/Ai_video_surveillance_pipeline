"""Types describing frame preprocessing configuration.

Scoped to this module only — the future detector never sees these; it only
ever receives `ProcessedFrame` objects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ColorMode(str, Enum):
    """Target color space for the optional color-conversion step.

    OpenCV frames are natively BGR, so BGR is intentionally not a member —
    disabling color_conversion already yields BGR untouched.
    """

    RGB = "RGB"
    GRAY = "GRAY"


@dataclass(frozen=True)
class PreprocessConfig:
    """Fully-resolved, validated parameters controlling FrameProcessor's behavior."""

    resize_enabled: bool = False
    resize_width: int = 640
    resize_height: int = 640

    color_conversion_enabled: bool = False
    color_mode: ColorMode = ColorMode.RGB

    normalize_enabled: bool = False
    normalize_scale: float = 1.0 / 255.0
    normalize_mean: Optional[Tuple[float, ...]] = None
    normalize_std: Optional[Tuple[float, ...]] = None

    frame_skip_enabled: bool = False
    frame_skip_interval: int = 1
