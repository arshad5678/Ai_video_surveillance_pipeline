"""Frame Processing module — prepares Frame objects for future AI inference.

Public API:
    FrameProcessor            — turns Frame objects into ProcessedFrame objects
    build_preprocess_config   — resolves PreprocessConfig from config.yaml
    PreprocessConfig, ColorMode
    FrameProcessingError, InvalidPreprocessConfigError
"""

from .config import build_preprocess_config
from .exceptions import FrameProcessingError, InvalidPreprocessConfigError
from .frame_processor import FrameProcessor
from .types import ColorMode, PreprocessConfig

__all__ = [
    "FrameProcessor",
    "build_preprocess_config",
    "PreprocessConfig",
    "ColorMode",
    "FrameProcessingError",
    "InvalidPreprocessConfigError",
]
