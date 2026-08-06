"""FrameProcessor: turns Frame objects into ProcessedFrame objects ready for inference.

Sole responsibility: optional resize / color-conversion / normalization /
frame-skipping. No detection, tracking, event, API, or persistence logic
lives here or is imported by this module.
"""

from typing import Iterable, Iterator, Optional

import cv2
import numpy as np
from loguru import logger

from ...models.domain.frame import Frame
from ...models.domain.processed_frame import ProcessedFrame
from .exceptions import InvalidPreprocessConfigError
from .types import ColorMode, PreprocessConfig

_COLOR_CONVERSION: dict = {
    ColorMode.RGB: cv2.COLOR_BGR2RGB,
    ColorMode.GRAY: cv2.COLOR_BGR2GRAY,
}


class FrameProcessor:
    """Applies configured preprocessing to Frame objects, yielding ProcessedFrame objects.

    Usage:
        processor = FrameProcessor(config)
        for processed in processor.process_stream(video_input.frames()):
            ...  # hand `processed` to a future detector
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self._validate_config(config)
        self._config = config
        self._processed_count = 0
        self._skipped_count = 0

    def process(self, frame: Frame) -> Optional[ProcessedFrame]:
        """Apply configured preprocessing to a single frame.

        Returns None if the frame is invalid or is skipped by the
        frame-skip interval — never raises for a bad individual frame.
        """
        if not self._is_valid_frame(frame):
            logger.warning("Discarding invalid frame index={} from processing.", frame.index)
            return None

        if self._should_skip(frame):
            self._skipped_count += 1
            logger.debug("Skipping frame index={} (frame-skip interval).", frame.index)
            return None

        image = frame.image.copy()  # never mutate the original Frame's image
        image = self._resize(image)
        image = self._convert_color(image)
        image = self._normalize(image)

        self._processed_count += 1
        logger.debug("Processed frame index={} -> shape={}, dtype={}", frame.index, image.shape, image.dtype)
        if self._processed_count % 100 == 0:
            logger.info(
                "Processed {} frames so far ({} skipped).", self._processed_count, self._skipped_count
            )

        return ProcessedFrame(original=frame, image=image)

    def process_stream(self, frames: Iterable[Frame]) -> Iterator[ProcessedFrame]:
        """Generator: consume an iterable of Frame objects, yield ProcessedFrame ones.

        Chains directly onto VideoInput.frames():
            processor.process_stream(video_input.frames())
        """
        logger.info("Frame processing started.")
        for frame in frames:
            processed = self.process(frame)
            if processed is not None:
                yield processed
        logger.info(
            "Frame processing completed: {} processed, {} skipped.",
            self._processed_count,
            self._skipped_count,
        )

    def _resize(self, image: np.ndarray) -> np.ndarray:
        if not self._config.resize_enabled:
            return image
        return cv2.resize(
            image,
            (self._config.resize_width, self._config.resize_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _convert_color(self, image: np.ndarray) -> np.ndarray:
        if not self._config.color_conversion_enabled:
            return image
        return cv2.cvtColor(image, _COLOR_CONVERSION[self._config.color_mode])

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if not self._config.normalize_enabled:
            return image

        normalized = image.astype(np.float32) * self._config.normalize_scale

        if self._config.normalize_mean is not None and self._config.normalize_std is not None:
            mean = np.array(self._config.normalize_mean, dtype=np.float32)
            std = np.array(self._config.normalize_std, dtype=np.float32)
            normalized = (normalized - mean) / std

        return normalized

    def _should_skip(self, frame: Frame) -> bool:
        if not self._config.frame_skip_enabled:
            return False
        return frame.index % self._config.frame_skip_interval != 0

    @staticmethod
    def _is_valid_frame(frame: Frame) -> bool:
        image = frame.image
        return (
            image is not None
            and isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim == 3
            and image.shape[0] > 0
            and image.shape[1] > 0
        )

    @staticmethod
    def _validate_config(config: PreprocessConfig) -> None:
        if config.resize_enabled and (config.resize_width <= 0 or config.resize_height <= 0):
            raise InvalidPreprocessConfigError(
                "resize.width and resize.height must be positive when resize is enabled."
            )
        if config.normalize_enabled and config.normalize_scale <= 0:
            raise InvalidPreprocessConfigError("normalize.scale must be positive when normalize is enabled.")
        if config.frame_skip_enabled and config.frame_skip_interval <= 0:
            raise InvalidPreprocessConfigError(
                "frame_skip.interval must be a positive integer when frame_skip is enabled."
            )
