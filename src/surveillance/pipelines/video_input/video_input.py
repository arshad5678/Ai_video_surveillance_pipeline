"""VideoInput: acquires frames from a webcam, local file, or RTSP stream.

Sole responsibility: open a validated source and yield `Frame` objects one
at a time. No detection, tracking, event, API, or persistence logic lives
here or is imported by this module.
"""

import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from loguru import logger

from ...models.domain.frame import Frame
from .exceptions import VideoSourceConnectionError, VideoSourceError, VideoSourceNotFoundError
from .types import VideoSourceConfig, VideoSourceType


class VideoInput:
    """Opens a configured video source and yields decoded frames.

    Usage:
        with VideoInput(config) as video_input:
            for frame in video_input.frames():
                ...  # hand `frame` to a downstream consumer
    """

    def __init__(self, config: VideoSourceConfig) -> None:
        self._config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        """Validate the source and open the underlying capture device.

        Raises:
            VideoSourceNotFoundError: the source fails basic pre-open validation.
            VideoSourceConnectionError: the source is valid but cannot be opened.
        """
        logger.info(
            "Initializing video source: type={}, uri={}",
            self._config.source_type.value,
            self._config.uri,
        )
        self._validate_source()

        capture = self._create_capture()
        if not capture.isOpened():
            capture.release()
            logger.error(
                "Failed to open video source: type={}, uri={}",
                self._config.source_type.value,
                self._config.uri,
            )
            raise VideoSourceConnectionError(
                f"Could not open {self._config.source_type.value} source: {self._config.uri!r}"
            )

        self._capture = capture
        self._is_open = True
        self._frame_index = 0
        logger.info(
            "Video source opened successfully: type={}",
            self._config.source_type.value,
        )

    def frames(self) -> Iterator[Frame]:
        """Yield frames one-by-one until the source ends or becomes unrecoverable.

        - Local files: stops cleanly (returns) at end-of-file.
        - Webcam/RTSP: attempts to reconnect on hard read failure, up to
          `reconnect_attempts`, then raises VideoSourceConnectionError.
        - A single corrupted/empty frame (capture still alive) is skipped
          and logged, without affecting reconnect state.
        """
        if not self._is_open or self._capture is None:
            raise VideoSourceError("VideoInput.open() must be called before frames().")

        consecutive_failures = 0

        while True:
            ok, image = self._capture.read()

            if not ok:
                # Hard read failure: for files this means EOF (clean stop); for
                # webcam/RTSP it may mean the device/stream dropped (reconnect).
                if self._config.source_type is VideoSourceType.FILE:
                    logger.info("End of video file reached after {} frames.", self._frame_index)
                    return

                consecutive_failures += 1
                logger.warning(
                    "Failed to read frame (attempt {}/{}) from {} source.",
                    consecutive_failures,
                    self._config.reconnect_attempts,
                    self._config.source_type.value,
                )

                if consecutive_failures > self._config.reconnect_attempts:
                    logger.error(
                        "Exceeded {} reconnect attempts; giving up on {} source.",
                        self._config.reconnect_attempts,
                        self._config.source_type.value,
                    )
                    raise VideoSourceConnectionError(
                        f"Lost connection to {self._config.source_type.value} source "
                        f"{self._config.uri!r} after {self._config.reconnect_attempts} attempts."
                    )

                if not self._reconnect():
                    time.sleep(self._config.reconnect_delay_seconds)
                continue

            if not self._is_valid_frame(image):
                # The capture is still alive — this is a single garbled/empty
                # frame (common with lossy RTSP), not a connection problem.
                # Skip it without touching reconnect state.
                logger.warning(
                    "Discarding corrupted or empty frame from {} source.",
                    self._config.source_type.value,
                )
                continue

            consecutive_failures = 0
            frame = Frame(
                index=self._frame_index,
                image=image,
                timestamp=time.time(),
                source_id=str(self._config.uri),
            )
            self._frame_index += 1
            logger.debug("Read frame index={} from {} source.", frame.index, self._config.source_type.value)
            if frame.index % 100 == 0:
                logger.info("Read {} frames so far from {} source.", frame.index, self._config.source_type.value)

            yield frame

    def close(self) -> None:
        """Release the underlying capture device. Safe to call multiple times."""
        if self._capture is not None:
            self._capture.release()
            logger.info(
                "Video source released and resources cleaned up: type={}",
                self._config.source_type.value,
            )
        self._capture = None
        self._is_open = False

    def __enter__(self) -> "VideoInput":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _validate_source(self) -> None:
        source_type = self._config.source_type
        uri = self._config.uri

        if source_type is VideoSourceType.WEBCAM:
            if not isinstance(uri, int) or uri < 0:
                raise VideoSourceNotFoundError(f"Invalid webcam index: {uri!r}")

        elif source_type is VideoSourceType.FILE:
            if not isinstance(uri, str) or not uri:
                raise VideoSourceNotFoundError(f"Invalid video file path: {uri!r}")
            path = Path(uri)
            if not path.exists() or not path.is_file():
                raise VideoSourceNotFoundError(f"Video file does not exist: {uri!r}")

        elif source_type is VideoSourceType.RTSP:
            if not isinstance(uri, str) or not uri.lower().startswith("rtsp://"):
                raise VideoSourceNotFoundError(f"Invalid RTSP URL: {uri!r}")

        else:  # pragma: no cover - exhaustive by VideoSourceType
            raise VideoSourceNotFoundError(f"Unsupported video source type: {source_type!r}")

    def _create_capture(self) -> cv2.VideoCapture:
        if self._config.source_type is VideoSourceType.RTSP:
            capture = cv2.VideoCapture(self._config.uri, cv2.CAP_FFMPEG)
            timeout_ms = self._config.read_timeout_seconds * 1000
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        else:
            capture = cv2.VideoCapture(self._config.uri)
        return capture

    def _reconnect(self) -> bool:
        logger.warning(
            "Attempting to reconnect to {} source: {}",
            self._config.source_type.value,
            self._config.uri,
        )
        if self._capture is not None:
            self._capture.release()

        time.sleep(self._config.reconnect_delay_seconds)
        capture = self._create_capture()

        if capture.isOpened():
            self._capture = capture
            logger.info("Reconnected to {} source.", self._config.source_type.value)
            return True

        capture.release()
        logger.warning("Reconnect attempt failed for {} source.", self._config.source_type.value)
        return False

    @staticmethod
    def _is_valid_frame(image: Optional[np.ndarray]) -> bool:
        return (
            image is not None
            and isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim == 3
            and image.shape[0] > 0
            and image.shape[1] > 0
        )
