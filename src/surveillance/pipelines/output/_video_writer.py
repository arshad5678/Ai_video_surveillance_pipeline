"""Private helper: lazy-opening cv2.VideoWriter lifecycle wrapper.

Shared by OutputGenerator for both the single long-running annotated
video and each short-lived event clip — same open/write/release
lifecycle, only the frame source differs (a live stream vs. an
already-collected list of frames).
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .exceptions import VideoWriterError


class _VideoWriterHandle:
    """Wraps a single cv2.VideoWriter, opened lazily on the first frame.

    Opening is deferred because the frame size isn't known until the
    first image arrives (the annotated frame's dimensions match the
    source video, which this module never reads directly).
    """

    def __init__(self, path: Path, codec: str, fps: float) -> None:
        self._path = path
        self._codec = codec
        self._fps = fps if fps > 0 else 30.0
        self._writer: "cv2.VideoWriter | None" = None

    def write(self, image: np.ndarray) -> None:
        if self._writer is None:
            self._open(width=image.shape[1], height=image.shape[0])
        self._writer.write(image)

    def _open(self, width: int, height: int) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (width, height))
        if not writer.isOpened():
            raise VideoWriterError(f"Failed to open video writer for {self._path} (codec={self._codec!r}).")
        self._writer = writer
        logger.info(
            "Video writer created: {} ({}x{} @ {}fps, codec={})", self._path, width, height, self._fps, self._codec
        )

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
