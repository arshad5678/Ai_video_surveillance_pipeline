"""Frame domain model — the interchange type between VideoInput and downstream consumers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """A single decoded video frame plus the metadata needed by any consumer.

    Framework-free by design: it depends only on numpy, so future stages
    (detection, tracking, output) can consume it without importing OpenCV
    or anything video-source-specific.
    """

    index: int
    image: np.ndarray
    timestamp: float
    source_id: str
