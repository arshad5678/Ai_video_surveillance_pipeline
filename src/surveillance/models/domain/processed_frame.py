"""ProcessedFrame domain model — the interchange type between FrameProcessor and inference."""

from dataclasses import dataclass

import numpy as np

from .frame import Frame


@dataclass(frozen=True)
class ProcessedFrame:
    """A preprocessed frame ready for AI inference, paired with its untouched origin.

    Keeping `original` alongside `image` (rather than copying index/timestamp/
    source_id) means downstream stages can always recover the un-preprocessed
    image — e.g. to draw detection boxes at native resolution even though
    inference ran on a resized/normalized copy.
    """

    original: Frame
    image: np.ndarray

    @property
    def index(self) -> int:
        return self.original.index

    @property
    def timestamp(self) -> float:
        return self.original.timestamp

    @property
    def source_id(self) -> str:
        return self.original.source_id
