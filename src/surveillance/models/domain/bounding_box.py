"""BoundingBox domain model — a pixel-space axis-aligned box."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """A detection's bounding box, in pixel coordinates of the image it was detected on.

    width/height/center_x/center_y are computed from (x1, y1, x2, y2) rather
    than stored separately, so there is exactly one source of truth and no
    risk of the derived values drifting out of sync with the corners.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0
