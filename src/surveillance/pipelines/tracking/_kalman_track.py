"""Internal constant-velocity Kalman filter wrapping a single tracked box.

Not part of the public API — only `_ByteTrackBackend` constructs these.
State space: [cx, cy, aspect_ratio, height, vcx, vcy, va, vh] (8-dim),
measurement space: [cx, cy, aspect_ratio, height] (4-dim) — the standard
formulation used by ByteTrack/DeepSORT-lineage trackers.
"""

from collections import deque
from typing import Deque, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter


def _bbox_to_measurement(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    aspect_ratio = width / height if height > 0 else 0.0
    return np.array([center_x, center_y, aspect_ratio, height], dtype=np.float64)


def _state_to_bbox(state: np.ndarray) -> Tuple[float, float, float, float]:
    center_x, center_y, aspect_ratio, height = state[0], state[1], state[2], state[3]
    width = aspect_ratio * height
    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    return float(x1), float(y1), float(x2), float(y2)


class _KalmanBoxTracker:
    """One tracked object's motion state, identity, and bookkeeping."""

    def __init__(
        self,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float,
        class_name: str,
        class_id: int,
        history_length: int,
    ) -> None:
        self.track_id = track_id
        self.confidence = confidence
        self.class_name = class_name
        self.class_id = class_id

        self.age = 0
        self.hits = 1
        self.time_since_update = 0

        self._kf = KalmanFilter(dim_x=8, dim_z=4)
        self._kf.F = np.eye(8)
        for i in range(4):
            self._kf.F[i, i + 4] = 1.0  # constant velocity: position += velocity * dt(=1)

        self._kf.H = np.zeros((4, 8))
        for i in range(4):
            self._kf.H[i, i] = 1.0

        self._kf.R[2:, 2:] *= 10.0  # less trust in measured aspect ratio/height
        self._kf.P[4:, 4:] *= 1000.0  # high initial uncertainty on unobserved velocities
        self._kf.P *= 10.0
        self._kf.Q[-1, -1] *= 0.01
        self._kf.Q[4:, 4:] *= 0.01

        self._kf.x[:4, 0] = _bbox_to_measurement(x1, y1, x2, y2)

        self.history: Deque[Tuple[float, float]] = deque(maxlen=history_length)
        self.history.append((float(self._kf.x[0, 0]), float(self._kf.x[1, 0])))

    def predict(self) -> None:
        """Advance motion state by one frame without a new measurement."""
        if self._kf.x[3, 0] + self._kf.x[7, 0] <= 0:
            self._kf.x[7, 0] = 0.0  # don't let predicted height go non-positive
        self._kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, x1: float, y1: float, x2: float, y2: float, confidence: float, class_name: str, class_id: int) -> None:
        """Incorporate a matched detection this frame."""
        self._kf.update(_bbox_to_measurement(x1, y1, x2, y2))
        self.confidence = confidence
        self.class_name = class_name
        self.class_id = class_id
        self.hits += 1
        self.time_since_update = 0
        self.history.append((float(self._kf.x[0, 0]), float(self._kf.x[1, 0])))

    def current_bbox(self) -> Tuple[float, float, float, float]:
        return _state_to_bbox(self._kf.x[:, 0])
