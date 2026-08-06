"""Internal ByteTrack-style two-stage association algorithm.

Self-contained (no Ultralytics dependency): Kalman motion prediction plus
IoU-based Hungarian matching, following Zhang et al. 2022 (ByteTrack) —
match high-confidence detections first, then recover tracks that would
otherwise be dropped using the low-confidence detections a naive
single-threshold tracker would discard entirely.

Not part of the public API — `MultiObjectTracker` is the only caller.
"""

from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._kalman_track import _KalmanBoxTracker
from .types import TrackingConfig

# ByteTrack's reference implementation uses a fixed second-stage IoU
# threshold, independent of the (generally stricter) first-stage
# match_thresh — recovering low-confidence detections only needs a loose
# spatial match, not the same confidence as a fresh association.
_SECOND_STAGE_IOU_THRESHOLD = 0.5


class _RawDetection(NamedTuple):
    """Backend-local detection representation — no Detection/BoundingBox import."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    class_id: int


def _box_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_matrix(
    boxes_a: Sequence[Tuple[float, float, float, float]],
    boxes_b: Sequence[Tuple[float, float, float, float]],
) -> np.ndarray:
    a = np.array(boxes_a, dtype=np.float64)
    b = np.array(boxes_b, dtype=np.float64)

    inter_x1 = np.maximum(a[:, 0:1], b[:, 0])
    inter_y1 = np.maximum(a[:, 1:2], b[:, 1])
    inter_x2 = np.minimum(a[:, 2:3], b[:, 2])
    inter_y2 = np.minimum(a[:, 3:4], b[:, 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).reshape(-1, 1)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    union = area_a + area_b.reshape(1, -1) - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


class _ByteTrackBackend:
    """Owns all active/lost tracks and advances them by one frame per update() call."""

    def __init__(self, config: TrackingConfig) -> None:
        self._config = config
        # ByteTrack expresses buffer size relative to a 30fps baseline,
        # scaled by the configured frame_rate.
        self._buffer_frames = max(1, int(config.frame_rate / 30.0 * config.track_buffer))
        self._tracks: List[_KalmanBoxTracker] = []
        self._next_id = 1

    def all_tracks(self) -> List[_KalmanBoxTracker]:
        """Snapshot of every track currently known (tracked + lost-but-buffered)."""
        return list(self._tracks)

    def update(self, detections: List[_RawDetection]) -> Tuple[List[_KalmanBoxTracker], List[int]]:
        """Advance the tracker by one frame.

        Returns (tracks active this frame, ids removed this frame).
        """
        for track in self._tracks:
            track.predict()

        valid = [d for d in detections if _box_area((d.x1, d.y1, d.x2, d.y2)) >= self._config.minimum_box_area]
        high = [d for d in valid if d.confidence >= self._config.track_high_thresh]
        low = [
            d
            for d in valid
            if self._config.track_low_thresh <= d.confidence < self._config.track_high_thresh
        ]

        all_track_indices = list(range(len(self._tracks)))

        # Stage 1: high-confidence detections vs every active track.
        matches1, unmatched_tracks, unmatched_high = self._associate(
            all_track_indices, high, self._config.match_thresh
        )
        for track_idx, det_idx in matches1:
            det = high[det_idx]
            self._tracks[track_idx].update(det.x1, det.y1, det.x2, det.y2, det.confidence, det.class_name, det.class_id)

        # Stage 2: low-confidence detections vs tracks that were active
        # last frame but are still unmatched (recovers brief misses/occlusions).
        recovery_candidates = [i for i in unmatched_tracks if self._tracks[i].time_since_update == 1]
        matches2, _, _ = self._associate(recovery_candidates, low, _SECOND_STAGE_IOU_THRESHOLD)
        for track_idx, det_idx in matches2:
            det = low[det_idx]
            self._tracks[track_idx].update(det.x1, det.y1, det.x2, det.y2, det.confidence, det.class_name, det.class_id)

        # New tracks from high-confidence detections nothing matched to.
        for det_idx in unmatched_high:
            det = high[det_idx]
            if det.confidence >= self._config.new_track_thresh:
                self._tracks.append(
                    _KalmanBoxTracker(
                        track_id=self._next_id,
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        confidence=det.confidence,
                        class_name=det.class_name,
                        class_id=det.class_id,
                        history_length=self._config.history_length,
                    )
                )
                self._next_id += 1

        removed_ids = [t.track_id for t in self._tracks if t.time_since_update > self._buffer_frames]
        if removed_ids:
            self._tracks = [t for t in self._tracks if t.time_since_update <= self._buffer_frames]

        active_this_frame = [t for t in self._tracks if t.time_since_update == 0]
        return active_this_frame, removed_ids

    def _associate(
        self,
        track_indices: List[int],
        dets: List[_RawDetection],
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match a subset of tracks (by global index) against a list of detections via IoU + Hungarian."""
        if not track_indices or not dets:
            return [], list(track_indices), list(range(len(dets)))

        track_boxes = [self._tracks[i].current_bbox() for i in track_indices]
        det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
        iou = _iou_matrix(track_boxes, det_boxes)

        row_idx, col_idx = linear_sum_assignment(1.0 - iou)

        matches: List[Tuple[int, int]] = []
        matched_tracks = set()
        matched_dets = set()
        for row, col in zip(row_idx, col_idx):
            if iou[row, col] >= iou_threshold:
                matches.append((track_indices[row], col))
                matched_tracks.add(track_indices[row])
                matched_dets.add(col)

        unmatched_tracks = [i for i in track_indices if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets
