"""MultiObjectTracker: assigns stable IDs to per-frame Detection lists.

Sole responsibility: turn independent per-frame Detection lists into
identity-consistent Track objects, using a self-contained ByteTrack-style
backend (Kalman motion + two-stage IoU association). No zone, event, API,
database, or alert logic lives here or is imported by this module — and
Ultralytics/YOLO are never imported either.
"""

from typing import Iterable, Iterator, List, Set, Tuple

from loguru import logger

from ...models.domain.bounding_box import BoundingBox
from ...models.domain.detection import Detection
from ...models.domain.track import Track
from ._bytetrack_backend import _ByteTrackBackend, _KalmanBoxTracker, _RawDetection
from .exceptions import InvalidTrackingConfigError, TrackerInitializationError, TrackingInferenceError
from .types import TrackingConfig

# A track is considered "confirmed" once it has been matched more than
# once — i.e. it survived at least one frame beyond creation. ByteTrack
# itself has no such concept (it activates immediately), but the Track
# domain model requires is_confirmed, so this is where that distinction
# is made, kept internal to this module rather than added to config.yaml.
_MIN_HITS_FOR_CONFIRMATION = 2


class MultiObjectTracker:
    """Assigns stable track IDs to per-frame Detection lists.

    The tracking backend is constructed exactly once, at construction
    time, and reused across every subsequent update() call — it carries
    state (active/lost tracks) between frames by design; this is not a
    stateless per-frame function.

    Usage:
        tracker = MultiObjectTracker(config)
        for tracks in tracker.track_stream(detections_stream):
            ...  # hand `tracks` to a future Zone Manager
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._validate_config(config)
        self._config = config

        try:
            self._backend = _ByteTrackBackend(config)
        except Exception as exc:
            logger.error("Failed to initialize tracking backend: {}", exc)
            raise TrackerInitializationError(f"Failed to initialize tracker: {exc}") from exc

        self._previous_active_ids: Set[int] = set()
        self._previous_all_ids: Set[int] = set()
        self._frame_count = 0

        logger.info(
            "Tracker initialized: type={}, high_thresh={}, low_thresh={}, "
            "new_track_thresh={}, match_thresh={}, buffer={} frames @ {}fps",
            config.tracker_type,
            config.track_high_thresh,
            config.track_low_thresh,
            config.new_track_thresh,
            config.match_thresh,
            config.track_buffer,
            config.frame_rate,
        )

    def update(self, detections: List[Detection]) -> List[Track]:
        """Advance tracking by one frame and return the tracks active this frame.

        `detections` must all belong to the same frame — this mirrors how
        PersonDetector.detect() emits one List[Detection] per frame. An
        empty list is valid (no people detected this frame); in that case
        no track can possibly be freshly matched or created, so an empty
        list is always returned regardless of frame context.

        Raises:
            TrackingInferenceError: the tracking backend update failed.
        """
        frame_index, timestamp, source_id = self._frame_context(detections)
        raw = [self._to_raw_detection(d) for d in detections]

        try:
            active_tracks, removed_ids = self._backend.update(raw)
        except Exception as exc:
            logger.error("Tracking update failed on frame index={}: {}", frame_index, exc)
            raise TrackingInferenceError(f"Tracking update failed on frame index={frame_index}: {exc}") from exc

        self._log_lifecycle_events(active_tracks, removed_ids)

        tracks = [self._to_track(kt, frame_index, timestamp, source_id) for kt in active_tracks]

        self._frame_count += 1
        logger.debug("Frame index={} -> {} active track(s).", frame_index, len(tracks))
        if self._frame_count % 100 == 0:
            logger.info("Tracked {} frames so far.", self._frame_count)

        return tracks

    def track_stream(self, detections_stream: Iterable[List[Detection]]) -> Iterator[List[Track]]:
        """Generator: run update() over a stream of per-frame Detection lists.

        Chains directly onto PersonDetector.detect_stream():
            tracker.track_stream(detector.detect_stream(processed_frames))
        """
        logger.info("Frame tracking started.")
        count = 0
        for detections in detections_stream:
            yield self.update(detections)
            count += 1
        logger.info("Tracking completed: {} frames processed.", count)

    def _log_lifecycle_events(self, active_tracks: List[_KalmanBoxTracker], removed_ids: List[int]) -> None:
        current_active_ids = {t.track_id for t in active_tracks}
        current_all_ids = {t.track_id for t in self._backend.all_tracks()}

        for new_id in sorted(current_all_ids - self._previous_all_ids):
            logger.info("New track created: id={}", new_id)

        lost_ids = self._previous_active_ids - current_active_ids - set(removed_ids)
        for lost_id in sorted(lost_ids):
            logger.info("Track lost: id={}", lost_id)

        for removed_id in sorted(removed_ids):
            logger.info("Track removed: id={}", removed_id)

        for track in active_tracks:
            logger.debug("Track updated: id={}", track.track_id)

        self._previous_active_ids = current_active_ids
        self._previous_all_ids = current_all_ids

    @staticmethod
    def _to_raw_detection(detection: Detection) -> _RawDetection:
        box = detection.bounding_box
        return _RawDetection(
            x1=box.x1,
            y1=box.y1,
            x2=box.x2,
            y2=box.y2,
            confidence=detection.confidence,
            class_name=detection.class_name,
            class_id=detection.class_id,
        )

    @staticmethod
    def _frame_context(detections: List[Detection]) -> Tuple[int, float, str]:
        if detections:
            first = detections[0]
            return first.frame_index, first.timestamp, first.source_id
        # No detections this frame => active_tracks is guaranteed empty (nothing
        # can be freshly matched or created), so these values are never read.
        return 0, 0.0, ""

    @staticmethod
    def _to_track(kt: _KalmanBoxTracker, frame_index: int, timestamp: float, source_id: str) -> Track:
        x1, y1, x2, y2 = kt.current_bbox()
        return Track(
            track_id=kt.track_id,
            bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            confidence=kt.confidence,
            class_name=kt.class_name,
            class_id=kt.class_id,
            timestamp=timestamp,
            frame_index=frame_index,
            source_id=source_id,
            is_confirmed=kt.hits >= _MIN_HITS_FOR_CONFIRMATION,
            age=kt.age,
            hits=kt.hits,
            time_since_update=kt.time_since_update,
            history=tuple(kt.history),
        )

    @staticmethod
    def _validate_config(config: TrackingConfig) -> None:
        if config.tracker_type != "bytetrack":
            raise InvalidTrackingConfigError(
                f"Unsupported tracker_type: {config.tracker_type!r} (only 'bytetrack' is supported)."
            )
        if not 0.0 <= config.track_low_thresh <= config.track_high_thresh <= 1.0:
            raise InvalidTrackingConfigError("Require 0 <= track_low_thresh <= track_high_thresh <= 1.")
        if not 0.0 <= config.new_track_thresh <= 1.0:
            raise InvalidTrackingConfigError("new_track_thresh must be between 0 and 1.")
        if not 0.0 <= config.match_thresh <= 1.0:
            raise InvalidTrackingConfigError("match_thresh must be between 0 and 1.")
        if config.track_buffer <= 0:
            raise InvalidTrackingConfigError("track_buffer must be positive.")
        if config.frame_rate <= 0:
            raise InvalidTrackingConfigError("frame_rate must be positive.")
        if config.minimum_box_area < 0:
            raise InvalidTrackingConfigError("minimum_box_area must be non-negative.")
        if config.history_length <= 0:
            raise InvalidTrackingConfigError("history_length must be positive.")
