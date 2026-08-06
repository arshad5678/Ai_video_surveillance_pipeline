"""Unit tests for MultiObjectTracker — the internal ByteTrack backend is mocked,
isolating the wrapper's own responsibilities: config validation, Detection->Track
field mapping, is_confirmed/lifecycle bookkeeping, and exception translation."""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from src.surveillance.models.domain.bounding_box import BoundingBox
from src.surveillance.models.domain.detection import Detection
from src.surveillance.pipelines.tracking import (
    InvalidTrackingConfigError,
    MultiObjectTracker,
    TrackerInitializationError,
    TrackingConfig,
    TrackingInferenceError,
)

PATCH_TARGET = "src.surveillance.pipelines.tracking.multi_object_tracker._ByteTrackBackend"


class FakeKalmanTrack:
    """Minimal stand-in for _KalmanBoxTracker, matching the attributes MultiObjectTracker reads."""

    def __init__(self, track_id, bbox=(10.0, 20.0, 60.0, 120.0), confidence=0.9, hits=1, age=1, time_since_update=0, history=None):
        self.track_id = track_id
        self._bbox = bbox
        self.confidence = confidence
        self.class_name = "person"
        self.class_id = 0
        self.hits = hits
        self.age = age
        self.time_since_update = time_since_update
        self.history = history if history is not None else deque([(35.0, 70.0)])

    def current_bbox(self):
        return self._bbox


def make_detection(frame_index: int = 3, timestamp: float = 42.0, source_id: str = "cam-1") -> Detection:
    return Detection(
        track_id=None,
        class_name="person",
        class_id=0,
        confidence=0.9,
        bounding_box=BoundingBox(x1=10.0, y1=20.0, x2=60.0, y2=120.0),
        timestamp=timestamp,
        frame_index=frame_index,
        source_id=source_id,
    )


def test_invalid_tracker_type_raises_before_backend_construction() -> None:
    with pytest.raises(InvalidTrackingConfigError):
        MultiObjectTracker(TrackingConfig(tracker_type="deepsort"))


def test_invalid_threshold_ordering_raises() -> None:
    with pytest.raises(InvalidTrackingConfigError):
        MultiObjectTracker(TrackingConfig(track_low_thresh=0.9, track_high_thresh=0.1))


def test_negative_minimum_box_area_raises() -> None:
    with pytest.raises(InvalidTrackingConfigError):
        MultiObjectTracker(TrackingConfig(minimum_box_area=-1))


def test_non_positive_history_length_raises() -> None:
    with pytest.raises(InvalidTrackingConfigError):
        MultiObjectTracker(TrackingConfig(history_length=0))


@patch(PATCH_TARGET)
def test_backend_initialization_failure_raises_tracker_initialization_error(mock_backend_cls: MagicMock) -> None:
    mock_backend_cls.side_effect = RuntimeError("boom")

    with pytest.raises(TrackerInitializationError):
        MultiObjectTracker(TrackingConfig())


@patch(PATCH_TARGET)
def test_update_maps_backend_track_to_domain_track_fields(mock_backend_cls: MagicMock) -> None:
    mock_backend = MagicMock()
    fake_track = FakeKalmanTrack(track_id=7, hits=3, age=5, time_since_update=0, history=deque([(100.0, 120.0), (105.0, 122.0)]))
    mock_backend.update.return_value = ([fake_track], [])
    mock_backend.all_tracks.return_value = [fake_track]
    mock_backend_cls.return_value = mock_backend

    tracker = MultiObjectTracker(TrackingConfig())
    detection = make_detection(frame_index=9, timestamp=99.0, source_id="cam-2")

    tracks = tracker.update([detection])

    assert len(tracks) == 1
    track = tracks[0]
    assert track.track_id == 7
    assert track.class_name == "person"
    assert track.class_id == 0
    assert track.confidence == 0.9
    assert track.bounding_box.x1 == 10.0
    assert track.frame_index == 9
    assert track.timestamp == 99.0
    assert track.source_id == "cam-2"
    assert track.hits == 3
    assert track.age == 5
    assert track.time_since_update == 0
    assert track.history == ((100.0, 120.0), (105.0, 122.0))


@patch(PATCH_TARGET)
def test_is_confirmed_false_for_single_hit_true_after_second_hit(mock_backend_cls: MagicMock) -> None:
    mock_backend = MagicMock()
    mock_backend.all_tracks.return_value = []
    mock_backend_cls.return_value = mock_backend
    tracker = MultiObjectTracker(TrackingConfig())

    new_track = FakeKalmanTrack(track_id=1, hits=1)
    mock_backend.update.return_value = ([new_track], [])
    result = tracker.update([make_detection()])
    assert result[0].is_confirmed is False

    confirmed_track = FakeKalmanTrack(track_id=1, hits=2)
    mock_backend.update.return_value = ([confirmed_track], [])
    result = tracker.update([make_detection()])
    assert result[0].is_confirmed is True


@patch(PATCH_TARGET)
def test_update_raises_tracking_inference_error_on_backend_failure(mock_backend_cls: MagicMock) -> None:
    mock_backend = MagicMock()
    mock_backend.update.side_effect = RuntimeError("backend exploded")
    mock_backend_cls.return_value = mock_backend

    tracker = MultiObjectTracker(TrackingConfig())

    with pytest.raises(TrackingInferenceError):
        tracker.update([make_detection()])


@patch(PATCH_TARGET)
def test_empty_detections_returns_empty_track_list(mock_backend_cls: MagicMock) -> None:
    mock_backend = MagicMock()
    mock_backend.update.return_value = ([], [])
    mock_backend.all_tracks.return_value = []
    mock_backend_cls.return_value = mock_backend

    tracker = MultiObjectTracker(TrackingConfig())

    assert tracker.update([]) == []


@patch(PATCH_TARGET)
def test_backend_constructed_exactly_once_across_multiple_updates(mock_backend_cls: MagicMock) -> None:
    mock_backend = MagicMock()
    mock_backend.update.return_value = ([], [])
    mock_backend.all_tracks.return_value = []
    mock_backend_cls.return_value = mock_backend

    tracker = MultiObjectTracker(TrackingConfig())
    for _ in range(5):
        tracker.update([make_detection()])

    mock_backend_cls.assert_called_once()
    assert mock_backend.update.call_count == 5


@patch(PATCH_TARGET)
def test_track_stream_yields_one_list_per_frame(mock_backend_cls: MagicMock) -> None:
    mock_backend = MagicMock()
    mock_backend.update.return_value = ([], [])
    mock_backend.all_tracks.return_value = []
    mock_backend_cls.return_value = mock_backend

    tracker = MultiObjectTracker(TrackingConfig())
    detections_stream = [[make_detection(frame_index=i)] for i in range(3)]

    results = list(tracker.track_stream(detections_stream))

    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)
