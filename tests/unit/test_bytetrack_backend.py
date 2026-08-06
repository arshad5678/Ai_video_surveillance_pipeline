"""Unit tests for the real _ByteTrackBackend algorithm — pure numpy/scipy/filterpy,
no GPU, no mocks, no Detection/YOLO involvement. Verifies the actual tracking
behaviors: stable IDs, new/lost/reappearing/removed lifecycle, and history."""

from src.surveillance.pipelines.tracking._bytetrack_backend import _ByteTrackBackend, _RawDetection
from src.surveillance.pipelines.tracking.types import TrackingConfig


def det(x1=100.0, y1=100.0, x2=150.0, y2=200.0, conf=0.9):
    return _RawDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_name="person", class_id=0)


def make_backend(**overrides) -> _ByteTrackBackend:
    config = TrackingConfig(**overrides)
    return _ByteTrackBackend(config)


def test_new_track_created_for_high_confidence_detection() -> None:
    backend = make_backend()

    active, removed = backend.update([det(conf=0.9)])

    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].hits == 1
    assert active[0].time_since_update == 0
    assert removed == []


def test_low_confidence_detection_alone_does_not_create_a_track() -> None:
    backend = make_backend(track_high_thresh=0.5, track_low_thresh=0.1)

    active, _ = backend.update([det(conf=0.3)])  # below high_thresh

    assert active == []
    assert backend.all_tracks() == []


def test_detection_below_low_thresh_is_discarded_entirely() -> None:
    backend = make_backend(track_low_thresh=0.2)

    active, _ = backend.update([det(conf=0.05)])

    assert active == []
    assert backend.all_tracks() == []


def test_detection_below_minimum_area_is_ignored() -> None:
    backend = make_backend(minimum_box_area=1000.0)

    active, _ = backend.update([det(x1=0, y1=0, x2=5, y2=5, conf=0.9)])  # area=25 < 1000

    assert active == []


def test_stable_id_across_consecutive_matching_frames() -> None:
    backend = make_backend()

    ids = []
    for i in range(5):
        active, _ = backend.update([det(x1=100 + i, y1=100, x2=150 + i, y2=200, conf=0.9)])
        assert len(active) == 1
        ids.append(active[0].track_id)

    assert len(set(ids)) == 1  # same id every frame
    assert backend.all_tracks()[0].hits == 5


def test_track_goes_lost_when_not_matched_but_is_not_removed() -> None:
    backend = make_backend(track_buffer=30)

    backend.update([det()])  # frame 1: create
    active, removed = backend.update([])  # frame 2: nothing detected

    assert active == []  # not active this frame
    assert removed == []  # but not removed either — still within buffer
    assert len(backend.all_tracks()) == 1
    assert backend.all_tracks()[0].time_since_update == 1


def test_track_reappears_with_same_id_after_a_gap() -> None:
    backend = make_backend(match_thresh=0.7)

    active1, _ = backend.update([det(x1=100, y1=100, x2=150, y2=200, conf=0.9)])
    original_id = active1[0].track_id

    backend.update([])  # missed one frame (occlusion)

    active3, _ = backend.update([det(x1=101, y1=100, x2=151, y2=200, conf=0.9)])  # reappears, near-identical box

    assert len(active3) == 1
    assert active3[0].track_id == original_id  # same identity, not a new track
    assert active3[0].hits == 2


def test_low_confidence_detection_recovers_a_track_missed_by_stage_one() -> None:
    backend = make_backend(track_high_thresh=0.6, track_low_thresh=0.1)

    backend.update([det(conf=0.9)])  # frame 1: create via high-conf

    # frame 2: only a low-confidence detection near the same box — should
    # recover the existing track via stage-2 matching, not create a new one.
    active, _ = backend.update([det(x1=101, y1=100, x2=151, y2=200, conf=0.3)])

    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].hits == 2


def test_track_removed_after_exceeding_buffer() -> None:
    backend = make_backend(track_buffer=2, frame_rate=30)  # buffer_frames = 2

    backend.update([det()])  # frame 1: create, time_since_update=0

    backend.update([])  # frame 2: time_since_update=1
    backend.update([])  # frame 3: time_since_update=2
    active, removed = backend.update([])  # frame 4: time_since_update=3 > buffer(2) => removed

    assert active == []
    assert removed == [1]
    assert backend.all_tracks() == []


def test_history_accumulates_center_points_across_updates() -> None:
    backend = make_backend()

    # Small per-frame shift (2px on a 50px-wide box keeps IoU ~0.92, well
    # above the default match_thresh=0.8) so this stays the *same* track
    # across all three frames rather than spawning a new one each time.
    for i in range(3):
        backend.update([det(x1=100 + i * 2, y1=100, x2=150 + i * 2, y2=200, conf=0.9)])

    assert len(backend.all_tracks()) == 1
    history = list(backend.all_tracks()[0].history)
    assert len(history) == 3
    # center_x should increase each frame, tracking the detection's movement
    assert history[0][0] < history[1][0] < history[2][0]


def test_history_length_is_capped_by_config() -> None:
    backend = make_backend(history_length=3)

    for i in range(10):
        backend.update([det(x1=100 + i, y1=100, x2=150 + i, y2=200, conf=0.9)])

    history = list(backend.all_tracks()[0].history)
    assert len(history) == 3  # capped, oldest points dropped
