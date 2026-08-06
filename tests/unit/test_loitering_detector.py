"""Unit tests for LoiteringDetector — pure state-machine logic, no I/O, no GPU.

IntrusionDetector is replaced by a minimal duck-typed fake exposing only
get_state(), so dwell-time scenarios are fully controllable independent
of IntrusionDetector's own transition logic (which is tested separately
in test_intrusion_detector.py).
"""

from typing import Dict, Optional, Tuple

import pytest

from src.surveillance.models.domain.bounding_box import BoundingBox
from src.surveillance.models.domain.intrusion_state import IntrusionState
from src.surveillance.models.domain.track import Track
from src.surveillance.models.domain.zone import Zone
from src.surveillance.models.domain.zone_membership import ZoneMembership
from src.surveillance.models.domain.zone_point import ZonePoint
from src.surveillance.pipelines.loitering import (
    LoiteringConfig,
    LoiteringConfigurationError,
    LoiteringDetector,
    LoiteringEvaluationError,
)

DUMMY_POLYGON = (ZonePoint(0, 0), ZonePoint(10, 0), ZonePoint(10, 10))


class FakeIntrusionDetector:
    """Duck-typed stand-in for IntrusionDetector — only get_state() is used by LoiteringDetector."""

    def __init__(self) -> None:
        self._states: Dict[Tuple[int, str], IntrusionState] = {}

    def set_first_seen(self, track_id: int, zone_id: str, first_seen: Optional[float]) -> None:
        self._states[(track_id, zone_id)] = IntrusionState(
            track_id=track_id,
            zone_id=zone_id,
            currently_inside=True,
            first_seen_inside_timestamp=first_seen,
            last_seen_timestamp=first_seen or 0.0,
        )

    def clear(self, track_id: int, zone_id: str) -> None:
        self._states.pop((track_id, zone_id), None)

    def get_state(self, track_id: int, zone_id: str) -> Optional[IntrusionState]:
        return self._states.get((track_id, zone_id))


def make_zone(zone_id: str = "restricted_area", zone_type: str = "intrusion") -> Zone:
    return Zone(
        zone_id=zone_id,
        zone_name=zone_id.replace("_", " ").title(),
        zone_type=zone_type,
        polygon=DUMMY_POLYGON,
        enabled=True,
    )


def make_track(track_id: int = 1, is_confirmed: bool = True) -> Track:
    return Track(
        track_id=track_id,
        bounding_box=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0),
        confidence=0.9,
        class_name="person",
        class_id=0,
        timestamp=0.0,
        frame_index=0,
        source_id="cam-1",
        is_confirmed=is_confirmed,
        age=5,
        hits=5 if is_confirmed else 1,
        time_since_update=0,
        history=((5.0, 10.0),),
    )


def make_membership(
    track_id: int = 1,
    zone_id: str = "restricted_area",
    inside: bool = True,
    timestamp: float = 0.0,
    frame_index: int = 0,
    source_id: str = "cam-1",
) -> ZoneMembership:
    return ZoneMembership(
        track_id=track_id, zone_id=zone_id, inside=inside, timestamp=timestamp, frame_index=frame_index, source_id=source_id
    )


def make_detector(zones=None, intrusion_detector=None, **config_overrides) -> Tuple[LoiteringDetector, FakeIntrusionDetector]:
    zones = zones if zones is not None else [make_zone()]
    fake_intrusion = intrusion_detector if intrusion_detector is not None else FakeIntrusionDetector()
    config = LoiteringConfig(threshold_seconds=10.0, **config_overrides)
    detector = LoiteringDetector(config, zones, fake_intrusion)
    return detector, fake_intrusion


# --- threshold behavior --------------------------------------------------


def test_below_threshold_produces_no_event() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    events = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=5.0)])

    assert events == []


def test_exactly_threshold_emits_event() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    events = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=10.0)])

    assert len(events) == 1
    assert events[0].dwell_time_seconds == 10.0
    assert events[0].threshold_seconds == 10.0


def test_above_threshold_emits_event() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    events = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=15.0)])

    assert len(events) == 1
    assert events[0].dwell_time_seconds == 15.0


# --- duplicate prevention -----------------------------------------------


def test_no_duplicate_event_while_continuing_to_stay_inside() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    first = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=12.0)])
    second = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=20.0)])
    third = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=30.0)])

    assert len(first) == 1
    assert second == []
    assert third == []


# --- exit resets timer --------------------------------------------------


def test_exit_resets_state_and_reentry_can_emit_again() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    first = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=12.0)])
    assert len(first) == 1

    # exits
    intrusion.clear(1, "restricted_area")
    exit_result = detector.evaluate([make_track()], [make_membership(inside=False, timestamp=13.0)])
    assert exit_result == []

    # re-enters with a fresh first_seen timestamp
    intrusion.set_first_seen(1, "restricted_area", 100.0)
    below_new_threshold = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=105.0)])
    assert below_new_threshold == []  # only 5s into the new stay

    above_new_threshold = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=111.0)])
    assert len(above_new_threshold) == 1  # new continuous stay crosses threshold again


# --- multiple tracks / zones ----------------------------------------------


def test_multiple_tracks_have_independent_dwell_state() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)
    intrusion.set_first_seen(2, "restricted_area", 5.0)

    events = detector.evaluate(
        [make_track(track_id=1), make_track(track_id=2)],
        [
            make_membership(track_id=1, inside=True, timestamp=11.0),  # dwell=11 >= 10
            make_membership(track_id=2, inside=True, timestamp=11.0),  # dwell=6 < 10
        ],
    )

    assert len(events) == 1
    assert events[0].track_id == 1


def test_multiple_zones_have_independent_dwell_state_for_same_track() -> None:
    zones = [make_zone(zone_id="zone_a"), make_zone(zone_id="zone_b")]
    detector, intrusion = make_detector(zones=zones)
    intrusion.set_first_seen(1, "zone_a", 0.0)
    intrusion.set_first_seen(1, "zone_b", 8.0)

    events = detector.evaluate(
        [make_track()],
        [
            make_membership(zone_id="zone_a", inside=True, timestamp=11.0),  # dwell=11 >= 10
            make_membership(zone_id="zone_b", inside=True, timestamp=11.0),  # dwell=3 < 10
        ],
    )

    assert len(events) == 1
    assert events[0].zone_id == "zone_a"


# --- disabled configuration ------------------------------------------------


def test_disabled_configuration_never_produces_events() -> None:
    detector, intrusion = make_detector(enabled=False)
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    events = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=100.0)])

    assert events == []


# --- unconfirmed tracks / unmonitored zones ------------------------------


def test_unconfirmed_track_does_not_start_dwell_timer() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    events = detector.evaluate(
        [make_track(is_confirmed=False)], [make_membership(inside=True, timestamp=20.0)]
    )

    assert events == []


def test_membership_for_unmonitored_zone_type_is_ignored() -> None:
    zone = make_zone(zone_type="monitoring")
    detector, intrusion = make_detector(zones=[zone], monitor_zone_types=("intrusion",))
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    events = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=100.0)])

    assert events == []


def test_missing_intrusion_state_falls_back_to_current_timestamp() -> None:
    detector, _ = make_detector()  # no first_seen ever set

    events = detector.evaluate([make_track()], [make_membership(inside=True, timestamp=100.0)])

    assert events == []  # treated as dwell_time=0, well below threshold


# --- config validation / exception handling --------------------------


def test_non_positive_threshold_raises_configuration_error() -> None:
    with pytest.raises(LoiteringConfigurationError):
        LoiteringDetector(LoiteringConfig(threshold_seconds=0), [make_zone()], FakeIntrusionDetector())


def test_empty_monitor_zone_types_raises_configuration_error() -> None:
    with pytest.raises(LoiteringConfigurationError):
        LoiteringDetector(LoiteringConfig(monitor_zone_types=()), [make_zone()], FakeIntrusionDetector())


def test_malformed_membership_raises_evaluation_error() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)
    bad_membership = make_membership(inside=True, timestamp="not-a-number")

    with pytest.raises(LoiteringEvaluationError):
        detector.evaluate([make_track()], [bad_membership])


# --- stale-state cleanup -------------------------------------------------


def test_stale_state_is_cleaned_up_after_ttl_elapses() -> None:
    detector, intrusion = make_detector(stale_state_ttl_seconds=5.0)
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    detector.evaluate([make_track(track_id=1)], [make_membership(track_id=1, inside=True, timestamp=1.0)])

    intrusion.set_first_seen(2, "restricted_area", 100.0)
    detector.evaluate([make_track(track_id=2)], [make_membership(track_id=2, inside=True, timestamp=100.0)])

    # Track 1's dwell state should now be pruned (100 - 1 = 99s > ttl=5s);
    # verified indirectly: a fresh short stay for track 1 must not
    # immediately re-trigger using stale bookkeeping.
    intrusion.set_first_seen(1, "restricted_area", 200.0)
    events = detector.evaluate([make_track(track_id=1)], [make_membership(track_id=1, inside=True, timestamp=201.0)])
    assert events == []  # only 1s into a brand-new stay, not resuming an old 99s dwell


# --- stream wrapper --------------------------------------------------------


def test_evaluate_stream_yields_one_list_per_frame() -> None:
    detector, intrusion = make_detector()
    intrusion.set_first_seen(1, "restricted_area", 0.0)

    frames = [
        ([make_track()], [make_membership(inside=True, timestamp=5.0)]),
        ([make_track()], [make_membership(inside=True, timestamp=15.0)]),
        ([make_track()], [make_membership(inside=True, timestamp=25.0)]),
    ]

    results = list(detector.evaluate_stream(frames))

    assert len(results) == 3
    assert results[0] == []
    assert len(results[1]) == 1
    assert results[2] == []
