"""Unit tests for IntrusionDetector — pure state-machine logic, no I/O, no GPU.

Zone and ZoneMembership objects are plain dataclass instances built by the
helpers below, the simplest reliable "mock" for immutable domain models.
"""

import pytest

from src.surveillance.models.domain.intrusion_event import IntrusionEventType
from src.surveillance.models.domain.zone import Zone
from src.surveillance.models.domain.zone_membership import ZoneMembership
from src.surveillance.models.domain.zone_point import ZonePoint
from src.surveillance.pipelines.intrusion import (
    IntrusionConfig,
    IntrusionConfigurationError,
    IntrusionDetector,
    IntrusionEvaluationError,
)

DUMMY_POLYGON = (ZonePoint(0, 0), ZonePoint(10, 0), ZonePoint(10, 10))


def make_zone(zone_id: str = "restricted_area", zone_type: str = "intrusion", zone_name: str = None) -> Zone:
    return Zone(
        zone_id=zone_id,
        zone_name=zone_name or zone_id.replace("_", " ").title(),
        zone_type=zone_type,
        polygon=DUMMY_POLYGON,
        enabled=True,
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
        track_id=track_id,
        zone_id=zone_id,
        inside=inside,
        timestamp=timestamp,
        frame_index=frame_index,
        source_id=source_id,
    )


def make_detector(zones=None, **config_overrides) -> IntrusionDetector:
    zones = zones if zones is not None else [make_zone()]
    return IntrusionDetector(IntrusionConfig(**config_overrides), zones)


# --- core transitions -------------------------------------------------


def test_first_observation_inside_produces_enter() -> None:
    detector = make_detector()

    events = detector.evaluate([make_membership(inside=True, timestamp=1.0)])

    assert len(events) == 1
    assert events[0].event_type is IntrusionEventType.ENTER
    assert events[0].track_id == 1
    assert events[0].zone_id == "restricted_area"
    assert events[0].zone_name == "Restricted Area"


def test_first_observation_outside_produces_no_event() -> None:
    detector = make_detector()

    events = detector.evaluate([make_membership(inside=False, timestamp=1.0)])

    assert events == []


def test_repeated_inside_does_not_duplicate_enter() -> None:
    detector = make_detector()

    first = detector.evaluate([make_membership(inside=True, timestamp=1.0)])
    second = detector.evaluate([make_membership(inside=True, timestamp=2.0)])
    third = detector.evaluate([make_membership(inside=True, timestamp=3.0)])

    assert len(first) == 1
    assert second == []
    assert third == []


def test_inside_to_outside_produces_exit() -> None:
    detector = make_detector()

    detector.evaluate([make_membership(inside=True, timestamp=1.0)])
    events = detector.evaluate([make_membership(inside=False, timestamp=2.0)])

    assert len(events) == 1
    assert events[0].event_type is IntrusionEventType.EXIT


def test_repeated_outside_does_not_duplicate_exit() -> None:
    detector = make_detector()

    detector.evaluate([make_membership(inside=True, timestamp=1.0)])
    detector.evaluate([make_membership(inside=False, timestamp=2.0)])
    second_outside = detector.evaluate([make_membership(inside=False, timestamp=3.0)])
    third_outside = detector.evaluate([make_membership(inside=False, timestamp=4.0)])

    assert second_outside == []
    assert third_outside == []


def test_full_enter_exit_reenter_cycle() -> None:
    detector = make_detector()

    e1 = detector.evaluate([make_membership(inside=True, timestamp=1.0)])
    e2 = detector.evaluate([make_membership(inside=True, timestamp=2.0)])
    e3 = detector.evaluate([make_membership(inside=False, timestamp=3.0)])
    e4 = detector.evaluate([make_membership(inside=False, timestamp=4.0)])
    e5 = detector.evaluate([make_membership(inside=True, timestamp=5.0)])

    types = [e[0].event_type if e else None for e in (e1, e2, e3, e4, e5)]
    assert types == [IntrusionEventType.ENTER, None, IntrusionEventType.EXIT, None, IntrusionEventType.ENTER]


# --- multiple tracks / zones -------------------------------------------


def test_multiple_tracks_have_independent_state() -> None:
    detector = make_detector()

    events = detector.evaluate(
        [
            make_membership(track_id=1, inside=True, timestamp=1.0),
            make_membership(track_id=2, inside=False, timestamp=1.0),
        ]
    )

    assert len(events) == 1
    assert events[0].track_id == 1

    events2 = detector.evaluate(
        [
            make_membership(track_id=1, inside=True, timestamp=2.0),  # still inside, no event
            make_membership(track_id=2, inside=True, timestamp=2.0),  # now enters
        ]
    )
    assert len(events2) == 1
    assert events2[0].track_id == 2


def test_multiple_zones_have_independent_state_for_same_track() -> None:
    zones = [make_zone(zone_id="zone_a"), make_zone(zone_id="zone_b")]
    detector = make_detector(zones=zones)

    events = detector.evaluate(
        [
            make_membership(zone_id="zone_a", inside=True, timestamp=1.0),
            make_membership(zone_id="zone_b", inside=False, timestamp=1.0),
        ]
    )
    assert {e.zone_id for e in events} == {"zone_a"}

    events2 = detector.evaluate(
        [
            make_membership(zone_id="zone_a", inside=True, timestamp=2.0),  # no change
            make_membership(zone_id="zone_b", inside=True, timestamp=2.0),  # enters
        ]
    )
    assert {e.zone_id for e in events2} == {"zone_b"}


# --- zone_type filtering -------------------------------------------------


def test_membership_for_unmonitored_zone_type_is_ignored() -> None:
    zone = make_zone(zone_type="monitoring")
    detector = make_detector(zones=[zone], monitor_zone_types=("intrusion",))

    events = detector.evaluate([make_membership(inside=True, timestamp=1.0)])

    assert events == []
    assert detector.get_state(1, "restricted_area") is None


# --- disabled configuration ------------------------------------------------


def test_disabled_configuration_never_produces_events() -> None:
    detector = make_detector(enabled=False)

    events = detector.evaluate([make_membership(inside=True, timestamp=1.0)])

    assert events == []
    assert detector.get_state(1, "restricted_area") is None


# --- emit_exit_events toggle -----------------------------------------------


def test_emit_exit_events_false_suppresses_exit_but_still_updates_state() -> None:
    detector = make_detector(emit_exit_events=False)

    detector.evaluate([make_membership(inside=True, timestamp=1.0)])
    events = detector.evaluate([make_membership(inside=False, timestamp=2.0)])

    assert events == []  # EXIT suppressed
    state = detector.get_state(1, "restricted_area")
    assert state.currently_inside is False  # state still correctly updated

    # A fresh re-entry should still produce a new ENTER, proving state
    # tracking wasn't corrupted by suppressing the EXIT event.
    reentry_events = detector.evaluate([make_membership(inside=True, timestamp=3.0)])
    assert len(reentry_events) == 1
    assert reentry_events[0].event_type is IntrusionEventType.ENTER


# --- get_state / first_seen_inside_timestamp --------------------------


def test_get_state_returns_none_for_unknown_pair() -> None:
    detector = make_detector()

    assert detector.get_state(999, "restricted_area") is None


def test_first_seen_inside_timestamp_persists_while_inside_and_clears_on_exit() -> None:
    detector = make_detector()

    detector.evaluate([make_membership(inside=True, timestamp=10.0)])
    state = detector.get_state(1, "restricted_area")
    assert state.first_seen_inside_timestamp == 10.0

    detector.evaluate([make_membership(inside=True, timestamp=20.0)])
    state = detector.get_state(1, "restricted_area")
    assert state.first_seen_inside_timestamp == 10.0  # unchanged while still inside

    detector.evaluate([make_membership(inside=False, timestamp=30.0)])
    state = detector.get_state(1, "restricted_area")
    assert state.first_seen_inside_timestamp is None  # cleared on exit


# --- stale-state cleanup -------------------------------------------------


def test_stale_state_is_cleaned_up_after_ttl_elapses() -> None:
    detector = make_detector(stale_state_ttl_seconds=5.0)

    detector.evaluate([make_membership(track_id=1, inside=True, timestamp=0.0)])
    assert detector.get_state(1, "restricted_area") is not None

    # A different track's membership much later advances the detector's
    # notion of "now" far past track 1's ttl window.
    detector.evaluate([make_membership(track_id=2, inside=True, timestamp=100.0)])

    assert detector.get_state(1, "restricted_area") is None  # pruned
    assert detector.get_state(2, "restricted_area") is not None  # still fresh


# --- exception handling / config validation --------------------------


def test_empty_monitor_zone_types_raises_configuration_error() -> None:
    with pytest.raises(IntrusionConfigurationError):
        IntrusionDetector(IntrusionConfig(monitor_zone_types=()), [make_zone()])


def test_non_positive_ttl_raises_configuration_error() -> None:
    with pytest.raises(IntrusionConfigurationError):
        IntrusionDetector(IntrusionConfig(stale_state_ttl_seconds=0), [make_zone()])


def test_malformed_membership_raises_evaluation_error() -> None:
    detector = make_detector()
    bad_membership = make_membership(timestamp="not-a-number")  # breaks max() comparison internally

    with pytest.raises(IntrusionEvaluationError):
        detector.evaluate([bad_membership])


# --- stream wrapper --------------------------------------------------------


def test_evaluate_stream_yields_one_list_per_frame() -> None:
    detector = make_detector()
    memberships_stream = [
        [make_membership(inside=True, timestamp=1.0)],
        [make_membership(inside=True, timestamp=2.0)],
        [make_membership(inside=False, timestamp=3.0)],
    ]

    results = list(detector.evaluate_stream(memberships_stream))

    assert len(results) == 3
    assert [e.event_type for e in results[0]] == [IntrusionEventType.ENTER]
    assert results[1] == []
    assert [e.event_type for e in results[2]] == [IntrusionEventType.EXIT]
