"""Unit tests for EventEngine — pure aggregation/filter/dedup logic, no I/O, no GPU."""

import pytest

from src.surveillance.models.domain.intrusion_event import IntrusionEvent, IntrusionEventType
from src.surveillance.models.domain.loitering_event import LoiteringEvent
from src.surveillance.models.domain.surveillance_event import (
    EventSeverity,
    EventSource,
    EventType,
    SurveillanceEvent,
)
from src.surveillance.pipelines.events import (
    EventAggregationError,
    EventConfigurationError,
    EventEngine,
    EventEngineConfig,
)

DEFAULT_SEVERITY_MAPPING = {
    "intrusion_enter": EventSeverity.HIGH,
    "intrusion_exit": EventSeverity.LOW,
    "loitering": EventSeverity.MEDIUM,
}


def make_engine(**overrides) -> EventEngine:
    overrides.setdefault("severity_mapping", dict(DEFAULT_SEVERITY_MAPPING))
    config = EventEngineConfig(**overrides)
    return EventEngine(config)


def make_intrusion_event(
    event_id: str = "evt-1",
    track_id: int = 1,
    zone_id: str = "zone_a",
    zone_name: str = "Zone A",
    event_type: IntrusionEventType = IntrusionEventType.ENTER,
    timestamp: float = 1.0,
    frame_index: int = 0,
    source_id: str = "cam-1",
) -> IntrusionEvent:
    return IntrusionEvent(
        event_id=event_id,
        track_id=track_id,
        zone_id=zone_id,
        zone_name=zone_name,
        event_type=event_type,
        timestamp=timestamp,
        frame_index=frame_index,
        source_id=source_id,
    )


def make_loitering_event(
    event_id: str = "evt-2",
    track_id: int = 1,
    zone_id: str = "zone_a",
    zone_name: str = "Zone A",
    dwell_time_seconds: float = 12.0,
    threshold_seconds: float = 10.0,
    timestamp: float = 2.0,
    frame_index: int = 1,
    source_id: str = "cam-1",
) -> LoiteringEvent:
    return LoiteringEvent(
        event_id=event_id,
        track_id=track_id,
        zone_id=zone_id,
        zone_name=zone_name,
        dwell_time_seconds=dwell_time_seconds,
        threshold_seconds=threshold_seconds,
        timestamp=timestamp,
        frame_index=frame_index,
        source_id=source_id,
    )


def make_surveillance_event(
    event_id: str,
    event_type: EventType = EventType.INTRUSION_ENTER,
    severity: EventSeverity = EventSeverity.HIGH,
    source: EventSource = EventSource.INTRUSION,
    track_id: int = 1,
    zone_id: str = "zone_a",
    zone_name: str = "Zone A",
    timestamp=1.0,
    frame_index: int = 0,
    source_id: str = "cam-1",
) -> SurveillanceEvent:
    return SurveillanceEvent(
        event_id=event_id,
        event_type=event_type,
        severity=severity,
        source=source,
        track_id=track_id,
        zone_id=zone_id,
        zone_name=zone_name,
        timestamp=timestamp,
        frame_index=frame_index,
        source_id=source_id,
        payload={},
    )


# --- aggregation / normalization -----------------------------------------


def test_evaluate_merges_intrusion_and_loitering_events() -> None:
    engine = make_engine()

    events = engine.evaluate(
        [make_intrusion_event(event_id="evt-1")],
        [make_loitering_event(event_id="evt-2")],
    )

    assert len(events) == 2
    assert {e.event_id for e in events} == {"evt-1", "evt-2"}


def test_intrusion_enter_normalizes_with_correct_type_source_and_payload() -> None:
    engine = make_engine()

    events = engine.evaluate([make_intrusion_event(event_type=IntrusionEventType.ENTER)], [])

    event = events[0]
    assert event.event_type is EventType.INTRUSION_ENTER
    assert event.source is EventSource.INTRUSION
    assert event.severity is EventSeverity.HIGH
    assert dict(event.payload) == {"event_type": "ENTER"}


def test_intrusion_exit_normalizes_with_correct_type_and_severity() -> None:
    engine = make_engine()

    events = engine.evaluate([make_intrusion_event(event_type=IntrusionEventType.EXIT)], [])

    event = events[0]
    assert event.event_type is EventType.INTRUSION_EXIT
    assert event.severity is EventSeverity.LOW
    assert dict(event.payload) == {"event_type": "EXIT"}


def test_loitering_normalizes_with_dwell_payload() -> None:
    engine = make_engine()

    events = engine.evaluate([], [make_loitering_event(dwell_time_seconds=15.5, threshold_seconds=10.0)])

    event = events[0]
    assert event.event_type is EventType.LOITERING
    assert event.source is EventSource.LOITERING
    assert event.severity is EventSeverity.MEDIUM
    assert dict(event.payload) == {"dwell_time_seconds": 15.5, "threshold_seconds": 10.0}


def test_empty_input_returns_empty_list() -> None:
    engine = make_engine()

    assert engine.evaluate([], []) == []


# --- sorting --------------------------------------------------------------


def test_aggregate_sorts_by_timestamp_then_frame_index_then_event_id() -> None:
    engine = make_engine()

    e1 = make_surveillance_event("evt-b", timestamp=2.0, frame_index=0)
    e2 = make_surveillance_event("evt-a", timestamp=1.0, frame_index=5)
    e3 = make_surveillance_event("evt-z", timestamp=1.0, frame_index=1)
    e4 = make_surveillance_event("evt-a", timestamp=1.0, frame_index=1)  # ties with e3 on (ts, frame) -> event_id breaks tie

    result = engine.aggregate([e1, e2, e3, e4])

    # (1.0, 1, "evt-a"), (1.0, 1, "evt-z"), (1.0, 5, "evt-a"), (2.0, 0, "evt-b")
    assert [(e.event_id, e.frame_index) for e in result] == [
        ("evt-a", 1),
        ("evt-z", 1),
        ("evt-a", 5),
        ("evt-b", 0),
    ]


# --- severity mapping --------------------------------------------------


def test_custom_severity_mapping_overrides_default() -> None:
    engine = make_engine(severity_mapping={"intrusion_enter": EventSeverity.LOW, "intrusion_exit": EventSeverity.LOW, "loitering": EventSeverity.LOW})

    events = engine.evaluate([make_intrusion_event(event_type=IntrusionEventType.ENTER)], [])

    assert events[0].severity is EventSeverity.LOW


def test_unmapped_event_type_defaults_to_low_severity() -> None:
    # severity_mapping present (non-empty, passes validation) but missing "loitering"
    engine = make_engine(severity_mapping={"intrusion_enter": EventSeverity.HIGH, "intrusion_exit": EventSeverity.LOW})

    events = engine.evaluate([], [make_loitering_event()])

    assert events[0].severity is EventSeverity.LOW


# --- filtering -------------------------------------------------------


def test_filter_by_enabled_event_types() -> None:
    engine = make_engine(enabled_event_types=("intrusion_enter",))

    events = engine.evaluate(
        [make_intrusion_event(event_id="e1", event_type=IntrusionEventType.ENTER),
         make_intrusion_event(event_id="e2", event_type=IntrusionEventType.EXIT, timestamp=2.0, frame_index=1)],
        [make_loitering_event(event_id="e3", timestamp=3.0, frame_index=2)],
    )

    assert len(events) == 1
    assert events[0].event_type is EventType.INTRUSION_ENTER


def test_filter_by_minimum_severity() -> None:
    engine = make_engine(minimum_severity=EventSeverity.HIGH)

    events = engine.evaluate(
        [make_intrusion_event(event_id="e1", event_type=IntrusionEventType.ENTER),  # HIGH
         make_intrusion_event(event_id="e2", event_type=IntrusionEventType.EXIT, timestamp=2.0, frame_index=1)],  # LOW
        [make_loitering_event(event_id="e3", timestamp=3.0, frame_index=2)],  # MEDIUM
    )

    assert len(events) == 1
    assert events[0].severity is EventSeverity.HIGH


def test_filter_by_zone() -> None:
    engine = make_engine(zone_filter=("zone_a",))

    events = engine.evaluate(
        [make_intrusion_event(event_id="e1", zone_id="zone_a"),
         make_intrusion_event(event_id="e2", zone_id="zone_b", timestamp=2.0, frame_index=1)],
        [],
    )

    assert len(events) == 1
    assert events[0].zone_id == "zone_a"


def test_filter_by_track_is_optional_and_permissive_when_unset() -> None:
    engine = make_engine()  # track_filter defaults to () -> no restriction

    events = engine.evaluate(
        [make_intrusion_event(event_id="e1", track_id=1),
         make_intrusion_event(event_id="e2", track_id=2, timestamp=2.0, frame_index=1)],
        [],
    )

    assert len(events) == 2


def test_filter_by_track_when_set() -> None:
    engine = make_engine(track_filter=(1,))

    events = engine.evaluate(
        [make_intrusion_event(event_id="e1", track_id=1),
         make_intrusion_event(event_id="e2", track_id=2, timestamp=2.0, frame_index=1)],
        [],
    )

    assert len(events) == 1
    assert events[0].track_id == 1


# --- deduplication --------------------------------------------------------


def test_deduplicates_identical_track_zone_timestamp_type() -> None:
    engine = make_engine()

    events = engine.evaluate(
        [
            make_intrusion_event(event_id="e1", track_id=1, zone_id="zone_a", timestamp=5.0, event_type=IntrusionEventType.ENTER),
            make_intrusion_event(event_id="e2", track_id=1, zone_id="zone_a", timestamp=5.0, event_type=IntrusionEventType.ENTER),
        ],
        [],
    )

    assert len(events) == 1
    assert events[0].event_id == "e1"  # first occurrence (post-sort) wins


def test_does_not_deduplicate_different_event_types_with_same_track_zone_timestamp() -> None:
    engine = make_engine()

    events = engine.evaluate(
        [make_intrusion_event(event_id="e1", track_id=1, zone_id="zone_a", timestamp=5.0, event_type=IntrusionEventType.ENTER)],
        [make_loitering_event(event_id="e2", track_id=1, zone_id="zone_a", timestamp=5.0)],
    )

    assert len(events) == 2


# --- mixed event types / generic process() entry point --------------------


def test_process_is_a_generic_entry_point_for_future_event_sources() -> None:
    engine = make_engine()

    # Simulates a hypothetical future module that builds SurveillanceEvent
    # directly and calls process() without going through evaluate() at all.
    hypothetical_future_event = make_surveillance_event(
        "evt-future", event_type=EventType.LOITERING, severity=EventSeverity.MEDIUM, source=EventSource.LOITERING
    )

    result = engine.process([hypothetical_future_event])

    assert len(result) == 1
    assert result[0].event_id == "evt-future"


# --- disabled configuration ------------------------------------------------


def test_disabled_configuration_always_returns_empty() -> None:
    engine = make_engine(enabled=False)

    events = engine.evaluate([make_intrusion_event()], [make_loitering_event()])

    assert events == []


# --- config validation / exception handling --------------------------


def test_empty_severity_mapping_raises_configuration_error() -> None:
    with pytest.raises(EventConfigurationError):
        EventEngine(EventEngineConfig(severity_mapping={}))


def test_invalid_severity_value_raises_configuration_error() -> None:
    with pytest.raises(EventConfigurationError):
        EventEngine(EventEngineConfig(severity_mapping={"intrusion_enter": "not-a-severity"}))


def test_incomparable_timestamps_raise_aggregation_error() -> None:
    engine = make_engine()
    bad_event = make_surveillance_event("evt-bad", timestamp=None)
    ok_event = make_surveillance_event("evt-ok", timestamp=1.0)

    with pytest.raises(EventAggregationError):
        engine.process([bad_event, ok_event])


# --- stream wrapper --------------------------------------------------------


def test_evaluate_stream_yields_one_list_per_frame() -> None:
    engine = make_engine()
    frames = [
        ([make_intrusion_event(event_id="e1")], []),
        ([], [make_loitering_event(event_id="e2", timestamp=2.0, frame_index=1)]),
    ]

    results = list(engine.evaluate_stream(frames))

    assert len(results) == 2
    assert [e.event_id for e in results[0]] == ["e1"]
    assert [e.event_id for e in results[1]] == ["e2"]
