"""EventEngine: merges, sorts, filters, and deduplicates events into one unified stream.

Sole responsibility: aggregation, ordering, filtering, and dispatch
preparation. No storage, notifications, REST APIs, dashboards, or alert
delivery live here or are imported by this module.
"""

from types import MappingProxyType
from typing import Dict, Iterable, Iterator, List, Set, Tuple

from loguru import logger

from ...models.domain.intrusion_event import IntrusionEvent, IntrusionEventType
from ...models.domain.loitering_event import LoiteringEvent
from ...models.domain.surveillance_event import EventSeverity, EventSource, EventType, SurveillanceEvent
from .exceptions import EventAggregationError, EventConfigurationError
from .types import EventEngineConfig

_SEVERITY_RANK: Dict[EventSeverity, int] = {
    EventSeverity.LOW: 0,
    EventSeverity.MEDIUM: 1,
    EventSeverity.HIGH: 2,
}


class EventEngine:
    """Normalizes IntrusionEvent/LoiteringEvent into SurveillanceEvent, then aggregates them.

    Usage:
        engine = EventEngine(config)
        events = engine.evaluate(intrusion_events, loitering_events)

    A future event source doesn't need EventEngine to change at all — it
    only needs its own normalization adapter producing SurveillanceEvent
    objects, then can call `engine.process(all_normalized_events)`
    directly; aggregate()/filter_events()/deduplicate() are already
    generic over SurveillanceEvent.
    """

    def __init__(self, config: EventEngineConfig) -> None:
        self._validate_config(config)
        self._config = config
        self._frame_count = 0

        logger.info(
            "EventEngine initialized: enabled={}, enabled_event_types={}, minimum_severity={}, "
            "zone_filter={}, track_filter={}",
            config.enabled,
            config.enabled_event_types or "(all)",
            config.minimum_severity.value,
            config.zone_filter or "(all)",
            config.track_filter or "(all)",
        )

    def evaluate(
        self, intrusion_events: List[IntrusionEvent], loitering_events: List[LoiteringEvent]
    ) -> List[SurveillanceEvent]:
        """Normalize this frame's typed events into SurveillanceEvent, then process() them."""
        if not self._config.enabled:
            return []

        normalized = [self._from_intrusion(event) for event in intrusion_events]
        normalized.extend(self._from_loitering(event) for event in loitering_events)

        return self.process(normalized)

    def process(self, events: List[SurveillanceEvent]) -> List[SurveillanceEvent]:
        """Aggregate, filter, and deduplicate an already-normalized event list.

        This is the generic core: it never inspects anything specific to
        intrusion or loitering, only the common SurveillanceEvent fields.
        """
        if not self._config.enabled:
            return []

        try:
            logger.debug("Events received: {}", len(events))

            aggregated = self.aggregate(events)

            filtered = self.filter_events(aggregated)
            logger.debug("Events filtered: {} -> {}", len(aggregated), len(filtered))

            deduplicated = self.deduplicate(filtered)
            logger.debug("Events deduplicated: {} -> {}", len(filtered), len(deduplicated))
        except Exception as exc:
            logger.error("Event aggregation failed: {}", exc)
            raise EventAggregationError(f"Event aggregation failed: {exc}") from exc

        self._frame_count += 1
        if deduplicated:
            logger.info("Events dispatched: {}", len(deduplicated))
        if self._frame_count % 100 == 0:
            logger.info("Processed {} frames of events so far.", self._frame_count)

        return deduplicated

    def evaluate_stream(
        self, frames: Iterable[Tuple[List[IntrusionEvent], List[LoiteringEvent]]]
    ) -> Iterator[List[SurveillanceEvent]]:
        """Generator: run evaluate() over a stream of (intrusion_events, loitering_events) tuples."""
        count = 0
        for intrusion_events, loitering_events in frames:
            yield self.evaluate(intrusion_events, loitering_events)
            count += 1
        logger.info("Evaluation completed: {} frames processed.", count)

    def aggregate(self, events: List[SurveillanceEvent]) -> List[SurveillanceEvent]:
        """Merge (already just one flat list) and sort by (timestamp, frame_index, event_id)."""
        return sorted(events, key=lambda event: (event.timestamp, event.frame_index, event.event_id))

    def filter_events(self, events: List[SurveillanceEvent]) -> List[SurveillanceEvent]:
        """Apply enabled-event-types, minimum-severity, zone, and (optional) track filters."""
        minimum_rank = _SEVERITY_RANK[self._config.minimum_severity]

        result = []
        for event in events:
            if self._config.enabled_event_types and event.event_type.value not in self._config.enabled_event_types:
                continue
            if _SEVERITY_RANK[event.severity] < minimum_rank:
                continue
            if self._config.zone_filter and event.zone_id not in self._config.zone_filter:
                continue
            if self._config.track_filter and event.track_id not in self._config.track_filter:
                continue
            result.append(event)

        return result

    def deduplicate(self, events: List[SurveillanceEvent]) -> List[SurveillanceEvent]:
        """Drop repeats sharing (track_id, zone_id, timestamp, event_type) — first occurrence wins."""
        seen: Set[Tuple[int, str, float, EventType]] = set()
        result = []
        for event in events:
            key = (event.track_id, event.zone_id, event.timestamp, event.event_type)
            if key in seen:
                continue
            seen.add(key)
            result.append(event)

        return result

    def _from_intrusion(self, event: IntrusionEvent) -> SurveillanceEvent:
        event_type = (
            EventType.INTRUSION_ENTER if event.event_type is IntrusionEventType.ENTER else EventType.INTRUSION_EXIT
        )
        return SurveillanceEvent(
            event_id=event.event_id,
            event_type=event_type,
            severity=self._resolve_severity(event_type),
            source=EventSource.INTRUSION,
            track_id=event.track_id,
            zone_id=event.zone_id,
            zone_name=event.zone_name,
            timestamp=event.timestamp,
            frame_index=event.frame_index,
            source_id=event.source_id,
            payload=MappingProxyType({"event_type": event.event_type.value}),
        )

    def _from_loitering(self, event: LoiteringEvent) -> SurveillanceEvent:
        event_type = EventType.LOITERING
        return SurveillanceEvent(
            event_id=event.event_id,
            event_type=event_type,
            severity=self._resolve_severity(event_type),
            source=EventSource.LOITERING,
            track_id=event.track_id,
            zone_id=event.zone_id,
            zone_name=event.zone_name,
            timestamp=event.timestamp,
            frame_index=event.frame_index,
            source_id=event.source_id,
            payload=MappingProxyType(
                {
                    "dwell_time_seconds": event.dwell_time_seconds,
                    "threshold_seconds": event.threshold_seconds,
                }
            ),
        )

    def _resolve_severity(self, event_type: EventType) -> EventSeverity:
        severity = self._config.severity_mapping.get(event_type.value)
        if severity is None:
            logger.warning("No severity mapping for event_type={}; defaulting to LOW.", event_type.value)
            return EventSeverity.LOW
        return severity

    @staticmethod
    def _validate_config(config: EventEngineConfig) -> None:
        if not config.severity_mapping:
            raise EventConfigurationError("severity_mapping must not be empty.")
        for event_type, severity in config.severity_mapping.items():
            if not isinstance(severity, EventSeverity):
                raise EventConfigurationError(f"Invalid severity for {event_type!r}: {severity!r}")
