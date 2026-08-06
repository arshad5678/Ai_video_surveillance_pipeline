"""IntrusionDetector: turns per-frame ZoneMembership lists into ENTER/EXIT IntrusionEvent objects.

Sole responsibility: intrusion state transitions for (track_id, zone_id)
pairs. No loitering-duration logic, no alerting, no persistence — those
belong to future modules built on top of the IntrusionEvent objects this
module produces. Depends only on ZoneMembership and Zone.
"""

from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from loguru import logger

from ...models.domain.intrusion_event import IntrusionEvent, IntrusionEventType
from ...models.domain.intrusion_state import IntrusionState
from ...models.domain.zone import Zone
from ...models.domain.zone_membership import ZoneMembership
from .exceptions import IntrusionConfigurationError, IntrusionEvaluationError
from .types import IntrusionConfig


class IntrusionDetector:
    """Detects ENTER/EXIT zone-occupancy transitions from ZoneMembership streams.

    A never-before-seen (track_id, zone_id) pair is treated as starting
    from an implicit "outside" state — so a track observed already inside
    a monitored zone on its very first appearance still produces an ENTER
    (better to flag an occupant who was already present than to silently
    miss them), while one first observed outside produces no event, since
    there is nothing to transition from yet.

    Usage:
        detector = IntrusionDetector(config, zones)
        for events in detector.evaluate_stream(memberships_stream):
            ...  # hand `events` to a future Event Engine
    """

    def __init__(self, config: IntrusionConfig, zones: List[Zone]) -> None:
        self._validate_config(config)
        self._config = config

        self._monitored_zone_ids: Set[str] = {
            zone.zone_id for zone in zones if zone.zone_type in config.monitor_zone_types
        }
        self._zone_names: Dict[str, str] = {zone.zone_id: zone.zone_name for zone in zones}

        self._states: Dict[Tuple[int, str], IntrusionState] = {}
        self._last_known_timestamp = 0.0
        self._event_counter = 1
        self._frame_count = 0

        logger.info(
            "IntrusionDetector initialized: enabled={}, monitor_zone_types={}, monitored_zones={}",
            config.enabled,
            config.monitor_zone_types,
            len(self._monitored_zone_ids),
        )

    def evaluate(self, memberships: List[ZoneMembership]) -> List[IntrusionEvent]:
        """Compute ENTER/EXIT events for monitored zones from this frame's memberships.

        Memberships for zones outside `monitor_zone_types` are ignored
        entirely — no state is created or updated for them.
        """
        if not self._config.enabled:
            return []

        events: List[IntrusionEvent] = []
        try:
            for membership in memberships:
                if membership.zone_id not in self._monitored_zone_ids:
                    continue

                event = self._process_membership(membership)
                if event is not None:
                    events.append(event)

                self._last_known_timestamp = max(self._last_known_timestamp, membership.timestamp)

            self._cleanup_stale_states()
        except Exception as exc:
            logger.error("Intrusion evaluation failed: {}", exc)
            raise IntrusionEvaluationError(f"Intrusion evaluation failed: {exc}") from exc

        self._frame_count += 1
        if self._frame_count % 100 == 0:
            logger.info("Evaluated intrusion state for {} frames so far.", self._frame_count)

        return events

    def evaluate_stream(
        self, memberships_stream: Iterable[List[ZoneMembership]]
    ) -> Iterator[List[IntrusionEvent]]:
        """Generator: run evaluate() over a stream of per-frame ZoneMembership lists.

        Chains directly onto ZoneManager.evaluate_stream():
            detector.evaluate_stream(zone_manager.evaluate_stream(tracks_stream))
        """
        logger.info("Frame evaluation started.")
        count = 0
        for memberships in memberships_stream:
            yield self.evaluate(memberships)
            count += 1
        logger.info("Evaluation completed: {} frames processed.", count)

    def get_state(self, track_id: int, zone_id: str) -> Optional[IntrusionState]:
        """Read-only snapshot of the current state for a (track_id, zone_id) pair, if any.

        Returns an immutable IntrusionState (or None) — never a reference
        into the internal dict — so a future module (e.g. Loitering
        Detection) can reuse `first_seen_inside_timestamp` without
        depending on this class's internals.
        """
        return self._states.get((track_id, zone_id))

    def _process_membership(self, membership: ZoneMembership) -> Optional[IntrusionEvent]:
        key = (membership.track_id, membership.zone_id)
        previous = self._states.get(key)
        was_inside = previous.currently_inside if previous is not None else False

        event: Optional[IntrusionEvent] = None
        first_seen_inside_timestamp = previous.first_seen_inside_timestamp if previous is not None else None

        if membership.inside and not was_inside:
            event = self._build_event(membership, IntrusionEventType.ENTER)
            first_seen_inside_timestamp = membership.timestamp
        elif not membership.inside and was_inside:
            if self._config.emit_exit_events:
                event = self._build_event(membership, IntrusionEventType.EXIT)
            first_seen_inside_timestamp = None
        # else: no transition — first_seen_inside_timestamp carries forward unchanged

        self._states[key] = IntrusionState(
            track_id=membership.track_id,
            zone_id=membership.zone_id,
            currently_inside=membership.inside,
            first_seen_inside_timestamp=first_seen_inside_timestamp,
            last_seen_timestamp=membership.timestamp,
        )

        return event

    def _build_event(self, membership: ZoneMembership, event_type: IntrusionEventType) -> IntrusionEvent:
        event = IntrusionEvent(
            event_id=f"evt-{self._event_counter}",
            track_id=membership.track_id,
            zone_id=membership.zone_id,
            zone_name=self._zone_names.get(membership.zone_id, membership.zone_id),
            event_type=event_type,
            timestamp=membership.timestamp,
            frame_index=membership.frame_index,
            source_id=membership.source_id,
        )
        self._event_counter += 1

        if event_type is IntrusionEventType.ENTER:
            logger.info("ENTER detected: track_id={}, zone_id={}", membership.track_id, membership.zone_id)
        else:
            logger.info("EXIT detected: track_id={}, zone_id={}", membership.track_id, membership.zone_id)

        return event

    def _cleanup_stale_states(self) -> None:
        ttl = self._config.stale_state_ttl_seconds
        stale_keys = [
            key
            for key, state in self._states.items()
            if self._last_known_timestamp - state.last_seen_timestamp > ttl
        ]
        for key in stale_keys:
            del self._states[key]
            logger.debug("Cleaned up stale intrusion state: track_id={}, zone_id={}", key[0], key[1])

    @staticmethod
    def _validate_config(config: IntrusionConfig) -> None:
        if not config.monitor_zone_types:
            raise IntrusionConfigurationError("monitor_zone_types must contain at least one zone type.")
        if config.stale_state_ttl_seconds <= 0:
            raise IntrusionConfigurationError("stale_state_ttl_seconds must be positive.")
