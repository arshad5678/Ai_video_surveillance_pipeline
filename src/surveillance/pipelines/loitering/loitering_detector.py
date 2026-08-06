"""LoiteringDetector: detects a person remaining inside a monitored zone past a dwell-time threshold.

Sole responsibility: loitering detection for one continuous inside-stay
per (track_id, zone_id) pair. No alerting, no persistence, no APIs —
those belong to future modules built on top of the LoiteringEvent
objects this module produces.

Reuses IntrusionDetector.get_state() for first_seen_inside_timestamp
rather than re-deriving it — this module must run *after*
IntrusionDetector has processed the same frame's memberships, since that
call is what keeps first_seen_inside_timestamp current.
"""

from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

from loguru import logger

from ...models.domain.loitering_event import LoiteringEvent
from ...models.domain.loitering_state import LoiteringState
from ...models.domain.track import Track
from ...models.domain.zone import Zone
from ...models.domain.zone_membership import ZoneMembership
from ..intrusion import IntrusionDetector
from .exceptions import LoiteringConfigurationError, LoiteringEvaluationError
from .types import LoiteringConfig


class LoiteringDetector:
    """Detects sustained zone occupancy from Track + ZoneMembership streams.

    Usage:
        detector = LoiteringDetector(config, zones, intrusion_detector)
        events = detector.evaluate(tracks, memberships)  # AFTER intrusion_detector.evaluate(memberships)
    """

    def __init__(
        self,
        config: LoiteringConfig,
        zones: List[Zone],
        intrusion_detector: IntrusionDetector,
    ) -> None:
        self._validate_config(config)
        self._config = config
        self._intrusion_detector = intrusion_detector

        self._monitored_zone_ids: Set[str] = {
            zone.zone_id for zone in zones if zone.zone_type in config.monitor_zone_types
        }
        self._zone_names: Dict[str, str] = {zone.zone_id: zone.zone_name for zone in zones}

        self._states: Dict[Tuple[int, str], LoiteringState] = {}
        self._last_known_timestamp = 0.0
        self._event_counter = 1
        self._frame_count = 0

        logger.info(
            "LoiteringDetector initialized: enabled={}, threshold_seconds={}, monitor_zone_types={}, monitored_zones={}",
            config.enabled,
            config.threshold_seconds,
            config.monitor_zone_types,
            len(self._monitored_zone_ids),
        )

    def evaluate(self, tracks: List[Track], memberships: List[ZoneMembership]) -> List[LoiteringEvent]:
        """Compute loitering events for monitored zones from this frame's tracks/memberships.

        Only confirmed tracks (Track.is_confirmed) are considered, so a
        dwell timer never starts on a single-frame flicker detection.
        """
        if not self._config.enabled:
            return []

        events: List[LoiteringEvent] = []
        try:
            confirmed_track_ids = {track.track_id for track in tracks if track.is_confirmed}

            for membership in memberships:
                if membership.zone_id not in self._monitored_zone_ids:
                    continue

                event = self._process_membership(membership, confirmed_track_ids)
                if event is not None:
                    events.append(event)

                self._last_known_timestamp = max(self._last_known_timestamp, membership.timestamp)

            self._cleanup_stale_states()
        except Exception as exc:
            logger.error("Loitering evaluation failed: {}", exc)
            raise LoiteringEvaluationError(f"Loitering evaluation failed: {exc}") from exc

        self._frame_count += 1
        if self._frame_count % 100 == 0:
            logger.info("Evaluated loitering state for {} frames so far.", self._frame_count)

        return events

    def evaluate_stream(
        self, frames: Iterable[Tuple[List[Track], List[ZoneMembership]]]
    ) -> Iterator[List[LoiteringEvent]]:
        """Generator: run evaluate() over a stream of (tracks, memberships) tuples, one per frame.

        Note: from this stage on, the pipeline is no longer a simple
        linear chain — LoiteringDetector needs both the tracks list and
        the memberships derived from it for the same frame, so the
        caller is responsible for pairing them (typically via a manual
        per-frame loop rather than nested generator chaining; see
        scripts/test_loitering_detection.py).
        """
        logger.info("Frame evaluation started.")
        count = 0
        for tracks, memberships in frames:
            yield self.evaluate(tracks, memberships)
            count += 1
        logger.info("Evaluation completed: {} frames processed.", count)

    def _process_membership(
        self, membership: ZoneMembership, confirmed_track_ids: Set[int]
    ) -> Optional[LoiteringEvent]:
        key = (membership.track_id, membership.zone_id)

        if not membership.inside:
            if key in self._states:
                del self._states[key]
                logger.debug(
                    "State reset: track_id={}, zone_id={} (exited zone)", membership.track_id, membership.zone_id
                )
            return None

        if membership.track_id not in confirmed_track_ids:
            return None  # don't start/continue a dwell timer on an unconfirmed track

        intrusion_state = self._intrusion_detector.get_state(membership.track_id, membership.zone_id)
        first_seen = intrusion_state.first_seen_inside_timestamp if intrusion_state else None
        if first_seen is None:
            # Defensive fallback: IntrusionDetector should always have this
            # set while inside=True, but if it doesn't (e.g. wiring order
            # issue), treat this frame as the start rather than crashing.
            first_seen = membership.timestamp

        dwell_time = membership.timestamp - first_seen
        logger.debug(
            "Current dwell time: track_id={}, zone_id={}, dwell_time={:.2f}s",
            membership.track_id,
            membership.zone_id,
            dwell_time,
        )

        previous = self._states.get(key)
        already_emitted = previous.event_emitted if previous is not None else False

        event = None
        if dwell_time >= self._config.threshold_seconds and not already_emitted:
            event = self._build_event(membership, dwell_time)
            already_emitted = True

        self._states[key] = LoiteringState(
            track_id=membership.track_id,
            zone_id=membership.zone_id,
            entered_timestamp=first_seen,
            current_dwell_time=dwell_time,
            event_emitted=already_emitted,
        )

        return event

    def _build_event(self, membership: ZoneMembership, dwell_time: float) -> LoiteringEvent:
        event = LoiteringEvent(
            event_id=f"evt-{self._event_counter}",
            track_id=membership.track_id,
            zone_id=membership.zone_id,
            zone_name=self._zone_names.get(membership.zone_id, membership.zone_id),
            dwell_time_seconds=dwell_time,
            threshold_seconds=self._config.threshold_seconds,
            timestamp=membership.timestamp,
            frame_index=membership.frame_index,
            source_id=membership.source_id,
        )
        self._event_counter += 1

        logger.info(
            "Loitering detected: track_id={}, zone_id={}, dwell_time={:.2f}s (threshold={:.2f}s)",
            membership.track_id,
            membership.zone_id,
            dwell_time,
            self._config.threshold_seconds,
        )

        return event

    def _cleanup_stale_states(self) -> None:
        ttl = self._config.stale_state_ttl_seconds
        stale_keys = [
            key
            for key, state in self._states.items()
            if self._last_known_timestamp - (state.entered_timestamp + state.current_dwell_time) > ttl
        ]
        for key in stale_keys:
            del self._states[key]
            logger.debug("State reset: track_id={}, zone_id={} (stale cleanup)", key[0], key[1])

    @staticmethod
    def _validate_config(config: LoiteringConfig) -> None:
        if config.threshold_seconds <= 0:
            raise LoiteringConfigurationError("threshold_seconds must be positive.")
        if not config.monitor_zone_types:
            raise LoiteringConfigurationError("monitor_zone_types must contain at least one zone type.")
        if config.stale_state_ttl_seconds <= 0:
            raise LoiteringConfigurationError("stale_state_ttl_seconds must be positive.")
