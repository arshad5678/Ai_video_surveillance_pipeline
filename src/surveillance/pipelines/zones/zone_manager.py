"""ZoneManager: determines whether each tracked person's center point falls inside configured zones.

Sole responsibility: spatial reasoning. It never decides that a
ZoneMembership constitutes an "intrusion" or "loitering" event, and never
raises alerts — that judgment belongs to a future module that consumes
ZoneMembership objects. No zone/event/API/database/alert logic beyond
point-in-polygon math lives here.
"""

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Union

from loguru import logger
from shapely.geometry import Point, Polygon

from ...core.constants import DEFAULT_ZONES_CONFIG_PATH
from ...models.domain.track import Track
from ...models.domain.zone import Zone
from ...models.domain.zone_membership import ZoneMembership
from .config import load_zones_config
from .exceptions import ZoneEvaluationError


class ZoneManager:
    """Evaluates Track center points against configured polygon zones.

    Zones are loaded once at construction and can be reloaded at any time
    via reload() — e.g. after an operator edits zones.yaml — without
    restarting the pipeline.

    Usage:
        zone_manager = ZoneManager()
        for memberships in zone_manager.evaluate_stream(tracks_stream):
            ...  # hand `memberships` to a future Intrusion Detection module
    """

    def __init__(self, zones_config_path: Union[str, Path] = DEFAULT_ZONES_CONFIG_PATH) -> None:
        self._zones_config_path = zones_config_path
        self._zones: List[Zone] = []
        self._polygons: Dict[str, Polygon] = {}
        self._frame_count = 0

        self.reload()

        logger.info("ZoneManager initialized with {} zone(s).", len(self._zones))

    @property
    def zones(self) -> Tuple[Zone, ...]:
        """Read-only snapshot of the currently loaded zones.

        Added so downstream modules (e.g. IntrusionDetector, which needs
        Zone objects to resolve zone_type/zone_name) can share this
        ZoneManager's authoritative, reload()-aware zone list instead of
        re-parsing zones.yaml themselves and risking drift after a reload.
        """
        return tuple(self._zones)

    def reload(self) -> None:
        """Reload zone definitions from disk without restarting the application."""
        zones = load_zones_config(self._zones_config_path)
        polygons = {zone.zone_id: Polygon([(p.x, p.y) for p in zone.polygon]) for zone in zones}

        self._zones = zones
        self._polygons = polygons

        logger.info("Zones loaded: {} zone(s) from {}", len(zones), self._zones_config_path)

    def evaluate(self, tracks: List[Track]) -> List[ZoneMembership]:
        """Compute one ZoneMembership per (Track, enabled Zone) pair.

        Disabled zones are skipped entirely — no ZoneMembership is
        produced for them.
        """
        memberships: List[ZoneMembership] = []
        try:
            enabled_zones = [zone for zone in self._zones if zone.enabled]
            for track in tracks:
                center = Point(track.bounding_box.center_x, track.bounding_box.center_y)
                for zone in enabled_zones:
                    inside = self._polygons[zone.zone_id].contains(center)
                    if inside:
                        logger.debug("Track {} entered polygon of zone '{}'.", track.track_id, zone.zone_id)
                    else:
                        logger.debug("Track {} outside polygon of zone '{}'.", track.track_id, zone.zone_id)

                    memberships.append(
                        ZoneMembership(
                            track_id=track.track_id,
                            zone_id=zone.zone_id,
                            inside=inside,
                            timestamp=track.timestamp,
                            frame_index=track.frame_index,
                            source_id=track.source_id,
                        )
                    )
        except Exception as exc:
            logger.error("Zone evaluation failed: {}", exc)
            raise ZoneEvaluationError(f"Zone evaluation failed: {exc}") from exc

        self._frame_count += 1
        logger.debug("Frame -> {} membership(s) evaluated.", len(memberships))
        if self._frame_count % 100 == 0:
            logger.info("Evaluated zones for {} frames so far.", self._frame_count)

        return memberships

    def evaluate_stream(self, tracks_stream: Iterable[List[Track]]) -> Iterator[List[ZoneMembership]]:
        """Generator: run evaluate() over a stream of per-frame Track lists.

        Chains directly onto MultiObjectTracker.track_stream():
            zone_manager.evaluate_stream(tracker.track_stream(detections_stream))
        """
        logger.info("Frame zone evaluation started.")
        count = 0
        for tracks in tracks_stream:
            yield self.evaluate(tracks)
            count += 1
        logger.info("Evaluation completed: {} frames processed.", count)
