"""Loads and validates zone definitions from a YAML file into Zone domain objects.

Unlike the other pipeline stages, this module's "configuration" isn't a
flat set of tuning knobs read from config.yaml — it's a list of domain
objects (zones) with their own file (config/zones.yaml), so there's no
types.py/TrackingConfig-style dataclass here; the loader's return type
*is* the configuration.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml
from shapely.geometry import Polygon

from ...models.domain.zone import Zone
from ...models.domain.zone_point import ZonePoint
from .exceptions import InvalidPolygonError, ZoneConfigurationError

_MIN_POLYGON_POINTS = 3


def load_zones_config(path: Union[str, Path]) -> List[Zone]:
    config_path = Path(path)
    if not config_path.exists():
        raise ZoneConfigurationError(f"Zones config file not found: {config_path}")

    with config_path.open("r") as f:
        try:
            raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise ZoneConfigurationError(f"Failed to parse {config_path}: {exc}") from exc

    raw_zones = raw.get("zones", [])
    if not isinstance(raw_zones, list):
        raise ZoneConfigurationError(f"'zones' must be a list in {config_path}.")

    zones: List[Zone] = []
    seen_ids = set()
    for entry in raw_zones:
        zone = _parse_zone(entry)
        if zone.zone_id in seen_ids:
            raise ZoneConfigurationError(f"Duplicate zone id: {zone.zone_id!r}")
        seen_ids.add(zone.zone_id)
        zones.append(zone)

    return zones


def _parse_zone(entry: Dict[str, Any]) -> Zone:
    if not isinstance(entry, dict):
        raise ZoneConfigurationError(f"Each zone entry must be a mapping, got: {entry!r}")

    try:
        zone_id = str(entry["id"])
        raw_polygon = entry["polygon"]
    except KeyError as exc:
        raise ZoneConfigurationError(f"Zone entry missing required field: {exc}") from exc

    zone_name = str(entry.get("name", zone_id))
    zone_type = str(entry.get("type", "monitoring"))
    enabled = bool(entry.get("enabled", True))

    polygon = _parse_polygon(raw_polygon, zone_id)

    return Zone(zone_id=zone_id, zone_name=zone_name, zone_type=zone_type, polygon=polygon, enabled=enabled)


def _parse_polygon(raw_polygon: Any, zone_id: str) -> Tuple[ZonePoint, ...]:
    if not isinstance(raw_polygon, list) or len(raw_polygon) < _MIN_POLYGON_POINTS:
        raise InvalidPolygonError(
            f"Zone {zone_id!r}: polygon must be a list of at least {_MIN_POLYGON_POINTS} [x, y] points."
        )

    points: List[ZonePoint] = []
    for raw_point in raw_polygon:
        try:
            x, y = raw_point
            points.append(ZonePoint(x=float(x), y=float(y)))
        except (TypeError, ValueError) as exc:
            raise InvalidPolygonError(f"Zone {zone_id!r}: invalid polygon point {raw_point!r}") from exc

    polygon = tuple(points)

    shapely_polygon = Polygon([(p.x, p.y) for p in polygon])
    if not shapely_polygon.is_valid:
        raise InvalidPolygonError(
            f"Zone {zone_id!r}: polygon geometry is invalid (self-intersecting or degenerate)."
        )

    return polygon
