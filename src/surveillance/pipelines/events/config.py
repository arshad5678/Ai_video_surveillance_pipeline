"""Resolves an EventEngineConfig from config.yaml."""

from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from ...models.domain.surveillance_event import EventSeverity
from .exceptions import EventConfigurationError
from .types import EventEngineConfig

_DEFAULT_SEVERITY_MAPPING = {
    "intrusion_enter": "HIGH",
    "intrusion_exit": "LOW",
    "loitering": "MEDIUM",
}


def build_event_engine_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> EventEngineConfig:
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    section = yaml_config.get("events", {}) if yaml_config else {}

    enabled_event_types = section.get("enabled_event_types", []) or []
    zone_filter = section.get("zone_filter", []) or []
    track_filter = section.get("track_filter", []) or []
    raw_severity_mapping = section.get("severity_mapping", _DEFAULT_SEVERITY_MAPPING) or {}

    try:
        severity_mapping = {str(k): EventSeverity(str(v).upper()) for k, v in raw_severity_mapping.items()}
        minimum_severity = EventSeverity(str(section.get("minimum_severity", "LOW")).upper())
    except ValueError as exc:
        raise EventConfigurationError(f"Invalid severity value in events config: {exc}") from exc

    return EventEngineConfig(
        enabled=bool(section.get("enabled", True)),
        enabled_event_types=tuple(str(t) for t in enabled_event_types),
        minimum_severity=minimum_severity,
        zone_filter=tuple(str(z) for z in zone_filter),
        track_filter=tuple(int(t) for t in track_filter),
        severity_mapping=severity_mapping,
        verbose=bool(section.get("verbose", False)),
    )
