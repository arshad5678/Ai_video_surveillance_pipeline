"""Resolves an IntrusionConfig from config.yaml.

Like frame_processing/detection/tracking, everything here is non-secret
pipeline tuning, so config.yaml alone is authoritative.
"""

from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from .types import IntrusionConfig


def build_intrusion_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> IntrusionConfig:
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    section = yaml_config.get("intrusion", {}) if yaml_config else {}

    monitor_zone_types = section.get("monitor_zone_types", ["intrusion"])
    if not isinstance(monitor_zone_types, list):
        monitor_zone_types = [monitor_zone_types]

    return IntrusionConfig(
        enabled=bool(section.get("enabled", True)),
        monitor_zone_types=tuple(str(t) for t in monitor_zone_types),
        emit_exit_events=bool(section.get("emit_exit_events", True)),
        verbose=bool(section.get("verbose", False)),
        stale_state_ttl_seconds=float(section.get("stale_state_ttl_seconds", 300.0)),
    )
