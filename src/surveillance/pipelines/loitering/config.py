"""Resolves a LoiteringConfig from config.yaml."""

from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from .types import LoiteringConfig


def build_loitering_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> LoiteringConfig:
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    section = yaml_config.get("loitering", {}) if yaml_config else {}

    monitor_zone_types = section.get("monitor_zone_types", ["intrusion"])
    if not isinstance(monitor_zone_types, list):
        monitor_zone_types = [monitor_zone_types]

    return LoiteringConfig(
        enabled=bool(section.get("enabled", True)),
        threshold_seconds=float(section.get("threshold_seconds", 10.0)),
        monitor_zone_types=tuple(str(t) for t in monitor_zone_types),
        verbose=bool(section.get("verbose", False)),
        stale_state_ttl_seconds=float(section.get("stale_state_ttl_seconds", 300.0)),
    )
