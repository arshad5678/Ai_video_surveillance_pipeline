"""Resolves a TrackingConfig from config.yaml.

Like frame_processing and detection, everything here is non-secret
pipeline tuning, so config.yaml alone is authoritative.
"""

from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from .types import TrackingConfig


def build_tracking_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> TrackingConfig:
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    section = yaml_config.get("tracking", {}) if yaml_config else {}

    return TrackingConfig(
        tracker_type=str(section.get("tracker_type", "bytetrack")),
        track_high_thresh=float(section.get("track_high_thresh", 0.5)),
        track_low_thresh=float(section.get("track_low_thresh", 0.1)),
        new_track_thresh=float(section.get("new_track_thresh", 0.6)),
        track_buffer=int(section.get("track_buffer", 30)),
        match_thresh=float(section.get("match_thresh", 0.8)),
        frame_rate=int(section.get("frame_rate", 30)),
        minimum_box_area=float(section.get("minimum_box_area", 10)),
        history_length=int(section.get("history_length", 30)),
        verbose=bool(section.get("verbose", False)),
    )
