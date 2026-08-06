"""Resolves a VideoSourceConfig from .env (Settings) + config.yaml.

Kept separate from VideoInput itself so the class stays a pure function of
an explicit VideoSourceConfig — easy to unit test without touching env vars
or the filesystem-backed YAML file.
"""

from typing import Any, Dict, Optional, Union

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from .types import VideoSourceConfig, VideoSourceType


def build_video_source_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> VideoSourceConfig:
    """Build a VideoSourceConfig from environment settings and config.yaml.

    - `.env` (via Settings) supplies WHICH source to use: type + URI.
    - `config.yaml` supplies non-secret tuning defaults: reconnect/timeout behavior.
    """
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    video_cfg = yaml_config.get("video_input", {}) if yaml_config else {}

    source_type = VideoSourceType(settings.video_source_type.strip().lower())

    uri: Union[int, str]
    if source_type is VideoSourceType.WEBCAM:
        uri = int(settings.video_source)
    else:
        uri = settings.video_source

    return VideoSourceConfig(
        source_type=source_type,
        uri=uri,
        reconnect_attempts=int(video_cfg.get("reconnect_attempts", 3)),
        reconnect_delay_seconds=float(video_cfg.get("reconnect_delay_seconds", 2.0)),
        read_timeout_seconds=float(video_cfg.get("read_timeout_seconds", 5.0)),
    )
