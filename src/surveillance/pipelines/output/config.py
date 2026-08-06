"""Resolves an OutputConfig from config.yaml."""

from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from .types import OutputConfig


def build_output_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> OutputConfig:
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    section = yaml_config.get("output", {}) if yaml_config else {}

    return OutputConfig(
        annotated_video=bool(section.get("annotated_video", True)),
        snapshots=bool(section.get("snapshots", True)),
        clips=bool(section.get("clips", True)),
        json_log=bool(section.get("json_log", True)),
        csv_log=bool(section.get("csv_log", True)),
        clip_pre_seconds=float(section.get("clip_pre_seconds", 5.0)),
        clip_post_seconds=float(section.get("clip_post_seconds", 5.0)),
        output_directory=str(section.get("output_directory", "output")),
        video_codec=str(section.get("video_codec", "mp4v")),
        jpeg_quality=int(section.get("jpeg_quality", 95)),
        frame_rate=float(section.get("frame_rate", 30.0)),
        verbose=bool(section.get("verbose", False)),
    )
