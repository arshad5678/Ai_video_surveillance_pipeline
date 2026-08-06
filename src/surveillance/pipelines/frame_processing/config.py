"""Resolves a PreprocessConfig from config.yaml.

Unlike video_input, this module has nothing environment-specific to
configure (no secrets, no "which source" decision) — every knob here is
non-secret pipeline tuning, so config.yaml alone is authoritative.
"""

from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from .exceptions import InvalidPreprocessConfigError
from .types import ColorMode, PreprocessConfig


def build_preprocess_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> PreprocessConfig:
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    section = yaml_config.get("frame_processing", {}) if yaml_config else {}

    resize = section.get("resize") or {}
    color = section.get("color_conversion") or {}
    normalize = section.get("normalize") or {}
    frame_skip = section.get("frame_skip") or {}

    try:
        color_mode = ColorMode(str(color.get("mode", "RGB")).upper())
    except ValueError as exc:
        raise InvalidPreprocessConfigError(
            f"Invalid frame_processing.color_conversion.mode: {color.get('mode')!r}"
        ) from exc

    mean = normalize.get("mean")
    std = normalize.get("std")

    return PreprocessConfig(
        resize_enabled=bool(resize.get("enabled", False)),
        resize_width=int(resize.get("width", 640)),
        resize_height=int(resize.get("height", 640)),
        color_conversion_enabled=bool(color.get("enabled", False)),
        color_mode=color_mode,
        normalize_enabled=bool(normalize.get("enabled", False)),
        normalize_scale=float(normalize.get("scale", 1.0 / 255.0)),
        normalize_mean=tuple(mean) if mean else None,
        normalize_std=tuple(std) if std else None,
        frame_skip_enabled=bool(frame_skip.get("enabled", False)),
        frame_skip_interval=int(frame_skip.get("interval", 1)),
    )
