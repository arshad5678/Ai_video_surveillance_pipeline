"""Resolves a DetectorConfig from config.yaml.

Like frame_processing, everything here is non-secret pipeline tuning, so
config.yaml alone is authoritative — no .env involvement needed. `device`
falls back to the shared `pipeline.device` value when not set explicitly
under `detection:`, so there's one place to change compute target unless
detection needs to differ from the rest of the pipeline.
"""

from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.settings import Settings, get_settings
from .types import DetectorConfig


def build_detector_config(
    settings: Optional[Settings] = None,
    yaml_config: Optional[Dict[str, Any]] = None,
) -> DetectorConfig:
    settings = settings or get_settings()
    if yaml_config is None:
        yaml_config = load_yaml_config(settings.config_path)

    section = yaml_config.get("detection", {}) if yaml_config else {}
    pipeline_section = yaml_config.get("pipeline", {}) if yaml_config else {}

    default_device = pipeline_section.get("device", "cpu")

    return DetectorConfig(
        model_path=str(section.get("model_path", "weights/yolov8n.pt")),
        confidence_threshold=float(section.get("confidence_threshold", 0.25)),
        iou_threshold=float(section.get("iou_threshold", 0.45)),
        device=str(section.get("device", default_device)),
        image_size=int(section.get("image_size", 640)),
        verbose=bool(section.get("verbose", False)),
    )
