"""Types describing person-detection configuration.

Scoped to this module only — future consumers (tracking, events, etc.)
never see these; they only ever receive `Detection` objects.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorConfig:
    """Fully-resolved parameters controlling PersonDetector's behavior."""

    model_path: str = "weights/yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    image_size: int = 640
    verbose: bool = False
