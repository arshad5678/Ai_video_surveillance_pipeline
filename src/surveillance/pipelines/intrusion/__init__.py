"""Intrusion Detection module — ENTER/EXIT zone-occupancy transitions only.

Does not implement loitering duration logic, alerting, or persistence —
those are future modules built on top of the IntrusionEvent objects this
module produces.

Public API:
    IntrusionDetector       — stateful ENTER/EXIT transition detector
    build_intrusion_config  — resolves IntrusionConfig from config.yaml
    IntrusionConfig
    IntrusionError, IntrusionConfigurationError, IntrusionEvaluationError
"""

from .config import build_intrusion_config
from .exceptions import IntrusionConfigurationError, IntrusionError, IntrusionEvaluationError
from .intrusion_detector import IntrusionDetector
from .types import IntrusionConfig

__all__ = [
    "IntrusionDetector",
    "build_intrusion_config",
    "IntrusionConfig",
    "IntrusionError",
    "IntrusionConfigurationError",
    "IntrusionEvaluationError",
]
