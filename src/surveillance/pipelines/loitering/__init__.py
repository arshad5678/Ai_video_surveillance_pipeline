"""Loitering Detection module — sustained zone-occupancy detection only.

Does not implement alerting, notification, event storage, or APIs — those
are future modules built on top of the LoiteringEvent objects this
module produces.

Public API:
    LoiteringDetector       — dwell-time threshold detector, reuses IntrusionDetector.get_state()
    build_loitering_config  — resolves LoiteringConfig from config.yaml
    LoiteringConfig
    LoiteringError, LoiteringConfigurationError, LoiteringEvaluationError
"""

from .config import build_loitering_config
from .exceptions import LoiteringConfigurationError, LoiteringError, LoiteringEvaluationError
from .loitering_detector import LoiteringDetector
from .types import LoiteringConfig

__all__ = [
    "LoiteringDetector",
    "build_loitering_config",
    "LoiteringConfig",
    "LoiteringError",
    "LoiteringConfigurationError",
    "LoiteringEvaluationError",
]
