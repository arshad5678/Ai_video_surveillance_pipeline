"""Zone Manager module — spatial reasoning only: is a Track's center inside a configured zone?

Does not generate intrusion/loitering events or alerts — that is a future
module's responsibility, built on top of the ZoneMembership objects this
module produces.

Public API:
    ZoneManager          — loads zones once (reloadable), evaluates tracks per frame
    load_zones_config    — parses/validates config/zones.yaml into Zone objects
    ZoneError, ZoneConfigurationError, InvalidPolygonError, ZoneEvaluationError
"""

from .config import load_zones_config
from .exceptions import (
    InvalidPolygonError,
    ZoneConfigurationError,
    ZoneError,
    ZoneEvaluationError,
)
from .zone_manager import ZoneManager

__all__ = [
    "ZoneManager",
    "load_zones_config",
    "ZoneError",
    "ZoneConfigurationError",
    "InvalidPolygonError",
    "ZoneEvaluationError",
]
