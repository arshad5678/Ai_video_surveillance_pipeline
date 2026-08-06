"""Event Engine module — aggregation, ordering, filtering, and dedup only.

Does not implement storage, notifications, REST APIs, dashboards, or
alert delivery — those are future modules built on top of the
SurveillanceEvent objects this module produces.

Public API:
    EventEngine              — normalizes, aggregates, filters, deduplicates events
    build_event_engine_config — resolves EventEngineConfig from config.yaml
    EventEngineConfig
    EventEngineError, EventConfigurationError, EventAggregationError
"""

from .config import build_event_engine_config
from .event_engine import EventEngine
from .exceptions import EventAggregationError, EventConfigurationError, EventEngineError
from .types import EventEngineConfig

__all__ = [
    "EventEngine",
    "build_event_engine_config",
    "EventEngineConfig",
    "EventEngineError",
    "EventConfigurationError",
    "EventAggregationError",
]
