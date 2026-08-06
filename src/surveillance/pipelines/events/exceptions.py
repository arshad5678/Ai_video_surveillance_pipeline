"""Exception hierarchy for the Event Engine module."""


class EventEngineError(Exception):
    """Base exception for all Event Engine failures."""


class EventConfigurationError(EventEngineError):
    """Raised when Event Engine configuration values are invalid."""


class EventAggregationError(EventEngineError):
    """Raised when aggregating/filtering/deduplicating events fails."""
