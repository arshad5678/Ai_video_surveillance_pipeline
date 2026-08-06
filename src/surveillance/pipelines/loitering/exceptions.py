"""Exception hierarchy for the Loitering Detection module."""


class LoiteringError(Exception):
    """Base exception for all loitering detection failures."""


class LoiteringConfigurationError(LoiteringError):
    """Raised when loitering detection configuration values are invalid."""


class LoiteringEvaluationError(LoiteringError):
    """Raised when evaluating tracks/memberships for loitering fails."""
