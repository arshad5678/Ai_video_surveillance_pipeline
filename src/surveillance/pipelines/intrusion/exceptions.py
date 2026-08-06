"""Exception hierarchy for the Intrusion Detection module."""


class IntrusionError(Exception):
    """Base exception for all intrusion detection failures."""


class IntrusionConfigurationError(IntrusionError):
    """Raised when intrusion detection configuration values are invalid."""


class IntrusionEvaluationError(IntrusionError):
    """Raised when evaluating zone memberships for intrusion events fails."""
