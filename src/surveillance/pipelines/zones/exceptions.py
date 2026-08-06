"""Exception hierarchy for the Zone Manager module."""


class ZoneError(Exception):
    """Base exception for all zone-management failures."""


class ZoneConfigurationError(ZoneError):
    """Raised when zones.yaml is missing, malformed, or structurally invalid."""


class InvalidPolygonError(ZoneConfigurationError):
    """Raised when a specific zone's polygon geometry is invalid (too few points, malformed
    coordinates, or self-intersecting)."""


class ZoneEvaluationError(ZoneError):
    """Raised when evaluating tracks against zones fails at runtime."""
