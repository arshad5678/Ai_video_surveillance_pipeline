"""API exception hierarchy + global exception handler registration."""

from .api_exceptions import ApiError, ConfigurationReloadError, ResourceNotFoundError
from .handlers import register_exception_handlers

__all__ = [
    "ApiError",
    "ResourceNotFoundError",
    "ConfigurationReloadError",
    "register_exception_handlers",
]
