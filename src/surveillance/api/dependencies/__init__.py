"""Dependency injection: ServiceContainer + FastAPI Depends() providers, no globals."""

from .container import ServiceContainer, build_container, reload_config
from .providers import (
    get_container,
    get_event_engine,
    get_output_generator,
    get_video_source_config,
    get_zone_manager,
)

__all__ = [
    "ServiceContainer",
    "build_container",
    "reload_config",
    "get_container",
    "get_output_generator",
    "get_event_engine",
    "get_zone_manager",
    "get_video_source_config",
]
