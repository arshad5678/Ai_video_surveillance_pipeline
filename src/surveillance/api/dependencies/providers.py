"""FastAPI `Depends()` providers.

Every downstream provider declares `container: ServiceContainer =
Depends(get_container)` rather than calling get_container() directly, so
FastAPI's own dependency-resolution/override mechanism (not a plain
Python call) is what wires them together. That matters for testability:
`app.dependency_overrides[get_container] = lambda: fake_container`
propagates through every provider below it automatically; a direct call
would silently bypass the override.

get_container() itself reads from `request.app.state.container` — set
once per app instance in app.py's lifespan handler — never a
module-level global.
"""

from fastapi import Depends, Request

from ...pipelines.events import EventEngine
from ...pipelines.output import OutputGenerator
from ...pipelines.video_input import VideoSourceConfig
from ...pipelines.zones import ZoneManager
from .container import ServiceContainer


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def get_output_generator(container: ServiceContainer = Depends(get_container)) -> OutputGenerator:
    return container.output_generator


def get_event_engine(container: ServiceContainer = Depends(get_container)) -> EventEngine:
    return container.event_engine


def get_zone_manager(container: ServiceContainer = Depends(get_container)) -> ZoneManager:
    return container.zone_manager


def get_video_source_config(container: ServiceContainer = Depends(get_container)) -> VideoSourceConfig:
    return container.video_source_config
