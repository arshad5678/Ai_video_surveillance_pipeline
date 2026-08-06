"""Business-endpoint routers, aggregated onto v1_router in api/v1/router.py."""

from .camera import router as camera_router
from .config import router as config_router
from .events import router as events_router
from .outputs import router as outputs_router
from .system import router as system_router

__all__ = [
    "config_router",
    "camera_router",
    "events_router",
    "outputs_router",
    "system_router",
]
