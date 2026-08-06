"""FastAPI application factory."""

from fastapi import FastAPI

from ..core.constants import API_V1_PREFIX, PROJECT_NAME, VERSION
from ..core.logging_config import configure_logging
from ..core.settings import get_settings
from .health import router as health_router
from .v1.router import api_router as v1_router


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level)

    app = FastAPI(title=PROJECT_NAME, version=VERSION)

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {"service": PROJECT_NAME, "version": VERSION, "status": "running"}

    app.include_router(health_router)
    app.include_router(v1_router, prefix=API_V1_PREFIX)

    return app
