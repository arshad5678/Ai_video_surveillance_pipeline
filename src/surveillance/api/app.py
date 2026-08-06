"""FastAPI application factory.

Wires together the DI container (built once at startup, held on
app.state — never a module-level global), middleware, global exception
handlers, and every router. See the Prompt 12 report for the full
architecture writeup.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from loguru import logger

from ..core.constants import API_V1_PREFIX, PROJECT_NAME, VERSION
from ..core.logging_config import configure_logging
from ..core.settings import get_settings
from .dependencies.container import build_container
from .exceptions.handlers import register_exception_handlers
from .health import router as health_router
from .middleware.logging_middleware import register_middleware
from .v1.router import api_router as v1_router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.container = build_container()
    logger.info("API started: {} v{}", PROJECT_NAME, VERSION)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level)

    app = FastAPI(
        title=PROJECT_NAME,
        version=VERSION,
        description="REST API exposing the AI Video Surveillance Pipeline's configuration, camera status, "
        "events, generated outputs, and system status.",
        lifespan=_lifespan,
    )

    register_middleware(app)
    register_exception_handlers(app)

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {"service": PROJECT_NAME, "version": VERSION, "status": "running"}

    app.include_router(health_router)
    app.include_router(v1_router, prefix=API_V1_PREFIX)

    return app
