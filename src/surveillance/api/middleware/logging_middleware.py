"""Request/response logging middleware.

Logs "Request received" / "Request completed" (with status + duration)
for every request. Exceptions are logged separately by the global
exception handlers (handlers.py), not here, so each failure is only
logged once.
"""

import time

from fastapi import FastAPI, Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info("Request received: {} {}", request.method, request.url.path)
        start = time.monotonic()

        response = await call_next(request)

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Request completed: {} {} -> {} ({:.1f}ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


def register_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestLoggingMiddleware)
