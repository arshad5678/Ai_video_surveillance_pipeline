"""Registers global exception handlers on the FastAPI app.

Every handler returns the same `ErrorResponse` shape, so API clients (the
future Prompt 13 dashboard included) only ever need to parse one error
format regardless of which failure mode they hit.
"""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from ..schemas.error import ErrorResponse
from .api_exceptions import ApiError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        logger.warning("{} on {} {}: {}", type(exc).__name__, request.method, request.url.path, exc.message)
        body = ErrorResponse(error=type(exc).__name__, detail=exc.message, status_code=exc.status_code)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Validation error on {} {}: {}", request.method, request.url.path, exc.errors())
        body = ErrorResponse(
            error="ValidationError",
            detail="Request validation failed.",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            context=exc.errors(),
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on {} {}", request.method, request.url.path)
        body = ErrorResponse(
            error=type(exc).__name__,
            detail="An unexpected internal error occurred.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body.model_dump())
