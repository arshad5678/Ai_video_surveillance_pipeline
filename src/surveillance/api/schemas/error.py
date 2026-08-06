"""Pydantic response model for every error the API returns."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Uniform error body returned by every global exception handler."""

    error: str = Field(..., description="Short machine-readable error category, e.g. 'ResourceNotFoundError'.")
    detail: str = Field(..., description="Human-readable explanation of what went wrong.")
    status_code: int = Field(..., description="The HTTP status code also set on the response.")
    context: Optional[Any] = Field(None, description="Optional extra structured detail (e.g. validation errors).")
