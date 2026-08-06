"""Pydantic response model for GET /health."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' if the process is up and serving requests.", examples=["healthy"])
    version: str = Field(..., description="Application version.", examples=["0.1.0"])
    uptime: float = Field(..., description="Seconds since the API process started.", examples=[123.45])
