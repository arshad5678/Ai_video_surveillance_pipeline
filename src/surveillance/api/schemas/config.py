"""Pydantic response models for GET /config and POST /config/reload."""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class ConfigResponse(BaseModel):
    """The fully-resolved non-secret pipeline configuration."""

    config_path: str = Field(..., description="Path config.yaml was loaded from.")
    zones_path: str = Field(..., description="Path zones.yaml was loaded from.")
    config: Dict[str, Any] = Field(..., description="The parsed contents of config.yaml.")
    zone_count: int = Field(..., description="Number of zones currently loaded from zones.yaml.")


class ConfigReloadResponse(BaseModel):
    status: str = Field(..., description="'reloaded' on success.", examples=["reloaded"])
    config_path: str
    zones_path: str
    zone_count: int
    reloaded_at: datetime
