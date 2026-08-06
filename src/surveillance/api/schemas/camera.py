"""Pydantic response model for GET /camera/status."""

from typing import Optional

from pydantic import BaseModel, Field


class CameraStatusResponse(BaseModel):
    source_type: str = Field(..., description="webcam | file | rtsp", examples=["webcam"])
    source: str = Field(..., description="The configured device index / file path / RTSP URL.")
    connected: bool = Field(..., description="Whether the source could be opened just now.")
    fps: Optional[float] = Field(None, description="Reported frames-per-second, if connected.")
    resolution: Optional[str] = Field(None, description="'WIDTHxHEIGHT', if connected.", examples=["1280x720"])
