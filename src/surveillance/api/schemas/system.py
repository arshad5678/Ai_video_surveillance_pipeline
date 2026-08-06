"""Pydantic response model for GET /system."""

from typing import List

from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    pipeline_status: str = Field(..., description="'ready' once all API-layer services are initialized.")
    modules_initialized: List[str] = Field(
        ..., description="Services actually constructed by this API process (see report for rationale)."
    )
    memory_usage_mb: float = Field(..., description="Resident memory used by this API process, in MB.")
    cpu_usage_percent: float = Field(..., description="CPU usage of this API process since the last call.")
    frame_count: int = Field(..., description="Frame count of the latest annotated video, if one exists.")
    track_count: int = Field(..., description="Distinct track_ids observed across the JSON event log.")
    event_count: int = Field(..., description="Total events recorded in the JSON event log.")
