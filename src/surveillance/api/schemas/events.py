"""Pydantic response models for GET /events and GET /events/{event_id}.

Deliberately a separate type from `SurveillanceEvent` (the domain model) —
the API never returns the internal dataclass directly, so the domain
model's fields can evolve without silently changing the public API
contract, and vice versa.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class EventResponse(BaseModel):
    event_id: str
    event_type: str = Field(..., examples=["intrusion_enter"])
    severity: str = Field(..., examples=["HIGH"])
    track_id: int
    zone_id: str
    timestamp: float
    frame_index: int
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventListResponse(BaseModel):
    events: List[EventResponse]
    count: int = Field(..., description="Number of events in this response (post-filter, post-limit).")
