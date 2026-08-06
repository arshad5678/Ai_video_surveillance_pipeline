"""Events Router — GET /events, GET /events/{event_id}."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..schemas.error import ErrorResponse
from ..schemas.events import EventListResponse, EventResponse
from ..services.event_service import EventService, get_event_service

router = APIRouter(prefix="/events", tags=["events"])


@router.get(
    "",
    response_model=EventListResponse,
    summary="List latest events",
    description="Returns the most recent SurveillanceEvents from the JSON event log, newest first, with optional filters.",
)
def list_events(
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of events to return."),
    severity: Optional[str] = Query(None, description="Filter by severity, e.g. HIGH."),
    event_type: Optional[str] = Query(None, description="Filter by event type, e.g. intrusion_enter."),
    zone_id: Optional[str] = Query(None, description="Filter by zone id."),
    track_id: Optional[int] = Query(None, description="Filter by track id."),
    service: EventService = Depends(get_event_service),
) -> EventListResponse:
    return service.list_events(
        limit=limit, severity=severity, event_type=event_type, zone_id=zone_id, track_id=track_id
    )


@router.get(
    "/{event_id}",
    response_model=EventResponse,
    summary="Get one event",
    description="Returns a single event by id.",
    responses={404: {"model": ErrorResponse, "description": "No event with this id exists in the log."}},
)
def get_event(event_id: str, service: EventService = Depends(get_event_service)) -> EventResponse:
    return service.get_event(event_id)
