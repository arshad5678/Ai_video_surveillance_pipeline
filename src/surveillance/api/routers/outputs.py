"""Output Router — GET /outputs/latest/{video,snapshot,json,csv}.

Every endpoint streams the file straight off disk via FileResponse
(chunked by Starlette, never loaded fully into memory) rather than
reading it into a Python object first — the point of this router is to
expose exactly what OutputGenerator already wrote, unchanged.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ..schemas.error import ErrorResponse
from ..services.output_service import OutputService, get_output_service

router = APIRouter(prefix="/outputs", tags=["outputs"])

_NOT_GENERATED_YET = {404: {"model": ErrorResponse, "description": "This output hasn't been generated yet."}}


@router.get(
    "/latest/video",
    summary="Download latest annotated video",
    description="Streams the most recent output/annotated_video/output.mp4.",
    responses=_NOT_GENERATED_YET,
)
def get_latest_video(service: OutputService = Depends(get_output_service)) -> FileResponse:
    path = service.get_latest_video()
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get(
    "/latest/snapshot",
    summary="Download latest event snapshot",
    description="Streams the most recently saved event snapshot JPEG.",
    responses=_NOT_GENERATED_YET,
)
def get_latest_snapshot(service: OutputService = Depends(get_output_service)) -> FileResponse:
    path = service.get_latest_snapshot()
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@router.get(
    "/latest/json",
    summary="Download JSON event log",
    description="Streams output/logs/events.json.",
    responses=_NOT_GENERATED_YET,
)
def get_latest_json(service: OutputService = Depends(get_output_service)) -> FileResponse:
    path = service.get_latest_json_log()
    return FileResponse(path, media_type="application/json", filename=path.name)


@router.get(
    "/latest/csv",
    summary="Download CSV event log",
    description="Streams output/logs/events.csv.",
    responses=_NOT_GENERATED_YET,
)
def get_latest_csv(service: OutputService = Depends(get_output_service)) -> FileResponse:
    path = service.get_latest_csv_log()
    return FileResponse(path, media_type="text/csv", filename=path.name)
