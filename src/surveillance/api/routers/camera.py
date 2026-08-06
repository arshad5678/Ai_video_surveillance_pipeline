"""Camera Router — GET /camera/status."""

from fastapi import APIRouter, Depends

from ..schemas.camera import CameraStatusResponse
from ..services.camera_service import CameraService, get_camera_service

router = APIRouter(prefix="/camera", tags=["camera"])


@router.get(
    "/status",
    response_model=CameraStatusResponse,
    summary="Get camera status",
    description="Performs a quick connectivity probe against the configured video source and reports source, fps, resolution, and connectivity.",
)
def get_camera_status(service: CameraService = Depends(get_camera_service)) -> CameraStatusResponse:
    return service.get_status()
