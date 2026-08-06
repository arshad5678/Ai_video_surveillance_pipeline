"""System Router — GET /system."""

from fastapi import APIRouter, Depends

from ..schemas.system import SystemStatusResponse
from ..services.system_service import SystemService, get_system_service

router = APIRouter(prefix="/system", tags=["system"])


@router.get(
    "",
    response_model=SystemStatusResponse,
    summary="Get system status",
    description="Reports this API process's status: services initialized, memory/CPU usage, and counts derived from already-generated outputs.",
)
def get_system_status(service: SystemService = Depends(get_system_service)) -> SystemStatusResponse:
    return service.get_status()
