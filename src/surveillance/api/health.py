"""Infrastructure-level health check — unversioned, for load balancers/orchestrators."""

import time

from fastapi import APIRouter, Depends

from ..core.constants import VERSION
from .dependencies.container import ServiceContainer
from .dependencies.providers import get_container
from .schemas.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    tags=["health"],
    response_model=HealthResponse,
    summary="Health check",
    description="Liveness/readiness probe: process status, version, and uptime in seconds.",
)
def health_check(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    return HealthResponse(status="healthy", version=VERSION, uptime=time.monotonic() - container.started_at)
