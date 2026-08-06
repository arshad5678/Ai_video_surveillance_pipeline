"""Infrastructure-level health check — unversioned, for load balancers/orchestrators."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health_check() -> dict:
    return {"status": "ok"}
