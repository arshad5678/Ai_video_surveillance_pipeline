"""Configuration Router — GET /config, POST /config/reload."""

from fastapi import APIRouter, Depends

from ..schemas.config import ConfigReloadResponse, ConfigResponse
from ..schemas.error import ErrorResponse
from ..services.config_service import ConfigService, get_config_service

router = APIRouter(prefix="/config", tags=["configuration"])


@router.get(
    "",
    response_model=ConfigResponse,
    summary="Get current configuration",
    description="Returns the fully-resolved config.yaml contents currently loaded in memory, plus the zone count from zones.yaml.",
)
def get_config(service: ConfigService = Depends(get_config_service)) -> ConfigResponse:
    return service.get_config()


@router.post(
    "/reload",
    response_model=ConfigReloadResponse,
    summary="Reload configuration",
    description="Reloads config.yaml and zones.yaml from disk and rebuilds the affected services, without restarting the API process.",
    responses={400: {"model": ErrorResponse, "description": "Reload failed; previous configuration is left untouched."}},
)
def reload_configuration(service: ConfigService = Depends(get_config_service)) -> ConfigReloadResponse:
    return service.reload()
