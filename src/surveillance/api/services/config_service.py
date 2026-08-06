"""Business logic behind the Configuration Router."""

from datetime import datetime, timezone

from fastapi import Depends
from loguru import logger

from ..dependencies.container import ServiceContainer, reload_config
from ..dependencies.providers import get_container
from ..schemas.config import ConfigReloadResponse, ConfigResponse


class ConfigService:
    def __init__(self, container: ServiceContainer) -> None:
        self._container = container

    def get_config(self) -> ConfigResponse:
        return ConfigResponse(
            config_path=self._container.config_path,
            zones_path=self._container.zones_path,
            config=self._container.yaml_config,
            zone_count=len(self._container.zone_manager.zones),
        )

    def reload(self) -> ConfigReloadResponse:
        reload_config(self._container)  # raises ConfigurationReloadError on failure; state is unchanged if so

        logger.info(
            "Configuration reloaded: config_path={}, zones_path={}, zone_count={}",
            self._container.config_path,
            self._container.zones_path,
            len(self._container.zone_manager.zones),
        )
        return ConfigReloadResponse(
            status="reloaded",
            config_path=self._container.config_path,
            zones_path=self._container.zones_path,
            zone_count=len(self._container.zone_manager.zones),
            reloaded_at=datetime.now(timezone.utc),
        )


def get_config_service(container: ServiceContainer = Depends(get_container)) -> ConfigService:
    return ConfigService(container)
