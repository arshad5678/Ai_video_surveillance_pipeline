"""The FastAPI backend's dependency-injection container.

Design note (see the Prompt 12 report for the full rationale): this API
is a thin read/status/reload layer over already-generated pipeline
outputs and configuration — it does not re-run detection or tracking.
Only the services routers actually need are constructed here:
ZoneManager (for zones.yaml + /config/reload), EventEngine (config
exposure), and OutputGenerator (file paths for the Output router). A
VideoSourceConfig is kept for the Camera router's on-demand probe.

Held on `app.state.container` (set once in app.py's lifespan handler) and
reached via `dependencies/providers.py` — never a module-level global, so
each `create_app()` call gets its own independent, testable instance.
"""

import time
from dataclasses import dataclass
from typing import Optional

from ...core.config_loader import load_yaml_config
from ...core.constants import DEFAULT_ZONES_CONFIG_PATH
from ...core.settings import Settings, get_settings
from ...pipelines.events import EventEngine, build_event_engine_config
from ...pipelines.output import OutputGenerator, build_output_config
from ...pipelines.video_input import VideoSourceConfig, build_video_source_config
from ...pipelines.zones import ZoneManager
from ..exceptions.api_exceptions import ConfigurationReloadError


@dataclass
class ServiceContainer:
    settings: Settings
    config_path: str
    zones_path: str
    yaml_config: dict
    zone_manager: ZoneManager
    event_engine: EventEngine
    output_generator: OutputGenerator
    video_source_config: VideoSourceConfig
    started_at: float


def build_container(settings: Optional[Settings] = None) -> ServiceContainer:
    settings = settings or get_settings()
    config_path = settings.config_path
    zones_path = DEFAULT_ZONES_CONFIG_PATH

    yaml_config = load_yaml_config(config_path)

    return ServiceContainer(
        settings=settings,
        config_path=config_path,
        zones_path=zones_path,
        yaml_config=yaml_config,
        zone_manager=ZoneManager(zones_path),
        event_engine=EventEngine(build_event_engine_config(settings, yaml_config)),
        output_generator=OutputGenerator(build_output_config(settings, yaml_config)),
        video_source_config=build_video_source_config(settings, yaml_config),
        started_at=time.monotonic(),
    )


def reload_config(container: ServiceContainer) -> None:
    """Reload config.yaml + zones.yaml in place, without restarting the app.

    Rebuilds EventEngine/OutputGenerator (their configs are immutable
    frozen dataclasses) and calls ZoneManager.reload() (which is already
    designed to re-read zones.yaml in place). Any failure leaves the
    container's previous, working state untouched — the reload either
    fully succeeds or fully raises, never partially applies.
    """
    try:
        new_yaml_config = load_yaml_config(container.config_path)
        new_event_engine = EventEngine(build_event_engine_config(container.settings, new_yaml_config))
        new_output_generator = OutputGenerator(build_output_config(container.settings, new_yaml_config))
        container.zone_manager.reload()
    except Exception as exc:
        raise ConfigurationReloadError(f"Failed to reload configuration: {exc}") from exc

    container.yaml_config = new_yaml_config
    container.event_engine = new_event_engine
    container.output_generator = new_output_generator
