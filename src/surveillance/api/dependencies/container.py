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

Important: OutputGenerator defaults to wiping snapshots/clips/logs from a
previous run on construction (`clean_previous_outputs=True`), since a
*pipeline* run constructing one really is starting fresh. This API is not
a pipeline run — its OutputGenerator is read-only and gets rebuilt on
every startup and every /config/reload while pointed at the same
directory a real pipeline run already populated, so both places below
explicitly build it with `clean_previous_outputs=False` via
`_read_only_output_config()`. Forgetting this would mean restarting the
API, or simply reloading its config, silently deletes real pipeline
output.
"""

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from ...core.config_loader import load_yaml_config
from ...core.constants import DEFAULT_ZONES_CONFIG_PATH
from ...core.settings import Settings, get_settings
from ...pipelines.events import EventEngine, build_event_engine_config
from ...pipelines.output import OutputConfig, OutputGenerator, build_output_config
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


def _read_only_output_config(settings: Settings, yaml_config: Dict[str, Any]) -> OutputConfig:
    """Same OutputConfig a pipeline run would get, except never wipes existing output/ files.

    This API's OutputGenerator only ever reads paths back (latest_video/
    latest_snapshot/latest_event_log) — it never calls write_frame() — so
    it must never run the "start of a new pipeline execution" cleanup a
    real run wants by default.
    """
    return replace(build_output_config(settings, yaml_config), clean_previous_outputs=False)


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
        output_generator=OutputGenerator(_read_only_output_config(settings, yaml_config)),
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
        new_output_generator = OutputGenerator(_read_only_output_config(container.settings, new_yaml_config))
        container.zone_manager.reload()
    except Exception as exc:
        raise ConfigurationReloadError(f"Failed to reload configuration: {exc}") from exc

    container.yaml_config = new_yaml_config
    container.event_engine = new_event_engine
    container.output_generator = new_output_generator
