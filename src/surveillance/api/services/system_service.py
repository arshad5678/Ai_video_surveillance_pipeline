"""Business logic behind the System Router.

frame_count/track_count/event_count are read from already-generated
outputs (the annotated video's own frame count, distinct track_ids seen
in the JSON event log) rather than from a live pipeline run — this API
process never runs detection/tracking itself. See the Prompt 12 report
for the full rationale.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import psutil
from fastapi import Depends

from ..dependencies.container import ServiceContainer
from ..dependencies.providers import get_container
from ..schemas.system import SystemStatusResponse
from .event_service import EventService, get_event_service

_MODULES_INITIALIZED: List[str] = ["settings", "zone_manager", "event_engine", "output_generator", "video_source_config"]


class SystemService:
    def __init__(self, container: ServiceContainer, event_service: EventService) -> None:
        self._container = container
        self._event_service = event_service
        self._process = psutil.Process()

    def get_status(self) -> SystemStatusResponse:
        memory_usage_mb = self._process.memory_info().rss / (1024 * 1024)
        cpu_usage_percent = self._process.cpu_percent(interval=None)

        records = self._event_service.read_records()
        track_count = len({record["track_id"] for record in records})

        return SystemStatusResponse(
            pipeline_status="ready",
            modules_initialized=list(_MODULES_INITIALIZED),
            memory_usage_mb=round(memory_usage_mb, 2),
            cpu_usage_percent=cpu_usage_percent,
            frame_count=self._latest_video_frame_count(),
            track_count=track_count,
            event_count=len(records),
        )

    def _latest_video_frame_count(self) -> int:
        video_path: Optional[Path] = self._container.output_generator.latest_video()
        if video_path is None or not video_path.exists():
            return 0

        capture = cv2.VideoCapture(str(video_path))
        try:
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()


def get_system_service(
    container: ServiceContainer = Depends(get_container),
    event_service: EventService = Depends(get_event_service),
) -> SystemService:
    return SystemService(container, event_service)
