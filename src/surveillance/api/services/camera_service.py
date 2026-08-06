"""Business logic behind the Camera Router.

Reuses VideoInput.probe_status() (added in Prompt 12 as an "integration
required" addition) rather than re-implementing device-opening logic
here — a fresh, dedicated VideoInput is probed per request so this never
interferes with any separately-running pipeline process holding its own
capture on the same source.
"""

from fastapi import Depends

from ...pipelines.video_input import VideoInput, VideoSourceConfig
from ..dependencies.providers import get_video_source_config
from ..schemas.camera import CameraStatusResponse


class CameraService:
    def __init__(self, video_source_config: VideoSourceConfig) -> None:
        self._video_source_config = video_source_config

    def get_status(self) -> CameraStatusResponse:
        status = VideoInput(self._video_source_config).probe_status()
        resolution = f"{status.width}x{status.height}" if status.width and status.height else None

        return CameraStatusResponse(
            source_type=status.source_type.value,
            source=str(status.uri),
            connected=status.connected,
            fps=status.fps,
            resolution=resolution,
        )


def get_camera_service(video_source_config: VideoSourceConfig = Depends(get_video_source_config)) -> CameraService:
    return CameraService(video_source_config)
