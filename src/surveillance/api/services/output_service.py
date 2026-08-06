"""Business logic behind the Output Router — resolves OutputGenerator's getters to real file paths.

Only path resolution + existence checks live here; the actual file
streaming (FileResponse) is the router's job, since that's a FastAPI/
HTTP concern, not business logic.
"""

from pathlib import Path

from fastapi import Depends

from ...pipelines.output import OutputGenerator
from ..dependencies.providers import get_output_generator
from ..exceptions.api_exceptions import ResourceNotFoundError


class OutputService:
    def __init__(self, output_generator: OutputGenerator) -> None:
        self._output_generator = output_generator

    def get_latest_video(self) -> Path:
        path = self._output_generator.latest_video()
        if path is None or not path.exists():
            raise ResourceNotFoundError("No annotated video has been generated yet.")
        return path

    def get_latest_snapshot(self) -> Path:
        path = self._output_generator.latest_snapshot()
        if path is None or not path.exists():
            raise ResourceNotFoundError("No event snapshot has been generated yet.")
        return path

    def get_latest_json_log(self) -> Path:
        log_paths = self._output_generator.latest_event_log()
        if log_paths.json_path is None or not log_paths.json_path.exists():
            raise ResourceNotFoundError("No JSON event log has been generated yet.")
        return log_paths.json_path

    def get_latest_csv_log(self) -> Path:
        log_paths = self._output_generator.latest_event_log()
        if log_paths.csv_path is None or not log_paths.csv_path.exists():
            raise ResourceNotFoundError("No CSV event log has been generated yet.")
        return log_paths.csv_path


def get_output_service(output_generator: OutputGenerator = Depends(get_output_generator)) -> OutputService:
    return OutputService(output_generator)
