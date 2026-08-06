"""Environment-driven application settings (see .env.example for all keys)."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import DEFAULT_CONFIG_PATH


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    device: str = "cpu"

    video_source_type: str = "webcam"  # webcam | file | rtsp
    video_source: str = "0"            # webcam index | file path | rtsp:// URL

    input_dir: str = "data/input"
    output_dir: str = "data/output"
    processed_dir: str = "data/processed"
    weights_dir: str = "weights"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: Optional[str] = None

    config_path: str = DEFAULT_CONFIG_PATH


@lru_cache
def get_settings() -> Settings:
    return Settings()
