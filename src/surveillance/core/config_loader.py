"""Loads non-secret YAML configuration (config/config.yaml) as a plain dict."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_yaml_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with config_path.open("r") as f:
        return yaml.safe_load(f) or {}
