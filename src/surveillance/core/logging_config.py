"""Loguru-based logging setup: console sink + rotating file sink under logs/."""

import sys
from pathlib import Path

from loguru import logger


def configure_logging(level: str = "INFO", log_dir: str = "logs") -> None:
    logger.remove()

    logger.add(sys.stderr, level=level, colorize=True)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "app.log",
        level=level,
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
    )
