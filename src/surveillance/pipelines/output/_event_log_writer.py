"""Private helper: incremental JSON + CSV event log writing.

JSON has no natural append mode, so the writer keeps the full record
list in memory (one small dict per event — negligible compared to
frame data) and rewrites the file each time an event occurs, not each
frame; CSV is append-friendly and gets one row written directly, with
the header written once on first use.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ...models.domain.surveillance_event import SurveillanceEvent
from .exceptions import LogExportError

_CSV_FIELDS = ["event_id", "event_type", "severity", "track_id", "zone_id", "timestamp", "frame_index"]


class _EventLogWriter:
    def __init__(self, json_path: Optional[Path], csv_path: Optional[Path]) -> None:
        self._json_path = json_path
        self._csv_path = csv_path
        self._records: List[Dict[str, Any]] = []
        self._csv_header_written = False

    def append(self, event: SurveillanceEvent) -> None:
        record = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "track_id": event.track_id,
            "zone_id": event.zone_id,
            "timestamp": event.timestamp,
            "frame_index": event.frame_index,
            "payload": dict(event.payload),
        }

        if self._json_path is not None:
            self._write_json(record)
        if self._csv_path is not None:
            self._write_csv(record)

    def _write_json(self, record: Dict[str, Any]) -> None:
        self._records.append(record)
        try:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._json_path, "w", encoding="utf-8") as handle:
                json.dump(self._records, handle, indent=2)
        except OSError as exc:
            raise LogExportError(f"Failed to write JSON event log {self._json_path}: {exc}") from exc
        logger.debug("JSON updated: {} ({} records)", self._json_path, len(self._records))

    def _write_csv(self, record: Dict[str, Any]) -> None:
        try:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self._csv_header_written and not self._csv_path.exists()
            with open(self._csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({field: record[field] for field in _CSV_FIELDS})
            self._csv_header_written = True
        except OSError as exc:
            raise LogExportError(f"Failed to write CSV event log {self._csv_path}: {exc}") from exc
        logger.debug("CSV updated: {}", self._csv_path)
