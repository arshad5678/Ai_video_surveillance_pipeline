"""SurveillanceEvent domain model — the single unified event type every source normalizes into.

Framework-free: no IntrusionEvent/LoiteringEvent types leak through this
class, so Prompt 11 (Output Generation) and any future consumer can work
with SurveillanceEvent without knowing intrusion or loitering detection
exist, let alone whichever future module produces the next event type.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class EventType(str, Enum):
    """Unified event kind. Adding a future type here (plus a new EventSource
    and a small normalization adapter) is the only change a new upstream
    module needs — EventEngine's aggregate/filter/deduplicate logic is
    already generic over EventType and never needs to change."""

    INTRUSION_ENTER = "intrusion_enter"
    INTRUSION_EXIT = "intrusion_exit"
    LOITERING = "loitering"


class EventSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EventSource(str, Enum):
    """Which upstream module originated the event."""

    INTRUSION = "intrusion"
    LOITERING = "loitering"


@dataclass(frozen=True)
class SurveillanceEvent:
    """A single normalized surveillance event, ready for filtering/dispatch.

    `payload` preserves the fields specific to the original event type
    (e.g. the raw ENTER/EXIT value for intrusion, dwell/threshold seconds
    for loitering) as a read-only mapping — never a plain mutable dict —
    so a consumer can inspect source-specific detail without needing to
    know the source's type.
    """

    event_id: str
    event_type: EventType
    severity: EventSeverity
    source: EventSource
    track_id: int
    zone_id: str
    zone_name: str
    timestamp: float
    frame_index: int
    source_id: str
    payload: Mapping[str, Any]
