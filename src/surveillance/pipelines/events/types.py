"""Types describing Event Engine configuration.

Scoped to this module only — downstream consumers (Output Generation,
etc.) never see this; they only ever receive `SurveillanceEvent` objects.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...models.domain.surveillance_event import EventSeverity


@dataclass(frozen=True)
class EventEngineConfig:
    """Fully-resolved parameters controlling EventEngine's behavior.

    Empty tuples for enabled_event_types/zone_filter/track_filter mean
    "no restriction" (permissive default), not "nothing allowed".
    """

    enabled: bool = True
    enabled_event_types: Tuple[str, ...] = ()
    minimum_severity: EventSeverity = EventSeverity.LOW
    zone_filter: Tuple[str, ...] = ()
    track_filter: Tuple[int, ...] = ()
    severity_mapping: Dict[str, EventSeverity] = field(default_factory=dict)
    verbose: bool = False
