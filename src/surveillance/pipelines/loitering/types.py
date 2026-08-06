"""Types describing loitering-detection configuration.

Scoped to this module only — future consumers (an Event Engine, etc.)
never see this; they only ever receive `LoiteringEvent` objects.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LoiteringConfig:
    """Fully-resolved parameters controlling LoiteringDetector's behavior.

    Mirrors the config.yaml `loitering:` block, plus one addition:
    `stale_state_ttl_seconds`, mirroring IntrusionConfig's same field —
    needed so state for a track that disappears while still inside a
    zone (and therefore never sends an explicit inside=False membership
    to trigger a reset) doesn't linger forever, per the "store minimal
    state" performance requirement.
    """

    enabled: bool = True
    threshold_seconds: float = 10.0
    monitor_zone_types: Tuple[str, ...] = ("intrusion",)
    verbose: bool = False
    stale_state_ttl_seconds: float = 300.0
