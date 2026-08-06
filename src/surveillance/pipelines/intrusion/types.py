"""Types describing intrusion-detection configuration.

Scoped to this module only — future consumers (an Event Engine, etc.)
never see this; they only ever receive `IntrusionEvent` objects.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class IntrusionConfig:
    """Fully-resolved parameters controlling IntrusionDetector's behavior.

    Mirrors the config.yaml `intrusion:` block, plus one addition:
    `stale_state_ttl_seconds`, needed to satisfy "automatically clean up
    state for tracks that disappear permanently" — the given config block
    had no key for it.
    """

    enabled: bool = True
    monitor_zone_types: Tuple[str, ...] = ("intrusion",)
    emit_exit_events: bool = True
    verbose: bool = False
    stale_state_ttl_seconds: float = 300.0
