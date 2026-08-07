"""Types describing Output Generation configuration.

Scoped to this module only — future consumers (Prompt 12's FastAPI
backend, etc.) never see this; they only ever call OutputGenerator's
getters (latest_video/latest_snapshot/latest_event_log) for file paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OutputConfig:
    """Fully-resolved parameters controlling OutputGenerator's behavior.

    Mirrors the config.yaml `output:` block, plus two additions:

    - `frame_rate` — not in the prompt's literal example block, but needed
      to (a) open the annotated-video/clip writers at the correct fps and
      (b) convert clip_pre_seconds/clip_post_seconds into frame counts for
      the circular pre-event buffer. Mirrors the same kind of addition as
      tracking.frame_rate.
    - `clean_previous_outputs` — also not in the literal block. Defaults to
      True so every OutputGenerator constructed to actually run a pipeline
      (scripts, integration tests) starts from a clean output/ tree,
      instead of mixing this run's events with a previous run's leftover
      snapshots/clips/logs. Prompt 12's FastAPI backend is the one
      legitimate exception: its OutputGenerator is read-only (never calls
      write_frame()) and gets reconstructed on every config reload while
      pointed at the *same* directory a real pipeline run already
      populated, so api/dependencies/container.py explicitly passes False
      there via dataclasses.replace() — see that module's docstring.
    """

    annotated_video: bool = True
    snapshots: bool = True
    clips: bool = True
    json_log: bool = True
    csv_log: bool = True
    clip_pre_seconds: float = 5.0
    clip_post_seconds: float = 5.0
    output_directory: str = "output"
    video_codec: str = "mp4v"
    jpeg_quality: int = 95
    frame_rate: float = 30.0
    verbose: bool = False
    clean_previous_outputs: bool = True


@dataclass(frozen=True)
class EventLogPaths:
    """Return type of OutputGenerator.latest_event_log() — one path per enabled log format."""

    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None
