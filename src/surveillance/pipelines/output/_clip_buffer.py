"""Private helper: the configurable circular frame buffer behind event clips.

`_ClipRecorder` owns two things:

1. A fixed-size circular buffer (`collections.deque(maxlen=...)`) of the
   most recent annotated frames — the "N seconds before" half of a clip.
   Old frames fall off the left automatically as new ones are appended,
   so memory use never grows past `pre_frame_count` frames regardless of
   how long the stream runs.
2. Zero or more in-progress "active" clips — started the moment an event
   fires, each still waiting to collect its "N seconds after" half.

Every frame is fed through `observe_frame()` exactly once: it appends
that frame to every active clip (advancing their post-event capture) and
then to the circular buffer, in that order — so a clip started this same
frame (via `start_clip()`, called after `observe_frame()`) sees a
pre-buffer whose *last* element is the current frame, without double
counting it.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List

import numpy as np


@dataclass
class _ActiveClip:
    event_id: str
    path: Path
    frames: List[np.ndarray] = field(default_factory=list)
    post_frames_needed: int = 0
    post_frames_collected: int = 0

    @property
    def is_complete(self) -> bool:
        return self.post_frames_collected >= self.post_frames_needed


class _ClipRecorder:
    def __init__(self, pre_frame_count: int, post_frame_count: int) -> None:
        self._pre_buffer: Deque[np.ndarray] = deque(maxlen=max(pre_frame_count, 0))
        self._post_frame_count = max(post_frame_count, 0)
        self._active: List[_ActiveClip] = []

    def observe_frame(self, frame: np.ndarray) -> List[_ActiveClip]:
        """Advance all active clips with this frame; return the ones that just completed."""
        completed: List[_ActiveClip] = []
        still_active: List[_ActiveClip] = []
        for clip in self._active:
            clip.frames.append(frame)
            clip.post_frames_collected += 1
            if clip.is_complete:
                completed.append(clip)
            else:
                still_active.append(clip)
        self._active = still_active
        self._pre_buffer.append(frame)
        return completed

    def start_clip(self, event_id: str, path: Path) -> None:
        """Start a new clip. Must be called after this frame's observe_frame()."""
        frames = list(self._pre_buffer)
        self._active.append(
            _ActiveClip(event_id=event_id, path=path, frames=frames, post_frames_needed=self._post_frame_count)
        )

    def drain(self) -> List[_ActiveClip]:
        """Return + clear any still-in-progress clips, e.g. at shutdown (partial clip)."""
        remaining = self._active
        self._active = []
        return remaining
