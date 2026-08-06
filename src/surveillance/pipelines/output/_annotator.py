"""Private helper: draws the overlay (boxes, zones, labels, timestamp) onto a frame.

Pure functions only — no state, no I/O. The returned image is a fresh
copy; the caller's original frame buffer is never mutated. This one
annotated image is reused for the video writer, the event snapshot, and
the event clip buffer, so the overlay only has to be computed once.
"""

from typing import Dict, List

import cv2
import numpy as np

from ...models.domain.surveillance_event import SurveillanceEvent
from ...models.domain.track import Track
from ...models.domain.zone import Zone

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TRACK_COLOR = (0, 200, 0)  # green, BGR
_ZONE_COLOR = (255, 128, 0)  # blue, BGR
_EVENT_COLOR = (0, 0, 255)  # red, BGR
_INFO_COLOR = (255, 255, 255)  # white, BGR


def annotate_frame(
    image: np.ndarray,
    tracks: List[Track],
    zones: List[Zone],
    events: List[SurveillanceEvent],
    frame_index: int,
    timestamp: float,
) -> np.ndarray:
    annotated = image.copy()
    _draw_zones(annotated, zones)
    _draw_tracks(annotated, tracks)
    _draw_events(annotated, tracks, events)
    _draw_frame_info(annotated, frame_index, timestamp)
    return annotated


def _draw_zones(image: np.ndarray, zones: List[Zone]) -> None:
    for zone in zones:
        if not zone.enabled or not zone.polygon:
            continue
        points = np.array([[point.x, point.y] for point in zone.polygon], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=_ZONE_COLOR, thickness=2)
        anchor = (int(zone.polygon[0].x), max(0, int(zone.polygon[0].y) - 8))
        cv2.putText(image, zone.zone_name, anchor, _FONT, 0.6, _ZONE_COLOR, 2, cv2.LINE_AA)


def _draw_tracks(image: np.ndarray, tracks: List[Track]) -> None:
    for track in tracks:
        box = track.bounding_box
        top_left = (int(box.x1), int(box.y1))
        bottom_right = (int(box.x2), int(box.y2))
        cv2.rectangle(image, top_left, bottom_right, _TRACK_COLOR, 2)
        cv2.putText(
            image, f"ID {track.track_id}", (top_left[0], max(0, top_left[1] - 8)), _FONT, 0.6, _TRACK_COLOR, 2, cv2.LINE_AA
        )


def _draw_events(image: np.ndarray, tracks: List[Track], events: List[SurveillanceEvent]) -> None:
    if not events:
        return
    tracks_by_id: Dict[int, Track] = {track.track_id: track for track in tracks}
    for event in events:
        track = tracks_by_id.get(event.track_id)
        if track is None:
            continue
        box = track.bounding_box
        top_left = (int(box.x1), int(box.y1))
        bottom_right = (int(box.x2), int(box.y2))
        cv2.rectangle(image, top_left, bottom_right, _EVENT_COLOR, 3)
        label = f"{event.event_type.value.upper()} | {event.zone_name}"
        cv2.putText(
            image, label, (top_left[0], max(0, top_left[1] - 28)), _FONT, 0.6, _EVENT_COLOR, 2, cv2.LINE_AA
        )


def _draw_frame_info(image: np.ndarray, frame_index: int, timestamp: float) -> None:
    text = f"Frame: {frame_index}  Time: {timestamp:.3f}s"
    cv2.putText(image, text, (10, 25), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (10, 25), _FONT, 0.6, _INFO_COLOR, 1, cv2.LINE_AA)
