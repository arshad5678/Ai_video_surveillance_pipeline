"""OutputGenerator: turns Track + Zone + SurveillanceEvent streams into visual/structured artifacts.

Sole responsibility: visualization and export — annotated video,
per-event snapshots, per-event video clips, and JSON/CSV event logs.
No REST APIs, dashboards, notifications, or databases live here or are
imported by this module; Prompt 12 (FastAPI) exposes what this module
writes to disk, it does not call into this module's drawing logic.
"""

from pathlib import Path
from typing import List, Optional

import cv2
from loguru import logger

from ...models.domain.processed_frame import ProcessedFrame
from ...models.domain.surveillance_event import SurveillanceEvent
from ...models.domain.track import Track
from ...models.domain.zone import Zone
from ._annotator import annotate_frame
from ._clip_buffer import _ActiveClip, _ClipRecorder
from ._event_log_writer import _EventLogWriter
from ._video_writer import _VideoWriterHandle
from .exceptions import ClipGenerationError, OutputGenerationError, SnapshotError
from .types import EventLogPaths, OutputConfig


class OutputGenerator:
    """Consumes one frame's Tracks/Zones/SurveillanceEvents at a time and writes outputs to disk.

    Usage:
        generator = OutputGenerator(config)
        for frame, tracks, zones, events in pipeline:
            generator.write_frame(frame, tracks, zones, events)
        generator.release()

    Or as a context manager:
        with OutputGenerator(config) as generator:
            ...
    """

    def __init__(self, config: OutputConfig) -> None:
        self._config = config
        self._output_root = Path(config.output_directory)
        self._annotated_video_dir = self._output_root / "annotated_video"
        self._snapshots_dir = self._output_root / "snapshots"
        self._clips_dir = self._output_root / "clips"
        self._logs_dir = self._output_root / "logs"
        for directory in (self._annotated_video_dir, self._snapshots_dir, self._clips_dir, self._logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self._video_path = self._annotated_video_dir / "output.mp4"
        self._video_writer: Optional[_VideoWriterHandle] = (
            _VideoWriterHandle(self._video_path, config.video_codec, config.frame_rate)
            if config.annotated_video
            else None
        )

        pre_frame_count = max(round(config.frame_rate * config.clip_pre_seconds), 0)
        post_frame_count = max(round(config.frame_rate * config.clip_post_seconds), 0)
        self._clip_recorder = _ClipRecorder(pre_frame_count, post_frame_count)

        json_path = self._logs_dir / "events.json" if config.json_log else None
        csv_path = self._logs_dir / "events.csv" if config.csv_log else None
        self._log_writer = _EventLogWriter(json_path, csv_path)
        self._log_paths = EventLogPaths(json_path=json_path, csv_path=csv_path)

        self._event_counter = 0
        self._latest_snapshot_path: Optional[Path] = None

        logger.info(
            "OutputGenerator initialized: output_directory={}, annotated_video={}, snapshots={}, clips={}, "
            "json_log={}, csv_log={}, clip_pre_seconds={}, clip_post_seconds={}",
            self._output_root,
            config.annotated_video,
            config.snapshots,
            config.clips,
            config.json_log,
            config.csv_log,
            config.clip_pre_seconds,
            config.clip_post_seconds,
        )

    def write_frame(
        self,
        processed_frame: ProcessedFrame,
        tracks: List[Track],
        zones: List[Zone],
        events: List[SurveillanceEvent],
    ) -> None:
        """Annotate one frame and dispatch it to every enabled output artifact."""
        annotated = annotate_frame(
            processed_frame.image, tracks, zones, events, processed_frame.index, processed_frame.timestamp
        )

        if self._video_writer is not None:
            self._video_writer.write(annotated)
            logger.debug("Annotated frame written: frame_index={}", processed_frame.index)

        if self._config.clips:
            for completed_clip in self._clip_recorder.observe_frame(annotated):
                self._write_clip(completed_clip)
        else:
            self._clip_recorder.observe_frame(annotated)

        for event in events:
            self._event_counter += 1
            self._handle_event(annotated, event)

    def _handle_event(self, annotated_frame, event: SurveillanceEvent) -> None:
        sequence = self._event_counter

        if self._config.snapshots:
            self._save_snapshot(annotated_frame, sequence)

        if self._config.clips:
            clip_path = self._clips_dir / f"event_{sequence:03d}.mp4"
            self._clip_recorder.start_clip(event.event_id, clip_path)

        if self._config.json_log or self._config.csv_log:
            self._log_writer.append(event)

    def _save_snapshot(self, annotated_frame, sequence: int) -> None:
        path = self._snapshots_dir / f"event_{sequence:03d}.jpg"
        try:
            ok = cv2.imwrite(str(path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality])
            if not ok:
                raise SnapshotError(f"cv2.imwrite returned False for {path}")
        except Exception as exc:
            logger.error("Failed to save snapshot {}: {}", path, exc)
            raise SnapshotError(f"Failed to save snapshot {path}: {exc}") from exc
        self._latest_snapshot_path = path
        logger.info("Snapshot saved: {}", path)

    def _write_clip(self, clip: "_ActiveClip") -> None:
        if not clip.frames:
            return
        handle = _VideoWriterHandle(clip.path, self._config.video_codec, self._config.frame_rate)
        try:
            for frame in clip.frames:
                handle.write(frame)
        except Exception as exc:
            logger.error("Failed to write clip {}: {}", clip.path, exc)
            raise ClipGenerationError(f"Failed to write clip {clip.path}: {exc}") from exc
        finally:
            handle.release()
        logger.info("Clip saved: {} ({} frames)", clip.path, len(clip.frames))

    def release(self) -> None:
        """Flush and release every open writer, finalizing any in-progress clips."""
        if self._video_writer is not None:
            self._video_writer.release()

        for clip in self._clip_recorder.drain():
            try:
                self._write_clip(clip)
            except OutputGenerationError as exc:
                logger.error("Failed to finalize in-progress clip on release: {}", exc)

        logger.info("OutputGenerator resources released.")

    def latest_video(self) -> Optional[Path]:
        """Path to the annotated video, if annotated_video output is enabled and has been written."""
        if self._video_writer is None or not self._video_path.exists():
            return None
        return self._video_path

    def latest_snapshot(self) -> Optional[Path]:
        """Path to the most recently saved event snapshot, or None if none exists yet.

        Falls back to scanning the snapshots directory (sequentially
        numbered event_NNN.jpg, so a lexicographic sort is also the
        chronological order) when this instance hasn't itself saved one
        -- e.g. a fresh API process reading snapshots a separate pipeline
        run already wrote to disk, per "consume outputs already generated
        by previous modules."
        """
        if self._latest_snapshot_path is not None:
            return self._latest_snapshot_path
        candidates = sorted(self._snapshots_dir.glob("event_*.jpg"))
        return candidates[-1] if candidates else None

    def latest_event_log(self) -> EventLogPaths:
        """Paths to the JSON/CSV event logs (None for whichever format is disabled)."""
        return self._log_paths

    def __enter__(self) -> "OutputGenerator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
