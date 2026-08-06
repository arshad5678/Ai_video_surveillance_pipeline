#!/usr/bin/env python
"""Manual smoke-test for MultiObjectTracker, chained onto the full pipeline so far.

Usage:
    python scripts/test_tracking.py                                   # use .env / config.yaml
    python scripts/test_tracking.py --source-type file --source data/input/sample.mp4
    python scripts/test_tracking.py --demo                            # self-contained, no video file needed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.surveillance.core.logging_config import configure_logging
from src.surveillance.core.settings import get_settings
from src.surveillance.pipelines.detection import PersonDetector, build_detector_config
from src.surveillance.pipelines.frame_processing import FrameProcessor, build_preprocess_config
from src.surveillance.pipelines.tracking import MultiObjectTracker, build_tracking_config
from src.surveillance.pipelines.video_input import (
    VideoInput,
    VideoSourceConfig,
    VideoSourceError,
    VideoSourceType,
    build_video_source_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the full VideoInput -> ... -> MultiObjectTracker chain.")
    parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in VideoSourceType],
        default=None,
        help="Override VIDEO_SOURCE_TYPE from .env",
    )
    parser.add_argument("--source", default=None, help="Override VIDEO_SOURCE from .env")
    parser.add_argument("--max-frames", type=int, default=50, help="Stop after this many source frames")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build a self-contained demo clip from ultralytics' bundled zidane.jpg (no video file needed).",
    )
    return parser.parse_args()


def build_demo_video() -> str:
    import cv2
    import ultralytics

    demo_image_path = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
    image = cv2.imread(str(demo_image_path))
    if image is None:
        raise RuntimeError(f"Could not read bundled demo image: {demo_image_path}")

    height, width = image.shape[:2]
    output_path = Path("data/input/_demo_tracking.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for _ in range(15):
        writer.write(image)
    writer.release()
    return str(output_path)


def resolve_video_config(args: argparse.Namespace) -> VideoSourceConfig:
    if args.demo:
        return VideoSourceConfig(source_type=VideoSourceType.FILE, uri=build_demo_video())
    if args.source_type is not None and args.source is not None:
        source_type = VideoSourceType(args.source_type)
        uri = int(args.source) if source_type is VideoSourceType.WEBCAM else args.source
        return VideoSourceConfig(source_type=source_type, uri=uri)
    return build_video_source_config()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level)

    video_config = resolve_video_config(args)
    preprocess_config = build_preprocess_config(settings)
    detector_config = build_detector_config(settings)
    tracking_config = build_tracking_config(settings)

    print(f"Video source: type={video_config.source_type.value}, uri={video_config.uri}")
    print(f"Tracker: type={tracking_config.tracker_type}, buffer={tracking_config.track_buffer} frames")

    processor = FrameProcessor(preprocess_config)
    detector = PersonDetector(detector_config)
    tracker = MultiObjectTracker(tracking_config)

    frame_count = 0

    try:
        with VideoInput(video_config) as video_input:

            def limited_frames():
                nonlocal frame_count
                for frame in video_input.frames():
                    frame_count += 1
                    yield frame
                    if frame_count >= args.max_frames:
                        return

            processed_frames = processor.process_stream(limited_frames())
            detections_stream = detector.detect_stream(processed_frames)
            for tracks in tracker.track_stream(detections_stream):
                for track in tracks:
                    print(
                        f"frame={track.frame_index} id={track.track_id} confirmed={track.is_confirmed} "
                        f"hits={track.hits} time_since_update={track.time_since_update} "
                        f"center={track.history[-1] if track.history else None}"
                    )
    except VideoSourceError as exc:
        print(f"VideoInput failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Source frames: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
