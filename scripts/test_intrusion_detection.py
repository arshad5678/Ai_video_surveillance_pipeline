#!/usr/bin/env python
"""Manual smoke-test for IntrusionDetector, chained onto the full pipeline so far.

Usage:
    python scripts/test_intrusion_detection.py                                   # use .env / config.yaml / config/zones.yaml
    python scripts/test_intrusion_detection.py --source-type file --source data/input/sample.mp4
    python scripts/test_intrusion_detection.py --demo                            # self-contained, shows a real ENTER then EXIT
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.surveillance.core.constants import DEFAULT_ZONES_CONFIG_PATH
from src.surveillance.core.logging_config import configure_logging
from src.surveillance.core.settings import get_settings
from src.surveillance.pipelines.detection import PersonDetector, build_detector_config
from src.surveillance.pipelines.frame_processing import FrameProcessor, build_preprocess_config
from src.surveillance.pipelines.intrusion import IntrusionDetector, build_intrusion_config
from src.surveillance.pipelines.tracking import MultiObjectTracker, build_tracking_config
from src.surveillance.pipelines.video_input import (
    VideoInput,
    VideoSourceConfig,
    VideoSourceError,
    VideoSourceType,
    build_video_source_config,
)
from src.surveillance.pipelines.zones import ZoneManager

DEMO_ZONES_YAML = """
zones:
  - id: sweep_zone
    name: Sweep Zone
    type: intrusion
    enabled: true
    polygon:
      - [550, 350]
      - [700, 350]
      - [700, 550]
      - [550, 550]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the full VideoInput -> ... -> IntrusionDetector chain.")
    parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in VideoSourceType],
        default=None,
        help="Override VIDEO_SOURCE_TYPE from .env",
    )
    parser.add_argument("--source", default=None, help="Override VIDEO_SOURCE from .env")
    parser.add_argument("--max-frames", type=int, default=50, help="Stop after this many source frames")
    parser.add_argument("--zones", default=None, help="Path to zones.yaml (defaults to config/zones.yaml, or the demo zones under --demo)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build a self-contained clip that pans ultralytics' bundled zidane.jpg across a "
        "zone, producing a real ENTER then EXIT (no video file needed).",
    )
    return parser.parse_args()


def build_demo_video() -> str:
    import cv2
    import numpy as np
    import ultralytics

    demo_image_path = Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"
    image = cv2.imread(str(demo_image_path))
    if image is None:
        raise RuntimeError(f"Could not read bundled demo image: {demo_image_path}")

    height, width = image.shape[:2]
    frame_count = 24
    output_path = Path("data/input/_demo_intrusion.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
    for i in range(frame_count):
        tx = -150.0 + i * (300.0 / (frame_count - 1))  # sweeps -150 -> +150
        translation = np.float32([[1, 0, tx], [0, 1, 0]])
        shifted = cv2.warpAffine(image, translation, (width, height), borderMode=cv2.BORDER_REPLICATE)
        writer.write(shifted)
    writer.release()
    return str(output_path)


def build_demo_zones() -> str:
    output_path = Path("data/input/_demo_intrusion_zones.yaml")
    output_path.write_text(DEMO_ZONES_YAML)
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
    zones_path = args.zones or (build_demo_zones() if args.demo else DEFAULT_ZONES_CONFIG_PATH)
    preprocess_config = build_preprocess_config(settings)
    detector_config = build_detector_config(settings)
    tracking_config = build_tracking_config(settings)
    intrusion_config = build_intrusion_config(settings)

    print(f"Video source: type={video_config.source_type.value}, uri={video_config.uri}")
    print(f"Zones config: {zones_path}")

    processor = FrameProcessor(preprocess_config)
    detector = PersonDetector(detector_config)
    tracker = MultiObjectTracker(tracking_config)
    zone_manager = ZoneManager(zones_path)
    intrusion_detector = IntrusionDetector(intrusion_config, list(zone_manager.zones))

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
            tracks_stream = tracker.track_stream(detections_stream)
            memberships_stream = zone_manager.evaluate_stream(tracks_stream)
            for events in intrusion_detector.evaluate_stream(memberships_stream):
                for event in events:
                    print(
                        f"frame={event.frame_index:<4} {event.event_type.value:<5} "
                        f"track={event.track_id} zone={event.zone_name} (event_id={event.event_id})"
                    )
    except VideoSourceError as exc:
        print(f"VideoInput failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Source frames: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
