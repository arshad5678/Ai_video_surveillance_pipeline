#!/usr/bin/env python
"""Manual smoke-test for OutputGenerator, chained onto the full pipeline so far.

Usage:
    python scripts/test_output_generator.py                                   # use .env / config.yaml / config/zones.yaml
    python scripts/test_output_generator.py --source-type file --source data/input/sample.mp4
    python scripts/test_output_generator.py --demo                            # self-contained, low threshold for a quick demo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.surveillance.core.constants import DEFAULT_ZONES_CONFIG_PATH
from src.surveillance.core.logging_config import configure_logging
from src.surveillance.core.settings import get_settings
from src.surveillance.pipelines.detection import PersonDetector, build_detector_config
from src.surveillance.pipelines.events import EventEngine, build_event_engine_config
from src.surveillance.pipelines.frame_processing import FrameProcessor, build_preprocess_config
from src.surveillance.pipelines.intrusion import IntrusionDetector, build_intrusion_config
from src.surveillance.pipelines.loitering import LoiteringConfig, LoiteringDetector
from src.surveillance.pipelines.output import OutputGenerator, build_output_config
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
  - id: watch_zone
    name: Watch Zone
    type: intrusion
    enabled: true
    polygon:
      - [850, 200]
      - [1200, 200]
      - [1200, 500]
      - [850, 500]
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the full VideoInput -> ... -> OutputGenerator chain.")
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
        help="Build a self-contained looped clip from ultralytics' bundled zidane.jpg, with a low "
        "loitering threshold so both an intrusion and a loitering event fire quickly.",
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
    output_path = Path("data/input/_demo_output_generator.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    for _ in range(20):
        writer.write(image)
    writer.release()
    return str(output_path)


def build_demo_zones() -> str:
    output_path = Path("data/input/_demo_output_generator_zones.yaml")
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
    loitering_config = LoiteringConfig(threshold_seconds=0.05) if args.demo else None
    event_engine_config = build_event_engine_config(settings)
    output_config = build_output_config(settings)

    print(f"Video source: type={video_config.source_type.value}, uri={video_config.uri}")
    print(f"Zones config: {zones_path}")
    print(f"Output directory: {output_config.output_directory}")

    processor = FrameProcessor(preprocess_config)
    detector = PersonDetector(detector_config)
    tracker = MultiObjectTracker(tracking_config)
    zone_manager = ZoneManager(zones_path)
    intrusion_detector = IntrusionDetector(intrusion_config, list(zone_manager.zones))
    if loitering_config is None:
        from src.surveillance.pipelines.loitering import build_loitering_config

        loitering_config = build_loitering_config(settings)
    loitering_detector = LoiteringDetector(loitering_config, list(zone_manager.zones), intrusion_detector)
    event_engine = EventEngine(event_engine_config)

    frame_count = 0
    total_events = 0

    try:
        with VideoInput(video_config) as video_input, OutputGenerator(output_config) as output_generator:
            for frame in video_input.frames():
                frame_count += 1

                processed = processor.process(frame)
                if processed is None:
                    if frame_count >= args.max_frames:
                        break
                    continue

                detections = detector.detect(processed)
                tracks = tracker.update(detections)
                memberships = zone_manager.evaluate(tracks)
                intrusion_events = intrusion_detector.evaluate(memberships)  # must run first
                loitering_events = loitering_detector.evaluate(tracks, memberships)
                events = event_engine.evaluate(intrusion_events, loitering_events)

                output_generator.write_frame(processed, tracks, list(zone_manager.zones), events)
                total_events += len(events)

                for event in events:
                    print(
                        f"frame={event.frame_index:<4} [{event.severity.value:<6}] {event.event_type.value:<16} "
                        f"track={event.track_id} zone={event.zone_name} (event_id={event.event_id})"
                    )

                if frame_count >= args.max_frames:
                    break
    except VideoSourceError as exc:
        print(f"VideoInput failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Source frames: {frame_count}, events: {total_events}")
    print(f"Annotated video: {output_generator.latest_video()}")
    print(f"Latest snapshot: {output_generator.latest_snapshot()}")
    log_paths = output_generator.latest_event_log()
    print(f"Event logs: json={log_paths.json_path} csv={log_paths.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
