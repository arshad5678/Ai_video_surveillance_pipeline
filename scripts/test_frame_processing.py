#!/usr/bin/env python
"""Manual smoke-test for FrameProcessor, chained onto VideoInput.

Usage:
    python scripts/test_frame_processing.py                                 # use .env / config.yaml
    python scripts/test_frame_processing.py --source-type file --source data/input/sample.mp4
    python scripts/test_frame_processing.py --source-type webcam --source 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.surveillance.core.logging_config import configure_logging
from src.surveillance.core.settings import get_settings
from src.surveillance.pipelines.frame_processing import FrameProcessor, build_preprocess_config
from src.surveillance.pipelines.video_input import (
    VideoInput,
    VideoSourceConfig,
    VideoSourceError,
    VideoSourceType,
    build_video_source_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test VideoInput -> FrameProcessor.")
    parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in VideoSourceType],
        default=None,
        help="Override VIDEO_SOURCE_TYPE from .env",
    )
    parser.add_argument("--source", default=None, help="Override VIDEO_SOURCE from .env")
    parser.add_argument("--max-frames", type=int, default=100, help="Stop after this many source frames")
    return parser.parse_args()


def resolve_video_config(args: argparse.Namespace) -> VideoSourceConfig:
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
    print(f"Video source: type={video_config.source_type.value}, uri={video_config.uri}")
    print(
        "Preprocessing: resize={} color_conversion={} normalize={} frame_skip={}".format(
            preprocess_config.resize_enabled,
            preprocess_config.color_conversion_enabled,
            preprocess_config.normalize_enabled,
            preprocess_config.frame_skip_enabled,
        )
    )

    processor = FrameProcessor(preprocess_config)
    source_frame_count = 0
    processed_count = 0

    try:
        with VideoInput(video_config) as video_input:

            def limited_frames():
                nonlocal source_frame_count
                for frame in video_input.frames():
                    source_frame_count += 1
                    yield frame
                    if source_frame_count >= args.max_frames:
                        return

            for processed in processor.process_stream(limited_frames()):
                processed_count += 1
                if processed_count % 30 == 0:
                    print(
                        f"Processed {processed_count} frames "
                        f"(last index={processed.index}, shape={processed.image.shape}, dtype={processed.image.dtype})"
                    )
    except VideoSourceError as exc:
        print(f"VideoInput failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Source frames read: {source_frame_count}, processed frames: {processed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
