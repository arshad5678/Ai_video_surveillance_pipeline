#!/usr/bin/env python
"""Manual smoke-test for the VideoInput module — reads frames and reports progress.

Usage:
    python scripts/test_video_input.py                                   # use .env / config.yaml
    python scripts/test_video_input.py --source-type webcam --source 0
    python scripts/test_video_input.py --source-type file --source data/input/sample.mp4
    python scripts/test_video_input.py --source-type rtsp --source rtsp://user:pass@host:554/stream
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.surveillance.core.logging_config import configure_logging
from src.surveillance.core.settings import get_settings
from src.surveillance.pipelines.video_input import (
    VideoInput,
    VideoSourceConfig,
    VideoSourceError,
    VideoSourceType,
    build_video_source_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the VideoInput module.")
    parser.add_argument(
        "--source-type",
        choices=[source_type.value for source_type in VideoSourceType],
        default=None,
        help="Override VIDEO_SOURCE_TYPE from .env",
    )
    parser.add_argument("--source", default=None, help="Override VIDEO_SOURCE from .env")
    parser.add_argument("--max-frames", type=int, default=100, help="Stop after this many frames")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> VideoSourceConfig:
    if args.source_type is not None and args.source is not None:
        source_type = VideoSourceType(args.source_type)
        uri = int(args.source) if source_type is VideoSourceType.WEBCAM else args.source
        return VideoSourceConfig(source_type=source_type, uri=uri)
    return build_video_source_config()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(level=settings.log_level)

    config = resolve_config(args)
    print(f"Using source: type={config.source_type.value}, uri={config.uri}")

    frame_count = 0
    try:
        with VideoInput(config) as video_input:
            for frame in video_input.frames():
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Read {frame_count} frames (last index={frame.index}, shape={frame.image.shape})")
                if frame_count >= args.max_frames:
                    print(f"Reached --max-frames={args.max_frames}, stopping.")
                    break
    except VideoSourceError as exc:
        print(f"VideoInput failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Total frames read: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
