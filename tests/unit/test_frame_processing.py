"""Unit tests for FrameProcessor — pure numpy/OpenCV operations, no I/O or mocks needed."""

import numpy as np
import pytest

from src.surveillance.models.domain.frame import Frame
from src.surveillance.pipelines.frame_processing import (
    ColorMode,
    FrameProcessor,
    InvalidPreprocessConfigError,
    PreprocessConfig,
)


def make_frame(index: int = 0, width: int = 100, height: int = 80) -> Frame:
    # Distinct per-channel values so BGR<->RGB reordering is verifiable.
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R
    return Frame(index=index, image=image, timestamp=1234.5, source_id="test-source")


def test_default_config_is_a_passthrough() -> None:
    frame = make_frame()
    processor = FrameProcessor(PreprocessConfig())

    processed = processor.process(frame)

    assert processed is not None
    assert processed.image.shape == frame.image.shape
    assert np.array_equal(processed.image, frame.image)


def test_resize_changes_shape() -> None:
    frame = make_frame(width=100, height=80)
    config = PreprocessConfig(resize_enabled=True, resize_width=50, resize_height=40)
    processor = FrameProcessor(config)

    processed = processor.process(frame)

    assert processed is not None
    assert processed.image.shape[:2] == (40, 50)  # (height, width)


def test_color_conversion_bgr_to_rgb_reorders_channels() -> None:
    frame = make_frame()
    config = PreprocessConfig(color_conversion_enabled=True, color_mode=ColorMode.RGB)
    processor = FrameProcessor(config)

    processed = processor.process(frame)

    assert processed is not None
    assert processed.image[0, 0, 0] == 30  # R now first
    assert processed.image[0, 0, 2] == 10  # B now last


def test_color_conversion_to_gray_reduces_channels() -> None:
    frame = make_frame()
    config = PreprocessConfig(color_conversion_enabled=True, color_mode=ColorMode.GRAY)
    processor = FrameProcessor(config)

    processed = processor.process(frame)

    assert processed is not None
    assert processed.image.ndim == 2


def test_normalize_scales_to_unit_range() -> None:
    frame = make_frame()
    config = PreprocessConfig(normalize_enabled=True, normalize_scale=1.0 / 255.0)
    processor = FrameProcessor(config)

    processed = processor.process(frame)

    assert processed is not None
    assert processed.image.dtype == np.float32
    assert processed.image.max() <= 1.0


def test_normalize_with_mean_std() -> None:
    frame = make_frame()
    config = PreprocessConfig(
        normalize_enabled=True,
        normalize_scale=1.0,
        normalize_mean=(10.0, 20.0, 30.0),
        normalize_std=(1.0, 1.0, 1.0),
    )
    processor = FrameProcessor(config)

    processed = processor.process(frame)

    assert processed is not None
    assert np.allclose(processed.image, 0.0)


def test_frame_skip_skips_every_other_frame() -> None:
    config = PreprocessConfig(frame_skip_enabled=True, frame_skip_interval=2)
    processor = FrameProcessor(config)

    results = [processor.process(make_frame(index=i)) for i in range(4)]

    assert [r is not None for r in results] == [True, False, True, False]


def test_original_frame_image_is_not_mutated() -> None:
    frame = make_frame()
    original_bytes = frame.image.copy()
    config = PreprocessConfig(
        resize_enabled=True,
        resize_width=10,
        resize_height=10,
        color_conversion_enabled=True,
        color_mode=ColorMode.GRAY,
        normalize_enabled=True,
    )
    processor = FrameProcessor(config)

    processor.process(frame)

    assert np.array_equal(frame.image, original_bytes)


def test_invalid_frame_returns_none() -> None:
    empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
    frame = Frame(index=0, image=empty_image, timestamp=0.0, source_id="test")
    processor = FrameProcessor(PreprocessConfig())

    assert processor.process(frame) is None


def test_processed_frame_preserves_index_timestamp_source_id() -> None:
    frame = make_frame(index=7)
    processor = FrameProcessor(PreprocessConfig())

    processed = processor.process(frame)

    assert processed is not None
    assert processed.index == 7
    assert processed.timestamp == frame.timestamp
    assert processed.source_id == frame.source_id
    assert processed.original is frame


def test_process_stream_yields_only_valid_non_skipped_frames() -> None:
    frames = [make_frame(index=i) for i in range(5)]
    config = PreprocessConfig(frame_skip_enabled=True, frame_skip_interval=2)
    processor = FrameProcessor(config)

    processed = list(processor.process_stream(frames))

    assert [p.index for p in processed] == [0, 2, 4]


def test_invalid_resize_config_raises() -> None:
    with pytest.raises(InvalidPreprocessConfigError):
        FrameProcessor(PreprocessConfig(resize_enabled=True, resize_width=0, resize_height=10))


def test_invalid_frame_skip_config_raises() -> None:
    with pytest.raises(InvalidPreprocessConfigError):
        FrameProcessor(PreprocessConfig(frame_skip_enabled=True, frame_skip_interval=0))


def test_invalid_normalize_scale_raises() -> None:
    with pytest.raises(InvalidPreprocessConfigError):
        FrameProcessor(PreprocessConfig(normalize_enabled=True, normalize_scale=0))
