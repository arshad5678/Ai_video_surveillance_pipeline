"""Unit tests for VideoInput — cv2.VideoCapture is mocked throughout, so these
tests require no real webcam, file, or network stream."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.surveillance.pipelines.video_input import (
    VideoInput,
    VideoSourceConfig,
    VideoSourceConnectionError,
    VideoSourceNotFoundError,
    VideoSourceType,
)


def make_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_file_source_validation_rejects_missing_path() -> None:
    config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri="does/not/exist.mp4")
    video_input = VideoInput(config)

    with pytest.raises(VideoSourceNotFoundError):
        video_input.open()


def test_webcam_source_validation_rejects_negative_index() -> None:
    config = VideoSourceConfig(source_type=VideoSourceType.WEBCAM, uri=-1)
    video_input = VideoInput(config)

    with pytest.raises(VideoSourceNotFoundError):
        video_input.open()


def test_rtsp_source_validation_rejects_malformed_url() -> None:
    config = VideoSourceConfig(source_type=VideoSourceType.RTSP, uri="http://not-rtsp")
    video_input = VideoInput(config)

    with pytest.raises(VideoSourceNotFoundError):
        video_input.open()


@patch("src.surveillance.pipelines.video_input.video_input.cv2.VideoCapture")
def test_open_raises_connection_error_when_capture_not_opened(mock_capture_cls: MagicMock) -> None:
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_capture_cls.return_value = mock_capture

    config = VideoSourceConfig(source_type=VideoSourceType.WEBCAM, uri=0)
    video_input = VideoInput(config)

    with pytest.raises(VideoSourceConnectionError):
        video_input.open()

    mock_capture.release.assert_called_once()


@patch("src.surveillance.pipelines.video_input.video_input.cv2.VideoCapture")
def test_frames_yields_frame_objects_with_incrementing_index(mock_capture_cls: MagicMock) -> None:
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.side_effect = [
        (True, make_frame()),
        (True, make_frame()),
        (False, None),  # end of file
    ]
    mock_capture_cls.return_value = mock_capture

    config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri="dummy.mp4")

    with patch("src.surveillance.pipelines.video_input.video_input.Path") as mock_path_cls:
        mock_path_cls.return_value.exists.return_value = True
        mock_path_cls.return_value.is_file.return_value = True

        video_input = VideoInput(config)
        video_input.open()
        frames = list(video_input.frames())

    assert [frame.index for frame in frames] == [0, 1]
    assert all(frame.source_id == "dummy.mp4" for frame in frames)


@patch("src.surveillance.pipelines.video_input.video_input.cv2.VideoCapture")
def test_frames_raises_after_exhausting_reconnect_attempts_on_webcam(mock_capture_cls: MagicMock) -> None:
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (False, None)  # always fails
    mock_capture_cls.return_value = mock_capture

    config = VideoSourceConfig(
        source_type=VideoSourceType.WEBCAM,
        uri=0,
        reconnect_attempts=2,
        reconnect_delay_seconds=0,
    )
    video_input = VideoInput(config)
    video_input.open()

    with pytest.raises(VideoSourceConnectionError):
        list(video_input.frames())


@patch("src.surveillance.pipelines.video_input.video_input.cv2.VideoCapture")
def test_corrupted_frame_is_skipped_not_yielded(mock_capture_cls: MagicMock) -> None:
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    corrupted = np.zeros((0, 0, 3), dtype=np.uint8)  # zero-size => invalid
    mock_capture.read.side_effect = [
        (True, corrupted),
        (True, make_frame()),
        (False, None),  # end of file — capture stays alive throughout, no reconnect involved
    ]
    mock_capture_cls.return_value = mock_capture

    config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri="dummy.mp4")

    with patch("src.surveillance.pipelines.video_input.video_input.Path") as mock_path_cls:
        mock_path_cls.return_value.exists.return_value = True
        mock_path_cls.return_value.is_file.return_value = True

        video_input = VideoInput(config)
        video_input.open()
        frames = list(video_input.frames())

    assert len(frames) == 1
    assert frames[0].index == 0


@patch("src.surveillance.pipelines.video_input.video_input.cv2.VideoCapture")
def test_close_releases_capture_and_resets_state(mock_capture_cls: MagicMock) -> None:
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture_cls.return_value = mock_capture

    config = VideoSourceConfig(source_type=VideoSourceType.WEBCAM, uri=0)
    video_input = VideoInput(config)
    video_input.open()
    assert video_input.is_open

    video_input.close()

    mock_capture.release.assert_called_once()
    assert not video_input.is_open


@patch("src.surveillance.pipelines.video_input.video_input.cv2.VideoCapture")
def test_probe_status_reports_connected_with_fps_and_resolution(mock_capture_cls: MagicMock) -> None:
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FPS: 25.0,
        cv2.CAP_PROP_FRAME_WIDTH: 1280,
        cv2.CAP_PROP_FRAME_HEIGHT: 720,
    }[prop]
    mock_capture_cls.return_value = mock_capture

    config = VideoSourceConfig(source_type=VideoSourceType.WEBCAM, uri=0)
    status = VideoInput(config).probe_status()

    assert status.connected is True
    assert status.fps == 25.0
    assert status.width == 1280
    assert status.height == 720
    mock_capture.release.assert_called_once()  # probe always closes again


def test_probe_status_reports_disconnected_without_raising() -> None:
    config = VideoSourceConfig(source_type=VideoSourceType.FILE, uri="does/not/exist.mp4")
    status = VideoInput(config).probe_status()

    assert status.connected is False
    assert status.fps is None
    assert status.width is None
    assert status.height is None
