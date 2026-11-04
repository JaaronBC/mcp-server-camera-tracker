from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from agents.dashboard.state import SharedDashboardState
from agents.dashboard.tracking_loop import run_live_tracking_loop


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], opened: bool = True) -> None:
        self.frames = iter(frames)
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = next(self.frames, None)
        return (frame is not None, frame)

    def get(self, prop: int) -> float:
        return 1.0

    def release(self) -> None:
        self.released = True


class FakeRuntime:
    def __init__(self, **_kwargs) -> None:
        pass

    def poll_latest_observation(self):
        return None

    def shutdown(self) -> None:
        pass


class TestTrackingLoopVideoInput(unittest.TestCase):
    def test_video_frames_are_shown_and_tracked(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        capture = FakeCapture([frame])
        state = SharedDashboardState()
        result = SimpleNamespace(
            boxes=SimpleNamespace(id=None),
            plot=lambda: frame.copy(),
        )

        with (
            patch("agents.dashboard.tracking_loop.cv2.VideoCapture", return_value=capture) as video_capture,
            patch("agents.dashboard.tracking_loop.YOLO") as yolo,
            patch("agents.dashboard.tracking_loop.load_ground_plane_calibration", return_value=None),
            patch("agents.dashboard.tracking_loop.socket.socket"),
            patch("agents.dashboard.tracking_loop.LLMVisualTrackingAgent") as tracking_agent,
            patch("agents.dashboard.tracking_loop.AsyncVisualTrackingRuntime", FakeRuntime),
            patch("agents.dashboard.tracking_loop.publish_tracking_frame_message"),
        ):
            model = yolo.return_value
            model.names = {}

            def track_frame(**kwargs):
                self.assertEqual(kwargs["source"].shape, frame.shape)
                self.assertTrue(np.array_equal(state.latest_annotated_frame, frame))
                return [result]

            model.track.side_effect = track_frame
            tracking_agent.return_value.model = "test-agent"

            run_live_tracking_loop(state, "sample.mp4")

        video_capture.assert_called_once_with("sample.mp4")
        yolo.assert_called_once()
        model.track.assert_called_once()
        self.assertTrue(capture.released)
        self.assertIsNotNone(state.latest_annotated_frame)
        self.assertIsNone(state.input_error)
        self.assertEqual(state.camera_status, "stopped")

    def test_unavailable_webcam_is_reported_without_initializing_yolo(self) -> None:
        capture = FakeCapture([], opened=False)
        state = SharedDashboardState()

        with (
            patch("agents.dashboard.tracking_loop.cv2.VideoCapture", return_value=capture) as video_capture,
            patch("agents.dashboard.tracking_loop.YOLO") as yolo,
        ):
            run_live_tracking_loop(state)

        video_capture.assert_called_once_with(0)
        yolo.assert_not_called()
        self.assertTrue(capture.released)
        self.assertEqual(state.camera_status, "unavailable")
        self.assertIn("webcam", state.input_error or "")

    def test_tracking_exception_is_reported_and_capture_is_released(self) -> None:
        capture = FakeCapture([np.zeros((24, 32, 3), dtype=np.uint8)])
        state = SharedDashboardState()

        with (
            patch("agents.dashboard.tracking_loop.cv2.VideoCapture", return_value=capture),
            patch("agents.dashboard.tracking_loop.YOLO") as yolo,
            patch("agents.dashboard.tracking_loop.load_ground_plane_calibration", return_value=None),
            patch("agents.dashboard.tracking_loop.socket.socket"),
            patch("agents.dashboard.tracking_loop.LLMVisualTrackingAgent") as tracking_agent,
            patch("agents.dashboard.tracking_loop.AsyncVisualTrackingRuntime", FakeRuntime),
        ):
            yolo.return_value.track.side_effect = RuntimeError("inference failed")
            tracking_agent.return_value.model = "test-agent"

            run_live_tracking_loop(state, "sample.mp4")

        self.assertTrue(capture.released)
        self.assertEqual(state.detector_status, "error")
        self.assertIn("inference failed", state.input_error or "")


if __name__ == "__main__":
    unittest.main()
