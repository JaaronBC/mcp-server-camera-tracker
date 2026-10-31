from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.dashboard.state import SharedDashboardState
from agents.dashboard import tracking_loop


class _FakeCamera:
    def __init__(self, opened: bool = True) -> None:
        self._opened = opened
        self.release_called = False

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.release_called = True


class _FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeTrackingRuntime:
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _RaisingLock:
    def __init__(self, raise_on_enter: int) -> None:
        self._raise_on_enter = raise_on_enter
        self._enter_count = 0

    def __enter__(self):
        self._enter_count += 1
        if self._enter_count == self._raise_on_enter:
            raise RuntimeError("lock_failure")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeAgent:
    def __init__(self, model: str = "qwen-3.8-27b") -> None:
        self.model = model


class TestDashboardTrackingLoopInitialization(unittest.TestCase):
    def test_initialize_runtime_closes_socket_and_releases_camera_when_agent_init_fails(self) -> None:
        state = SharedDashboardState()
        fake_camera = _FakeCamera(opened=True)
        fake_socket = _FakeSocket()

        with (
            patch("agents.dashboard.tracking_loop.YOLO", return_value=object()),
            patch("agents.dashboard.tracking_loop.cv2.VideoCapture", return_value=fake_camera),
            patch("agents.dashboard.tracking_loop.load_ground_plane_calibration", return_value=None),
            patch("agents.dashboard.tracking_loop.socket.socket", return_value=fake_socket),
            patch("agents.dashboard.tracking_loop.LLMVisualTrackingAgent", side_effect=RuntimeError("agent_init_failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "agent_init_failed"):
                tracking_loop._initialize_tracking_runtime(state)

        self.assertTrue(fake_camera.release_called)
        self.assertTrue(fake_socket.closed)

    def test_initialize_runtime_shuts_down_runtime_and_cleans_resources_when_late_step_fails(self) -> None:
        state = SharedDashboardState()
        state.lock = _RaisingLock(raise_on_enter=3)

        fake_camera = _FakeCamera(opened=True)
        fake_socket = _FakeSocket()
        fake_runtime = _FakeTrackingRuntime()

        with (
            patch("agents.dashboard.tracking_loop.YOLO", return_value=object()),
            patch("agents.dashboard.tracking_loop.cv2.VideoCapture", return_value=fake_camera),
            patch("agents.dashboard.tracking_loop.load_ground_plane_calibration", return_value=None),
            patch("agents.dashboard.tracking_loop.socket.socket", return_value=fake_socket),
            patch("agents.dashboard.tracking_loop.LLMVisualTrackingAgent", return_value=_FakeAgent()),
            patch("agents.dashboard.tracking_loop.AsyncVisualTrackingRuntime", return_value=fake_runtime),
        ):
            with self.assertRaisesRegex(RuntimeError, "lock_failure"):
                tracking_loop._initialize_tracking_runtime(state)

        self.assertTrue(fake_runtime.shutdown_called)
        self.assertTrue(fake_socket.closed)
        self.assertTrue(fake_camera.release_called)

    def test_run_live_tracking_loop_finally_cleans_all_initialized_resources(self) -> None:
        state = SharedDashboardState()
        state.running = False

        fake_camera = _FakeCamera(opened=True)
        fake_socket = _FakeSocket()
        fake_runtime = _FakeTrackingRuntime()
        initialized = (object(), fake_camera, None, fake_socket, fake_runtime)

        with patch("agents.dashboard.tracking_loop._initialize_tracking_runtime", return_value=initialized):
            tracking_loop.run_live_tracking_loop(state)

        self.assertTrue(fake_socket.closed)
        self.assertTrue(fake_runtime.shutdown_called)
        self.assertTrue(fake_camera.release_called)
        self.assertEqual(state.camera_status, "stopped")
        self.assertEqual(state.detector_status, "stopped")


if __name__ == "__main__":
    unittest.main()
