# Handles test orchestration process behavior.
from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from tools.orchestration import process as orchestration_process


class DummyProcess:
    def __init__(self, pid: int = 1234, poll_value: int | None = None) -> None:
        # Performs init logic.
        self.pid = pid
        self._poll_value = poll_value
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        # Performs poll logic.
        return self._poll_value

    def terminate(self) -> None:
        # Performs terminate logic.
        self.terminated = True
        self._poll_value = 0

    def kill(self) -> None:
        # Performs kill logic.
        self.killed = True
        self._poll_value = -9

    def wait(self, timeout: float | None = None) -> int:
        # Performs wait logic.
        self.wait_calls.append(timeout)
        return 0


class TestOrchestrationProcess(unittest.TestCase):
    def tearDown(self) -> None:
        # Performs tearDown logic.
        orchestration_process._dashboard_process = None

    @patch("tools.orchestration.process.DASHBOARD_SCRIPT_PATH")
    def test_start_dashboard_process_when_script_missing(self, mock_path) -> None:
        # Performs test start dashboard process when script missing logic.
        mock_path.is_file.return_value = False
        with self.assertRaises(FileNotFoundError):
            orchestration_process.start_dashboard_process()

    @patch("tools.orchestration.process.subprocess.Popen")
    @patch("tools.orchestration.process.DASHBOARD_SCRIPT_PATH")
    def test_start_dashboard_process_starts_subprocess(self, mock_path, mock_popen) -> None:
        # Performs test start dashboard process starts subprocess logic.
        mock_path.is_file.return_value = True
        mock_path.parent = "/tmp"
        mock_process = DummyProcess(pid=6789)
        mock_popen.return_value = mock_process

        started, message = orchestration_process.start_dashboard_process()

        self.assertTrue(started)
        self.assertIn("Process ID: 6789", message)
        self.assertIs(orchestration_process._dashboard_process, mock_process)

    def test_start_dashboard_process_returns_existing_running_message(self) -> None:
        # Performs test start dashboard process returns existing running message logic.
        mock_process = DummyProcess(pid=7777, poll_value=None)
        orchestration_process._dashboard_process = mock_process

        started, message = orchestration_process.start_dashboard_process()

        self.assertFalse(started)
        self.assertIn("already running", message)
        self.assertIn("7777", message)

    def test_stop_dashboard_process_when_not_running(self) -> None:
        # Performs test stop dashboard process when not running logic.
        orchestration_process._dashboard_process = None
        message = orchestration_process.stop_dashboard_process()
        self.assertEqual(message, "Dashboard demo is not running.")

    def test_stop_dashboard_process_terminate(self) -> None:
        # Performs test stop dashboard process terminate logic.
        mock_process = DummyProcess(pid=1111, poll_value=None)
        orchestration_process._dashboard_process = mock_process

        message = orchestration_process.stop_dashboard_process(force=False)

        self.assertEqual(message, "Dashboard demo stopped.")
        self.assertTrue(mock_process.terminated)
        self.assertFalse(mock_process.killed)
        self.assertIsNone(orchestration_process._dashboard_process)

    def test_stop_dashboard_process_force(self) -> None:
        # Performs test stop dashboard process force logic.
        mock_process = DummyProcess(pid=2222, poll_value=None)
        orchestration_process._dashboard_process = mock_process

        message = orchestration_process.stop_dashboard_process(force=True)

        self.assertEqual(message, "Dashboard demo stopped.")
        self.assertTrue(mock_process.killed)
        self.assertIsNone(orchestration_process._dashboard_process)

    def test_stop_dashboard_process_timeout_fallback_to_kill(self) -> None:
        # Performs test stop dashboard process timeout fallback to kill logic.
        class TimeoutProcess(DummyProcess):
            def wait(self, timeout: float | None = None) -> int:
                # Performs wait logic.
                self.wait_calls.append(timeout)
                if len(self.wait_calls) == 1:
                    raise subprocess.TimeoutExpired(cmd="demo", timeout=timeout)
                return 0

        mock_process = TimeoutProcess(pid=3333, poll_value=None)
        orchestration_process._dashboard_process = mock_process

        message = orchestration_process.stop_dashboard_process(force=False)

        self.assertEqual(message, "Dashboard demo stopped.")
        self.assertTrue(mock_process.terminated)
        self.assertTrue(mock_process.killed)

    def test_dashboard_process_status(self) -> None:
        # Performs test orchestration process status logic.
        status = orchestration_process.dashboard_process_status()
        self.assertFalse(status["running"])
        self.assertIsNone(status["pid"])

        mock_process = DummyProcess(pid=4444, poll_value=None)
        orchestration_process._dashboard_process = mock_process
        running_status = orchestration_process.dashboard_process_status()

        self.assertTrue(running_status["running"])
        self.assertEqual(running_status["pid"], 4444)
        self.assertIsNone(running_status["return_code"])


if __name__ == "__main__":
    unittest.main()
