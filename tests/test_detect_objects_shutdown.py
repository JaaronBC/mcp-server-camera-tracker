from __future__ import annotations

import asyncio
import json
import types
import unittest
from unittest.mock import patch

from agents.orchestration.tools.detect_objects import detect_objects
from agents.tracking import tracking_runtime


class _FakeSocket:
    def __init__(self, responses=None, timeout_error: bool = False):
        self._responses = list(responses or [])
        self._timeout_error = timeout_error
        self.bound = None
        self.timeout = None
        self.non_blocking = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def bind(self, address):
        self.bound = address

    def settimeout(self, value):
        self.timeout = value

    def setblocking(self, value):
        self.non_blocking = not value

    def recvfrom(self, _max_bytes):
        if self._timeout_error:
            raise TimeoutError("timed out")
        if not self._responses:
            raise BlockingIOError()
        return self._responses.pop(0), ("127.0.0.1", 5005)


class _FakeSession:
    def __init__(self, result):
        self._result = result

    async def call_tool(self, _tool_name, arguments):
        _ = arguments
        return self._result


def _make_tool_result(text: str, is_error: bool = False):
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(text=text)],
        is_error=is_error,
    )


class TestDetectObjectsShutdownHandling(unittest.TestCase):
    def test_detect_objects_returns_dashboard_stopped_on_timeout_when_not_running(self) -> None:
        fake_socket = _FakeSocket(timeout_error=True)
        with (
            patch("agents.orchestration.tools.detect_objects.socket.socket", return_value=fake_socket),
            patch(
                "agents.orchestration.tools.detect_objects.dashboard_process_status",
                return_value={"running": False},
            ),
        ):
            payload = json.loads(detect_objects(["127.0.0.1", 5005]))

        self.assertEqual(payload["status"], "dashboard_stopped")
        self.assertEqual(payload["error"], "dashboard_not_running")
        self.assertFalse(payload["dashboard_running"])

    def test_detect_objects_returns_frame_timeout_when_dashboard_running(self) -> None:
        fake_socket = _FakeSocket(timeout_error=True)
        with (
            patch("agents.orchestration.tools.detect_objects.socket.socket", return_value=fake_socket),
            patch(
                "agents.orchestration.tools.detect_objects.dashboard_process_status",
                return_value={"running": True},
            ),
        ):
            payload = json.loads(detect_objects(["127.0.0.1", 5005]))

        self.assertEqual(payload["status"], "frame_timeout")
        self.assertEqual(payload["error"], "udp_frame_timeout")
        self.assertTrue(payload["dashboard_running"])

    def test_run_tracking_cycle_emits_stopped_message_for_dashboard_stopped(self) -> None:
        messages: list[str] = []
        result = _make_tool_result(
            json.dumps(
                {
                    "time": 1.0,
                    "detections": [],
                    "status": "dashboard_stopped",
                    "error": "dashboard_not_running",
                    "dashboard_running": False,
                }
            )
        )
        session = _FakeSession(result)

        asyncio.run(
            tracking_runtime.run_tracking_cycle(
                session,
                address=["127.0.0.1", 5005],
                emit=messages.append,
            )
        )

        self.assertEqual(messages, ["Movement analysis stopped: dashboard not running."])

    def test_run_tracking_cycle_emits_timeout_message_for_frame_timeout(self) -> None:
        messages: list[str] = []
        result = _make_tool_result(
            json.dumps(
                {
                    "time": 1.0,
                    "detections": [],
                    "status": "frame_timeout",
                    "error": "udp_frame_timeout",
                    "dashboard_running": True,
                }
            )
        )
        session = _FakeSession(result)

        asyncio.run(
            tracking_runtime.run_tracking_cycle(
                session,
                address=["127.0.0.1", 5005],
                emit=messages.append,
            )
        )

        self.assertEqual(
            messages,
            ["Movement analysis timeout: dashboard running but no frame received."],
        )


if __name__ == "__main__":
    unittest.main()
