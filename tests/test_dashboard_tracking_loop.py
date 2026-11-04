from __future__ import annotations

from collections import deque
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
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


class _TrackingLock:
    def __init__(self) -> None:
        self.held = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.held = False
        return False


class _GuardedTrailsDict(dict):
    def __init__(self, lock: _TrackingLock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = lock

    def __iter__(self):
        if not self._lock.held:
            raise RuntimeError("track_trails_iterated_without_lock")
        return super().__iter__()


class _FakeFrame:
    shape = (720, 1280, 3)


class _FakeResult:
    boxes = object()
    names = {0: "person"}


class _FakeModel:
    names = {0: "person"}


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
        initialized = (_FakeModel(), fake_camera, None, fake_socket, fake_runtime)

        with patch("agents.dashboard.tracking_loop._initialize_tracking_runtime", return_value=initialized):
            tracking_loop.run_live_tracking_loop(state)

        self.assertTrue(fake_socket.closed)
        self.assertTrue(fake_runtime.shutdown_called)
        self.assertTrue(fake_camera.release_called)
        self.assertEqual(state.camera_status, "stopped")
        self.assertEqual(state.detector_status, "stopped")


class TestDashboardTrackingLoopTrailSnapshot(unittest.TestCase):
    def test_snapshot_for_transformer_is_taken_while_holding_state_lock(self) -> None:
        state = SharedDashboardState()
        state.running = True

        tracking_lock = _TrackingLock()
        state.lock = tracking_lock
        state.track_trails = _GuardedTrailsDict(
            tracking_lock,
            {
                "7": deque([(1.0, 2.0)], maxlen=20),
            },
        )

        fake_camera = _FakeCamera(opened=True)
        fake_socket = _FakeSocket()
        fake_runtime = _FakeTrackingRuntime()
        initialized = (_FakeModel(), fake_camera, None, fake_socket, fake_runtime)

        def _publish_side_effect(**kwargs):
            with state.lock:
                state.running = False

        with (
            patch("agents.dashboard.tracking_loop._initialize_tracking_runtime", return_value=initialized),
            patch("agents.dashboard.tracking_loop._capture_frame_or_mark_error", return_value=(True, _FakeFrame())),
            patch("agents.dashboard.tracking_loop._run_detection_or_mark_error", return_value=_FakeResult()),
            patch(
                "agents.dashboard.tracking_loop.build_tracking_frame_outputs",
                return_value=({}, [], [], []),
            ),
            patch("agents.dashboard.tracking_loop._apply_tracking_agent_runtime_cycle", return_value=None),
            patch("agents.dashboard.tracking_loop._publish_tracking_outputs", side_effect=_publish_side_effect),
        ):
            tracking_loop.run_live_tracking_loop(state)

    def test_transformer_receives_snapshot_dict_not_shared_track_trails_dict(self) -> None:
        state = SharedDashboardState()
        state.running = True
        state.track_trails = {
            "7": deque([(1.0, 2.0)], maxlen=20),
        }

        fake_camera = _FakeCamera(opened=True)
        fake_socket = _FakeSocket()
        fake_runtime = _FakeTrackingRuntime()
        initialized = (_FakeModel(), fake_camera, None, fake_socket, fake_runtime)

        captured_track_trails = {}

        def _build_outputs_side_effect(**kwargs):
            captured_track_trails["value"] = kwargs["track_trails"]
            return {}, [], [], []

        def _publish_side_effect(**kwargs):
            with state.lock:
                state.running = False

        with (
            patch("agents.dashboard.tracking_loop._initialize_tracking_runtime", return_value=initialized),
            patch("agents.dashboard.tracking_loop._capture_frame_or_mark_error", return_value=(True, _FakeFrame())),
            patch("agents.dashboard.tracking_loop._run_detection_or_mark_error", return_value=_FakeResult()),
            patch("agents.dashboard.tracking_loop.build_tracking_frame_outputs", side_effect=_build_outputs_side_effect),
            patch("agents.dashboard.tracking_loop._apply_tracking_agent_runtime_cycle", return_value=None),
            patch("agents.dashboard.tracking_loop._publish_tracking_outputs", side_effect=_publish_side_effect),
        ):
            tracking_loop.run_live_tracking_loop(state)

        self.assertIn("value", captured_track_trails)
        snapshot = captured_track_trails["value"]
        self.assertIsNot(snapshot, state.track_trails)
        self.assertEqual(snapshot.keys(), state.track_trails.keys())
        self.assertIs(snapshot["7"], state.track_trails["7"])


class TestTrackingFeedbackDecisionLogging(unittest.TestCase):
    def test_empty_poll_does_not_append_decision_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_feedback_consumer_decisions.jsonl"
            tracking_loop._append_tracking_feedback_consumer_decisions(
                feedback_summary={
                    "accepted": [],
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "error": None,
                    "evaluated_records": [],
                    "consumer_outcome": {
                        "status": "ok",
                        "evaluated_count": 0,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 0,
                    },
                },
                now=1001.0,
                diagnostics_path=diagnostics_path,
            )

            self.assertFalse(diagnostics_path.exists())

    def test_poll_error_appends_error_diagnostic_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_feedback_consumer_decisions.jsonl"
            tracking_loop._append_tracking_feedback_consumer_decisions(
                feedback_summary={
                    "accepted": [],
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "error": "feedback_poll_failed",
                    "evaluated_records": [],
                },
                now=1002.0,
                diagnostics_path=diagnostics_path,
            )

            self.assertTrue(diagnostics_path.is_file())
            lines = diagnostics_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn('"decision":"error"', lines[0])
            self.assertIn('"reason":"feedback_poll_failed"', lines[0])


class TestTrackingFeedbackPathResolution(unittest.TestCase):
    def _reload_dashboard_modules(self) -> tuple[object, object]:
        config_module = importlib.import_module("agents.dashboard.config")
        tracking_module = importlib.import_module("agents.dashboard.tracking_loop")
        config_module = importlib.reload(config_module)
        tracking_module = importlib.reload(tracking_module)
        return config_module, tracking_module

    def test_default_feedback_path_used_when_env_not_set(self) -> None:
        original_env = os.environ.pop("PHASE16E_OPERATOR_FEEDBACK_PATH", None)
        try:
            config_module, tracking_module = self._reload_dashboard_modules()
            expected_default = config_module.CAPTURES_DIR / "operator_tracking_feedback.json"
            self.assertEqual(config_module.LATEST_OPERATOR_TRACKING_FEEDBACK_PATH, expected_default)
            self.assertEqual(tracking_module.LATEST_OPERATOR_TRACKING_FEEDBACK_PATH, expected_default)
        finally:
            if original_env is None:
                os.environ.pop("PHASE16E_OPERATOR_FEEDBACK_PATH", None)
            else:
                os.environ["PHASE16E_OPERATOR_FEEDBACK_PATH"] = original_env
            self._reload_dashboard_modules()

    def test_env_feedback_path_used_when_set_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_local_path = Path(temp_dir) / "operator_tracking_feedback.json"
            original_env = os.environ.get("PHASE16E_OPERATOR_FEEDBACK_PATH")
            try:
                os.environ["PHASE16E_OPERATOR_FEEDBACK_PATH"] = str(run_local_path)
                config_module, tracking_module = self._reload_dashboard_modules()
                self.assertEqual(config_module.LATEST_OPERATOR_TRACKING_FEEDBACK_PATH, run_local_path)
                self.assertEqual(tracking_module.LATEST_OPERATOR_TRACKING_FEEDBACK_PATH, run_local_path)
            finally:
                if original_env is None:
                    os.environ.pop("PHASE16E_OPERATOR_FEEDBACK_PATH", None)
                else:
                    os.environ["PHASE16E_OPERATOR_FEEDBACK_PATH"] = original_env
                self._reload_dashboard_modules()

    def test_run_local_feedback_record_is_discovered_and_evaluated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "operator_tracking_feedback.json"
            feedback_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "records": [
                            {
                                "schema_version": "1.0",
                                "feedback_id": "fb-1",
                                "verification_request_id": "req-1",
                                "track_id": "track-1",
                                "evidence_message_id": "msg-1",
                                "verification_observation_id": "obs-1",
                                "verification_message_id": "msg-1",
                                "verification_timestamp": 900.0,
                                "source": "external_source",
                                "current_applicability": "historical",
                                "historical_observation": True,
                                "stale_for_current_decisions": True,
                                "coordinator_valid": True,
                                "validation_outcome": "accepted",
                                "submitted_at": 901.0,
                                "started_at": 902.0,
                                "finished_at": 903.0,
                                "published_at": 904.0,
                                "expires_at": 1100.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            original_env = os.environ.get("PHASE16E_OPERATOR_FEEDBACK_PATH")
            try:
                os.environ["PHASE16E_OPERATOR_FEEDBACK_PATH"] = str(feedback_path)
                _, tracking_module = self._reload_dashboard_modules()
                consumer = tracking_module.TrackingObservationFeedbackConsumer(
                    artifact_path=tracking_module.LATEST_OPERATOR_TRACKING_FEEDBACK_PATH,
                    max_seen_feedback_ids=16,
                    max_artifact_bytes=128_000,
                )
                summary = consumer.poll(
                    current_track_ids={"track-1"},
                    temporarily_lost_track_ids=set(),
                    latest_message_id_by_track={"track-1": "msg-1"},
                    now_timestamp=1000.0,
                )
                self.assertEqual(summary.get("accepted_count"), 1)
                evaluated = summary.get("evaluated_records")
                self.assertIsInstance(evaluated, list)
                self.assertEqual(len(evaluated), 1)
                self.assertEqual(evaluated[0].get("verification_observation_id"), "obs-1")
                self.assertEqual(evaluated[0].get("decision"), "accepted")
            finally:
                if original_env is None:
                    os.environ.pop("PHASE16E_OPERATOR_FEEDBACK_PATH", None)
                else:
                    os.environ["PHASE16E_OPERATOR_FEEDBACK_PATH"] = original_env
                self._reload_dashboard_modules()


    def test_initialize_runtime_uses_configured_feedback_artifact_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_local_path = Path(temp_dir) / "operator_tracking_feedback.json"
            original_env = os.environ.get("PHASE16E_OPERATOR_FEEDBACK_PATH")
            captured_artifact_path: dict[str, Path] = {}

            def _consumer_factory(*, artifact_path: Path, max_seen_feedback_ids: int, max_artifact_bytes: int):
                captured_artifact_path["value"] = artifact_path
                return object()

            try:
                os.environ["PHASE16E_OPERATOR_FEEDBACK_PATH"] = str(run_local_path)
                _, tracking_module = self._reload_dashboard_modules()

                state = SharedDashboardState()
                fake_camera = _FakeCamera(opened=True)
                fake_socket = _FakeSocket()
                fake_runtime = _FakeTrackingRuntime()

                with (
                    patch.object(tracking_module, "YOLO", return_value=object()),
                    patch.object(tracking_module.cv2, "VideoCapture", return_value=fake_camera),
                    patch.object(tracking_module, "load_ground_plane_calibration", return_value=None),
                    patch.object(tracking_module.socket, "socket", return_value=fake_socket),
                    patch.object(tracking_module, "AgentCoordinator", return_value=object()),
                    patch.object(tracking_module, "LLMVisualTrackingAgent", return_value=_FakeAgent()),
                    patch.object(tracking_module, "AsyncVisualTrackingRuntime", return_value=fake_runtime),
                    patch.object(tracking_module, "TrackingEvidenceStore", return_value=object()),
                    patch.object(tracking_module, "TrackingEvidencePackager", return_value=object()),
                    patch.object(
                        tracking_module,
                        "TrackingObservationFeedbackConsumer",
                        side_effect=_consumer_factory,
                    ),
                    patch.object(tracking_module, "TrackingEstimateProposalEngine", return_value=object()),
                ):
                    initialized = tracking_module._initialize_tracking_runtime(state)

                self.assertIsNotNone(initialized)
                self.assertEqual(tracking_module.LATEST_OPERATOR_TRACKING_FEEDBACK_PATH, run_local_path)
                self.assertEqual(captured_artifact_path.get("value"), run_local_path)
            finally:
                if original_env is None:
                    os.environ.pop("PHASE16E_OPERATOR_FEEDBACK_PATH", None)
                else:
                    os.environ["PHASE16E_OPERATOR_FEEDBACK_PATH"] = original_env
                self._reload_dashboard_modules()


if __name__ == "__main__":
    unittest.main()
