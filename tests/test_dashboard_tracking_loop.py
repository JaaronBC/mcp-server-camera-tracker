from __future__ import annotations

from collections import deque
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from agents.contracts import BoundingBox, DetectionCandidate
from agents.dashboard import rendering
from agents.dashboard.state import SharedDashboardState
from agents.dashboard import tracking_loop
from agents.tracking.identity_registry import PersistentIdentityRegistry
from agents.tracking.tracking_types import GroundPosition, RadarTrackSnapshot


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


class _FakePlotResult:
    def __init__(self) -> None:
        self.plot_calls: list[dict] = []

    def plot(self, *args, **kwargs):
        self.plot_calls.append(dict(kwargs))
        return np.zeros((120, 160, 3), dtype=np.uint8)


class _FakeAnnotator:
    def __init__(self, *args, **kwargs) -> None:
        self.text_calls: list[dict[str, object]] = []
        self.box_label_calls: list[dict[str, object]] = []

    def text(self, xy, text, txt_color=(255, 255, 255), anchor="top", box_color=()):
        self.text_calls.append(
            {
                "xy": xy,
                "text": text,
                "txt_color": txt_color,
                "anchor": anchor,
                "box_color": box_color,
            }
        )

    def box_label(self, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
        self.box_label_calls.append(
            {
                "box": box,
                "label": label,
                "color": color,
                "txt_color": txt_color,
            }
        )


def _candidate(
    *,
    detection_id: str,
    yolo_track_id: str,
    timestamp: float,
    detection_reference: str,
) -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        timestamp=timestamp,
        yolo_track_id=yolo_track_id,
        label="person",
        detection_confidence=0.9,
        pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0),
        pixel_center_x=30.0,
        pixel_center_y=80.0,
        relative_depth=0.4,
        ground_x=1.0,
        ground_z=2.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
        detection_reference=detection_reference,
    )


class TestStableIdentityDisplayMapping(unittest.TestCase):
    def test_stable_display_identity_persists_across_track_switch_4_to_12(self) -> None:
        registry = PersistentIdentityRegistry(retirement_seconds=20.0)
        registry.process_detections(
            now=100.0,
            detections=[
                _candidate(
                    detection_id="det-4",
                    yolo_track_id="4",
                    timestamp=100.0,
                    detection_reference="fdet-4",
                )
            ],
        )
        app_identity_id = registry.get_identity_for_yolo_track("4")
        self.assertIsNotNone(app_identity_id)
        assert isinstance(app_identity_id, str)

        registry.process_detections(now=101.0, detections=[])
        snapshot = registry.snapshot()
        mutation = registry.apply_identity_mutation(
            app_identity_id=app_identity_id,
            new_yolo_track_id="12",
            now=102.0,
            expected_generation_at_least=int(snapshot["generation"]),
            expected_version=int(snapshot["version"]),
        )
        self.assertTrue(mutation.accepted)

        display_map, summary = tracking_loop._build_stable_identity_display_map(
            registry=registry,
            tracking_candidates=[
                _candidate(
                    detection_id="det-12",
                    yolo_track_id="12",
                    timestamp=102.0,
                    detection_reference="fdet-12",
                )
            ],
        )

        self.assertEqual(summary["mapped_count"], 1)
        self.assertEqual(display_map["12"]["app_identity_id"], app_identity_id)
        self.assertNotIn("4", display_map)

    def test_conflicting_display_mapping_falls_back_safely(self) -> None:
        class _ConflictingRegistry:
            def get_identity_for_yolo_track(self, yolo_track_id: str) -> str | None:
                if yolo_track_id in {"4", "12"}:
                    return "app-1"
                return None

            def get_record(self, app_identity_id: str):
                if app_identity_id == "app-1":
                    return SimpleNamespace(status="visible", provisional=False)
                return None

        display_map, summary = tracking_loop._build_stable_identity_display_map(
            registry=_ConflictingRegistry(),
            tracking_candidates=[
                _candidate(
                    detection_id="det-4",
                    yolo_track_id="4",
                    timestamp=200.0,
                    detection_reference="fdet-4",
                ),
                _candidate(
                    detection_id="det-12",
                    yolo_track_id="12",
                    timestamp=200.0,
                    detection_reference="fdet-12",
                ),
            ],
        )

        self.assertEqual(summary["conflict_count"], 1)
        self.assertEqual(summary["mapped_count"], 0)
        self.assertEqual(display_map, {})

    def test_missing_mapping_falls_back_to_raw_track_label(self) -> None:
        registry = PersistentIdentityRegistry()
        display_map, summary = tracking_loop._build_stable_identity_display_map(
            registry=registry,
            tracking_candidates=[
                _candidate(
                    detection_id="det-1",
                    yolo_track_id="1",
                    timestamp=300.0,
                    detection_reference="fdet-1",
                )
            ],
        )

        self.assertEqual(display_map, {})
        self.assertEqual(summary["fallback_count"], 1)

    def test_disabled_feature_flag_preserves_legacy_labels(self) -> None:
        state = SharedDashboardState()
        state.tracking_stable_identity_visualization_enabled = False
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = PersistentIdentityRegistry()

        display_map = tracking_loop._update_stable_identity_display_mapping_if_enabled(
            state=state,
            tracking_candidates=[
                _candidate(
                    detection_id="det-1",
                    yolo_track_id="1",
                    timestamp=400.0,
                    detection_reference="fdet-1",
                )
            ],
        )

        self.assertEqual(display_map, {})
        self.assertEqual(state.latest_tracking_display_identity_by_track, {})
        self.assertIsInstance(state.latest_tracking_display_identity_summary, dict)
        assert isinstance(state.latest_tracking_display_identity_summary, dict)
        self.assertEqual(
            state.latest_tracking_display_identity_summary.get("reason"),
            "feature_disabled",
        )


class TestStableIdentityDisplayIntegration(unittest.TestCase):
    def _track_snapshot(self, *, track_id: str, timestamp: float) -> RadarTrackSnapshot:
        return RadarTrackSnapshot(
            track_id=track_id,
            timestamp=timestamp,
            pixel_position=None,
            ground_position=GroundPosition(x=1.0, z=2.0),
            ground_coordinate_unit="relative_units",
            position_kind="relative_ground_estimate",
            quality_flags=[],
            confidence=0.9,
            relative_depth=0.4,
        )

    def test_complete_4_to_12_display_transition_uses_same_mapping_for_webcam_and_radar(self) -> None:
        state = SharedDashboardState()
        state.tracking_stable_identity_visualization_enabled = True

        registry = PersistentIdentityRegistry(retirement_seconds=20.0)
        registry.process_detections(
            now=100.0,
            detections=[
                _candidate(
                    detection_id="det-4",
                    yolo_track_id="4",
                    timestamp=100.0,
                    detection_reference="fdet-4",
                )
            ],
        )
        app_identity_id = registry.get_identity_for_yolo_track("4")
        self.assertIsNotNone(app_identity_id)
        assert isinstance(app_identity_id, str)

        registry.process_detections(now=101.0, detections=[])
        snapshot = registry.snapshot()
        mutation = registry.apply_identity_mutation(
            app_identity_id=app_identity_id,
            new_yolo_track_id="12",
            now=102.0,
            expected_generation_at_least=int(snapshot["generation"]),
            expected_version=int(snapshot["version"]),
        )
        self.assertTrue(mutation.accepted)

        tracking_candidates = [
            _candidate(
                detection_id="det-12",
                yolo_track_id="12",
                timestamp=102.0,
                detection_reference="fdet-12",
            )
        ]
        display_map, summary = tracking_loop._build_stable_identity_display_map(
            registry=registry,
            tracking_candidates=tracking_candidates,
        )
        self.assertEqual(summary["mapped_count"], 1)
        self.assertEqual(display_map["12"]["app_identity_id"], app_identity_id)

        tracks = {
            "12": self._track_snapshot(track_id="12", timestamp=102.0),
        }
        fake_result = _FakePlotResult()
        fake_annotator = _FakeAnnotator()

        with (
            patch.object(tracking_loop, "publish_tracking_frame_message", return_value={}),
            patch.object(tracking_loop, "Annotator", return_value=fake_annotator),
        ):
            tracking_loop._publish_tracking_outputs(
                state=state,
                result=fake_result,
                now=102.0,
                tracks=tracks,
                tracking_candidates=tracking_candidates,
                tracking_agent_observation=None,
                trail_length=20,
                detections_for_file=[],
                udp_socket=None,
                stable_identity_visualization_enabled=True,
                display_identity_by_track=display_map,
            )

        self.assertEqual(fake_result.plot_calls[0].get("labels"), False)
        drawn_texts = [str(call["text"]) for call in fake_annotator.text_calls]
        self.assertTrue(
            any(text.startswith(f"person {app_identity_id}") for text in drawn_texts)
        )
        self.assertTrue(any(text.endswith("0.90") for text in drawn_texts))
        self.assertEqual(fake_annotator.box_label_calls, [])

        state_snapshot = rendering._read_dashboard_state_snapshot(state)
        self.assertEqual(state_snapshot["tracks"].keys(), {"12"})
        self.assertEqual(
            state_snapshot["display_identity_by_track"].get("12", {}).get("app_identity_id"),
            app_identity_id,
        )
        radar_display_label = rendering._resolve_track_display_label(
            "12",
            state_snapshot["display_identity_by_track"],
        )
        self.assertTrue(radar_display_label.startswith(app_identity_id))

    def test_enabled_mode_with_missing_mapping_falls_back_to_raw_overlay_label(self) -> None:
        state = SharedDashboardState()
        state.tracking_stable_identity_visualization_enabled = True
        fake_result = _FakePlotResult()
        tracking_candidates = [
            _candidate(
                detection_id="det-12",
                yolo_track_id="12",
                timestamp=200.0,
                detection_reference="fdet-12",
            )
        ]

        fake_annotator = _FakeAnnotator()

        with (
            patch.object(tracking_loop, "publish_tracking_frame_message", return_value={}),
            patch.object(tracking_loop, "Annotator", return_value=fake_annotator),
        ):
            tracking_loop._publish_tracking_outputs(
                state=state,
                result=fake_result,
                now=200.0,
                tracks={"12": self._track_snapshot(track_id="12", timestamp=200.0)},
                tracking_candidates=tracking_candidates,
                tracking_agent_observation=None,
                trail_length=20,
                detections_for_file=[],
                udp_socket=None,
                stable_identity_visualization_enabled=True,
                display_identity_by_track={},
            )

        self.assertEqual(fake_result.plot_calls[0].get("labels"), False)
        drawn_texts = [str(call["text"]) for call in fake_annotator.text_calls]
        self.assertIn("person id:12 0.90", drawn_texts)
        self.assertEqual(fake_annotator.box_label_calls, [])

    def test_disabled_mode_preserves_legacy_result_plot_behavior(self) -> None:
        state = SharedDashboardState()
        state.tracking_stable_identity_visualization_enabled = False
        fake_result = _FakePlotResult()

        with patch.object(tracking_loop, "publish_tracking_frame_message", return_value={}):
            tracking_loop._publish_tracking_outputs(
                state=state,
                result=fake_result,
                now=300.0,
                tracks={"12": self._track_snapshot(track_id="12", timestamp=300.0)},
                tracking_candidates=[
                    _candidate(
                        detection_id="det-12",
                        yolo_track_id="12",
                        timestamp=300.0,
                        detection_reference="fdet-12",
                    )
                ],
                tracking_agent_observation=None,
                trail_length=20,
                detections_for_file=[],
                udp_socket=None,
                stable_identity_visualization_enabled=False,
                display_identity_by_track={},
            )

        self.assertEqual(fake_result.plot_calls[0], {})

    def test_stable_overlay_formats_provisional_confirmed_and_fallback_labels(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        fake_annotator = _FakeAnnotator()
        candidates = [
            _candidate(
                detection_id="det-1",
                yolo_track_id="1",
                timestamp=500.0,
                detection_reference="fdet-1",
            ),
            _candidate(
                detection_id="det-2",
                yolo_track_id="2",
                timestamp=500.0,
                detection_reference="fdet-2",
            ),
            _candidate(
                detection_id="det-3",
                yolo_track_id="3",
                timestamp=500.0,
                detection_reference="fdet-3",
            ),
        ]
        display_map = {
            "1": {
                "display_label": "app-1 (P)",
                "app_identity_id": "app-1",
                "status": "visible",
                "provisional": True,
            },
            "2": {
                "display_label": "app-2",
                "app_identity_id": "app-2",
                "status": "visible",
                "provisional": False,
            },
        }

        with patch.object(tracking_loop, "Annotator", return_value=fake_annotator):
            tracking_loop._apply_stable_identity_webcam_overlays(
                frame=frame,
                tracking_candidates=candidates,
                stable_identity_visualization_enabled=True,
                display_identity_by_track=display_map,
                result_names={0: "person"},
            )

        drawn_texts = [str(call["text"]) for call in fake_annotator.text_calls]
        self.assertIn("person app-1 (P) 0.90", drawn_texts)
        self.assertIn("person app-2 0.90", drawn_texts)
        self.assertIn("person id:3 0.90", drawn_texts)
        self.assertEqual(fake_annotator.box_label_calls, [])

    def test_stable_overlay_omits_confidence_when_missing(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        fake_annotator = _FakeAnnotator()
        candidate = _candidate(
            detection_id="det-1",
            yolo_track_id="1",
            timestamp=510.0,
            detection_reference="fdet-1",
        )
        candidate.detection_confidence = None

        with patch.object(tracking_loop, "Annotator", return_value=fake_annotator):
            tracking_loop._apply_stable_identity_webcam_overlays(
                frame=frame,
                tracking_candidates=[candidate],
                stable_identity_visualization_enabled=True,
                display_identity_by_track={
                    "1": {
                        "display_label": "app-1",
                        "app_identity_id": "app-1",
                        "status": "visible",
                        "provisional": False,
                    }
                },
                result_names={0: "person"},
            )

        drawn_texts = [str(call["text"]) for call in fake_annotator.text_calls]
        self.assertIn("person app-1", drawn_texts)
        self.assertNotIn("person app-1 0.90", drawn_texts)


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
