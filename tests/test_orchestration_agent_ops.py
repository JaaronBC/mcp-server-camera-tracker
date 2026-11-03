# Handles test orchestration agent ops behavior.
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agents.orchestration.tools import ops as orchestration_agent_ops
from agents.reasoning.tools import _shared as orchestration_reasoning
from agents.tracking.state_publisher import publish_tracking_radar_state
from agents.tracking.tracking_types import GroundPosition, RadarTrackSnapshot


class TestOrchestrationAgentOps(unittest.TestCase):
    def tearDown(self) -> None:
        # Performs tearDown logic.
        orchestration_reasoning._reset_reasoning_state()

    def test_run_reasoning_cycle_and_log_missing_radar_state(self) -> None:
        # Performs test run reasoning cycle and log missing radar state logic.
        with TemporaryDirectory() as temp_dir:
            missing_radar = Path(temp_dir) / "latest_radar_state.json"
            events_path = Path(temp_dir) / "agent_events.jsonl"

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", missing_radar):
                with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                    result = orchestration_agent_ops.run_reasoning_cycle_and_log()

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_run_reasoning_cycle_and_log_writes_events(self) -> None:
        # Performs test run reasoning cycle and log writes events logic.
        payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "confidence": 0.9,
                },
                {
                    "track_id": "track-2",
                    "timestamp": 1000.1,
                    "ground_position": {"x": 2.0, "z": 2.0},
                    "confidence": 0.8,
                },
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            events_path = Path(temp_dir) / "agent_events.jsonl"

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                    with patch(
                        "agents.reasoning.tools._shared.simulate_reasoning_assessment",
                        return_value={
                            "available": True,
                            "summary": {
                                "needs_more_observations": False,
                                "latest_track_snapshot": {
                                    "available": True,
                                    "track_id": "track-1",
                                    "ground_position": {"x": 1.0, "z": 2.0},
                                    "ground_coordinate_unit": "meters",
                                    "position_kind": "calibrated_ground_plane",
                                    "relative_depth": 0.55,
                                },
                                "tracking_agent_context": {
                                    "available": True,
                                    "vlm_model": "qwen-3.8-27b",
                                    "selected_updates": [
                                        {
                                            "detection_id": "track-1",
                                            "resolved_track_id": "track-1",
                                        }
                                    ],
                                },
                                "coordinate_context": {
                                    "available": True,
                                    "coordinate_system": "calibrated_ground_plane",
                                    "calibration_status": "loaded",
                                },
                            },
                            "assessment": {"assessment_label": "stable_motion"},
                            "evidence_request": None,
                            "handoff_validation": {
                                "status": "ok",
                                "error_code": None,
                                "error": None,
                            },
                            "agents": {
                                "observation": {
                                    "status": "ok",
                                    "model": "gpt-oss-120b",
                                    "latency_ms": 7.1,
                                    "error_code": None,
                                    "attempts": 1,
                                    "retry_count": 0,
                                },
                                "reasoning": {
                                    "status": "ok",
                                    "model": "gpt-oss-120b",
                                    "latency_ms": 9.4,
                                    "error_code": None,
                                    "attempts": 1,
                                    "retry_count": 0,
                                },
                            },
                            "error": None,
                        },
                    ):
                        result = orchestration_agent_ops.run_reasoning_cycle_and_log(max_tracks=1)

                    self.assertTrue(events_path.is_file())
                    lines = events_path.read_text(encoding="utf-8").splitlines()

        self.assertTrue(result["available"])
        self.assertIn("cycle_id", result["cycle"])
        self.assertIsInstance(result["cycle"]["cycle_id"], str)
        self.assertTrue(result["cycle"]["cycle_id"].startswith("cycle-"))
        self.assertEqual(result["cycle"]["evaluated_tracks"], 1)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(len(lines), 1)
        event_payload = json.loads(lines[0])
        self.assertEqual(event_payload["event_type"], "reasoning_cycle")
        self.assertEqual(event_payload["cycle_id"], result["cycle"]["cycle_id"])
        self.assertIsInstance(event_payload["task_id"], str)
        self.assertTrue(event_payload["task_id"].startswith(f"{result['cycle']['cycle_id']}:track-"))
        self.assertIsNone(event_payload["parent_task_id"])
        self.assertIn("latency_ms", event_payload)
        self.assertEqual(event_payload["observation_agent_model"], "gpt-oss-120b")
        self.assertEqual(event_payload["reasoning_agent_model"], "gpt-oss-120b")
        self.assertEqual(event_payload["observation_agent_error_code"], None)
        self.assertEqual(event_payload["reasoning_agent_error_code"], None)
        self.assertEqual(event_payload["observation_agent_retry_count"], 0)
        self.assertEqual(event_payload["reasoning_agent_retry_count"], 0)
        self.assertEqual(event_payload["handoff_validation_status"], "ok")
        self.assertEqual(event_payload["handoff_validation_error_code"], None)
        self.assertEqual(event_payload["handoff_validation_error"], None)
        self.assertEqual(event_payload["handoff_quality"], "healthy")
        self.assertEqual(event_payload["agent_error_digest"], "obs(none),rea(none)")
        self.assertEqual(event_payload["conflicts"], [])
        self.assertEqual(event_payload["limitations"], [])
        self.assertIn("evidence_trace", event_payload)
        self.assertTrue(event_payload["evidence_trace"]["latest_track_snapshot"]["available"])
        self.assertEqual(
            event_payload["evidence_trace"]["tracking_agent_context"]["vlm_model"],
            "qwen-3.8-27b",
        )
        self.assertEqual(
            event_payload["evidence_trace"]["coordinate_context"]["coordinate_system"],
            "calibrated_ground_plane",
        )

    def test_run_reasoning_cycle_and_log_survives_event_log_write_error(self) -> None:
        # Performs test run reasoning cycle and log survives event log write error logic.
        payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                with patch(
                    "agents.reasoning.tools._shared.simulate_reasoning_assessment",
                    return_value={
                        "available": True,
                        "summary": {"needs_more_observations": False},
                        "assessment": {"assessment_label": "stable_motion"},
                        "evidence_request": None,
                        "handoff_validation": {
                            "status": "ok",
                            "error_code": None,
                            "error": None,
                        },
                        "agents": {
                            "observation": {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 7.0},
                            "reasoning": {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 9.0},
                        },
                        "error": None,
                    },
                ):
                    with patch(
                        "agents.orchestration.tools.ops._append_event",
                        side_effect=OSError("disk full"),
                    ):
                        result = orchestration_agent_ops.run_reasoning_cycle_and_log(max_tracks=1)

        self.assertTrue(result["available"])
        self.assertEqual(result["cycle"]["success_count"], 1)
        self.assertFalse(result["event_log_writable"])
        self.assertEqual(result["event_log_error_count"], 1)
        self.assertIn("OSError:disk full", result["event_log_last_error"])
        self.assertEqual(len(result["events"]), 1)

    def test_run_reasoning_cycle_and_log_emits_evidence_followup_task_links(self) -> None:
        # Performs test run reasoning cycle and log emits evidence followup task links logic.
        payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            events_path = Path(temp_dir) / "agent_events.jsonl"

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                    with patch(
                        "agents.reasoning.tools._shared.simulate_reasoning_assessment",
                        return_value={
                            "available": True,
                            "summary": {"needs_more_observations": True},
                            "assessment": {"assessment_label": "requires_more_observation"},
                            "evidence_request": {
                                "track_id": "track-1",
                                "reason": "llm_requested_more_observations",
                                "requested_at": 1000.1,
                                "request_index": 2,
                            },
                            "handoff_validation": {
                                "status": "ok",
                                "error_code": None,
                                "error": None,
                            },
                            "agents": {
                                "observation": {
                                    "status": "ok",
                                    "model": "gpt-oss-120b",
                                    "latency_ms": 7.1,
                                    "error_code": None,
                                    "attempts": 1,
                                    "retry_count": 0,
                                },
                                "reasoning": {
                                    "status": "ok",
                                    "model": "gpt-oss-120b",
                                    "latency_ms": 9.4,
                                    "error_code": None,
                                    "attempts": 1,
                                    "retry_count": 0,
                                },
                            },
                            "error": None,
                        },
                    ):
                        result = orchestration_agent_ops.run_reasoning_cycle_and_log(max_tracks=1)

        self.assertTrue(result["available"])
        self.assertEqual(len(result["events"]), 1)
        event_payload = result["events"][0]
        self.assertIn("evidence_followup_task_id", event_payload)
        self.assertIn("evidence_followup_parent_task_id", event_payload)
        self.assertIn("evidence_followup_previous_task_id", event_payload)
        self.assertEqual(
            event_payload["evidence_followup_parent_task_id"],
            event_payload["task_id"],
        )
        self.assertTrue(event_payload["evidence_followup_task_id"].endswith(":evidence-2"))
        self.assertTrue(event_payload["evidence_followup_previous_task_id"].endswith(":evidence-1"))

    def test_run_reasoning_cycle_and_log_emits_conflicts_and_limitations(self) -> None:
        # Performs test run reasoning cycle and log emits conflicts and limitations logic.
        payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                with patch(
                    "agents.reasoning.tools._shared.simulate_reasoning_assessment",
                    return_value={
                        "available": False,
                        "summary": {
                            "needs_more_observations": True,
                            "uncertainties": ["insufficient_history"],
                        },
                        "assessment": {
                            "assessment_label": "requires_more_observation",
                            "uncertainties": ["conflicting_inputs"],
                        },
                        "evidence_request": None,
                        "handoff_validation": {
                            "status": "error",
                            "error_code": "invalid_handoff_payload",
                            "error": "missing_supporting_observation_ids",
                        },
                        "agents": {
                            "observation": {
                                "status": "ok",
                                "model": "gpt-oss-120b",
                                "latency_ms": 7.1,
                                "error_code": "timeout",
                                "attempts": 1,
                                "retry_count": 0,
                            },
                            "reasoning": {
                                "status": "error",
                                "model": "gpt-oss-120b",
                                "latency_ms": 9.4,
                                "error_code": "invalid_response_format",
                                "attempts": 1,
                                "retry_count": 0,
                            },
                        },
                        "error": "radar_state_stale",
                    },
                ):
                    result = orchestration_agent_ops.run_reasoning_cycle_and_log(max_tracks=1)

        self.assertTrue(result["available"])
        self.assertEqual(len(result["events"]), 1)
        event_payload = result["events"][0]
        self.assertIn("handoff_validation_failed", event_payload["conflicts"])
        self.assertIn(
            "handoff_validation_error:invalid_handoff_payload",
            event_payload["conflicts"],
        )
        self.assertIn(
            "agent_status_mismatch:obs=ok,rea=error",
            event_payload["conflicts"],
        )
        self.assertIn(
            "agent_error_code_divergence:obs=timeout,rea=invalid_response_format",
            event_payload["conflicts"],
        )
        self.assertIn("cycle_error:radar_state_stale", event_payload["limitations"])
        self.assertIn("bounded_out", event_payload["limitations"])
        self.assertIn(
            "handoff_validation_error:invalid_handoff_payload",
            event_payload["limitations"],
        )
        self.assertIn(
            "summary_uncertainty:insufficient_history",
            event_payload["limitations"],
        )
        self.assertIn(
            "assessment_uncertainty:conflicting_inputs",
            event_payload["limitations"],
        )

    def test_get_recent_agent_events_parses_and_drops_invalid(self) -> None:
        # Performs test get recent agent events parses and drops invalid logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            events_path.write_text(
                "\n".join(
                    [
                        json.dumps({"event_type": "a", "index": 1}),
                        "not-json",
                        json.dumps({"event_type": "b", "index": 2}),
                    ]
                ),
                encoding="utf-8",
            )

            with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                result = orchestration_agent_ops.get_recent_agent_events(limit=10)

        self.assertTrue(result["available"])
        self.assertEqual(result["returned_events"], 2)
        self.assertEqual(result["dropped_lines"], 1)
        self.assertEqual(result["events"][0]["event_type"], "a")
        self.assertEqual(result["events"][1]["event_type"], "b")

    def test_invalid_limits_raise(self) -> None:
        # Performs test invalid limits raise logic.
        with self.assertRaises(ValueError):
            orchestration_agent_ops.get_recent_agent_events(limit=0)
        with self.assertRaises(ValueError):
            orchestration_agent_ops.run_reasoning_cycle_and_log(max_tracks=0)

    def test_summarize_agent_event_health_missing_file(self) -> None:
        # Performs test summarize agent event health missing file logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                result = orchestration_agent_ops.summarize_agent_event_health(window_size=10)

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "agent_events_missing")

    def test_summarize_agent_event_health_computes_rates_and_percentiles(self) -> None:
        # Performs test summarize agent event health computes rates and percentiles logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            events_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_type": "reasoning_cycle",
                                "timestamp": 100.0,
                                "status": "ok",
                                "latency_ms": 10.0,
                                "assessment_label": "stable_motion",
                                "bounded_out": False,
                            }
                        ),
                        json.dumps(
                            {
                                "event_type": "reasoning_cycle",
                                "timestamp": 101.0,
                                "status": "ok",
                                "latency_ms": 20.0,
                                "assessment_label": "requires_review",
                                "bounded_out": True,
                            }
                        ),
                        "bad-json-line",
                        json.dumps(
                            {
                                "event_type": "reasoning_cycle",
                                "timestamp": 102.0,
                                "status": "error",
                                "latency_ms": 40.0,
                                "assessment_label": "requires_review",
                                "bounded_out": False,
                                "cycle_error_code": "radar_state_stale",
                                "observation_agent_error_code": "timeout",
                                "reasoning_agent_error_code": "invalid_response_format",
                                "handoff_validation_status": "error",
                                "handoff_validation_error_code": "invalid_handoff_payload",
                                "conflicts": [
                                    "handoff_validation_failed",
                                    "agent_status_mismatch:obs=ok,rea=error",
                                ],
                                "limitations": [
                                    "cycle_error:radar_state_stale",
                                    "assessment_uncertainty:conflicting_inputs",
                                    "bounded_out",
                                ],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                with patch("agents.orchestration.tools.ops.time.time", return_value=120.0):
                    result = orchestration_agent_ops.summarize_agent_event_health(window_size=10)

        self.assertTrue(result["available"])
        self.assertEqual(result["returned_events"], 3)
        self.assertEqual(result["dropped_lines"], 1)
        self.assertEqual(result["status"]["ok_count"], 2)
        self.assertEqual(result["status"]["error_count"], 1)
        self.assertAlmostEqual(result["status"]["success_rate"], 2 / 3, places=6)
        self.assertAlmostEqual(result["status"]["bounded_out_rate"], 1 / 3, places=6)
        self.assertAlmostEqual(result["latency"]["p50_ms"], 20.0, places=6)
        self.assertAlmostEqual(result["latency"]["p95_ms"], 38.0, places=6)
        self.assertEqual(result["latency"]["max_ms"], 40.0)
        self.assertEqual(result["assessment_label_counts"]["stable_motion"], 1)
        self.assertEqual(result["assessment_label_counts"]["requires_review"], 2)
        self.assertEqual(result["agent_error_code_counts"]["observation"]["timeout"], 1)
        self.assertEqual(result["agent_error_code_counts"]["reasoning"]["invalid_response_format"], 1)
        self.assertEqual(result["agent_error_code_counts"]["cycle"]["radar_state_stale"], 1)
        self.assertEqual(result["agent_error_trends"]["observation"]["top_error_code"], "timeout")
        self.assertEqual(result["agent_error_trends"]["observation"]["top_error_count"], 1)
        self.assertAlmostEqual(result["agent_error_trends"]["observation"]["top_error_rate"], 1 / 3, places=6)
        self.assertEqual(
            result["agent_error_trends"]["reasoning"]["top_error_code"],
            "invalid_response_format",
        )
        self.assertEqual(result["agent_error_trends"]["reasoning"]["top_error_count"], 1)
        self.assertAlmostEqual(result["agent_error_trends"]["reasoning"]["top_error_rate"], 1 / 3, places=6)
        self.assertEqual(result["agent_error_trends"]["cycle"]["top_error_code"], "radar_state_stale")
        self.assertEqual(result["agent_error_trends"]["cycle"]["top_error_count"], 1)
        self.assertAlmostEqual(result["agent_error_trends"]["cycle"]["top_error_rate"], 1 / 3, places=6)
        self.assertEqual(result["handoff_validation"]["status_counts"]["error"], 1)
        self.assertEqual(
            result["handoff_validation"]["error_code_counts"]["invalid_handoff_payload"],
            1,
        )
        self.assertEqual(result["handoff_validation"]["quality_counts"]["healthy"], 2)
        self.assertEqual(result["handoff_validation"]["quality_counts"]["degraded"], 1)
        self.assertEqual(result["handoff_validation"]["degraded_count"], 1)
        self.assertAlmostEqual(result["handoff_validation"]["degraded_rate"], 1 / 3, places=6)
        self.assertEqual(result["collaboration_signals"]["conflict_event_count"], 1)
        self.assertAlmostEqual(result["collaboration_signals"]["conflict_event_rate"], 1 / 3, places=6)
        self.assertEqual(result["collaboration_signals"]["conflict_item_count"], 2)
        self.assertEqual(result["collaboration_signals"]["limitation_event_count"], 1)
        self.assertAlmostEqual(result["collaboration_signals"]["limitation_event_rate"], 1 / 3, places=6)
        self.assertEqual(result["collaboration_signals"]["limitation_item_count"], 3)
        self.assertEqual(
            result["agent_error_trends"]["handoff_validation"]["top_error_code"],
            "invalid_handoff_payload",
        )
        self.assertEqual(result["agent_error_trends"]["handoff_validation"]["top_error_count"], 1)
        self.assertAlmostEqual(
            result["agent_error_trends"]["handoff_validation"]["top_error_rate"],
            1 / 3,
            places=6,
        )
        self.assertTrue(result["freshness"]["is_fresh"])
        self.assertAlmostEqual(result["freshness"]["age_seconds"], 18.0, places=6)
        self.assertEqual(result["freshness"]["threshold_seconds"], 30.0)

    def test_summarize_agent_event_health_marks_stale_when_old(self) -> None:
        # Performs test summarize agent event health marks stale when old logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            events_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_type": "reasoning_cycle",
                                "timestamp": 100.0,
                                "status": "ok",
                                "latency_ms": 10.0,
                                "assessment_label": "stable_motion",
                                "bounded_out": False,
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                with patch("agents.orchestration.tools.ops.time.time", return_value=200.0):
                    result = orchestration_agent_ops.summarize_agent_event_health(window_size=10)

        self.assertTrue(result["available"])
        self.assertFalse(result["freshness"]["is_fresh"])
        self.assertAlmostEqual(result["freshness"]["age_seconds"], 100.0, places=6)

    def test_format_agent_error_digest_from_codes(self) -> None:
        # Performs test format agent error digest from codes logic.
        self.assertEqual(
            orchestration_agent_ops._format_agent_error_digest("timeout", "invalid_response_format"),
            "obs(timeout),rea(invalid_response_format)",
        )
        self.assertEqual(
            orchestration_agent_ops._format_agent_error_digest(None, ""),
            "obs(none),rea(none)",
        )

    def test_derive_handoff_quality(self) -> None:
        # Performs test derive handoff quality logic.
        self.assertEqual(orchestration_agent_ops._derive_handoff_quality("ok", None), "healthy")
        self.assertEqual(orchestration_agent_ops._derive_handoff_quality("error", None), "degraded")
        self.assertEqual(
            orchestration_agent_ops._derive_handoff_quality(None, "invalid_handoff_payload"),
            "degraded",
        )

    def test_extract_evidence_trace_bounds_selected_updates_and_lists(self) -> None:
        # Performs test extract evidence trace bounds selected updates and lists logic.
        summary = {
            "latest_track_snapshot": {
                "available": True,
                "track_id": "track-1",
                "ground_position": {"x": 1.0, "z": 2.0, "y": 3.0},
                "quality_flags": ["a", "b", "c", "d", "e", "f"],
            },
            "tracking_agent_context": {
                "available": True,
                "source_agent": "visual_tracking",
                "vlm_model": "qwen-3.8-27b",
                "uncertainties": ["u1", "u2", "u3", "u4", "u5", "u6"],
                "selected_updates": [
                    {"detection_id": "d1", "resolved_track_id": "t1", "supporting_evidence": ["e1", "e2", "e3", "e4", "e5", "e6"]},
                    {"detection_id": "d2", "resolved_track_id": "t2"},
                    {"detection_id": "d3", "resolved_track_id": "t3"},
                    {"detection_id": "d4", "resolved_track_id": "t4"},
                ],
            },
            "coordinate_context": {
                "available": True,
                "coordinate_system": "calibrated_ground_plane",
                "calibration_status": "loaded",
                "camera_status": "connected",
                "detector_status": "running",
            },
        }

        trace = orchestration_agent_ops._extract_evidence_trace(summary)

        self.assertEqual(trace["latest_track_snapshot"]["ground_position"], {"x": 1.0, "z": 2.0})
        self.assertEqual(len(trace["latest_track_snapshot"]["quality_flags"]), 5)
        self.assertEqual(len(trace["tracking_agent_context"]["uncertainties"]), 5)
        self.assertEqual(len(trace["tracking_agent_context"]["selected_updates"]), 3)
        self.assertEqual(
            len(trace["tracking_agent_context"]["selected_updates"][0]["supporting_evidence"]),
            5,
        )

    def test_summarize_agent_event_health_invalid_window_size(self) -> None:
        # Performs test summarize agent event health invalid window size logic.
        with self.assertRaises(ValueError):
            orchestration_agent_ops.summarize_agent_event_health(window_size=0)

    def test_prune_agent_event_log_missing_file(self) -> None:
        # Performs test prune agent event log missing file logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                result = orchestration_agent_ops.prune_agent_event_log(max_events_keep=10)

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "agent_events_missing")

    def test_prune_agent_event_log_noop_when_under_limit(self) -> None:
        # Performs test prune agent event log noop when under limit logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            events_path.write_text(
                "\n".join([json.dumps({"index": 1}), json.dumps({"index": 2})]) + "\n",
                encoding="utf-8",
            )

            with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                result = orchestration_agent_ops.prune_agent_event_log(max_events_keep=5)

        self.assertTrue(result["available"])
        self.assertFalse(result["pruned"])
        self.assertEqual(result["removed_lines"], 0)

    def test_prune_agent_event_log_prunes_and_archives(self) -> None:
        # Performs test prune agent event log prunes and archives logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            events_path.write_text(
                "\n".join(
                    [
                        json.dumps({"index": 1}),
                        json.dumps({"index": 2}),
                        json.dumps({"index": 3}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                with patch("agents.orchestration.tools.ops.time.strftime", return_value="20261026_120000"):
                    result = orchestration_agent_ops.prune_agent_event_log(max_events_keep=2)

            remaining_lines = events_path.read_text(encoding="utf-8").splitlines()
            archive_path = result["archive_path"]
            self.assertIsNotNone(archive_path)
            archive_lines = Path(archive_path).read_text(encoding="utf-8").splitlines()

        self.assertTrue(result["available"])
        self.assertTrue(result["pruned"])
        self.assertEqual(result["removed_lines"], 1)
        self.assertEqual(result["total_lines_after"], 2)
        self.assertEqual(len(remaining_lines), 2)
        self.assertIn('"index": 2', remaining_lines[0])
        self.assertIn('"index": 3', remaining_lines[1])

        self.assertEqual(len(archive_lines), 1)
        self.assertIn('"index": 1', archive_lines[0])

    def test_prune_agent_event_log_invalid_max_events_keep(self) -> None:
        # Performs test prune agent event log invalid max events keep logic.
        with self.assertRaises(ValueError):
            orchestration_agent_ops.prune_agent_event_log(max_events_keep=-1)


    def test_production_cycle_reads_fresh_published_real_radar_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            radar_path = temp_root / "latest_radar_state.json"
            history_path = temp_root / "radar_depth_coverage_history.jsonl"
            events_path = temp_root / "agent_events.jsonl"

            now = 2000.0
            real_track_id = "track-real-1"
            real_track = RadarTrackSnapshot(
                track_id=real_track_id,
                timestamp=now,
                pixel_position=None,
                ground_position=GroundPosition(x=1.2, z=3.4),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=["position_kind:relative_ground_estimate"],
                confidence=0.95,
                relative_depth=0.42,
            )

            publish_tracking_radar_state(
                radar_state_path=radar_path,
                radar_depth_coverage_history_path=history_path,
                now=now,
                coordinate_system="calibrated_ground_plane",
                camera_status="connected",
                detector_status="running",
                calibration_status="loaded",
                tracking_agent_status={},
                last_observation_timestamp=now,
                tracking_agent_observation=None,
                tracks=[real_track],
            )

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                with patch.dict("os.environ", {"DASHBOARD_RADAR_STATE_MAX_AGE_SECONDS": "10"}, clear=False):
                    with patch("agents.orchestration.tools.ops.AGENT_EVENTS_PATH", events_path):
                        with patch("agents.reasoning.tools._shared.time.time", return_value=now + 0.5):
                            with patch(
                                "agents.reasoning.tools._shared._call_observation_agent_with_retry",
                                return_value=(
                                    None,
                                    {
                                        "status": "error",
                                        "model": "gpt-oss-120b",
                                        "latency_ms": 0.0,
                                        "error": "provider_disabled_for_test",
                                        "error_code": "provider_disabled_for_test",
                                        "attempts": 0,
                                        "retry_count": 0,
                                    },
                                ),
                            ):
                                with patch(
                                    "agents.reasoning.tools._shared._call_reasoning_agent_with_retry",
                                    return_value=(
                                        None,
                                        {
                                            "status": "error",
                                            "model": "gpt-oss-120b",
                                            "latency_ms": 0.0,
                                            "error": "provider_disabled_for_test",
                                            "error_code": "provider_disabled_for_test",
                                            "attempts": 0,
                                            "retry_count": 0,
                                        },
                                    ),
                                ):
                                    result = orchestration_agent_ops.run_reasoning_cycle_and_log(
                                        max_tracks=5,
                                        event_origin="production_monitor",
                                    )

        self.assertTrue(result["available"])
        self.assertEqual(result["cycle"]["evaluated_tracks"], 1)
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["event_origin"], "production_monitor")
        self.assertEqual(event["track_id"], real_track_id)

        evidence_trace = event.get("evidence_trace", {})
        latest_snapshot = evidence_trace.get("latest_track_snapshot", {})
        coordinate_context = evidence_trace.get("coordinate_context", {})
        self.assertEqual(latest_snapshot.get("track_id"), real_track_id)
        self.assertEqual(latest_snapshot.get("timestamp"), now)
        self.assertEqual(coordinate_context.get("coordinate_system"), "calibrated_ground_plane")
        self.assertNotEqual(coordinate_context.get("coordinate_system"), "simulation_fallback")


if __name__ == "__main__":
    unittest.main()
