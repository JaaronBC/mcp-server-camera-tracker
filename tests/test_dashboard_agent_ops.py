# Handles test dashboard agent ops behavior.
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import dashboardAgentOps
from tools import dashboardReasoning


class TestDashboardAgentOps(unittest.TestCase):
    def tearDown(self) -> None:
        # Performs tearDown logic.
        dashboardReasoning._reset_reasoning_state()

    def test_run_reasoning_cycle_and_log_missing_radar_state(self) -> None:
        # Performs test run reasoning cycle and log missing radar state logic.
        with TemporaryDirectory() as temp_dir:
            missing_radar = Path(temp_dir) / "latest_radar_state.json"
            events_path = Path(temp_dir) / "agent_events.jsonl"

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_radar):
                with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                    result = dashboardAgentOps.run_reasoning_cycle_and_log()

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

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                    with patch(
                        "tools.dashboardReasoning.simulate_reasoning_assessment",
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
                        result = dashboardAgentOps.run_reasoning_cycle_and_log(max_tracks=1)

                    self.assertTrue(events_path.is_file())
                    lines = events_path.read_text(encoding="utf-8").splitlines()

        self.assertTrue(result["available"])
        self.assertEqual(result["cycle"]["evaluated_tracks"], 1)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(len(lines), 1)
        event_payload = json.loads(lines[0])
        self.assertEqual(event_payload["event_type"], "reasoning_cycle")
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

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                with patch(
                    "tools.dashboardReasoning.simulate_reasoning_assessment",
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
                        "tools.dashboardAgentOps._append_event",
                        side_effect=OSError("disk full"),
                    ):
                        result = dashboardAgentOps.run_reasoning_cycle_and_log(max_tracks=1)

        self.assertTrue(result["available"])
        self.assertEqual(result["cycle"]["success_count"], 1)
        self.assertFalse(result["event_log_writable"])
        self.assertEqual(result["event_log_error_count"], 1)
        self.assertIn("OSError:disk full", result["event_log_last_error"])
        self.assertEqual(len(result["events"]), 1)

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

            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                result = dashboardAgentOps.get_recent_agent_events(limit=10)

        self.assertTrue(result["available"])
        self.assertEqual(result["returned_events"], 2)
        self.assertEqual(result["dropped_lines"], 1)
        self.assertEqual(result["events"][0]["event_type"], "a")
        self.assertEqual(result["events"][1]["event_type"], "b")

    def test_invalid_limits_raise(self) -> None:
        # Performs test invalid limits raise logic.
        with self.assertRaises(ValueError):
            dashboardAgentOps.get_recent_agent_events(limit=0)
        with self.assertRaises(ValueError):
            dashboardAgentOps.run_reasoning_cycle_and_log(max_tracks=0)

    def test_summarize_agent_event_health_missing_file(self) -> None:
        # Performs test summarize agent event health missing file logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                result = dashboardAgentOps.summarize_agent_event_health(window_size=10)

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
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                with patch("tools.dashboardAgentOps.time.time", return_value=120.0):
                    result = dashboardAgentOps.summarize_agent_event_health(window_size=10)

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

            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                with patch("tools.dashboardAgentOps.time.time", return_value=200.0):
                    result = dashboardAgentOps.summarize_agent_event_health(window_size=10)

        self.assertTrue(result["available"])
        self.assertFalse(result["freshness"]["is_fresh"])
        self.assertAlmostEqual(result["freshness"]["age_seconds"], 100.0, places=6)

    def test_format_agent_error_digest_from_codes(self) -> None:
        # Performs test format agent error digest from codes logic.
        self.assertEqual(
            dashboardAgentOps._format_agent_error_digest("timeout", "invalid_response_format"),
            "obs(timeout),rea(invalid_response_format)",
        )
        self.assertEqual(
            dashboardAgentOps._format_agent_error_digest(None, ""),
            "obs(none),rea(none)",
        )

    def test_derive_handoff_quality(self) -> None:
        # Performs test derive handoff quality logic.
        self.assertEqual(dashboardAgentOps._derive_handoff_quality("ok", None), "healthy")
        self.assertEqual(dashboardAgentOps._derive_handoff_quality("error", None), "degraded")
        self.assertEqual(
            dashboardAgentOps._derive_handoff_quality(None, "invalid_handoff_payload"),
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

        trace = dashboardAgentOps._extract_evidence_trace(summary)

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
            dashboardAgentOps.summarize_agent_event_health(window_size=0)

    def test_prune_agent_event_log_missing_file(self) -> None:
        # Performs test prune agent event log missing file logic.
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                result = dashboardAgentOps.prune_agent_event_log(max_events_keep=10)

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

            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                result = dashboardAgentOps.prune_agent_event_log(max_events_keep=5)

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

            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                with patch("tools.dashboardAgentOps.time.strftime", return_value="20261026_120000"):
                    result = dashboardAgentOps.prune_agent_event_log(max_events_keep=2)

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
            dashboardAgentOps.prune_agent_event_log(max_events_keep=-1)


if __name__ == "__main__":
    unittest.main()
