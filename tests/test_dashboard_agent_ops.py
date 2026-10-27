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
        dashboardReasoning._reset_reasoning_state()

    def test_run_reasoning_cycle_and_log_missing_radar_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_radar = Path(temp_dir) / "latest_radar_state.json"
            events_path = Path(temp_dir) / "agent_events.jsonl"

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_radar):
                with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                    result = dashboardAgentOps.run_reasoning_cycle_and_log()

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_run_reasoning_cycle_and_log_writes_events(self) -> None:
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

    def test_get_recent_agent_events_parses_and_drops_invalid(self) -> None:
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
        with self.assertRaises(ValueError):
            dashboardAgentOps.get_recent_agent_events(limit=0)
        with self.assertRaises(ValueError):
            dashboardAgentOps.run_reasoning_cycle_and_log(max_tracks=0)

    def test_summarize_agent_event_health_missing_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                result = dashboardAgentOps.summarize_agent_event_health(window_size=10)

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "agent_events_missing")

    def test_summarize_agent_event_health_computes_rates_and_percentiles(self) -> None:
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
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
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

    def test_summarize_agent_event_health_invalid_window_size(self) -> None:
        with self.assertRaises(ValueError):
            dashboardAgentOps.summarize_agent_event_health(window_size=0)

    def test_prune_agent_event_log_missing_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            with patch("tools.dashboardAgentOps.AGENT_EVENTS_PATH", events_path):
                result = dashboardAgentOps.prune_agent_event_log(max_events_keep=10)

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "agent_events_missing")

    def test_prune_agent_event_log_noop_when_under_limit(self) -> None:
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
        with self.assertRaises(ValueError):
            dashboardAgentOps.prune_agent_event_log(max_events_keep=-1)


if __name__ == "__main__":
    unittest.main()
