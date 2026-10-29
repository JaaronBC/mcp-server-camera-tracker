# Handles test dashboard metrics behavior.
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.dashboard import metrics as dashboardMetrics


class TestDashboardMetrics(unittest.TestCase):
    def test_metrics_reports_unavailable_when_snapshots_missing(self) -> None:
        # Performs test metrics reports unavailable when snapshots missing logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", captures_dir / "latest_radar_state.json"):
                    with patch("tools.dashboard.metrics.TRACKING_FRAME_PATH", captures_dir / "latest_tracking_frame.json"):
                        with patch("tools.dashboard.metrics.time.time", return_value=100.0):
                            result = dashboardMetrics.dashboard_latency_continuity_metrics()

        self.assertFalse(result["available"])
        self.assertIn("radar_state_missing", result["errors"])
        self.assertIn("tracking_frame_missing", result["errors"])
        self.assertFalse(result["continuity"]["available"])

    def test_metrics_latency_and_counts_from_current_snapshots(self) -> None:
        # Performs test metrics latency and counts from current snapshots logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)
            radar_path = captures_dir / "latest_radar_state.json"
            tracking_path = captures_dir / "latest_tracking_frame.json"

            radar_path.write_text(
                json.dumps(
                    {
                        "timestamp": 95.0,
                        "last_observation_timestamp": 94.0,
                        "tracks": [{"track_id": "a"}, {"track_id": "b"}],
                    }
                ),
                encoding="utf-8",
            )
            tracking_path.write_text(
                json.dumps(
                    {
                        "time": 96.0,
                        "detections": [{"id": 1}, {"id": 2}, {"id": 3}],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", radar_path):
                    with patch("tools.dashboard.metrics.TRACKING_FRAME_PATH", tracking_path):
                        with patch("tools.dashboard.metrics.time.time", return_value=100.0):
                            result = dashboardMetrics.dashboard_latency_continuity_metrics()

        self.assertTrue(result["available"])
        self.assertEqual(result["counts"]["active_track_count"], 2)
        self.assertEqual(result["counts"]["detection_count"], 3)
        self.assertAlmostEqual(result["latency"]["state_update_latency_seconds"], 5.0, places=6)
        self.assertAlmostEqual(result["latency"]["tracking_update_latency_seconds"], 4.0, places=6)
        self.assertAlmostEqual(result["latency"]["end_to_end_delay_seconds"], 6.0, places=6)

    def test_metrics_continuity_from_latest_tracking_history(self) -> None:
        # Performs test metrics continuity from latest tracking history logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)
            (captures_dir / "latest_radar_state.json").write_text(
                json.dumps({"timestamp": 10.0, "tracks": []}),
                encoding="utf-8",
            )
            (captures_dir / "latest_tracking_frame.json").write_text(
                json.dumps({"time": 10.0, "detections": []}),
                encoding="utf-8",
            )

            history_path = captures_dir / "tracking_20261020_000000_000001.json"
            history_path.write_text(
                json.dumps(
                    {
                        "1": [
                            {"frame": 1, "elapsed_seconds": 0.0, "center_x": 10.0, "center_y": 10.0},
                            {"frame": 2, "elapsed_seconds": 0.1, "center_x": 12.0, "center_y": 10.0},
                            {"frame": 4, "elapsed_seconds": 0.2, "center_x": 15.0, "center_y": 10.0},
                        ],
                        "2": [{"frame": 10, "elapsed_seconds": 0.0, "center_x": 1.0, "center_y": 1.0}],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", captures_dir / "latest_radar_state.json"):
                    with patch("tools.dashboard.metrics.TRACKING_FRAME_PATH", captures_dir / "latest_tracking_frame.json"):
                        with patch("tools.dashboard.metrics.time.time", return_value=20.0):
                            result = dashboardMetrics.dashboard_latency_continuity_metrics()

        continuity = result["continuity"]
        self.assertTrue(continuity["available"])
        self.assertEqual(continuity["evaluated_tracks"], 2)
        self.assertEqual(continuity["transitions"], 2)
        self.assertEqual(continuity["gap_transitions"], 1)
        self.assertAlmostEqual(continuity["continuity_ratio"], 0.5, places=6)

    def test_depth_coverage_metrics_reports_unavailable_when_files_missing(self) -> None:
        # Performs test depth coverage metrics reports unavailable when files missing logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", captures_dir / "latest_radar_state.json"):
                    with patch(
                        "tools.dashboard.metrics.DEPTH_COVERAGE_HISTORY_PATH",
                        captures_dir / "radar_depth_coverage_history.jsonl",
                    ):
                        with patch("tools.dashboard.metrics.time.time", return_value=100.0):
                            result = dashboardMetrics.dashboard_depth_coverage_metrics()

        self.assertFalse(result["available"])
        self.assertIn("radar_state_missing", result["errors"])
        self.assertIn("depth_coverage_history_missing", result["errors"])
        self.assertFalse(result["trend"]["available"])

    def test_depth_coverage_metrics_uses_latest_radar_state(self) -> None:
        # Performs test depth coverage metrics uses latest radar state logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)
            radar_path = captures_dir / "latest_radar_state.json"
            history_path = captures_dir / "radar_depth_coverage_history.jsonl"

            radar_path.write_text(
                json.dumps(
                    {
                        "timestamp": 95.0,
                        "relative_depth_coverage": {
                            "tracks_with_depth": 3,
                            "total_tracks": 4,
                            "coverage_ratio": 0.75,
                            "coverage_percent": 75.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            history_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": 90.0,
                                "coverage": {
                                    "tracks_with_depth": 1,
                                    "total_tracks": 4,
                                    "coverage_ratio": 0.25,
                                    "coverage_percent": 25.0,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": 95.0,
                                "coverage": {
                                    "tracks_with_depth": 3,
                                    "total_tracks": 4,
                                    "coverage_ratio": 0.75,
                                    "coverage_percent": 75.0,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", radar_path):
                    with patch("tools.dashboard.metrics.DEPTH_COVERAGE_HISTORY_PATH", history_path):
                        with patch("tools.dashboard.metrics.time.time", return_value=100.0):
                            result = dashboardMetrics.dashboard_depth_coverage_metrics(sample_limit=10)

        self.assertTrue(result["available"])
        self.assertEqual(result["latest_coverage"]["source"], "latest_radar_state")
        self.assertEqual(result["latest_coverage"]["coverage_percent"], 75.0)
        self.assertTrue(result["trend"]["available"])
        self.assertEqual(result["trend"]["direction"], "improving")
        self.assertEqual(result["trend"]["delta_coverage_percent"], 50.0)

    def test_depth_coverage_metrics_falls_back_to_history_when_radar_has_no_coverage(self) -> None:
        # Performs test depth coverage metrics falls back to history when radar has no coverage logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)
            radar_path = captures_dir / "latest_radar_state.json"
            history_path = captures_dir / "radar_depth_coverage_history.jsonl"

            radar_path.write_text(json.dumps({"timestamp": 50.0}), encoding="utf-8")
            history_path.write_text(
                json.dumps(
                    {
                        "timestamp": 50.0,
                        "coverage": {
                            "tracks_with_depth": 2,
                            "total_tracks": 5,
                            "coverage_ratio": 0.4,
                            "coverage_percent": 40.0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", radar_path):
                    with patch("tools.dashboard.metrics.DEPTH_COVERAGE_HISTORY_PATH", history_path):
                        with patch("tools.dashboard.metrics.time.time", return_value=60.0):
                            result = dashboardMetrics.dashboard_depth_coverage_metrics(sample_limit=10)

        self.assertTrue(result["available"])
        self.assertEqual(result["latest_coverage"]["source"], "depth_coverage_history")
        self.assertEqual(result["latest_coverage"]["coverage_percent"], 40.0)
        self.assertFalse(result["trend"]["available"])

    def test_tracking_quality_report_healthy(self) -> None:
        # Performs test tracking quality report healthy logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)
            radar_path = captures_dir / "latest_radar_state.json"
            tracking_path = captures_dir / "latest_tracking_frame.json"
            depth_history_path = captures_dir / "radar_depth_coverage_history.jsonl"
            event_log_path = captures_dir / "agent_events.jsonl"
            tracking_history_path = captures_dir / "tracking_20261020_000000_000001.json"

            radar_path.write_text(
                json.dumps(
                    {
                        "timestamp": 99.5,
                        "last_observation_timestamp": 99.4,
                        "active_track_count": 2,
                        "relative_depth_coverage": {
                            "tracks_with_depth": 2,
                            "total_tracks": 2,
                            "coverage_ratio": 1.0,
                            "coverage_percent": 100.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            tracking_path.write_text(json.dumps({"time": 99.6, "detections": [{"id": 1}]}), encoding="utf-8")
            depth_history_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": 95.0,
                                "coverage": {
                                    "tracks_with_depth": 1,
                                    "total_tracks": 2,
                                    "coverage_ratio": 0.5,
                                    "coverage_percent": 50.0,
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": 99.0,
                                "coverage": {
                                    "tracks_with_depth": 2,
                                    "total_tracks": 2,
                                    "coverage_ratio": 1.0,
                                    "coverage_percent": 100.0,
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            event_log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_type": "reasoning_cycle",
                                "timestamp": 99.0,
                                "track_id": "1",
                                "status": "ok",
                                "latency_ms": 4.0,
                                "bounded_out": False,
                            }
                        ),
                        json.dumps(
                            {
                                "event_type": "reasoning_cycle",
                                "timestamp": 99.1,
                                "track_id": "2",
                                "status": "ok",
                                "latency_ms": 6.0,
                                "bounded_out": False,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            tracking_history_path.write_text(
                json.dumps(
                    {
                        "1": [{"frame": 1}, {"frame": 2}, {"frame": 3}],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", radar_path):
                    with patch("tools.dashboard.metrics.TRACKING_FRAME_PATH", tracking_path):
                        with patch("tools.dashboard.metrics.DEPTH_COVERAGE_HISTORY_PATH", depth_history_path):
                            with patch("tools.dashboard.metrics.time.time", return_value=100.0):
                                with patch("tools.dashboard.agent_ops.AGENT_EVENTS_PATH", event_log_path):
                                    result = dashboardMetrics.dashboard_tracking_quality_report(
                                        event_window_size=10,
                                        depth_sample_limit=10,
                                    )

        self.assertTrue(result["available"])
        self.assertEqual(result["quality_status"], "healthy")
        self.assertIn("good_continuity", result["signals"])
        self.assertIn("good_depth_coverage", result["signals"])
        self.assertIn("good_agent_success_rate", result["signals"])
        self.assertEqual(result["errors"], [])

    def test_tracking_quality_report_degraded_with_missing_inputs(self) -> None:
        # Performs test tracking quality report degraded with missing inputs logic.
        with TemporaryDirectory() as temp_dir:
            captures_dir = Path(temp_dir)
            radar_path = captures_dir / "latest_radar_state.json"
            tracking_path = captures_dir / "latest_tracking_frame.json"
            depth_history_path = captures_dir / "radar_depth_coverage_history.jsonl"
            event_log_path = captures_dir / "agent_events.jsonl"

            radar_path.write_text(
                json.dumps(
                    {
                        "timestamp": 90.0,
                        "last_observation_timestamp": 85.0,
                        "active_track_count": 4,
                        "relative_depth_coverage": {
                            "tracks_with_depth": 1,
                            "total_tracks": 4,
                            "coverage_ratio": 0.25,
                            "coverage_percent": 25.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            tracking_path.write_text(json.dumps({"time": 90.0, "detections": [{"id": 1}]}), encoding="utf-8")
            event_log_path.write_text(
                json.dumps(
                    {
                        "event_type": "reasoning_cycle",
                        "timestamp": 89.0,
                        "track_id": "1",
                        "status": "error",
                        "latency_ms": 120.0,
                        "bounded_out": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("tools.dashboard.metrics.CAPTURES_DIR", captures_dir):
                with patch("tools.dashboard.metrics.RADAR_STATE_PATH", radar_path):
                    with patch("tools.dashboard.metrics.TRACKING_FRAME_PATH", tracking_path):
                        with patch("tools.dashboard.metrics.DEPTH_COVERAGE_HISTORY_PATH", depth_history_path):
                            with patch("tools.dashboard.metrics.time.time", return_value=100.0):
                                with patch("tools.dashboard.agent_ops.AGENT_EVENTS_PATH", event_log_path):
                                    result = dashboardMetrics.dashboard_tracking_quality_report(
                                        event_window_size=10,
                                        depth_sample_limit=10,
                                    )

        self.assertTrue(result["available"])
        self.assertEqual(result["quality_status"], "degraded")
        self.assertIn("low_depth_coverage", result["signals"])
        self.assertIn("high_end_to_end_delay", result["signals"])
        self.assertIn("low_agent_success_rate", result["signals"])
        self.assertIn("depth_coverage_history_missing", result["errors"])


if __name__ == "__main__":
    unittest.main()
