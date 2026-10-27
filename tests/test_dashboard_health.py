from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import dashboardHealth


class TestDashboardHealth(unittest.TestCase):
    def test_parse_radar_state_with_explicit_count(self) -> None:
        with patch("tools.dashboardHealth.time.time", return_value=100.0):
            result = dashboardHealth._parse_radar_state(
                {
                    "timestamp": 98.0,
                    "coordinate_system": "relative_image_plane",
                    "camera_status": "connected",
                    "detector_status": "running",
                    "active_track_count": 3,
                    "last_observation_timestamp": 99.5,
                    "tracks": [{}, {}],
                }
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["active_track_count"], 3)
        self.assertAlmostEqual(result["last_observation_age_seconds"], 0.5, places=6)

    def test_parse_radar_state_falls_back_to_tracks_length(self) -> None:
        with patch("tools.dashboardHealth.time.time", return_value=200.0):
            result = dashboardHealth._parse_radar_state(
                {
                    "timestamp": 190.0,
                    "tracks": [{"id": 1}, {"id": 2}],
                }
            )

        self.assertEqual(result["active_track_count"], 2)
        self.assertEqual(result["camera_status"], "unknown")
        self.assertEqual(result["detector_status"], "unknown")
        self.assertAlmostEqual(result["last_observation_age_seconds"], 10.0, places=6)

    def test_dashboard_health_status_missing_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "missing.json"
            with patch("tools.dashboardHealth.RADAR_STATE_PATH", fake_path):
                result = dashboardHealth.dashboard_health_status()

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_missing")

    def test_dashboard_health_status_invalid_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "bad.json"
            fake_path.write_text("not-json", encoding="utf-8")
            with patch("tools.dashboardHealth.RADAR_STATE_PATH", fake_path):
                result = dashboardHealth.dashboard_health_status()

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_invalid_json")

    def test_dashboard_health_status_valid_payload(self) -> None:
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "state.json"
            fake_path.write_text(
                json.dumps(
                    {
                        "timestamp": 1000.0,
                        "coordinate_system": "simulation_fallback",
                        "camera_status": "unavailable",
                        "detector_status": "idle",
                        "active_track_count": 1,
                        "last_observation_timestamp": 1001.0,
                        "tracks": [{"track_id": "sim-1"}],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.dashboardHealth.RADAR_STATE_PATH", fake_path):
                with patch("tools.dashboardHealth.time.time", return_value=1003.0):
                    result = dashboardHealth.dashboard_health_status()

        self.assertTrue(result["available"])
        self.assertEqual(result["mapping_mode"], "simulation_fallback")
        self.assertEqual(result["active_track_count"], 1)
        self.assertAlmostEqual(result["last_observation_age_seconds"], 2.0, places=6)

    @patch("tools.dashboardHealth.dashboard_health_status")
    def test_wait_for_dashboard_ready_immediate_success(self, mock_health_status) -> None:
        mock_health_status.return_value = {
            "available": True,
            "last_observation_age_seconds": 0.2,
        }

        result = dashboardHealth.wait_for_dashboard_ready(
            timeout_seconds=1.0,
            max_observation_age_seconds=1.0,
            poll_interval_seconds=0.1,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["attempts"], 1)

    @patch("tools.dashboardHealth.time.sleep")
    @patch("tools.dashboardHealth.dashboard_health_status")
    def test_wait_for_dashboard_ready_eventual_success(
        self,
        mock_health_status,
        mock_sleep,
    ) -> None:
        mock_health_status.side_effect = [
            {"available": False, "error": "radar_state_missing"},
            {"available": True, "last_observation_age_seconds": 4.0},
            {"available": True, "last_observation_age_seconds": 0.4},
        ]

        result = dashboardHealth.wait_for_dashboard_ready(
            timeout_seconds=2.0,
            max_observation_age_seconds=1.0,
            poll_interval_seconds=0.01,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["attempts"], 3)
        self.assertGreaterEqual(mock_sleep.call_count, 2)

    @patch("tools.dashboardHealth.time.sleep")
    @patch("tools.dashboardHealth.dashboard_health_status")
    def test_wait_for_dashboard_ready_timeout(
        self,
        mock_health_status,
        mock_sleep,
    ) -> None:
        mock_health_status.return_value = {
            "available": True,
            "last_observation_age_seconds": 999.0,
        }

        with patch("tools.dashboardHealth.time.time", side_effect=[0.0, 0.0, 0.05, 0.1, 0.11]):
            result = dashboardHealth.wait_for_dashboard_ready(
                timeout_seconds=0.1,
                max_observation_age_seconds=1.0,
                poll_interval_seconds=0.05,
            )

        self.assertFalse(result["ready"])
        self.assertGreaterEqual(result["attempts"], 2)
        self.assertIn("status", result)

    def test_wait_for_dashboard_ready_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            dashboardHealth.wait_for_dashboard_ready(timeout_seconds=0)
        with self.assertRaises(ValueError):
            dashboardHealth.wait_for_dashboard_ready(max_observation_age_seconds=-1)
        with self.assertRaises(ValueError):
            dashboardHealth.wait_for_dashboard_ready(poll_interval_seconds=0)


if __name__ == "__main__":
    unittest.main()
