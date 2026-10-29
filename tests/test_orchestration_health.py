# Handles test orchestration health behavior.
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.orchestration import health as orchestrationHealth


class TestOrchestrationHealth(unittest.TestCase):
    def test_parse_radar_state_with_explicit_count(self) -> None:
        # Performs test parse radar state with explicit count logic.
        with patch("tools.orchestration.health.time.time", return_value=100.0):
            result = orchestrationHealth._parse_radar_state(
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
        # Performs test parse radar state falls back to tracks length logic.
        with patch("tools.orchestration.health.time.time", return_value=200.0):
            result = orchestrationHealth._parse_radar_state(
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
        # Performs test orchestration health status missing file logic.
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "missing.json"
            with patch("tools.orchestration.health.RADAR_STATE_PATH", fake_path):
                result = orchestrationHealth.dashboard_health_status()

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_missing")

    def test_dashboard_health_status_invalid_json(self) -> None:
        # Performs test orchestration health status invalid json logic.
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "bad.json"
            fake_path.write_text("not-json", encoding="utf-8")
            with patch("tools.orchestration.health.RADAR_STATE_PATH", fake_path):
                result = orchestrationHealth.dashboard_health_status()

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_invalid_json")

    def test_dashboard_health_status_valid_payload(self) -> None:
        # Performs test orchestration health status valid payload logic.
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

            with patch("tools.orchestration.health.RADAR_STATE_PATH", fake_path):
                with patch("tools.orchestration.health.time.time", return_value=1003.0):
                    result = orchestrationHealth.dashboard_health_status()

        self.assertTrue(result["available"])
        self.assertEqual(result["mapping_mode"], "simulation_fallback")
        self.assertEqual(result["active_track_count"], 1)
        self.assertAlmostEqual(result["last_observation_age_seconds"], 2.0, places=6)

    @patch("tools.orchestration.health.dashboard_health_status")
    def test_wait_for_dashboard_ready_immediate_success(self, mock_health_status) -> None:
        # Performs test wait for dashboard ready immediate success logic.
        mock_health_status.return_value = {
            "available": True,
            "last_observation_age_seconds": 0.2,
        }

        result = orchestrationHealth.wait_for_dashboard_ready(
            timeout_seconds=1.0,
            max_observation_age_seconds=1.0,
            poll_interval_seconds=0.1,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["attempts"], 1)

    @patch("tools.orchestration.health.time.sleep")
    @patch("tools.orchestration.health.dashboard_health_status")
    def test_wait_for_dashboard_ready_eventual_success(
        self,
        mock_health_status,
        mock_sleep,
    ) -> None:
        # Performs test wait for dashboard ready eventual success logic.
        mock_health_status.side_effect = [
            {"available": False, "error": "radar_state_missing"},
            {"available": True, "last_observation_age_seconds": 4.0},
            {"available": True, "last_observation_age_seconds": 0.4},
        ]

        result = orchestrationHealth.wait_for_dashboard_ready(
            timeout_seconds=2.0,
            max_observation_age_seconds=1.0,
            poll_interval_seconds=0.01,
        )

        self.assertTrue(result["ready"])
        self.assertEqual(result["attempts"], 3)
        self.assertGreaterEqual(mock_sleep.call_count, 2)

    @patch("tools.orchestration.health.time.sleep")
    @patch("tools.orchestration.health.dashboard_health_status")
    def test_wait_for_dashboard_ready_timeout(
        self,
        mock_health_status,
        mock_sleep,
    ) -> None:
        # Performs test wait for dashboard ready timeout logic.
        mock_health_status.return_value = {
            "available": True,
            "last_observation_age_seconds": 999.0,
        }

        with patch("tools.orchestration.health.time.time", side_effect=[0.0, 0.0, 0.05, 0.1, 0.11]):
            result = orchestrationHealth.wait_for_dashboard_ready(
                timeout_seconds=0.1,
                max_observation_age_seconds=1.0,
                poll_interval_seconds=0.05,
            )

        self.assertFalse(result["ready"])
        self.assertGreaterEqual(result["attempts"], 2)
        self.assertIn("status", result)

    def test_wait_for_dashboard_ready_invalid_arguments(self) -> None:
        # Performs test wait for dashboard ready invalid arguments logic.
        with self.assertRaises(ValueError):
            orchestrationHealth.wait_for_dashboard_ready(timeout_seconds=0)
        with self.assertRaises(ValueError):
            orchestrationHealth.wait_for_dashboard_ready(max_observation_age_seconds=-1)
        with self.assertRaises(ValueError):
            orchestrationHealth.wait_for_dashboard_ready(poll_interval_seconds=0)

    def test_verify_dashboard_pipeline_contract_passes_for_fresh_valid_payload(self) -> None:
        # Performs test verify dashboard pipeline contract passes for fresh valid payload logic.
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "state.json"
            fake_path.write_text(
                json.dumps(
                    {
                        "timestamp": 1000.0,
                        "coordinate_system": "calibrated_ground_plane",
                        "camera_status": "connected",
                        "detector_status": "running",
                        "active_track_count": 1,
                        "last_observation_timestamp": 1001.0,
                        "tracks": [
                            {
                                "track_id": "track-1",
                                "timestamp": 1001.0,
                                "ground_position": {"x": 1.0, "z": 2.0},
                                "position_kind": "calibrated_ground_plane",
                                "ground_coordinate_unit": "meters",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.orchestration.health.RADAR_STATE_PATH", fake_path):
                with patch("tools.orchestration.health.time.time", return_value=1002.0):
                    result = orchestrationHealth.verify_dashboard_pipeline_contract()

        self.assertTrue(result["available"])
        self.assertTrue(result["ready"])
        self.assertEqual(result["summary"]["error_count"], 0)

    def test_verify_dashboard_pipeline_contract_flags_stale_observation(self) -> None:
        # Performs test verify dashboard pipeline contract flags stale observation logic.
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "state.json"
            fake_path.write_text(
                json.dumps(
                    {
                        "timestamp": 1000.0,
                        "last_observation_timestamp": 1000.0,
                        "tracks": [],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.orchestration.health.RADAR_STATE_PATH", fake_path):
                with patch("tools.orchestration.health.time.time", return_value=1005.0):
                    result = orchestrationHealth.verify_dashboard_pipeline_contract(
                        max_observation_age_seconds=3.0,
                    )

        self.assertTrue(result["available"])
        self.assertFalse(result["ready"])
        codes = {item["code"] for item in result["findings"]}
        self.assertIn("stale_observation", codes)

    def test_verify_dashboard_pipeline_contract_flags_coordinate_mismatch(self) -> None:
        # Performs test verify dashboard pipeline contract flags coordinate mismatch logic.
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "state.json"
            fake_path.write_text(
                json.dumps(
                    {
                        "timestamp": 1000.0,
                        "last_observation_timestamp": 1000.0,
                        "tracks": [
                            {
                                "track_id": "track-1",
                                "timestamp": 1000.0,
                                "ground_position": {"x": 1.0, "z": 2.0},
                                "position_kind": "relative_ground_estimate",
                                "ground_coordinate_unit": "meters",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.orchestration.health.RADAR_STATE_PATH", fake_path):
                with patch("tools.orchestration.health.time.time", return_value=1000.5):
                    result = orchestrationHealth.verify_dashboard_pipeline_contract()

        self.assertTrue(result["available"])
        self.assertFalse(result["ready"])
        codes = {item["code"] for item in result["findings"]}
        self.assertIn("coordinate_unit_mismatch", codes)

    def test_verify_dashboard_pipeline_contract_simulation_warning_vs_error(self) -> None:
        # Performs test verify dashboard pipeline contract simulation warning vs error logic.
        with TemporaryDirectory() as temp_dir:
            fake_path = Path(temp_dir) / "state.json"
            fake_path.write_text(
                json.dumps(
                    {
                        "timestamp": 1000.0,
                        "coordinate_system": "simulation_fallback",
                        "last_observation_timestamp": 1000.0,
                        "tracks": [
                            {
                                "track_id": "sim-1",
                                "timestamp": 1000.0,
                                "ground_position": {"x": 1.0, "z": 2.0},
                                "position_kind": "simulated_ground_track",
                                "ground_coordinate_unit": "relative_units",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("tools.orchestration.health.RADAR_STATE_PATH", fake_path):
                with patch("tools.orchestration.health.time.time", return_value=1000.5):
                    warn_result = orchestrationHealth.verify_dashboard_pipeline_contract(
                        require_live_pipeline=False,
                    )
                    error_result = orchestrationHealth.verify_dashboard_pipeline_contract(
                        require_live_pipeline=True,
                    )

        self.assertTrue(warn_result["available"])
        self.assertTrue(warn_result["ready"])
        self.assertEqual(warn_result["summary"]["warn_count"], 1)

        self.assertTrue(error_result["available"])
        self.assertFalse(error_result["ready"])
        self.assertEqual(error_result["summary"]["error_count"], 1)


if __name__ == "__main__":
    unittest.main()
