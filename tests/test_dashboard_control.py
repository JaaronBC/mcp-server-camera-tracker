from __future__ import annotations

import unittest
from unittest.mock import patch

from tools import dashboardControl


class TestDashboardControl(unittest.TestCase):
    @patch("tools.dashboardControl.wait_for_dashboard_ready")
    @patch("tools.dashboardControl.start_dashboard_process")
    def test_start_dashboard_and_wait_ready_success(
        self,
        mock_start,
        mock_wait,
    ) -> None:
        mock_start.return_value = (True, "started")
        mock_wait.return_value = {
            "ready": True,
            "attempts": 2,
            "status": {"available": True},
        }

        result = dashboardControl.start_dashboard_and_wait_ready(
            timeout_seconds=5.0,
            max_observation_age_seconds=2.0,
            poll_interval_seconds=0.1,
        )

        self.assertTrue(result["started"])
        self.assertEqual(result["start_message"], "started")
        self.assertTrue(result["readiness"]["ready"])

        mock_wait.assert_called_once_with(
            timeout_seconds=5.0,
            max_observation_age_seconds=2.0,
            poll_interval_seconds=0.1,
        )

    @patch("tools.dashboardControl.wait_for_dashboard_ready")
    @patch("tools.dashboardControl.start_dashboard_process")
    def test_start_dashboard_and_wait_ready_timeout(
        self,
        mock_start,
        mock_wait,
    ) -> None:
        mock_start.return_value = (False, "already running")
        mock_wait.return_value = {
            "ready": False,
            "attempts": 6,
            "timeout_seconds": 1.0,
            "status": {"available": False, "error": "radar_state_missing"},
        }

        result = dashboardControl.start_dashboard_and_wait_ready(
            timeout_seconds=1.0,
            max_observation_age_seconds=0.5,
            poll_interval_seconds=0.1,
        )

        self.assertFalse(result["started"])
        self.assertEqual(result["start_message"], "already running")
        self.assertFalse(result["readiness"]["ready"])

    @patch("tools.dashboardControl.dashboard_process_status")
    @patch("tools.dashboardControl.stop_dashboard_process")
    def test_stop_dashboard_and_confirm_success(
        self,
        mock_stop,
        mock_status,
    ) -> None:
        mock_stop.return_value = "stopped"
        mock_status.side_effect = [
            {"running": True, "pid": 10, "return_code": None},
            {"running": False, "pid": 10, "return_code": 0},
        ]

        result = dashboardControl.stop_dashboard_and_confirm(
            force=False,
            timeout_seconds=1.0,
            poll_interval_seconds=0.01,
        )

        self.assertTrue(result["stopped"])
        self.assertEqual(result["stop_message"], "stopped")

    @patch("tools.dashboardControl.time.sleep")
    @patch("tools.dashboardControl.time.time")
    @patch("tools.dashboardControl.dashboard_process_status")
    @patch("tools.dashboardControl.stop_dashboard_process")
    def test_stop_dashboard_and_confirm_timeout(
        self,
        mock_stop,
        mock_status,
        mock_time,
        mock_sleep,
    ) -> None:
        mock_stop.return_value = "stopping"
        mock_status.return_value = {"running": True, "pid": 10, "return_code": None}
        mock_time.side_effect = [0.0, 0.0, 0.05, 0.11]

        result = dashboardControl.stop_dashboard_and_confirm(
            force=True,
            timeout_seconds=0.1,
            poll_interval_seconds=0.05,
        )

        self.assertFalse(result["stopped"])
        self.assertEqual(result["stop_message"], "stopping")
        self.assertIn("timeout_seconds", result)
        self.assertGreaterEqual(mock_sleep.call_count, 1)

    def test_stop_dashboard_and_confirm_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            dashboardControl.stop_dashboard_and_confirm(timeout_seconds=0)
        with self.assertRaises(ValueError):
            dashboardControl.stop_dashboard_and_confirm(poll_interval_seconds=0)

    @patch("tools.dashboardControl.start_dashboard_and_wait_ready")
    @patch("tools.dashboardControl.stop_dashboard_and_confirm")
    def test_restart_dashboard_and_wait_ready_success(
        self,
        mock_stop_confirm,
        mock_start_wait,
    ) -> None:
        mock_stop_confirm.return_value = {
            "stopped": True,
            "stop_message": "stopped",
            "attempts": 1,
            "status": {"running": False},
        }
        mock_start_wait.return_value = {
            "started": True,
            "start_message": "started",
            "readiness": {"ready": True, "attempts": 2},
        }

        result = dashboardControl.restart_dashboard_and_wait_ready(
            force_stop=False,
            stop_timeout_seconds=4.0,
            start_timeout_seconds=8.0,
            max_observation_age_seconds=2.0,
            poll_interval_seconds=0.2,
        )

        self.assertTrue(result["restarted"])
        self.assertTrue(result["start"]["started"])
        self.assertTrue(result["readiness"]["ready"])

    @patch("tools.dashboardControl.start_dashboard_and_wait_ready")
    @patch("tools.dashboardControl.stop_dashboard_and_confirm")
    def test_restart_dashboard_and_wait_ready_stop_failure(
        self,
        mock_stop_confirm,
        mock_start_wait,
    ) -> None:
        mock_stop_confirm.return_value = {
            "stopped": False,
            "stop_message": "timeout",
            "attempts": 3,
            "status": {"running": True},
        }

        result = dashboardControl.restart_dashboard_and_wait_ready()

        self.assertFalse(result["restarted"])
        self.assertIsNone(result["start"])
        self.assertIsNone(result["readiness"])
        mock_start_wait.assert_not_called()

    @patch("tools.dashboardControl.validate_ground_plane_calibration")
    @patch("tools.dashboardControl.dashboard_health_status")
    @patch("tools.dashboardControl.dashboard_process_status")
    def test_dashboard_lifecycle_status_ready(
        self,
        mock_process_status,
        mock_health_status,
        mock_calibration,
    ) -> None:
        mock_process_status.return_value = {
            "running": True,
            "pid": 123,
            "return_code": None,
        }
        mock_health_status.return_value = {
            "available": True,
            "last_observation_age_seconds": 0.8,
            "camera_status": "connected",
        }
        mock_calibration.return_value = {
            "valid": True,
            "ground_coordinate_unit": "meters",
        }

        result = dashboardControl.dashboard_lifecycle_status(max_observation_age_seconds=1.0)

        self.assertTrue(result["ready"])
        self.assertTrue(result["running"])
        self.assertTrue(result["health_available"])
        self.assertTrue(result["observations_fresh"])

    @patch("tools.dashboardControl.validate_ground_plane_calibration")
    @patch("tools.dashboardControl.dashboard_health_status")
    @patch("tools.dashboardControl.dashboard_process_status")
    def test_dashboard_lifecycle_status_not_ready_when_stale(
        self,
        mock_process_status,
        mock_health_status,
        mock_calibration,
    ) -> None:
        mock_process_status.return_value = {
            "running": True,
            "pid": 123,
            "return_code": None,
        }
        mock_health_status.return_value = {
            "available": True,
            "last_observation_age_seconds": 9.0,
        }
        mock_calibration.return_value = {
            "valid": False,
            "error": "calibration_missing",
        }

        result = dashboardControl.dashboard_lifecycle_status(max_observation_age_seconds=1.0)

        self.assertFalse(result["ready"])
        self.assertTrue(result["running"])
        self.assertTrue(result["health_available"])
        self.assertFalse(result["observations_fresh"])

    def test_dashboard_lifecycle_status_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            dashboardControl.dashboard_lifecycle_status(max_observation_age_seconds=-0.1)

    @patch("tools.dashboardControl.dashboard_lifecycle_status")
    @patch("tools.dashboardControl.start_dashboard_and_wait_ready")
    @patch("tools.dashboardControl.set_ground_plane_calibration")
    def test_dashboard_bootstrap_start_path_with_calibration(
        self,
        mock_set_calibration,
        mock_start_wait,
        mock_lifecycle,
    ) -> None:
        mock_set_calibration.return_value = {
            "saved": True,
            "ground_coordinate_unit": "meters",
            "matrix_shape": [3, 3],
        }
        mock_start_wait.return_value = {
            "started": True,
            "start_message": "started",
            "readiness": {"ready": True, "attempts": 1},
        }
        mock_lifecycle.return_value = {
            "ready": True,
            "running": True,
        }

        result = dashboardControl.dashboard_bootstrap(
            calibration={
                "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "ground_points": [[0, 0], [10, 0], [10, 20], [0, 20]],
                "ground_coordinate_unit": "meters",
            },
            restart_if_running=False,
            start_timeout_seconds=5.0,
            max_observation_age_seconds=1.0,
            poll_interval_seconds=0.1,
        )

        self.assertTrue(result["bootstrapped"])
        self.assertFalse(result["restart_if_running"])
        self.assertIsNotNone(result["calibration_result"])
        mock_start_wait.assert_called_once()

    @patch("tools.dashboardControl.dashboard_lifecycle_status")
    @patch("tools.dashboardControl.restart_dashboard_and_wait_ready")
    @patch("tools.dashboardControl.start_dashboard_and_wait_ready")
    def test_dashboard_bootstrap_restart_path(
        self,
        mock_start_wait,
        mock_restart_wait,
        mock_lifecycle,
    ) -> None:
        mock_restart_wait.return_value = {
            "restarted": True,
            "stop": {"stopped": True},
            "start": {"started": True},
            "readiness": {"ready": True},
        }
        mock_lifecycle.return_value = {
            "ready": True,
            "running": True,
        }

        result = dashboardControl.dashboard_bootstrap(
            restart_if_running=True,
            force_stop=True,
            stop_timeout_seconds=2.0,
            start_timeout_seconds=3.0,
            max_observation_age_seconds=1.5,
            poll_interval_seconds=0.2,
        )

        self.assertTrue(result["bootstrapped"])
        self.assertTrue(result["restart_if_running"])
        mock_restart_wait.assert_called_once()
        mock_start_wait.assert_not_called()


if __name__ == "__main__":
    unittest.main()
