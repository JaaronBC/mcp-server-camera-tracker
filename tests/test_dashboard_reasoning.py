from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools import dashboardReasoning


class TestDashboardReasoning(unittest.TestCase):
    def tearDown(self) -> None:
        dashboardReasoning._reset_reasoning_state()

    def test_reasoning_returns_unavailable_when_no_radar_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_path):
                result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_observation_summary_returns_unavailable_when_no_radar_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_path):
                result = dashboardReasoning.simulate_observation_summary("track-1")

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_reasoning_requests_evidence_for_insufficient_history(self) -> None:
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
                result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertTrue(result["available"])
        self.assertEqual(result["assessment"]["assessment_label"], "requires_more_observation")
        self.assertIsNotNone(result["evidence_request"])

    def test_reasoning_detects_fast_motion_after_two_updates(self) -> None:
        first_payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 0.0, "z": 0.0},
                    "confidence": 0.9,
                }
            ],
        }
        second_payload = {
            "timestamp": 1001.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1001.0,
                    "ground_position": {"x": 3.0, "z": 0.0},
                    "confidence": 0.95,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                radar_path.write_text(json.dumps(first_payload), encoding="utf-8")
                dashboardReasoning.simulate_reasoning_assessment("track-1")

                radar_path.write_text(json.dumps(second_payload), encoding="utf-8")
                result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertTrue(result["available"])
        self.assertEqual(result["assessment"]["assessment_label"], "requires_review")
        self.assertIsNotNone(result["summary"]["speed"])

    def test_observation_summary_reports_velocity_after_two_updates(self) -> None:
        first_payload = {
            "timestamp": 2000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 2000.0,
                    "ground_position": {"x": 1.0, "z": 1.0},
                    "confidence": 0.8,
                }
            ],
        }
        second_payload = {
            "timestamp": 2001.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 2001.0,
                    "ground_position": {"x": 2.0, "z": 3.0},
                    "confidence": 0.85,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                radar_path.write_text(json.dumps(first_payload), encoding="utf-8")
                dashboardReasoning.simulate_observation_summary("track-1")

                radar_path.write_text(json.dumps(second_payload), encoding="utf-8")
                result = dashboardReasoning.simulate_observation_summary("track-1")

        self.assertTrue(result["available"])
        summary = result["summary"]
        self.assertIsNotNone(summary["velocity_x"])
        self.assertIsNotNone(summary["velocity_z"])
        self.assertIsNotNone(summary["speed"])

    def test_evidence_request_returns_unavailable_when_no_radar_state(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_path):
                result = dashboardReasoning.simulate_evidence_request(
                    "track-1",
                    "need_more_observations",
                )

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_evidence_request_is_bounded_over_multiple_calls(self) -> None:
        payload = {
            "timestamp": 3000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 3000.0,
                    "ground_position": {"x": 0.0, "z": 0.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                requests = []
                for index in range(4):
                    response = dashboardReasoning.simulate_evidence_request(
                        "track-1",
                        f"reason-{index}",
                    )
                    requests.append(response)

        self.assertTrue(requests[0]["available"])
        self.assertIsNotNone(requests[0]["request"])
        self.assertIsNotNone(requests[1]["request"])
        self.assertIsNotNone(requests[2]["request"])
        self.assertIsNone(requests[3]["request"])
        self.assertTrue(requests[3]["bounded_out"])


if __name__ == "__main__":
    unittest.main()
