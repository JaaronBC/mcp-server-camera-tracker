# Handles test orchestration calibration behavior.
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.orchestration import calibration as orchestrationCalibration


class TestOrchestrationCalibrationTools(unittest.TestCase):
    def test_get_calibration_missing(self) -> None:
        # Performs test get calibration missing logic.
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ground_plane_calibration.json"
            with patch("tools.orchestration.calibration.CALIBRATION_PATH", path):
                result = orchestrationCalibration.get_ground_plane_calibration()

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "calibration_missing")

    def test_set_and_get_and_validate_calibration(self) -> None:
        # Performs test set and get and validate calibration logic.
        payload = {
            "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "ground_points": [[0, 0], [10, 0], [10, 20], [0, 20]],
            "ground_coordinate_unit": "meters",
        }

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ground_plane_calibration.json"
            with patch("tools.orchestration.calibration.CALIBRATION_PATH", path):
                set_result = orchestrationCalibration.set_ground_plane_calibration(payload)
                get_result = orchestrationCalibration.get_ground_plane_calibration()
                validate_result = orchestrationCalibration.validate_ground_plane_calibration()

        self.assertTrue(set_result["saved"])
        self.assertEqual(set_result["ground_coordinate_unit"], "meters")
        self.assertTrue(get_result["available"])
        self.assertEqual(get_result["calibration"]["ground_coordinate_unit"], "meters")
        self.assertTrue(validate_result["valid"])

    def test_set_calibration_rejects_non_object(self) -> None:
        # Performs test set calibration rejects non object logic.
        with self.assertRaises(ValueError):
            orchestrationCalibration.set_ground_plane_calibration([1, 2, 3])  # type: ignore[arg-type]

    def test_validate_calibration_invalid_payload(self) -> None:
        # Performs test validate calibration invalid payload logic.
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ground_plane_calibration.json"
            path.write_text(json.dumps({"matrix": [[1, 2], [3, 4]]}), encoding="utf-8")

            with patch("tools.orchestration.calibration.CALIBRATION_PATH", path):
                result = orchestrationCalibration.validate_ground_plane_calibration()

        self.assertFalse(result["valid"])
        self.assertIn("error", result)

    def test_clear_calibration(self) -> None:
        # Performs test clear calibration logic.
        payload = {
            "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "ground_points": [[0, 0], [10, 0], [10, 20], [0, 20]],
            "ground_coordinate_unit": "meters",
        }

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ground_plane_calibration.json"
            with patch("tools.orchestration.calibration.CALIBRATION_PATH", path):
                orchestrationCalibration.set_ground_plane_calibration(payload)
                clear_result = orchestrationCalibration.clear_ground_plane_calibration()
                get_result = orchestrationCalibration.get_ground_plane_calibration()

        self.assertTrue(clear_result["cleared"])
        self.assertFalse(get_result["available"])


if __name__ == "__main__":
    unittest.main()
