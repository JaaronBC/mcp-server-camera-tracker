# Handles test calibration behavior.
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from agents.components.calibration import (
    compute_homography,
    load_ground_plane_calibration,
    map_image_to_ground,
)


class TestCalibration(unittest.TestCase):
    def test_compute_homography_and_map_point(self) -> None:
        # Performs test compute homography and map point logic.
        image_points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        ground_points = [[0, 0], [10, 0], [10, 20], [0, 20]]

        matrix = compute_homography(image_points, ground_points)
        x_ground, z_ground = map_image_to_ground(matrix, 50, 50)

        self.assertAlmostEqual(x_ground, 5.0, places=3)
        self.assertAlmostEqual(z_ground, 10.0, places=3)

    def test_load_calibration_from_points(self) -> None:
        # Performs test load calibration from points logic.
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ground_plane_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                        "ground_points": [[0, 0], [10, 0], [10, 20], [0, 20]],
                        "ground_coordinate_unit": "meters",
                    }
                ),
                encoding="utf-8",
            )

            calibration = load_ground_plane_calibration(path)

        assert calibration is not None
        self.assertEqual(calibration.unit, "meters")
        self.assertEqual(calibration.matrix.shape, (3, 3))

    def test_load_calibration_from_matrix(self) -> None:
        # Performs test load calibration from matrix logic.
        matrix = np.array(
            [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ground_plane_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "matrix": matrix.tolist(),
                        "ground_coordinate_unit": "meters",
                    }
                ),
                encoding="utf-8",
            )

            calibration = load_ground_plane_calibration(path)

        assert calibration is not None
        self.assertEqual(calibration.unit, "meters")
        self.assertEqual(calibration.matrix.shape, (3, 3))

    def test_load_calibration_missing_file(self) -> None:
        # Performs test load calibration missing file logic.
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "does_not_exist.json"
            calibration = load_ground_plane_calibration(path)

        self.assertIsNone(calibration)


if __name__ == "__main__":
    unittest.main()
