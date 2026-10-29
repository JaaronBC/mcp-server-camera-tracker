from __future__ import annotations

import unittest

import numpy as np

from agents.tracking.calibration import GroundPlaneCalibration
from agents.tracking.frame_transformer import build_tracking_frame_outputs


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values

    def int(self):
        return self


class _FakeBoxes:
    def __init__(self, xyxy, track_ids, classes, confidences):
        self.xyxy = _FakeTensor(xyxy)
        self.id = _FakeTensor(track_ids)
        self.cls = _FakeTensor(classes)
        self.conf = _FakeTensor(confidences)


class TestTrackingFrameTransformer(unittest.TestCase):
    def test_out_of_bounds_calibration_falls_back_to_relative_estimate(self) -> None:
        boxes = _FakeBoxes(
            xyxy=[[100.0, 100.0, 200.0, 300.0]],
            track_ids=[1],
            classes=[0],
            confidences=[0.9],
        )

        calibration = GroundPlaneCalibration(
            matrix=np.array(
                [
                    [1.0, 0.0, 1000.0],
                    [0.0, 1.0, 1000.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            unit="meters",
        )

        tracks, detections_for_file, _candidates, _reasons = build_tracking_frame_outputs(
            boxes=boxes,
            model_names={0: "person"},
            now=1000.0,
            frame_shape=(720, 1280, 3),
            calibration=calibration,
            track_trails={},
            max_ground_x=10.0,
            max_ground_z=20.0,
        )

        self.assertEqual(len(tracks), 1)
        snapshot = tracks["1"]
        self.assertEqual(snapshot.position_kind, "relative_ground_estimate")
        self.assertEqual(snapshot.ground_coordinate_unit, "relative_units")
        self.assertIn("calibration_projection_out_of_bounds", snapshot.quality_flags)

        self.assertEqual(detections_for_file[0]["position_kind"], "relative_ground_estimate")
        self.assertEqual(detections_for_file[0]["ground_coordinate_unit"], "relative_units")


if __name__ == "__main__":
    unittest.main()
