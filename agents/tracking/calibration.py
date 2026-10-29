from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class GroundPlaneCalibration:
    matrix: np.ndarray
    unit: str


def _to_point_array(points: list[list[float]], name: str) -> np.ndarray:
    if not isinstance(points, list) or len(points) < 4:
        raise ValueError(f"{name} must contain at least 4 points")

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be shaped as Nx2")
    return arr


def compute_homography(
    image_points: list[list[float]],
    ground_points: list[list[float]],
) -> np.ndarray:
    image_arr = _to_point_array(image_points, "image_points")
    ground_arr = _to_point_array(ground_points, "ground_points")
    if image_arr.shape[0] != ground_arr.shape[0]:
        raise ValueError("image_points and ground_points must have same length")

    matrix, _ = cv2.findHomography(image_arr, ground_arr)
    if matrix is None or matrix.shape != (3, 3):
        raise ValueError("could not compute valid homography matrix")
    return matrix


def map_image_to_ground(
    matrix: np.ndarray,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float]:
    points = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(points, matrix)
    ground_x = float(transformed[0][0][0])
    ground_z = float(transformed[0][0][1])
    return (ground_x, ground_z)


def load_ground_plane_calibration(path: Path) -> GroundPlaneCalibration | None:
    if not path.is_file():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration file must be a JSON object")

    if "matrix" in payload:
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("matrix must be a 3x3 numeric array")
    else:
        image_points = payload.get("image_points")
        ground_points = payload.get("ground_points")
        matrix = compute_homography(image_points, ground_points)

    unit = payload.get("ground_coordinate_unit", "calibrated_units")
    if not isinstance(unit, str) or not unit:
        raise ValueError("ground_coordinate_unit must be a non-empty string")

    return GroundPlaneCalibration(matrix=matrix, unit=unit)
