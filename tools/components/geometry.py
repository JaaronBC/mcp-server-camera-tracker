from __future__ import annotations

import math


def image_to_relative_ground(
    pixel_x: float,
    pixel_y: float,
    frame_width: int,
    frame_height: int,
    max_ground_x: float = 10.0,
    max_ground_z: float = 20.0,
) -> tuple[float, float]:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    normalized_x = (pixel_x / frame_width) - 0.5
    normalized_y = 1.0 - (pixel_y / frame_height)

    ground_x = normalized_x * 2.0 * max_ground_x
    ground_z = max(0.0, normalized_y * max_ground_z)
    return (ground_x, ground_z)


def ground_to_radar_screen(
    ground_x: float,
    ground_z: float,
    panel_width: int,
    panel_height: int,
    max_ground_x: float = 10.0,
    max_ground_z: float = 20.0,
    margin: int = 40,
) -> tuple[int, int]:
    if panel_width <= 2 * margin or panel_height <= 2 * margin:
        raise ValueError("radar panel is too small for requested margin")

    usable_width = panel_width - (2 * margin)
    usable_height = panel_height - (2 * margin)

    center_x = margin + (usable_width // 2)
    camera_edge_y = panel_height - margin

    x_scale = usable_width / (2.0 * max_ground_x)
    z_scale = usable_height / max_ground_z

    screen_x = int(round(center_x + (ground_x * x_scale)))
    screen_y = int(round(camera_edge_y - (ground_z * z_scale)))
    return (screen_x, screen_y)


def planar_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])
