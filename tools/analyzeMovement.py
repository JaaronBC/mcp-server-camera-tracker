import json
from pathlib import Path

from .registry import mcp

@mcp.tool()
def analyze_object_movement(tracking_data: str) -> str:
    """Analyze tracked object positions and determine each object's movement direction."""

    if not tracking_data.strip():
        raise ValueError("tracking_data cannot be empty.")

    tracking_file = Path(tracking_data)

    if not tracking_file.exists():
        raise ValueError(f"Tracking file not found: {tracking_data}")

    with open(tracking_file, "r") as file:
        data = json.load(file)

    if not data:
        return "No objects were detected in the tracking data."

    results = {}

    # Movement threshold in pixels
    movement_threshold = 30

    for object_id, positions in data.items():

        if len(positions) < 2:
            results[object_id] = {
                "movement": "insufficient data",
                "displacement_x": 0,
            }
            continue

        start_x = positions[0]["center_x"]
        end_x = positions[-1]["center_x"]

        displacement_x = end_x - start_x

        if displacement_x > movement_threshold:
            movement = "right"

        elif displacement_x < -movement_threshold:
            movement = "left"

        else:
            movement = "stationary"

        results[object_id] = {
            "movement": movement,
            "displacement_x": round(displacement_x, 2),
            "frames_tracked": len(positions),
        }

    return json.dumps(results, indent=2)
