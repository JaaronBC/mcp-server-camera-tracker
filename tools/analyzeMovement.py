import json
import math
import socket

from .registry import mcp

FRAME_RECEIVE_TIMEOUT_SECONDS = 4
STATIONARY_DISTANCE_PIXELS = 10
STATIONARY_DURATION_SECONDS = 1


@mcp.tool()
def analyze_movement(address: list[str | int], id: int) -> str:
    """Collect frames for a YOLO track until it remains nearly stationary."""
    if (
        len(address) != 2
        or not isinstance(address[0], str)
        or not address[0]
        or not isinstance(address[1], int)
        or isinstance(address[1], bool)
        or not 1 <= address[1] <= 65535
    ):
        raise ValueError(
            "address must contain a non-empty host string and a UDP port "
            "integer between 1 and 65535"
        )
    if not isinstance(id, int) or isinstance(id, bool):
        raise ValueError("id must be the integer track ID from the YOLO stream")

    frames: list[dict[str, object]] = []
    stable_position: tuple[float, float] | None = None
    stable_since: float | None = None

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((address[0], address[1]))
        sock.settimeout(FRAME_RECEIVE_TIMEOUT_SECONDS)

        while True:
            data, _ = sock.recvfrom(65535)
            message = json.loads(data.decode("utf-8"))
            if not isinstance(message, dict):
                raise ValueError("YOLO frame must be a JSON object")

            timestamp = message.get("time")
            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not math.isfinite(timestamp)
            ):
                raise ValueError("YOLO frame must contain a finite time value")

            detections = message.get("detections")
            if not isinstance(detections, list):
                raise ValueError("YOLO frame must contain a detections list")

            detection = next(
                (
                    item
                    for item in detections
                    if isinstance(item, dict) and item.get("id") == id
                ),
                None,
            )
            if detection is None:
                stable_position = None
                stable_since = None
                continue

            x = detection.get("x")
            y = detection.get("y")
            if (
                not isinstance(x, (int, float))
                or isinstance(x, bool)
                or not isinstance(y, (int, float))
                or isinstance(y, bool)
                or not math.isfinite(x)
                or not math.isfinite(y)
            ):
                raise ValueError(
                    f"YOLO detection for track {id} must have finite x and y coordinates"
                )

            frames.append(
                {
                    "time": timestamp,
                    "detection": detection,
                }
            )

            if stable_position is None or stable_since is None:
                stable_position = (x, y)
                stable_since = timestamp
                continue

            distance = math.hypot(
                x - stable_position[0],
                y - stable_position[1],
            )
            if distance > STATIONARY_DISTANCE_PIXELS:
                stable_position = (x, y)
                stable_since = timestamp
            elif timestamp - stable_since >= STATIONARY_DURATION_SECONDS:
                break

    return json.dumps(
        {
            "id": id,
            "stationary": True,
            "frames": frames,
        }
    )
