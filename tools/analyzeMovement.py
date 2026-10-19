
import json
from pathlib import Path
from .registry import mcp


@mcp.tool()
def analyze_object_movement(tracking_data: str) -> str:
    """Load raw tracking observations, including timestamps, for agent analysis."""

    try:
        event = json.loads(tracking_data)
        file_path = Path(event["file_path"])

        if not file_path.is_file():
            return json.dumps({
                "status": "error",
                "message": "Tracking capture file not found",
                "file_path": str(file_path),
            })

        with file_path.open("r", encoding="utf-8") as file:
            capture_data = json.load(file)

        # Explicitly return the timestamps and positions for each track.
        tracks = {}

        for track_id, observations in capture_data.items():
            tracks[track_id] = [
                {
                    "frame": obs.get("frame"),
                    "elapsed_seconds": obs.get("elapsed_seconds"),
                    "timestamp": obs.get("timestamp"),
                    "center_x": obs.get("center_x"),
                    "center_y": obs.get("center_y"),
                }
                for obs in observations
            ]

        return json.dumps({
            "status": "success",
            "trigger_track_id": event.get("track_id"),
            "trigger_distance_pixels": event.get("distance_pixels"),
            "observation_count": sum(
                len(observations) for observations in tracks.values()
            ),
            "tracks": tracks,
        })

    except (json.JSONDecodeError, KeyError, TypeError, OSError) as exc:
        return json.dumps({
            "status": "error",
            "message": str(exc),
        })
