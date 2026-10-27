from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADAR_STATE_PATH = PROJECT_ROOT / "captures" / "latest_radar_state.json"


def _parse_radar_state(payload: dict[str, Any]) -> dict[str, Any]:
    now = time.time()

    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        timestamp = None

    last_observation_timestamp = payload.get("last_observation_timestamp")
    if not isinstance(last_observation_timestamp, (int, float)):
        last_observation_timestamp = timestamp

    active_track_count = payload.get("active_track_count")
    if not isinstance(active_track_count, int):
        tracks = payload.get("tracks")
        active_track_count = len(tracks) if isinstance(tracks, list) else 0

    last_observation_age_seconds = None
    if isinstance(last_observation_timestamp, (int, float)):
        last_observation_age_seconds = max(0.0, now - float(last_observation_timestamp))

    return {
        "available": True,
        "radar_state_path": str(RADAR_STATE_PATH),
        "camera_status": payload.get("camera_status", "unknown"),
        "detector_status": payload.get("detector_status", "unknown"),
        "calibration_status": payload.get("calibration_status", "unknown"),
        "mapping_mode": payload.get("coordinate_system", "unknown"),
        "active_track_count": active_track_count,
        "state_timestamp": timestamp,
        "last_observation_timestamp": last_observation_timestamp,
        "last_observation_age_seconds": last_observation_age_seconds,
    }


@mcp.tool()
def dashboard_health_status() -> dict[str, Any]:
    """Return dashboard health summary derived from captures/latest_radar_state.json."""
    if not RADAR_STATE_PATH.is_file():
        return {
            "available": False,
            "radar_state_path": str(RADAR_STATE_PATH),
            "error": "radar_state_missing",
        }

    try:
        payload = json.loads(RADAR_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "available": False,
            "radar_state_path": str(RADAR_STATE_PATH),
            "error": "radar_state_invalid_json",
        }

    if not isinstance(payload, dict):
        return {
            "available": False,
            "radar_state_path": str(RADAR_STATE_PATH),
            "error": "radar_state_not_object",
        }

    return _parse_radar_state(payload)


@mcp.tool()
def wait_for_dashboard_ready(
    timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    """Poll dashboard health until fresh observations are available or timeout occurs."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if max_observation_age_seconds < 0:
        raise ValueError("max_observation_age_seconds must be non-negative")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero")

    deadline = time.time() + timeout_seconds
    attempts = 0
    last_status: dict[str, Any] | None = None

    while True:
        attempts += 1
        status = dashboard_health_status()
        last_status = status

        if status.get("available"):
            age = status.get("last_observation_age_seconds")
            if isinstance(age, (int, float)) and age <= max_observation_age_seconds:
                return {
                    "ready": True,
                    "attempts": attempts,
                    "status": status,
                }

        now = time.time()
        if now >= deadline:
            return {
                "ready": False,
                "attempts": attempts,
                "timeout_seconds": timeout_seconds,
                "status": last_status,
            }

        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))
