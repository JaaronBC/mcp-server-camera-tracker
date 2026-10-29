from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = PROJECT_ROOT / "captures"
LATEST_TRACKING_FRAME_PATH = CAPTURES_DIR / "latest_tracking_frame.json"
LATEST_RADAR_STATE_PATH = CAPTURES_DIR / "latest_radar_state.json"
RADAR_DEPTH_COVERAGE_HISTORY_PATH = CAPTURES_DIR / "radar_depth_coverage_history.jsonl"
GROUND_CALIBRATION_PATH = CAPTURES_DIR / "ground_plane_calibration.json"

WINDOW_NAME = "Counter-UAS Dashboard Demo"
RADAR_TRAIL_LENGTH = 20
RADAR_STALE_SECONDS = 1.5
RADAR_LOST_SECONDS = 3.5
LIVE_DATA_STALE_FOR_SIM_SECONDS = 2.0
SIM_UPDATE_HZ = 10.0
RENDER_HZ = 20.0
MAX_GROUND_X = 10.0
AGENT_GROUND_BLEND_FACTOR = 0.35
AGENT_GROUND_MAX_CORRECTION = 1.5


def _env_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


MAX_GROUND_Z = _env_positive_float("DASHBOARD_MAX_GROUND_Z", 20.0)
UDP_TRACKING_HOST = "127.0.0.1"
UDP_TRACKING_PORT = 5005
