from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = PROJECT_ROOT / "captures"
LATEST_TRACKING_FRAME_PATH = CAPTURES_DIR / "latest_tracking_frame.json"
LATEST_RADAR_STATE_PATH = CAPTURES_DIR / "latest_radar_state.json"
LATEST_TRACKING_ESTIMATE_DIAGNOSTICS_PATH = CAPTURES_DIR / "latest_tracking_estimate_diagnostics.json"
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


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    return Path(value).expanduser()


TRACKING_FEEDBACK_CONSUMER_DECISIONS_PATH = _env_path(
    "PHASE16E_TRACKING_FEEDBACK_CONSUMER_DECISIONS_PATH",
    CAPTURES_DIR / "tracking_feedback_consumer_decisions.jsonl",
)
LATEST_OPERATOR_TRACKING_FEEDBACK_PATH = _env_path(
    "PHASE16E_OPERATOR_FEEDBACK_PATH",
    CAPTURES_DIR / "operator_tracking_feedback.json",
)
MAX_GROUND_Z = _env_positive_float("DASHBOARD_MAX_GROUND_Z", 20.0)
TRACKING_MISSING_TRACK_GRACE_SECONDS = _env_positive_float(
    "TRACKING_MISSING_TRACK_GRACE_SECONDS",
    1.5,
)
TRACKING_MAX_EFFECTIVE_HORIZON_SECONDS = _env_positive_float(
    "TRACKING_MAX_EFFECTIVE_HORIZON_SECONDS",
    0.5,
)
UDP_TRACKING_HOST = "127.0.0.1"
UDP_TRACKING_PORT = 5005
