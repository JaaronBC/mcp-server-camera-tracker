# Live dashboard runtime orchestration: capture/tracking loops + dashboard entrypoint.
from __future__ import annotations

import threading
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.tracking.runtime_support import (
    apply_tracking_observation_update,
    apply_tracking_agent_updates_to_tracks,
    build_relative_depth_coverage_text,
    build_tracking_detection_id,
    compute_relative_depth_coverage,
    estimate_relative_depth_from_image_geometry,
    normalize_position_metadata,
    summarize_tracking_agent_observation_metrics,
)
from agents.tracking.state_publisher import (
    build_tracking_frame_message,
)
from agents.tracking.status_runtime import (
    build_tracking_agent_health_legend_text,
    build_tracking_agent_status_text,
    build_tracking_debug_toggle_hint,
    classify_tracking_agent_health,
    tracking_agent_health_color,
)

from .config import (
    CAPTURES_DIR,
)
from .radar_loop import run_radar_loop
from .rendering import render_dashboard
from .state import SharedDashboardState
from .tracking_loop import resolve_model_path, run_live_tracking_loop

__all__ = [
    "apply_tracking_observation_update",
    "apply_tracking_agent_updates_to_tracks",
    "build_relative_depth_coverage_text",
    "build_tracking_detection_id",
    "compute_relative_depth_coverage",
    "estimate_relative_depth_from_image_geometry",
    "normalize_position_metadata",
    "summarize_tracking_agent_observation_metrics",
    "build_tracking_agent_health_legend_text",
    "build_tracking_agent_status_text",
    "build_tracking_debug_toggle_hint",
    "classify_tracking_agent_health",
    "tracking_agent_health_color",
    "resolve_model_path",
    "run_live_tracking_loop",
    "run_radar_loop",
    "build_tracking_message",
    "main",
]


def build_tracking_message(timestamp: float, detections: list[dict]) -> dict:
    return build_tracking_frame_message(timestamp, detections)


def main() -> None:
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    state = SharedDashboardState()

    tracking_thread = threading.Thread(target=run_live_tracking_loop, args=(state,), daemon=True)
    radar_thread = threading.Thread(target=run_radar_loop, args=(state,), daemon=True)

    tracking_thread.start()
    radar_thread.start()

    try:
        render_dashboard(state)
    finally:
        with state.lock:
            state.running = False

        tracking_thread.join(timeout=2.0)
        radar_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
