from __future__ import annotations

import time

from agents.tracking.state_publisher import (
    apply_simulation_fallback_state,
    build_tracking_simulated_tracks,
    collect_radar_publish_snapshot,
    publish_tracking_radar_state,
)

from .config import (
    LATEST_RADAR_STATE_PATH,
    LIVE_DATA_STALE_FOR_SIM_SECONDS,
    MAX_GROUND_X,
    MAX_GROUND_Z,
    RADAR_DEPTH_COVERAGE_HISTORY_PATH,
    RADAR_TRAIL_LENGTH,
    SIM_UPDATE_HZ,
)
from .state import SharedDashboardState


def run_radar_loop(state: SharedDashboardState) -> None:
    sim_step = 0.0
    sim_interval = 1.0 / SIM_UPDATE_HZ

    while True:
        with state.lock:
            if not state.running:
                break
            now = time.time()
            live_is_stale = (now - state.last_live_frame_time) > LIVE_DATA_STALE_FOR_SIM_SECONDS
            camera_status = state.camera_status

        using_simulation = live_is_stale or camera_status in {"unavailable", "error", "frame_error", "stopped"}

        if using_simulation:
            simulated_tracks = build_tracking_simulated_tracks(
                sim_step,
                max_ground_x=MAX_GROUND_X,
                max_ground_z=MAX_GROUND_Z,
            )
            apply_simulation_fallback_state(
                state=state,
                now=now,
                simulated_tracks=simulated_tracks,
                trail_length=RADAR_TRAIL_LENGTH,
            )
            sim_step += sim_interval
        else:
            with state.lock:
                state.mapping_mode = "relative_image_plane"

        radar_publish_snapshot = collect_radar_publish_snapshot(state)

        try:
            publish_tracking_radar_state(
                radar_state_path=LATEST_RADAR_STATE_PATH,
                radar_depth_coverage_history_path=RADAR_DEPTH_COVERAGE_HISTORY_PATH,
                now=now,
                coordinate_system=radar_publish_snapshot["coordinate_system"],
                camera_status=radar_publish_snapshot["camera_status"],
                detector_status=radar_publish_snapshot["detector_status"],
                calibration_status=radar_publish_snapshot["calibration_status"],
                tracking_agent_status=radar_publish_snapshot["tracking_agent_status"],
                last_observation_timestamp=radar_publish_snapshot["last_observation_time"],
                tracking_agent_observation=radar_publish_snapshot["tracking_agent_observation"],
                tracks=radar_publish_snapshot["tracks"],
            )
        except OSError:
            pass

        time.sleep(sim_interval)
