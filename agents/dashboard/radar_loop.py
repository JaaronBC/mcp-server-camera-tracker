from __future__ import annotations

import time

from agents.tracking.state_publisher import (
    apply_simulation_fallback_state,
    build_tracking_simulated_tracks,
    collect_radar_publish_snapshot,
    publish_tracking_radar_state,
)

from .identity_snapshot_bridge import (
    build_identity_snapshot_payload,
    build_identity_snapshot_unavailable_payload,
    publish_identity_snapshot,
)

from .config import (
    LATEST_RADAR_STATE_PATH,
    LIVE_DATA_STALE_FOR_SIM_SECONDS,
    MAX_GROUND_X,
    MAX_GROUND_Z,
    RADAR_DEPTH_COVERAGE_HISTORY_PATH,
    LATEST_TRACKING_IDENTITY_SNAPSHOT_PATH,
    RADAR_TRAIL_LENGTH,
    SIM_UPDATE_HZ,
    IDENTITY_SNAPSHOT_FRESHNESS_SECONDS,
    IDENTITY_SNAPSHOT_MAX_IDENTITIES,
    IDENTITY_SNAPSHOT_MAX_HISTORY_PER_IDENTITY,
    IDENTITY_SNAPSHOT_MAX_HISTORICAL_TRACK_IDS,
    IDENTITY_SNAPSHOT_MAX_PAYLOAD_BYTES,
)
from .state import SharedDashboardState


def _publish_identity_snapshot_if_enabled(state: SharedDashboardState, now: float) -> None:
    with state.lock:
        bridge_enabled = bool(state.tracking_identity_snapshot_bridge_enabled)
        registry_enabled = bool(state.tracking_identity_registry_enabled)
        registry = state.tracking_identity_registry
        last_source_version = state.latest_tracking_identity_snapshot_source_version
        last_published_at = float(state.latest_tracking_identity_snapshot_published_at)

    if not bridge_enabled:
        return

    should_refresh = (now - last_published_at) >= max(0.1, IDENTITY_SNAPSHOT_FRESHNESS_SECONDS / 2.0)
    source_version = registry.version if (registry_enabled and registry is not None) else None
    if not should_refresh and source_version is not None and last_source_version == source_version:
        return

    if not registry_enabled or registry is None:
        payload = build_identity_snapshot_unavailable_payload(
            now=now,
            freshness_window_seconds=IDENTITY_SNAPSHOT_FRESHNESS_SECONDS,
            state="unavailable",
            reason="identity_registry_unavailable",
        )
    else:
        payload = build_identity_snapshot_payload(
            registry=registry,
            now=now,
            freshness_window_seconds=IDENTITY_SNAPSHOT_FRESHNESS_SECONDS,
            max_identities=IDENTITY_SNAPSHOT_MAX_IDENTITIES,
            max_history_per_identity=IDENTITY_SNAPSHOT_MAX_HISTORY_PER_IDENTITY,
            max_historical_track_ids=IDENTITY_SNAPSHOT_MAX_HISTORICAL_TRACK_IDS,
            max_payload_bytes=IDENTITY_SNAPSHOT_MAX_PAYLOAD_BYTES,
        )

    try:
        publish_identity_snapshot(LATEST_TRACKING_IDENTITY_SNAPSHOT_PATH, payload)
    except OSError:
        with state.lock:
            state.latest_tracking_identity_snapshot_metadata = {
                "available": False,
                "state": "unavailable",
                "reason": "snapshot_write_failed",
                "published_at": float(now),
            }
        return

    with state.lock:
        state.latest_tracking_identity_snapshot_metadata = {
            "available": payload.get("state") == "available",
            "state": payload.get("state"),
            "reason": payload.get("reason"),
            "source_generation": payload.get("source_generation"),
            "source_version": payload.get("source_version"),
            "identity_count": payload.get("identity_count"),
            "payload_bytes": payload.get("payload_bytes"),
            "snapshot_created_at": payload.get("snapshot_created_at"),
            "fresh_until": payload.get("fresh_until"),
            "published_at": float(now),
        }
        source_version_value = payload.get("source_version")
        state.latest_tracking_identity_snapshot_source_version = (
            int(source_version_value)
            if isinstance(source_version_value, int)
            else None
        )
        state.latest_tracking_identity_snapshot_published_at = float(now)


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

        _publish_identity_snapshot_if_enabled(state, now)

        time.sleep(sim_interval)
