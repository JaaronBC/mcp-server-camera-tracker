from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from agents.components.dashboard_types import GroundPosition, RadarTrackSnapshot
from agents.tracking.runtime_support import compute_relative_depth_coverage


def write_json_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, separators=(",", ":")) + "\n")


def build_tracking_simulated_tracks(
    sim_time: float,
    *,
    max_ground_x: float,
    max_ground_z: float,
) -> dict[str, RadarTrackSnapshot]:
    simulated: dict[str, RadarTrackSnapshot] = {}

    sim_tracks = [
        ("sim-1", -4.0 + 0.8 * sim_time, 8.0 + 1.0 * (sim_time % 4.0)),
        ("sim-2", 3.0 + 1.2 * (sim_time % 2.5), 5.5 + 0.8 * sim_time),
    ]

    for track_id, x_position, z_position in sim_tracks:
        wrapped_x = ((x_position + max_ground_x) % (2 * max_ground_x)) - max_ground_x
        wrapped_z = z_position % max_ground_z

        simulated[track_id] = RadarTrackSnapshot(
            track_id=track_id,
            timestamp=time.time(),
            pixel_position=None,
            ground_position=GroundPosition(x=wrapped_x, z=wrapped_z),
            ground_coordinate_unit="relative_units",
            position_kind="simulated_ground_track",
            quality_flags=["simulated_data"],
            confidence=None,
            relative_depth=None,
        )

    return simulated


def apply_simulation_fallback_state(
    *,
    state: Any,
    now: float,
    simulated_tracks: dict[str, RadarTrackSnapshot],
    trail_length: int,
) -> None:
    with state.lock:
        state.mapping_mode = "simulation_fallback"
        state.active_tracks = simulated_tracks
        state.last_observation_time = now
        for track_key, snapshot in simulated_tracks.items():
            if track_key not in state.track_trails:
                state.track_trails[track_key] = deque(maxlen=trail_length)
            state.track_trails[track_key].append(
                (snapshot.ground_position.x, snapshot.ground_position.z)
            )


def build_tracking_frame_message(timestamp: float, detections: list[dict]) -> dict:
    return {
        "time": timestamp,
        "detections": detections,
    }


def publish_tracking_frame_message(
    *,
    tracking_frame_path: Path,
    timestamp: float,
    detections: list[dict],
    udp_socket: Any | None = None,
    udp_host: str | None = None,
    udp_port: int | None = None,
) -> dict:
    tracking_message = build_tracking_frame_message(timestamp, detections)
    write_json_payload(tracking_frame_path, tracking_message)

    if udp_socket is not None and isinstance(udp_host, str) and isinstance(udp_port, int):
        try:
            udp_socket.sendto(
                json.dumps(tracking_message).encode("utf-8"),
                (udp_host, udp_port),
            )
        except OSError:
            pass

    return tracking_message


def build_tracking_agent_status_from_state(state: Any) -> dict[str, Any]:
    return {
        "model": state.tracking_agent_model,
        "submissions": state.tracking_agent_submissions,
        "completions": state.tracking_agent_completions,
        "unavailable_count": state.tracking_agent_unavailable_count,
        "total_updates_accepted": state.tracking_agent_total_updates_accepted,
        "total_updates_rejected": state.tracking_agent_total_updates_rejected,
        "total_updates_unmatched": state.tracking_agent_total_updates_unmatched,
        "last_latency_ms": state.tracking_agent_last_latency_ms,
        "last_error": state.tracking_agent_last_error,
    }


def collect_radar_publish_snapshot(state: Any) -> dict[str, Any]:
    with state.lock:
        snapshot_tracks = list(state.active_tracks.values())
        coordinate_system = state.mapping_mode
        camera_status = state.camera_status
        detector_status = state.detector_status
        calibration_status = state.calibration_status
        last_observation_time = state.last_observation_time
        tracking_agent_observation = state.latest_tracking_agent_observation
        tracking_agent_status = build_tracking_agent_status_from_state(state)

    if calibration_status == "loaded" and coordinate_system != "simulation_fallback":
        coordinate_system = "calibrated_ground_plane"

    return {
        "tracks": snapshot_tracks,
        "coordinate_system": coordinate_system,
        "camera_status": camera_status,
        "detector_status": detector_status,
        "calibration_status": calibration_status,
        "last_observation_time": last_observation_time,
        "tracking_agent_observation": tracking_agent_observation,
        "tracking_agent_status": tracking_agent_status,
    }


def publish_tracking_radar_state(
    *,
    radar_state_path: Path,
    radar_depth_coverage_history_path: Path,
    now: float,
    coordinate_system: str,
    camera_status: str,
    detector_status: str,
    calibration_status: str,
    tracking_agent_status: dict,
    last_observation_timestamp: float,
    tracking_agent_observation: dict | None,
    tracks: list[RadarTrackSnapshot],
) -> dict[str, float | int]:
    relative_depth_coverage = compute_relative_depth_coverage(tracks)

    write_json_payload(
        radar_state_path,
        {
            "timestamp": now,
            "coordinate_system": coordinate_system,
            "camera_status": camera_status,
            "detector_status": detector_status,
            "calibration_status": calibration_status,
            "tracking_agent_status": tracking_agent_status,
            "active_track_count": len(tracks),
            "last_observation_timestamp": last_observation_timestamp,
            "relative_depth_coverage": relative_depth_coverage,
            "tracking_agent_observation": tracking_agent_observation,
            "tracks": [track.to_dict() for track in tracks],
        },
    )

    append_jsonl_payload(
        radar_depth_coverage_history_path,
        {
            "timestamp": now,
            "coverage": relative_depth_coverage,
        },
    )

    return relative_depth_coverage
