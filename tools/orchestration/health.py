# MCP health and pipeline-contract checks over latest radar state artifacts.
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RADAR_STATE_PATH = PROJECT_ROOT / "captures" / "latest_radar_state.json"


def _read_radar_state_payload() -> tuple[dict[str, Any] | None, str | None]:
    # Implements read radar state payload behavior for this module.
    if not RADAR_STATE_PATH.is_file():
        return None, "radar_state_missing"

    try:
        payload = json.loads(RADAR_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "radar_state_invalid_json"

    if not isinstance(payload, dict):
        return None, "radar_state_not_object"

    return payload, None


def _parse_radar_state(payload: dict[str, Any]) -> dict[str, Any]:
    # Implements parse radar state behavior for this module.
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
    # Implements dashboard health status behavior for this module.
    """Return dashboard health summary derived from captures/latest_radar_state.json."""
    payload, error = _read_radar_state_payload()
    if payload is None:
        return {
            "available": False,
            "radar_state_path": str(RADAR_STATE_PATH),
            "error": error,
        }

    return _parse_radar_state(payload)


@mcp.tool()
def verify_dashboard_pipeline_contract(
    max_observation_age_seconds: float = 3.0,
    require_live_pipeline: bool = False,
) -> dict[str, Any]:
    # Verifies dashboard pipeline contract contract/availability requirements.
    """Validate latest radar-state payload against critical connected-pipeline contract checks."""
    if max_observation_age_seconds < 0:
        raise ValueError("max_observation_age_seconds must be non-negative")

    payload, error = _read_radar_state_payload()
    if payload is None:
        return {
            "available": False,
            "radar_state_path": str(RADAR_STATE_PATH),
            "ready": False,
            "error": error,
            "findings": [
                {
                    "severity": "error",
                    "code": error,
                    "field": "latest_radar_state",
                    "message": "Radar state payload is unavailable for contract verification.",
                }
            ],
        }

    status = _parse_radar_state(payload)
    findings: list[dict[str, Any]] = []

    age = status.get("last_observation_age_seconds")
    if not isinstance(age, (int, float)):
        findings.append(
            {
                "severity": "error",
                "code": "missing_observation_age",
                "field": "last_observation_age_seconds",
                "message": "Missing last observation timestamp/age in radar state.",
            }
        )
    elif float(age) > max_observation_age_seconds:
        findings.append(
            {
                "severity": "error",
                "code": "stale_observation",
                "field": "last_observation_age_seconds",
                "message": f"Last observation age {float(age):.3f}s exceeds threshold {max_observation_age_seconds:.3f}s.",
            }
        )

    mapping_mode = status.get("mapping_mode")
    if mapping_mode == "simulation_fallback":
        severity = "error" if require_live_pipeline else "warn"
        code = "simulation_fallback_active"
        findings.append(
            {
                "severity": severity,
                "code": code,
                "field": "coordinate_system",
                "message": "Radar state currently indicates simulation fallback mode.",
            }
        )

    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        findings.append(
            {
                "severity": "error",
                "code": "tracks_not_list",
                "field": "tracks",
                "message": "Radar state tracks payload must be a list.",
            }
        )
        tracks = []

    valid_kinds = {"relative_ground_estimate", "calibrated_ground_plane", "simulated_ground_track"}
    for index, track in enumerate(tracks):
        field_prefix = f"tracks[{index}]"
        if not isinstance(track, dict):
            findings.append(
                {
                    "severity": "error",
                    "code": "track_not_object",
                    "field": field_prefix,
                    "message": "Track entry must be an object.",
                }
            )
            continue

        track_id = track.get("track_id")
        if not isinstance(track_id, str) or not track_id.strip():
            findings.append(
                {
                    "severity": "error",
                    "code": "track_id_missing",
                    "field": f"{field_prefix}.track_id",
                    "message": "Track is missing a non-empty track_id.",
                }
            )

        timestamp = track.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            findings.append(
                {
                    "severity": "error",
                    "code": "track_timestamp_missing",
                    "field": f"{field_prefix}.timestamp",
                    "message": "Track is missing numeric timestamp.",
                }
            )

        ground_position = track.get("ground_position")
        if not isinstance(ground_position, dict):
            findings.append(
                {
                    "severity": "error",
                    "code": "ground_position_missing",
                    "field": f"{field_prefix}.ground_position",
                    "message": "Track is missing ground_position object.",
                }
            )
        else:
            x_value = ground_position.get("x")
            z_value = ground_position.get("z")
            if not isinstance(x_value, (int, float)) or not isinstance(z_value, (int, float)):
                findings.append(
                    {
                        "severity": "error",
                        "code": "ground_position_invalid",
                        "field": f"{field_prefix}.ground_position",
                        "message": "ground_position.x and ground_position.z must be numeric.",
                    }
                )

        position_kind = track.get("position_kind")
        unit = track.get("ground_coordinate_unit")
        if not isinstance(position_kind, str) or position_kind not in valid_kinds:
            findings.append(
                {
                    "severity": "error",
                    "code": "position_kind_invalid",
                    "field": f"{field_prefix}.position_kind",
                    "message": "position_kind is missing or invalid.",
                }
            )
        else:
            if position_kind in {"relative_ground_estimate", "simulated_ground_track"} and unit != "relative_units":
                findings.append(
                    {
                        "severity": "error",
                        "code": "coordinate_unit_mismatch",
                        "field": f"{field_prefix}.ground_coordinate_unit",
                        "message": "Relative/simulated position_kind must use ground_coordinate_unit='relative_units'.",
                    }
                )
            if position_kind == "calibrated_ground_plane":
                if not isinstance(unit, str) or not unit.strip() or unit == "relative_units":
                    findings.append(
                        {
                            "severity": "error",
                            "code": "coordinate_unit_mismatch",
                            "field": f"{field_prefix}.ground_coordinate_unit",
                            "message": "Calibrated position_kind requires a non-relative ground_coordinate_unit.",
                        }
                    )

    error_count = sum(1 for finding in findings if finding.get("severity") == "error")
    warn_count = sum(1 for finding in findings if finding.get("severity") == "warn")

    return {
        "available": True,
        "ready": error_count == 0,
        "radar_state_path": str(RADAR_STATE_PATH),
        "summary": {
            "error_count": error_count,
            "warn_count": warn_count,
            "max_observation_age_seconds": max_observation_age_seconds,
            "require_live_pipeline": require_live_pipeline,
        },
        "status": status,
        "findings": findings,
    }


@mcp.tool()
def wait_for_dashboard_ready(
    timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    # Waits for for dashboard ready conditions within configured limits.
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


@mcp.tool()
def orchestration_health_status() -> dict[str, Any]:
    # Implements orchestration health status behavior for this module.
    return dashboard_health_status()


@mcp.tool()
def verify_orchestration_pipeline_contract(
    max_observation_age_seconds: float = 3.0,
    require_live_pipeline: bool = False,
) -> dict[str, Any]:
    # Verifies orchestration pipeline contract contract/availability requirements.
    return verify_dashboard_pipeline_contract(
        max_observation_age_seconds=max_observation_age_seconds,
        require_live_pipeline=require_live_pipeline,
    )


@mcp.tool()
def wait_for_orchestration_ready(
    timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict[str, Any]:
    # Waits for for orchestration ready conditions within configured limits.
    return wait_for_dashboard_ready(
        timeout_seconds=timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
