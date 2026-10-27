from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .agent_contracts import GroundPoint, TrackingObservationRecord
from .agent_coordinator import AgentCoordinator
from .registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADAR_STATE_PATH = PROJECT_ROOT / "captures" / "latest_radar_state.json"

_COORDINATOR = AgentCoordinator()
_INGESTED_OBSERVATION_IDS: set[str] = set()


def _reset_reasoning_state() -> None:
    global _COORDINATOR
    _COORDINATOR = AgentCoordinator()
    _INGESTED_OBSERVATION_IDS.clear()


def _load_radar_state() -> dict[str, Any] | None:
    if not RADAR_STATE_PATH.is_file():
        return None

    payload = json.loads(RADAR_STATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _ingest_latest_tracks(radar_state: dict[str, Any]) -> int:
    tracks = radar_state.get("tracks")
    if not isinstance(tracks, list):
        return 0

    ingested = 0
    for track in tracks:
        if not isinstance(track, dict):
            continue

        track_id = track.get("track_id")
        timestamp = track.get("timestamp")
        ground = track.get("ground_position")

        if not isinstance(track_id, str):
            continue
        if not isinstance(timestamp, (int, float)):
            continue
        if not isinstance(ground, dict):
            continue

        x_value = ground.get("x")
        z_value = ground.get("z")
        if not isinstance(x_value, (int, float)) or not isinstance(z_value, (int, float)):
            continue

        observation_id = f"{track_id}:{float(timestamp):.6f}"
        if observation_id in _INGESTED_OBSERVATION_IDS:
            continue

        pixel_position = track.get("pixel_position")
        pixel_x = None
        pixel_y = None
        if isinstance(pixel_position, dict):
            if isinstance(pixel_position.get("x"), (int, float)):
                pixel_x = float(pixel_position["x"])
            if isinstance(pixel_position.get("y"), (int, float)):
                pixel_y = float(pixel_position["y"])

        confidence = track.get("confidence")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None

        quality_flags = track.get("quality_flags")
        if not isinstance(quality_flags, list):
            quality_flags = []
        normalized_flags = [str(flag) for flag in quality_flags]

        _COORDINATOR.add_tracking_observation(
            TrackingObservationRecord(
                observation_id=observation_id,
                track_id=track_id,
                timestamp=float(timestamp),
                source="radar_state",
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                ground_point=GroundPoint(
                    x=float(x_value),
                    z=float(z_value),
                    unit="unknown",
                ),
                confidence=confidence_value,
                quality_flags=normalized_flags,
            )
        )
        _INGESTED_OBSERVATION_IDS.add(observation_id)
        ingested += 1

    return ingested


def _prepare_track_analysis(track_id: str) -> tuple[int, float] | None:
    if not isinstance(track_id, str) or not track_id.strip():
        raise ValueError("track_id must be a non-empty string")

    radar_state = _load_radar_state()
    if radar_state is None:
        return None

    ingested_count = _ingest_latest_tracks(radar_state)
    return (ingested_count, time.time())


@mcp.tool()
def simulate_evidence_request(track_id: str, reason: str) -> dict[str, Any]:
    """Issue a bounded evidence request for a track through the coordinator."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    prepared = _prepare_track_analysis(track_id)
    if prepared is None:
        return {
            "available": False,
            "error": "radar_state_unavailable",
            "track_id": track_id,
        }

    ingested_count, now = prepared
    request = _COORDINATOR.request_evidence(
        track_id=track_id,
        reason=reason,
        requested_at=now,
    )

    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "request": None if request is None else request.to_dict(),
        "bounded_out": request is None,
    }


@mcp.tool()
def simulate_observation_summary(track_id: str) -> dict[str, Any]:
    """Build only the observation/behavior summary for one track from latest radar state."""
    prepared = _prepare_track_analysis(track_id)
    if prepared is None:
        return {
            "available": False,
            "error": "radar_state_unavailable",
            "track_id": track_id,
        }

    ingested_count, now = prepared
    summary = _COORDINATOR.build_observation_summary(track_id, created_at=now)
    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "summary": summary.to_dict(),
    }


@mcp.tool()
def simulate_reasoning_assessment(
    track_id: str,
    request_evidence_on_uncertainty: bool = True,
) -> dict[str, Any]:
    """Build an observation summary and reasoning assessment for one track from latest radar state."""
    prepared = _prepare_track_analysis(track_id)
    if prepared is None:
        return {
            "available": False,
            "error": "radar_state_unavailable",
            "track_id": track_id,
        }

    ingested_count, now = prepared
    summary = _COORDINATOR.build_observation_summary(track_id, created_at=now)
    assessment = _COORDINATOR.build_reasoning_assessment(track_id, created_at=now)

    evidence_request = None
    if request_evidence_on_uncertainty and summary.speed is None:
        request = _COORDINATOR.request_evidence(
            track_id=track_id,
            reason="insufficient_history_for_motion_assessment",
            requested_at=now,
        )
        if request is not None:
            evidence_request = request.to_dict()

    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "summary": summary.to_dict(),
        "assessment": assessment.to_dict(),
        "evidence_request": evidence_request,
    }
