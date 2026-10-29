# MCP reasoning tools that build observation/reasoning outputs from radar state with LLM-first and deterministic fallback paths.
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ..contracts import GroundPoint, TrackingObservationRecord
from ..coordinator import AgentCoordinator
from ..observation import agent_observation as observation_agent_module
from . import agent_reasoning as reasoning_agent_module

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RADAR_STATE_PATH = PROJECT_ROOT / "captures" / "latest_radar_state.json"

DEFAULT_RADAR_STATE_MAX_AGE_SECONDS = 5.0
MAX_HISTORY_RECORDS_FOR_LLM = 12

_COORDINATOR = AgentCoordinator()
_INGESTED_OBSERVATION_IDS: set[str] = set()


def _build_unavailable_track_response(track_id: str, prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": False,
        "error": prepared.get("error", "radar_state_unavailable"),
        "track_id": track_id,
        "radar_state_age_seconds": prepared.get("radar_state_age_seconds"),
        "radar_state_max_age_seconds": prepared.get("radar_state_max_age_seconds"),
    }


def _attach_llm_meta(payload: dict[str, Any], llm_payload: dict[str, Any] | None, llm_meta: dict[str, Any]) -> None:
    payload["llm_available"] = llm_payload is not None
    payload["llm_model"] = llm_meta.get("model")
    payload["llm_latency_ms"] = llm_meta.get("latency_ms")
    payload["llm_error"] = llm_meta.get("error")


def _default_handoff_validation() -> dict[str, Any]:
    return {
        "status": "ok",
        "error": None,
        "error_code": None,
    }


def _handoff_validation_error(error: Exception) -> dict[str, Any]:
    return {
        "status": "error",
        "error": f"{type(error).__name__}:{error}",
        "error_code": "invalid_handoff_payload",
    }


def _resolve_radar_state_max_age_seconds() -> float:
    # Resolves radar state max age seconds from configuration with safe defaults.
    value = os.getenv("DASHBOARD_RADAR_STATE_MAX_AGE_SECONDS")
    if value is None:
        return DEFAULT_RADAR_STATE_MAX_AGE_SECONDS
    try:
        max_age = float(value)
    except ValueError:
        return DEFAULT_RADAR_STATE_MAX_AGE_SECONDS
    if max_age <= 0:
        return DEFAULT_RADAR_STATE_MAX_AGE_SECONDS
    return max_age


def _normalize_track_coordinate_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    # Delegates coordinate metadata normalization to observation agent module.
    return observation_agent_module.normalize_track_coordinate_metadata(snapshot)


def _build_observation_llm_input(
    track_id: str,
    created_at: float,
    radar_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    # Delegates observation input construction to observation agent module.
    return observation_agent_module.build_observation_llm_input(
        track_id,
        created_at,
        coordinator=_COORDINATOR,
        max_history_records=MAX_HISTORY_RECORDS_FOR_LLM,
        radar_state=radar_state,
    )


def _extract_latest_track_snapshot(
    radar_state: dict[str, Any] | None,
    track_id: str,
) -> dict[str, Any]:
    # Delegates latest track extraction to observation agent module.
    return observation_agent_module.extract_latest_track_snapshot(radar_state, track_id)


def _extract_tracking_agent_context(
    radar_state: dict[str, Any] | None,
    track_id: str,
) -> dict[str, Any]:
    # Delegates tracking agent context extraction to observation agent module.
    return observation_agent_module.extract_tracking_agent_context(radar_state, track_id)


def _extract_coordinate_context(radar_state: dict[str, Any] | None) -> dict[str, Any]:
    # Delegates coordinate context extraction to observation agent module.
    return observation_agent_module.extract_coordinate_context(radar_state)


def _validate_observation_llm_payload(
    payload: dict[str, Any],
    fallback_summary: dict[str, Any],
    valid_observation_ids: set[str],
) -> dict[str, Any]:
    # Delegates observation payload validation to observation agent module.
    return observation_agent_module.validate_observation_llm_payload(
        payload,
        fallback_summary,
        valid_observation_ids,
    )


def _validate_reasoning_llm_payload(
    payload: dict[str, Any],
    track_id: str,
    created_at: float,
    valid_observation_ids: set[str],
) -> tuple[dict[str, Any], bool]:
    # Delegates reasoning payload validation to reasoning agent module.
    return reasoning_agent_module.validate_reasoning_llm_payload(
        payload,
        track_id,
        created_at,
        valid_observation_ids,
    )


def _build_reasoning_handoff_payload(
    track_id: str,
    created_at: float,
    summary: dict[str, Any],
    history_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Delegates reasoning handoff validation to reasoning agent module.
    return reasoning_agent_module.build_reasoning_handoff_payload(
        track_id,
        created_at,
        summary,
        history_records,
    )


def _skipped_reasoning_meta(error: Exception) -> dict[str, Any]:
    # Implements skipped reasoning meta behavior for this module.
    model = reasoning_agent_module._resolve_model()
    return {
        "status": "skipped",
        "model": model,
        "latency_ms": 0.0,
        "error": f"{type(error).__name__}:{error}",
        "error_code": "invalid_handoff_payload",
        "attempts": 0,
        "retry_count": 0,
    }


def _call_observation_agent_with_retry(
    user_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    # Delegates retry + invoke policy to the observation agent module.
    return observation_agent_module.call_observation_agent_with_retry(user_payload)


def _call_reasoning_agent_with_retry(
    user_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    # Delegates retry + invoke policy to the reasoning agent module.
    return reasoning_agent_module.call_reasoning_agent_with_retry(user_payload)


def _fallback_observation_summary(
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    # Implements fallback observation summary behavior for this module.
    fallback = dict(summary_payload)
    fallback.update(
        {
            "summary": "Fallback observation summary: deterministic measurements only.",
            "measured_facts": [],
            "interpretations": [],
            "hypotheses": [],
            "conflicting_evidence": [],
            "needs_more_observations": fallback.get("speed") is None,
        }
    )
    return fallback


def _fallback_reasoning_assessment(
    track_id: str,
    created_at: float,
    summary: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    # Implements fallback reasoning assessment behavior for this module.
    deterministic = _COORDINATOR.build_reasoning_assessment(track_id, created_at=created_at).to_dict()
    deterministic.update(
        {
            "rationale": "Fallback reasoning assessment: deterministic rule path.",
            "recommendation": "requires_operator_review",
            "limitations": ["llm_unavailable"],
            "alternative_interpretations": [],
            "missing_evidence": ["additional_history"],
            "contradictory_evidence": [],
        }
    )
    request_more = bool(summary.get("needs_more_observations", False))
    return deterministic, request_more


def _reset_reasoning_state() -> None:
    # Implements reset reasoning state behavior for this module.
    global _COORDINATOR
    _COORDINATOR = AgentCoordinator()
    _INGESTED_OBSERVATION_IDS.clear()


def _load_radar_state() -> dict[str, Any] | None:
    # Loads radar state data from storage for downstream processing.
    if not RADAR_STATE_PATH.is_file():
        return None

    payload = json.loads(RADAR_STATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _radar_state_age_seconds(now_timestamp: float | None = None) -> float | None:
    # Implements radar state age seconds behavior for this module.
    if not RADAR_STATE_PATH.is_file():
        return None

    current_timestamp = time.time() if now_timestamp is None else float(now_timestamp)
    try:
        modified_at = RADAR_STATE_PATH.stat().st_mtime
    except OSError:
        return None

    age_seconds = current_timestamp - float(modified_at)
    if age_seconds < 0:
        return 0.0
    return age_seconds


def _ingest_latest_tracks(radar_state: dict[str, Any]) -> int:
    # Implements ingest latest tracks behavior for this module.
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


def _prepare_track_analysis(track_id: str) -> dict[str, Any]:
    # Implements prepare track analysis behavior for this module.
    if not isinstance(track_id, str) or not track_id.strip():
        raise ValueError("track_id must be a non-empty string")

    radar_state = _load_radar_state()
    if radar_state is None:
        return {
            "available": False,
            "error": "radar_state_unavailable",
        }

    now = time.time()
    max_age_seconds = _resolve_radar_state_max_age_seconds()
    age_seconds = _radar_state_age_seconds(now_timestamp=now)
    if isinstance(age_seconds, (int, float)) and age_seconds > max_age_seconds:
        return {
            "available": False,
            "error": "radar_state_stale",
            "radar_state_age_seconds": round(float(age_seconds), 3),
            "radar_state_max_age_seconds": max_age_seconds,
        }

    ingested_count = _ingest_latest_tracks(radar_state)
    return {
        "available": True,
        "radar_state": radar_state,
        "ingested_count": ingested_count,
        "now": now,
    }


def simulate_evidence_request(track_id: str, reason: str) -> dict[str, Any]:
    # Issues bounded evidence requests for a track through coordinator policy.
    """Issue a bounded evidence request for a track through the coordinator."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    prepared = _prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return _build_unavailable_track_response(track_id, prepared)

    _radar_state = prepared["radar_state"]
    ingested_count = prepared["ingested_count"]
    now = prepared["now"]
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


def simulate_observation_summary(track_id: str) -> dict[str, Any]:
    # Builds observation-agent summary output for one track from latest ingested radar state.
    """Build only the observation/behavior summary for one track from latest radar state."""
    prepared = _prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return _build_unavailable_track_response(track_id, prepared)

    radar_state = prepared["radar_state"]
    ingested_count = prepared["ingested_count"]
    now = prepared["now"]
    base_summary, history_records, supporting_ids, llm_inputs = _build_observation_llm_input(
        track_id,
        created_at=now,
        radar_state=radar_state,
    )
    observation_payload, observation_meta = _call_observation_agent_with_retry(llm_inputs[0])
    if observation_payload is None:
        summary = _fallback_observation_summary(base_summary)
    else:
        summary = _validate_observation_llm_payload(
            observation_payload,
            fallback_summary=base_summary,
            valid_observation_ids=set(supporting_ids),
        )

    _attach_llm_meta(summary, observation_payload, observation_meta)

    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "summary": summary,
        "agents": {
            "observation": observation_meta,
        },
        "history_records": [record.to_dict() for record in history_records],
    }


def simulate_reasoning_assessment(
    track_id: str,
    request_evidence_on_uncertainty: bool = True,
) -> dict[str, Any]:
    # Builds full observation+reasoning assessment output for one track with bounded evidence requests.
    """Build an observation summary and reasoning assessment for one track from latest radar state."""
    prepared = _prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return _build_unavailable_track_response(track_id, prepared)

    radar_state = prepared["radar_state"]
    ingested_count = prepared["ingested_count"]
    now = prepared["now"]
    base_summary, history_records, supporting_ids, llm_inputs = _build_observation_llm_input(
        track_id,
        created_at=now,
        radar_state=radar_state,
    )

    observation_payload, observation_meta = _call_observation_agent_with_retry(llm_inputs[0])
    if observation_payload is None:
        summary = _fallback_observation_summary(base_summary)
    else:
        summary = _validate_observation_llm_payload(
            observation_payload,
            fallback_summary=base_summary,
            valid_observation_ids=set(supporting_ids),
        )

    _attach_llm_meta(summary, observation_payload, observation_meta)

    handoff_validation = _default_handoff_validation()
    try:
        reasoning_input, handoff_validation = _build_reasoning_handoff_payload(
            track_id=track_id,
            created_at=now,
            summary=summary,
            history_records=[record.to_dict() for record in history_records],
        )
        reasoning_payload, reasoning_meta = _call_reasoning_agent_with_retry(reasoning_input)
    except ValueError as error:
        reasoning_payload = None
        reasoning_meta = _skipped_reasoning_meta(error)
        handoff_validation = _handoff_validation_error(error)

    if reasoning_payload is None:
        assessment, request_more_observations = _fallback_reasoning_assessment(
            track_id,
            created_at=now,
            summary=summary,
        )
    else:
        assessment, request_more_observations = _validate_reasoning_llm_payload(
            reasoning_payload,
            track_id=track_id,
            created_at=now,
            valid_observation_ids=set(summary.get("supporting_observation_ids", [])),
        )

    _attach_llm_meta(assessment, reasoning_payload, reasoning_meta)

    evidence_request = None
    if request_evidence_on_uncertainty and request_more_observations:
        request = _COORDINATOR.request_evidence(
            track_id=track_id,
            reason="llm_requested_more_observations",
            requested_at=now,
        )
        if request is not None:
            evidence_request = request.to_dict()

    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "summary": summary,
        "assessment": assessment,
        "evidence_request": evidence_request,
        "agents": {
            "observation": observation_meta,
            "reasoning": reasoning_meta,
        },
        "handoff_validation": handoff_validation,
    }
