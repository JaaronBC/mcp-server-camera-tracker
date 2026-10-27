# MCP reasoning tools that build observation/reasoning outputs from radar state with LLM-first and deterministic fallback paths.
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .agent_contracts import GroundPoint, TrackingObservationRecord
from .agent_coordinator import AgentCoordinator
from .registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADAR_STATE_PATH = PROJECT_ROOT / "captures" / "latest_radar_state.json"

DEFAULT_OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
DEFAULT_OBSERVATION_AGENT_MODEL = "gpt-oss-120b"
DEFAULT_REASONING_AGENT_MODEL = "gpt-oss-120b"
DEFAULT_LLM_TIMEOUT_SECONDS = 8.0
DEFAULT_LLM_RETRY_LIMIT = 1
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 0.1
DEFAULT_RADAR_STATE_MAX_AGE_SECONDS = 5.0
MAX_HISTORY_RECORDS_FOR_LLM = 12

_COORDINATOR = AgentCoordinator()
_INGESTED_OBSERVATION_IDS: set[str] = set()


def _resolve_model(env_name: str, default_model: str) -> str:
    # Resolves model from configuration with safe defaults.
    value = os.getenv(env_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default_model


def _resolve_timeout_seconds() -> float:
    # Resolves timeout seconds from configuration with safe defaults.
    value = os.getenv("DASHBOARD_AGENT_LLM_TIMEOUT_SECONDS")
    if value is None:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    if timeout <= 0:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    return timeout


def _resolve_retry_limit() -> int:
    # Resolves retry limit from configuration with safe defaults.
    value = os.getenv("DASHBOARD_AGENT_LLM_RETRY_LIMIT")
    if value is None:
        return DEFAULT_LLM_RETRY_LIMIT
    try:
        retry_limit = int(value)
    except ValueError:
        return DEFAULT_LLM_RETRY_LIMIT
    if retry_limit < 0:
        return DEFAULT_LLM_RETRY_LIMIT
    return retry_limit


def _resolve_retry_backoff_seconds() -> float:
    # Resolves retry backoff seconds from configuration with safe defaults.
    value = os.getenv("DASHBOARD_AGENT_LLM_RETRY_BACKOFF_SECONDS")
    if value is None:
        return DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    try:
        backoff_seconds = float(value)
    except ValueError:
        return DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    if backoff_seconds < 0:
        return DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    return backoff_seconds


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


def _classify_llm_error(error: Exception) -> str:
    # Classifies llm error into normalized categories.
    text = f"{type(error).__name__}:{error}".lower()
    if "timeout" in text:
        return "timeout"
    if "api_key" in text or "unauthorized" in text or "authentication" in text:
        return "authentication"
    if "llm_invalid_json_response" in text or "json" in text:
        return "invalid_response_format"
    if "llm_empty_response" in text or "empty_response" in text:
        return "empty_response"
    return "runtime_error"


def _create_openai_client() -> AsyncOpenAI:
    # Implements create openai client behavior for this module.
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    # Extracts json object from structured payloads.
    text = raw_text.strip()
    if not text:
        return None

    if text.startswith("```"):
        blocks = text.split("```")
        text = ""
        for block in blocks:
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
        if not text:
            return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_str_list(value: Any) -> list[str]:
    # Implements coerce str list behavior for this module.
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _coerce_bool(value: Any, default: bool = False) -> bool:
    # Implements coerce bool behavior for this module.
    if isinstance(value, bool):
        return value
    return default


def _coerce_str(value: Any, default: str) -> str:
    # Implements coerce str behavior for this module.
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _normalize_track_coordinate_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    # Normalizes track coordinate metadata into contract-safe values.
    normalized = dict(snapshot)
    quality_flags = normalized.get("quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = []
    else:
        quality_flags = [item for item in quality_flags if isinstance(item, str) and item.strip()]

    position_kind = normalized.get("position_kind")
    ground_coordinate_unit = normalized.get("ground_coordinate_unit")

    valid_kinds = {"relative_ground_estimate", "calibrated_ground_plane", "simulated_ground_track"}
    if not isinstance(position_kind, str) or position_kind not in valid_kinds:
        position_kind = "relative_ground_estimate"
        ground_coordinate_unit = "relative_units"
        quality_flags.append("position_kind_normalized:unknown_kind")

    if position_kind == "relative_ground_estimate":
        if ground_coordinate_unit != "relative_units":
            ground_coordinate_unit = "relative_units"
            quality_flags.append("ground_unit_normalized:relative_units")
    elif position_kind == "calibrated_ground_plane":
        if (
            not isinstance(ground_coordinate_unit, str)
            or not ground_coordinate_unit.strip()
            or ground_coordinate_unit == "relative_units"
        ):
            position_kind = "relative_ground_estimate"
            ground_coordinate_unit = "relative_units"
            quality_flags.append("position_kind_normalized:invalid_calibrated_unit")
    elif position_kind == "simulated_ground_track":
        if ground_coordinate_unit != "relative_units":
            ground_coordinate_unit = "relative_units"
            quality_flags.append("ground_unit_normalized:relative_units")

    normalized["position_kind"] = position_kind
    normalized["ground_coordinate_unit"] = ground_coordinate_unit
    normalized["quality_flags"] = quality_flags
    return normalized


def _build_observation_llm_input(
    track_id: str,
    created_at: float,
    radar_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    # Builds observation llm input payloads for pipeline consumers.
    history = _COORDINATOR.get_track_history(track_id)
    relevant_history = history[-MAX_HISTORY_RECORDS_FOR_LLM:]
    deterministic_summary = _COORDINATOR.build_observation_summary(track_id, created_at=created_at)
    supporting_ids = [record.observation_id for record in relevant_history]

    records_payload = [record.to_dict() for record in relevant_history]

    summary_payload = deterministic_summary.to_dict()
    summary_payload["supporting_observation_ids"] = supporting_ids

    latest_track_snapshot = _extract_latest_track_snapshot(radar_state, track_id)
    tracking_agent_context = _extract_tracking_agent_context(radar_state, track_id)
    coordinate_context = _extract_coordinate_context(radar_state)
    summary_payload["latest_track_snapshot"] = latest_track_snapshot
    summary_payload["tracking_agent_context"] = tracking_agent_context
    summary_payload["coordinate_context"] = coordinate_context

    measurements_payload = {
        "velocity_x": deterministic_summary.velocity_x,
        "velocity_z": deterministic_summary.velocity_z,
        "speed": deterministic_summary.speed,
        "confidence": deterministic_summary.confidence,
        "sample_count": deterministic_summary.sample_count,
    }

    llm_input = {
        "track_id": track_id,
        "created_at": created_at,
        "summary_measurements": measurements_payload,
        "quality_uncertainties": deterministic_summary.uncertainties,
        "history_records": records_payload,
        "latest_track_snapshot": latest_track_snapshot,
        "tracking_agent_context": tracking_agent_context,
        "coordinate_context": coordinate_context,
    }

    return summary_payload, relevant_history, supporting_ids, [llm_input]


def _extract_latest_track_snapshot(
    radar_state: dict[str, Any] | None,
    track_id: str,
) -> dict[str, Any]:
    # Extracts latest track snapshot from structured payloads.
    if not isinstance(radar_state, dict):
        return {"available": False}

    tracks = radar_state.get("tracks")
    if not isinstance(tracks, list):
        return {"available": False}

    selected_track = None
    for item in tracks:
        if not isinstance(item, dict):
            continue
        if item.get("track_id") == track_id:
            selected_track = item
            break

    if not isinstance(selected_track, dict):
        return {"available": False}

    snapshot = {
        "available": True,
        "track_id": selected_track.get("track_id"),
        "timestamp": selected_track.get("timestamp"),
        "pixel_position": selected_track.get("pixel_position"),
        "ground_position": selected_track.get("ground_position"),
        "ground_coordinate_unit": selected_track.get("ground_coordinate_unit"),
        "position_kind": selected_track.get("position_kind"),
        "relative_depth": selected_track.get("relative_depth"),
        "confidence": selected_track.get("confidence"),
        "quality_flags": selected_track.get("quality_flags"),
    }
    return _normalize_track_coordinate_metadata(snapshot)


def _extract_tracking_agent_context(
    radar_state: dict[str, Any] | None,
    track_id: str,
) -> dict[str, Any]:
    # Extracts tracking agent context from structured payloads.
    if not isinstance(radar_state, dict):
        return {"available": False}

    observation = radar_state.get("tracking_agent_observation")
    if not isinstance(observation, dict):
        return {"available": False}

    raw_updates = observation.get("updates")
    selected_updates: list[dict[str, Any]] = []
    if isinstance(raw_updates, list):
        for item in raw_updates:
            if not isinstance(item, dict):
                continue

            detection_id = item.get("detection_id")
            resolved_track_id = item.get("resolved_track_id")
            if detection_id != track_id and resolved_track_id != track_id:
                continue

            selected_updates.append(
                {
                    "detection_id": detection_id,
                    "resolved_track_id": resolved_track_id,
                    "accepted": item.get("accepted"),
                    "rejection_reasons": item.get("rejection_reasons"),
                    "uncertainties": item.get("uncertainties"),
                    "supporting_evidence": item.get("supporting_evidence"),
                }
            )

    return {
        "available": True,
        "source_agent": observation.get("source_agent"),
        "vlm_model": observation.get("vlm_model"),
        "timestamp": observation.get("timestamp"),
        "proposal_latency_ms": observation.get("proposal_latency_ms"),
        "uncertainties": observation.get("uncertainties"),
        "selected_updates": selected_updates,
    }


def _extract_coordinate_context(radar_state: dict[str, Any] | None) -> dict[str, Any]:
    # Extracts coordinate context from structured payloads.
    if not isinstance(radar_state, dict):
        return {"available": False}

    return {
        "available": True,
        "coordinate_system": radar_state.get("coordinate_system"),
        "calibration_status": radar_state.get("calibration_status"),
        "camera_status": radar_state.get("camera_status"),
        "detector_status": radar_state.get("detector_status"),
        "last_observation_timestamp": radar_state.get("last_observation_timestamp"),
    }


def _validate_observation_llm_payload(
    payload: dict[str, Any],
    fallback_summary: dict[str, Any],
    valid_observation_ids: set[str],
) -> dict[str, Any]:
    # Implements validate observation llm payload behavior for this module.
    interpretation = {
        "summary": _coerce_str(
            payload.get("summary"),
            "Observation agent could not produce a structured interpretation.",
        ),
        "measured_facts": _coerce_str_list(payload.get("measured_facts")),
        "interpretations": _coerce_str_list(payload.get("interpretations")),
        "hypotheses": _coerce_str_list(payload.get("hypotheses")),
        "conflicting_evidence": _coerce_str_list(payload.get("conflicting_evidence")),
        "uncertainties": _coerce_str_list(payload.get("uncertainties")),
        "needs_more_observations": _coerce_bool(payload.get("needs_more_observations"), default=False),
    }

    candidate_ids = _coerce_str_list(payload.get("supporting_observation_ids"))
    filtered_ids = [item for item in candidate_ids if item in valid_observation_ids]
    if not filtered_ids:
        filtered_ids = list(fallback_summary.get("supporting_observation_ids", []))

    summary = dict(fallback_summary)
    summary.update(interpretation)
    summary["supporting_observation_ids"] = filtered_ids
    return summary


def _validate_reasoning_llm_payload(
    payload: dict[str, Any],
    track_id: str,
    created_at: float,
    valid_observation_ids: set[str],
) -> tuple[dict[str, Any], bool]:
    # Implements validate reasoning llm payload behavior for this module.
    label = _coerce_str(payload.get("assessment_label"), "requires_more_observation")
    if label not in {"stable_motion", "requires_review", "requires_more_observation"}:
        label = "requires_more_observation"

    supporting_ids = [
        item for item in _coerce_str_list(payload.get("supporting_observation_ids"))
        if item in valid_observation_ids
    ]

    request_more_observations = _coerce_bool(
        payload.get("request_more_observations"),
        default=(label == "requires_more_observation"),
    )

    assessment = {
        "track_id": track_id,
        "created_at": created_at,
        "assessment_label": label,
        "summary": _coerce_str(payload.get("summary"), "Reasoning agent produced no summary."),
        "rationale": _coerce_str(payload.get("rationale"), "No rationale provided."),
        "recommendation": _coerce_str(payload.get("recommendation"), "requires_operator_review"),
        "supporting_observation_ids": supporting_ids,
        "uncertainties": _coerce_str_list(payload.get("uncertainties")),
        "limitations": _coerce_str_list(payload.get("limitations")),
        "alternative_interpretations": _coerce_str_list(payload.get("alternative_interpretations")),
        "missing_evidence": _coerce_str_list(payload.get("missing_evidence")),
        "contradictory_evidence": _coerce_str_list(payload.get("contradictory_evidence")),
        "source_agent": "reasoning",
        "score": None,
        "score_method": None,
    }

    return assessment, request_more_observations


def _build_reasoning_handoff_payload(
    track_id: str,
    created_at: float,
    summary: dict[str, Any],
    history_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Builds reasoning handoff payload payloads for pipeline consumers.
    if not isinstance(summary, dict):
        raise ValueError("observation_summary must be a dict")
    if not isinstance(history_records, list):
        raise ValueError("history_records must be a list")

    summary_supporting_ids = summary.get("supporting_observation_ids")
    if not isinstance(summary_supporting_ids, list):
        raise ValueError("supporting_observation_ids must be a list")

    valid_history_ids: set[str] = set()
    normalized_history_records: list[dict[str, Any]] = []
    for record in history_records:
        if not isinstance(record, dict):
            raise ValueError("history record must be an object")
        observation_id = record.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("history record missing observation_id")
        valid_history_ids.add(observation_id)
        normalized_history_records.append(record)

    normalized_supporting_ids = [
        item
        for item in _coerce_str_list(summary_supporting_ids)
        if item in valid_history_ids
    ]
    if not normalized_supporting_ids:
        raise ValueError("no valid supporting observation ids for reasoning handoff")

    normalized_summary = dict(summary)
    normalized_summary["supporting_observation_ids"] = normalized_supporting_ids

    handoff_payload = {
        "track_id": track_id,
        "created_at": created_at,
        "observation_summary": normalized_summary,
        "history_records": normalized_history_records,
    }
    return handoff_payload, {
        "status": "ok",
        "error": None,
        "error_code": None,
    }


def _skipped_reasoning_meta(error: Exception) -> dict[str, Any]:
    # Implements skipped reasoning meta behavior for this module.
    return {
        "status": "skipped",
        "model": _resolve_model("REASONING_AGENT_MODEL", DEFAULT_REASONING_AGENT_MODEL),
        "latency_ms": 0.0,
        "error": f"{type(error).__name__}:{error}",
        "error_code": "invalid_handoff_payload",
        "attempts": 0,
        "retry_count": 0,
    }


async def _invoke_llm_json_response(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    # Implements invoke llm json response behavior for this module.
    client = _create_openai_client()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            ),
            timeout=timeout_seconds,
        )

        if not response.choices:
            raise RuntimeError("llm_empty_response")

        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            raise RuntimeError("llm_invalid_json_response")
        return parsed
    finally:
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_result = close_method()
            if asyncio.iscoroutine(close_result):
                await close_result


def _call_observation_agent_with_retry(
    user_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    # Implements call observation agent with retry behavior for this module.
    model = _resolve_model("OBSERVATION_AGENT_MODEL", DEFAULT_OBSERVATION_AGENT_MODEL)
    system_prompt = (
        "You are the Observation/Behavior Agent. Use provided measurements and timestamped observations "
        "to interpret movement patterns without inventing measurements. "
        "Return strict JSON with keys: summary, measured_facts, interpretations, hypotheses, "
        "supporting_observation_ids, conflicting_evidence, uncertainties, needs_more_observations."
    )

    timeout_seconds = _resolve_timeout_seconds()
    attempts = _resolve_retry_limit() + 1
    retry_backoff_seconds = _resolve_retry_backoff_seconds()
    last_error: str | None = None
    last_error_code: str | None = None
    started = time.perf_counter()
    for attempt_index in range(attempts):
        try:
            payload = asyncio.run(
                _invoke_llm_json_response(
                    model=model,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    timeout_seconds=timeout_seconds,
                )
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return payload, {
                "status": "ok",
                "model": model,
                "latency_ms": latency_ms,
                "error": None,
                "error_code": None,
                "attempts": attempt_index + 1,
                "retry_count": attempt_index,
            }
        except Exception as error:
            last_error_code = _classify_llm_error(error)
            last_error = f"{type(error).__name__}:{error}"
            has_more_attempts = attempt_index < attempts - 1
            if has_more_attempts and retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (2**attempt_index))

    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return None, {
        "status": "error",
        "model": model,
        "latency_ms": latency_ms,
        "error": last_error or "unknown_error",
        "error_code": last_error_code or "runtime_error",
        "attempts": attempts,
        "retry_count": max(0, attempts - 1),
    }


def _call_reasoning_agent_with_retry(
    user_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    # Implements call reasoning agent with retry behavior for this module.
    model = _resolve_model("REASONING_AGENT_MODEL", DEFAULT_REASONING_AGENT_MODEL)
    system_prompt = (
        "You are the Reasoning/Decision-Support Agent. Synthesize observation summaries and evidence "
        "into an explainable operator-facing assessment. Do not infer hostile intent from movement alone. "
        "Return strict JSON with keys: assessment_label, summary, rationale, recommendation, "
        "supporting_observation_ids, uncertainties, limitations, alternative_interpretations, "
        "missing_evidence, contradictory_evidence, request_more_observations."
    )

    timeout_seconds = _resolve_timeout_seconds()
    attempts = _resolve_retry_limit() + 1
    retry_backoff_seconds = _resolve_retry_backoff_seconds()
    last_error: str | None = None
    last_error_code: str | None = None
    started = time.perf_counter()
    for attempt_index in range(attempts):
        try:
            payload = asyncio.run(
                _invoke_llm_json_response(
                    model=model,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    timeout_seconds=timeout_seconds,
                )
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return payload, {
                "status": "ok",
                "model": model,
                "latency_ms": latency_ms,
                "error": None,
                "error_code": None,
                "attempts": attempt_index + 1,
                "retry_count": attempt_index,
            }
        except Exception as error:
            last_error_code = _classify_llm_error(error)
            last_error = f"{type(error).__name__}:{error}"
            has_more_attempts = attempt_index < attempts - 1
            if has_more_attempts and retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (2**attempt_index))

    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return None, {
        "status": "error",
        "model": model,
        "latency_ms": latency_ms,
        "error": last_error or "unknown_error",
        "error_code": last_error_code or "runtime_error",
        "attempts": attempts,
        "retry_count": max(0, attempts - 1),
    }


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


@mcp.tool()
def simulate_evidence_request(track_id: str, reason: str) -> dict[str, Any]:
    # Issues bounded evidence requests for a track through coordinator policy.
    """Issue a bounded evidence request for a track through the coordinator."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    prepared = _prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return {
            "available": False,
            "error": prepared.get("error", "radar_state_unavailable"),
            "track_id": track_id,
            "radar_state_age_seconds": prepared.get("radar_state_age_seconds"),
            "radar_state_max_age_seconds": prepared.get("radar_state_max_age_seconds"),
        }

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


@mcp.tool()
def simulate_observation_summary(track_id: str) -> dict[str, Any]:
    # Builds observation-agent summary output for one track from latest ingested radar state.
    """Build only the observation/behavior summary for one track from latest radar state."""
    prepared = _prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return {
            "available": False,
            "error": prepared.get("error", "radar_state_unavailable"),
            "track_id": track_id,
            "radar_state_age_seconds": prepared.get("radar_state_age_seconds"),
            "radar_state_max_age_seconds": prepared.get("radar_state_max_age_seconds"),
        }

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

    summary["llm_available"] = observation_payload is not None
    summary["llm_model"] = observation_meta.get("model")
    summary["llm_latency_ms"] = observation_meta.get("latency_ms")
    summary["llm_error"] = observation_meta.get("error")

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


@mcp.tool()
def simulate_reasoning_assessment(
    track_id: str,
    request_evidence_on_uncertainty: bool = True,
) -> dict[str, Any]:
    # Builds full observation+reasoning assessment output for one track with bounded evidence requests.
    """Build an observation summary and reasoning assessment for one track from latest radar state."""
    prepared = _prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return {
            "available": False,
            "error": prepared.get("error", "radar_state_unavailable"),
            "track_id": track_id,
            "radar_state_age_seconds": prepared.get("radar_state_age_seconds"),
            "radar_state_max_age_seconds": prepared.get("radar_state_max_age_seconds"),
        }

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

    summary["llm_available"] = observation_payload is not None
    summary["llm_model"] = observation_meta.get("model")
    summary["llm_latency_ms"] = observation_meta.get("latency_ms")
    summary["llm_error"] = observation_meta.get("error")

    handoff_validation = {
        "status": "ok",
        "error": None,
        "error_code": None,
    }
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
        handoff_validation = {
            "status": "error",
            "error": f"{type(error).__name__}:{error}",
            "error_code": "invalid_handoff_payload",
        }

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

    assessment["llm_available"] = reasoning_payload is not None
    assessment["llm_model"] = reasoning_meta.get("model")
    assessment["llm_latency_ms"] = reasoning_meta.get("latency_ms")
    assessment["llm_error"] = reasoning_meta.get("error")

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
