from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from .. import llm_call

DEFAULT_OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
DEFAULT_OBSERVATION_AGENT_MODEL = "gpt-oss-120b"
DEFAULT_LLM_TIMEOUT_SECONDS = 8.0
DEFAULT_LLM_RETRY_LIMIT = 1
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 0.1
DEFAULT_OBSERVATION_SYSTEM_PROMPT = (
    "You are the Observation/Behavior Agent. Use provided measurements and timestamped observations "
    "to interpret movement patterns without inventing measurements. "
    "Return strict JSON with keys: summary, measured_facts, interpretations, hypotheses, "
    "supporting_observation_ids, conflicting_evidence, uncertainties, needs_more_observations."
)


def _resolve_model(env_name: str = "OBSERVATION_AGENT_MODEL", default_model: str = DEFAULT_OBSERVATION_AGENT_MODEL) -> str:
    return llm_call.resolve_model(env_name, default_model)


def _resolve_timeout_seconds() -> float:
    return llm_call.resolve_timeout_seconds(DEFAULT_LLM_TIMEOUT_SECONDS)


def _resolve_retry_limit() -> int:
    return llm_call.resolve_retry_limit(DEFAULT_LLM_RETRY_LIMIT)


def _resolve_retry_backoff_seconds() -> float:
    return llm_call.resolve_retry_backoff_seconds(DEFAULT_LLM_RETRY_BACKOFF_SECONDS)


def _classify_llm_error(error: Exception) -> str:
    return llm_call.classify_llm_error(error)


def _create_openai_client() -> AsyncOpenAI:
    return llm_call.create_openai_client(DEFAULT_OPENAI_BASE_URL)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    payload = llm_call.extract_json_object(raw_text)
    if isinstance(payload, dict):
        return payload
    return None


def _coerce_str_list(value: Any) -> list[str]:
    return llm_call.coerce_str_list(value)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return llm_call.coerce_bool(value, default)


def _coerce_str(value: Any, default: str) -> str:
    return llm_call.coerce_str(value, default)


def normalize_track_coordinate_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
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


def extract_latest_track_snapshot(
    radar_state: dict[str, Any] | None,
    track_id: str,
) -> dict[str, Any]:
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
    return normalize_track_coordinate_metadata(snapshot)


def extract_tracking_agent_context(
    radar_state: dict[str, Any] | None,
    track_id: str,
) -> dict[str, Any]:
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


def extract_coordinate_context(radar_state: dict[str, Any] | None) -> dict[str, Any]:
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


def build_observation_llm_input(
    track_id: str,
    created_at: float,
    *,
    coordinator: Any,
    max_history_records: int,
    radar_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[Any], list[str], list[dict[str, Any]]]:
    history = coordinator.get_track_history(track_id)
    relevant_history = history[-max_history_records:]
    deterministic_summary = coordinator.build_observation_summary(track_id, created_at=created_at)
    supporting_ids = [record.observation_id for record in relevant_history]

    records_payload = [record.to_dict() for record in relevant_history]

    summary_payload = deterministic_summary.to_dict()
    summary_payload["supporting_observation_ids"] = supporting_ids

    latest_track_snapshot = extract_latest_track_snapshot(radar_state, track_id)
    tracking_agent_context = extract_tracking_agent_context(radar_state, track_id)
    coordinate_context = extract_coordinate_context(radar_state)
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


def validate_observation_llm_payload(
    payload: dict[str, Any],
    fallback_summary: dict[str, Any],
    valid_observation_ids: set[str],
) -> dict[str, Any]:
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


async def invoke_llm_json_response(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
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


def call_observation_agent_with_retry(
    user_payload: dict[str, Any],
    *,
    invoke_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    retry_limit: int | None = None,
    retry_backoff_seconds: float | None = None,
    classify_error_fn: Callable[[Exception], str] | None = None,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    invoke = invoke_fn or invoke_llm_json_response
    selected_model = model or _resolve_model()
    selected_timeout = _resolve_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    selected_retry_limit = _resolve_retry_limit() if retry_limit is None else int(retry_limit)
    selected_backoff = _resolve_retry_backoff_seconds() if retry_backoff_seconds is None else float(retry_backoff_seconds)
    classify_error = classify_error_fn or _classify_llm_error
    selected_prompt = system_prompt or DEFAULT_OBSERVATION_SYSTEM_PROMPT

    attempts = selected_retry_limit + 1
    last_error: str | None = None
    last_error_code: str | None = None
    started = time.perf_counter()
    for attempt_index in range(attempts):
        try:
            payload = asyncio.run(
                invoke(
                    model=selected_model,
                    system_prompt=selected_prompt,
                    user_payload=user_payload,
                    timeout_seconds=selected_timeout,
                )
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return payload, {
                "status": "ok",
                "model": selected_model,
                "latency_ms": latency_ms,
                "error": None,
                "error_code": None,
                "attempts": attempt_index + 1,
                "retry_count": attempt_index,
            }
        except Exception as error:
            last_error_code = classify_error(error)
            last_error = f"{type(error).__name__}:{error}"
            has_more_attempts = attempt_index < attempts - 1
            if has_more_attempts and selected_backoff > 0:
                time.sleep(selected_backoff * (2**attempt_index))

    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return None, {
        "status": "error",
        "model": selected_model,
        "latency_ms": latency_ms,
        "error": last_error or "unknown_error",
        "error_code": last_error_code or "runtime_error",
        "attempts": attempts,
        "retry_count": max(0, attempts - 1),
    }
