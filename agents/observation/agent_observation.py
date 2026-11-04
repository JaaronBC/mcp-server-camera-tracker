from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from ..contracts import TrackingEvidenceEnvelope, VisualVerificationEnvelope
from ..coordinator import AgentCoordinator
from .. import llm_call
from .. import provider_call_budget
from .evidence_consumption import (
    ObservationEvidenceConsumptionError,
    ObservationEvidenceRequest,
    consume_tracking_evidence_by_message_id,
)

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
DEFAULT_OBSERVATION_VISUAL_MODEL = "qwen-3.8-27b"
DEFAULT_OBSERVATION_VISUAL_SYSTEM_PROMPT = (
    "You are the Observation Visual Verification Agent. Verify whether the provided image evidence supports, "
    "contradicts, or is inconclusive for the referenced tracking identity claim. "
    "Never invent evidence and return strict JSON only with keys: "
    "verification_status, visibility_status, identity_consistency, possible_id_switch, confidence, "
    "uncertainty, limitations, supporting_evidence_refs, contradicting_evidence_refs, recommended_follow_up, "
    "visual_content_inspected."
)

_KNOWN_TEXT_ONLY_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-oss",
    "o1",
    "o3",
)

_KNOWN_MULTIMODAL_MODEL_PREFIXES: tuple[str, ...] = (
    "qwen",
    "gpt-4o",
    "gpt-4.1",
    "gemini",
    "claude-3",
)


@dataclass(slots=True)
class ObservationVisualVerificationRequest:
    message_id: str
    now_timestamp: float | None = None
    max_history_records: int = 20
    observation_id: str | None = None


@dataclass(slots=True)
class ObservationVisualVerificationResult:
    envelope: VisualVerificationEnvelope
    validation: dict[str, Any]
    llm_meta: dict[str, Any]
    preflight: dict[str, Any]
    referenced_tracking_evidence: TrackingEvidenceEnvelope
    measurement_provenance: dict[str, dict[str, Any]]


class ObservationVisualVerificationError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = {} if details is None else details


def _resolve_model(env_name: str = "OBSERVATION_AGENT_MODEL", default_model: str = DEFAULT_OBSERVATION_AGENT_MODEL) -> str:
    return llm_call.resolve_model(env_name, default_model)


def _resolve_visual_model() -> str:
    return llm_call.resolve_model("OBSERVATION_VISUAL_AGENT_MODEL", DEFAULT_OBSERVATION_VISUAL_MODEL)


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


def _derive_timeout_source(error: Exception) -> str | None:
    if isinstance(error, asyncio.TimeoutError):
        return "local_asyncio_deadline"

    exception_type = type(error)
    name = exception_type.__name__.lower()
    module = exception_type.__module__.lower()
    text = f"{exception_type.__name__}:{error}".lower()

    if "timeout" not in name and "timeout" not in text:
        return None

    if "httpx" in module or "openai" in module:
        return "sdk_http_timeout"

    if any(token in name for token in ("readtimeout", "connecttimeout", "writetimeout", "pooltimeout", "apitimeouterror")):
        return "sdk_http_timeout"

    return "timeout_unknown_source"


def _supports_vision_inputs(model: str) -> bool:
    normalized = model.strip().lower()
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in _KNOWN_TEXT_ONLY_MODEL_PREFIXES):
        return False
    if any(normalized.startswith(prefix) for prefix in _KNOWN_MULTIMODAL_MODEL_PREFIXES):
        return True
    return False


def _encode_image_bytes_as_data_url(image_bytes: bytes) -> str:
    if not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
        raise ObservationVisualVerificationError(
            "image_bytes_missing",
            "Image bytes are required for visual verification",
        )
    payload = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_status(value: Any, *, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    return default


def _require_visual_payload_shape(payload: dict[str, Any]) -> None:
    required_keys = {
        "verification_status",
        "visibility_status",
        "identity_consistency",
        "possible_id_switch",
        "uncertainty",
        "limitations",
        "supporting_evidence_refs",
        "contradicting_evidence_refs",
        "visual_content_inspected",
    }
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise ObservationVisualVerificationError(
            "malformed_verification_response",
            "Visual verification response is missing required keys",
            details={"missing_keys": missing},
        )


def _build_visual_verification_envelope(
    *,
    request: ObservationVisualVerificationRequest,
    llm_payload: dict[str, Any],
    referenced_evidence: TrackingEvidenceEnvelope,
    source_evidence_refs: list[str],
) -> VisualVerificationEnvelope:
    _require_visual_payload_shape(llm_payload)

    observation_id = request.observation_id
    if not isinstance(observation_id, str) or not observation_id.strip():
        observation_id = (
            f"verify-{referenced_evidence.message_id}-"
            f"{int(float(referenced_evidence.timestamp) * 1_000_000)}"
        )

    verification_status = _coerce_status(
        llm_payload.get("verification_status"),
        allowed={"supported", "contradicted", "inconclusive"},
        default="inconclusive",
    )
    visibility_status = _coerce_status(
        llm_payload.get("visibility_status"),
        allowed={"visible", "partially_occluded", "occluded", "not_visible", "unknown"},
        default="unknown",
    )
    identity_consistency = _coerce_status(
        llm_payload.get("identity_consistency"),
        allowed={"consistent", "inconsistent", "unknown"},
        default="unknown",
    )

    supporting_refs = _coerce_str_list(llm_payload.get("supporting_evidence_refs"))
    contradicting_refs = _coerce_str_list(llm_payload.get("contradicting_evidence_refs"))

    return VisualVerificationEnvelope(
        schema_version=referenced_evidence.schema_version,
        observation_id=observation_id.strip(),
        referenced_track_id=referenced_evidence.track_id,
        referenced_message_id=referenced_evidence.message_id,
        timestamp=(
            float(request.now_timestamp)
            if isinstance(request.now_timestamp, (int, float))
            else float(referenced_evidence.timestamp)
        ),
        source_evidence_refs=list(source_evidence_refs),
        verification_status=verification_status,
        visibility_status=visibility_status,
        identity_consistency=identity_consistency,
        possible_id_switch=_coerce_bool(llm_payload.get("possible_id_switch"), default=False),
        uncertainty=_coerce_str_list(llm_payload.get("uncertainty")),
        limitations=_coerce_str_list(llm_payload.get("limitations")),
        supporting_evidence_refs=supporting_refs,
        contradicting_evidence_refs=contradicting_refs,
        confidence=_coerce_optional_float(llm_payload.get("confidence")),
        recommended_follow_up=(
            llm_payload.get("recommended_follow_up")
            if isinstance(llm_payload.get("recommended_follow_up"), str)
            else None
        ),
        visual_content_inspected=_coerce_bool(llm_payload.get("visual_content_inspected"), default=True),
    )


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
    reservation = provider_call_budget.reserve_or_raise(provider_call_budget.ROLE_OBSERVATION_TEXT)
    started = time.perf_counter()
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
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="ok",
            latency_ms=latency_ms,
            usage=llm_call.extract_usage_dict(response),
        )
        if not response.choices:
            raise RuntimeError("llm_empty_response")
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            raise RuntimeError("llm_invalid_json_response")
        return parsed
    except Exception as error:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="error",
            latency_ms=latency_ms,
            error_code=llm_call.classify_llm_error(error),
        )
        raise
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


async def invoke_visual_verification_llm_response(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    image_data_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    reservation = provider_call_budget.reserve_or_raise(provider_call_budget.ROLE_OBSERVATION_VISUAL)
    started = time.perf_counter()
    client: AsyncOpenAI | None = None
    try:
        client = _create_openai_client()
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": json.dumps(user_payload)},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    },
                ],
            ),
            timeout=timeout_seconds,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="ok",
            latency_ms=latency_ms,
            usage=llm_call.extract_usage_dict(response),
        )
        if not response.choices:
            raise RuntimeError("llm_empty_response")
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            raise RuntimeError("llm_invalid_json_response")
        return parsed
    except Exception as error:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="error",
            latency_ms=latency_ms,
            error_code=llm_call.classify_llm_error(error),
        )
        raise
    finally:
        if client is not None:
            close_method = getattr(client, "close", None)
            if callable(close_method):
                close_result = close_method()
                if asyncio.iscoroutine(close_result):
                    await close_result


def verify_tracking_evidence_for_observation(
    request: ObservationVisualVerificationRequest,
    *,
    evidence_root_dir: Path,
    coordinator: AgentCoordinator | None = None,
    consume_fn: Callable[..., Any] | None = None,
    invoke_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    retry_limit: int | None = None,
    retry_backoff_seconds: float | None = None,
    classify_error_fn: Callable[[Exception], str] | None = None,
    system_prompt: str | None = None,
) -> ObservationVisualVerificationResult:
    selected_coordinator = coordinator or AgentCoordinator()
    selected_model = model or _resolve_visual_model()
    if not _supports_vision_inputs(selected_model):
        raise ObservationVisualVerificationError(
            "unsupported_vision_model",
            "Selected observation visual model does not support image inputs",
            details={"model": selected_model},
        )

    consume_tracking_evidence = consume_fn or consume_tracking_evidence_by_message_id

    try:
        evidence_package = consume_tracking_evidence(
            ObservationEvidenceRequest(
                message_id=request.message_id,
                now_timestamp=request.now_timestamp,
                require_visual_evidence=True,
                max_history_records=request.max_history_records,
            ),
            evidence_root_dir=evidence_root_dir,
            coordinator=selected_coordinator,
        )
    except ObservationEvidenceConsumptionError as error:
        raise ObservationVisualVerificationError(
            "evidence_unavailable",
            "Unable to consume tracking evidence for visual verification",
            details={"upstream_code": error.code, "upstream_details": error.details},
        ) from error

    if evidence_package.image_evidence is None:
        raise ObservationVisualVerificationError(
            "image_evidence_missing",
            "Visual verification requires retrievable image evidence",
            details={"message_id": request.message_id},
        )

    image_data_url = _encode_image_bytes_as_data_url(evidence_package.image_evidence.image_bytes)

    visual_input = {
        "message_id": evidence_package.message_id,
        "track_id": evidence_package.envelope.track_id,
        "tracking_claims": evidence_package.envelope.tracking_claims.to_dict(),
        "measurement_provenance": evidence_package.measurement_provenance,
        "bounded_history": [record.to_dict() for record in evidence_package.bounded_history],
        "source_evidence_refs": list(evidence_package.source_references),
    }

    invoke = invoke_fn or invoke_visual_verification_llm_response
    selected_timeout = _resolve_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    selected_retry_limit = _resolve_retry_limit() if retry_limit is None else int(retry_limit)
    selected_backoff = _resolve_retry_backoff_seconds() if retry_backoff_seconds is None else float(retry_backoff_seconds)
    classify_error = classify_error_fn or _classify_llm_error
    selected_prompt = system_prompt or DEFAULT_OBSERVATION_VISUAL_SYSTEM_PROMPT

    attempts = selected_retry_limit + 1
    last_error: str | None = None
    last_error_code: str | None = None
    last_exception_class: str | None = None
    last_timeout_source: str | None = None
    last_attempt_latency_ms: float | None = None
    llm_payload: dict[str, Any] | None = None
    started = time.perf_counter()
    for attempt_index in range(attempts):
        attempt_started = time.perf_counter()
        try:
            llm_payload = asyncio.run(
                invoke(
                    model=selected_model,
                    system_prompt=selected_prompt,
                    user_payload=visual_input,
                    image_data_url=image_data_url,
                    timeout_seconds=selected_timeout,
                )
            )
            llm_meta = {
                "status": "ok",
                "model": selected_model,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error": None,
                "error_code": None,
                "attempts": attempt_index + 1,
                "retry_count": attempt_index,
            }
            break
        except Exception as error:
            last_error_code = classify_error(error)
            last_error = f"{type(error).__name__}:{error}"
            last_exception_class = type(error).__name__
            last_timeout_source = _derive_timeout_source(error)
            last_attempt_latency_ms = round((time.perf_counter() - attempt_started) * 1000.0, 3)
            if attempt_index < attempts - 1 and selected_backoff > 0:
                time.sleep(selected_backoff * (2**attempt_index))
    else:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        raise ObservationVisualVerificationError(
            "provider_failure",
            "Visual verification provider invocation failed",
            details={
                "error": last_error or "unknown_error",
                "error_code": last_error_code or "runtime_error",
                "model": selected_model,
                "attempts": attempts,
                "retry_count": max(0, attempts - 1),
                "latency_ms": latency_ms,
                "attempt_latency_ms": last_attempt_latency_ms,
                "configured_timeout_seconds": selected_timeout,
                "exception_class": last_exception_class,
                "timeout_source": last_timeout_source,
            },
        )

    if not isinstance(llm_payload, dict):
        raise ObservationVisualVerificationError(
            "malformed_verification_response",
            "Visual verification response was not a JSON object",
            details={"model": selected_model},
        )

    envelope = _build_visual_verification_envelope(
        request=request,
        llm_payload=llm_payload,
        referenced_evidence=evidence_package.envelope,
        source_evidence_refs=list(evidence_package.source_references),
    )

    validation = selected_coordinator.validate_visual_verification_envelope(
        envelope,
        evidence_by_message_id={
            evidence_package.envelope.message_id: evidence_package.envelope,
        },
    )
    if not bool(validation.get("valid", False)):
        raise ObservationVisualVerificationError(
            "coordinator_validation_rejected",
            "Visual verification envelope rejected by coordinator validation",
            details={
                "validation": validation,
                "observation_id": envelope.observation_id,
                "referenced_message_id": envelope.referenced_message_id,
            },
        )

    return ObservationVisualVerificationResult(
        envelope=envelope,
        validation=validation,
        llm_meta=llm_meta,
        preflight=evidence_package.preflight,
        referenced_tracking_evidence=evidence_package.envelope,
        measurement_provenance=evidence_package.measurement_provenance,
    )
