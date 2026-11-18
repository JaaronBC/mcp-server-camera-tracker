# MCP tools for reasoning-cycle event logging, health summaries, and event log maintenance.
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from ... import latency_instrumentation
from ...observation.tools.capabilities import (
    OBSERVATION_ACTION_ASSESS_NOW,
    OBSERVATION_ACTION_INCONCLUSIVE,
    OBSERVATION_ACTION_PREDICT_BOUNDED,
    OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
    OBSERVATION_DECISION_REQUEST_ACTIONS,
    OBSERVATION_REQUEST_ACTION_TO_TOOL_NAME,
    OBSERVATION_TOOL_GET_TRACK_EVIDENCE_HISTORY,
)
from ...observation.tools.track_evidence_history_contract import (
    build_track_evidence_history_argument_diagnostics,
    validate_track_evidence_history_arguments,
)
from ...orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport
from ...reasoning.tools import _shared as dashboard_reasoning
from ...registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGENT_EVENTS_PATH = PROJECT_ROOT / "captures" / "agent_events.jsonl"
AGENT_EVENT_HEALTH_FRESHNESS_SECONDS = 30.0
MAX_EVIDENCE_STR_LIST_ITEMS = 5
MAX_EVIDENCE_SELECTED_UPDATES = 3
VALID_INVESTIGATION_STATES = {
    "open",
    "awaiting_evidence",
    "verifying",
    "awaiting_reassociation",
    "resolved",
    "closed",
}
OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS = 1.5
OBSERVATION_TOOL_REQUEST_TTL_SECONDS = 30.0
OBSERVATION_TOOL_REQUESTS_ENABLED_ENV = "OBSERVATION_TOOL_REQUESTS_ENABLED"
OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS_ENV = "OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS"
OBSERVATION_TOOL_REQUESTS_PATH_ENV = "PHASE2D_TRACKING_DECISION_TOOL_REQUESTS_PATH"
OBSERVATION_TOOL_RESULTS_PATH_ENV = "PHASE2D_TRACKING_DECISION_TOOL_RESULTS_PATH"
OBSERVATION_TOOL_REQUEST_INDEX_PATH_ENV = "PHASE2D_TRACKING_DECISION_REQUEST_INDEX_PATH"
OBSERVATION_TOOL_RESULT_INDEX_PATH_ENV = "PHASE2D_TRACKING_DECISION_RESULT_INDEX_PATH"

_PENDING_OBSERVATION_TOOL_FOLLOWUPS: dict[str, dict[str, Any]] = {}
_OBSERVATION_FOLLOWUP_RESULT_ALLOWED_STATUSES = frozenset(
    {
        "ok",
        "timeout",
        "unavailable",
        "rejected",
        "expired",
        "terminal",
        "incompatible",
        "stale",
    }
)


def _resolve_observation_tool_request_timeout_seconds(default: float = 30.0) -> float:
    raw = os.getenv(OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS_ENV, str(default)).strip()
    try:
        parsed = float(raw)
    except ValueError:
        parsed = default
    if not math.isfinite(parsed) or parsed <= 0:
        parsed = default
    return max(1.0, min(300.0, parsed))


OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS = _resolve_observation_tool_request_timeout_seconds(30.0)


def _observation_tool_requests_enabled() -> bool:
    raw = os.getenv(OBSERVATION_TOOL_REQUESTS_ENABLED_ENV, "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_observation_ipc_paths() -> tuple[Path, Path, Path, Path]:
    requests_path = Path(
        os.getenv(
            OBSERVATION_TOOL_REQUESTS_PATH_ENV,
            str(PROJECT_ROOT / "captures" / "tracking_decision_tool_requests.jsonl"),
        )
    ).expanduser().resolve()
    results_path = Path(
        os.getenv(
            OBSERVATION_TOOL_RESULTS_PATH_ENV,
            str(PROJECT_ROOT / "captures" / "tracking_decision_tool_results.jsonl"),
        )
    ).expanduser().resolve()
    request_index_path = Path(
        os.getenv(
            OBSERVATION_TOOL_REQUEST_INDEX_PATH_ENV,
            str(PROJECT_ROOT / "captures" / "latest_tracking_decision_request_index.json"),
        )
    ).expanduser().resolve()
    result_index_path = Path(
        os.getenv(
            OBSERVATION_TOOL_RESULT_INDEX_PATH_ENV,
            str(PROJECT_ROOT / "captures" / "latest_tracking_decision_result_index.json"),
        )
    ).expanduser().resolve()
    return requests_path, results_path, request_index_path, result_index_path


def _observation_ipc_transport() -> TrackingDecisionIpcTransport:
    requests_path, results_path, request_index_path, result_index_path = _resolve_observation_ipc_paths()
    return TrackingDecisionIpcTransport(
        requests_path=requests_path,
        results_path=results_path,
        request_index_path=request_index_path,
        result_index_path=result_index_path,
        request_ttl_seconds=OBSERVATION_TOOL_REQUEST_TTL_SECONDS,
        writer_role="dashboard",
    )


def _build_investigation_id(*, cycle_id: str, track_id: str) -> str:
    fingerprint = f"{cycle_id}|{track_id}".encode("utf-8")
    return f"inv-{hashlib.sha1(fingerprint).hexdigest()[:20]}"


def _agent_events_path() -> Path:
    configured = os.getenv("PHASE16E_AGENT_EVENTS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return AGENT_EVENTS_PATH


def _run_id_from_env() -> str | None:
    run_id = os.getenv("PHASE16E_RUN_ID", "").strip()
    return run_id or None


def _append_event(event: dict[str, Any]) -> None:
    # Implements append event behavior for this module.
    event_path = _agent_events_path()
    payload = dict(event)
    if "run_id" not in payload:
        run_id = _run_id_from_env()
        if run_id:
            payload["run_id"] = run_id
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload)}\n")


def _extract_track_ids(radar_state: dict[str, Any]) -> list[str]:
    # Extracts track ids from structured payloads.
    tracks = radar_state.get("tracks")
    if not isinstance(tracks, list):
        return []

    track_ids: list[str] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_id = track.get("track_id")
        if isinstance(track_id, str) and track_id.strip():
            track_ids.append(track_id)

    return track_ids


def _load_events(limit: int, event_path: Path) -> tuple[list[dict[str, Any]], int]:
    # Loads events data from storage for downstream processing.
    lines = event_path.read_text(encoding="utf-8").splitlines()
    selected = lines[-limit:]
    events: list[dict[str, Any]] = []
    dropped_lines = 0

    for line in selected:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            dropped_lines += 1
            continue
        if isinstance(payload, dict):
            events.append(payload)
        else:
            dropped_lines += 1

    return events, dropped_lines


def _percentile(values: list[float], percentile: float) -> float | None:
    # Implements percentile behavior for this module.
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = percentile * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * fraction


def _top_count_entry(counts: dict[str, int]) -> tuple[str | None, int]:
    # Implements top count entry behavior for this module.
    best_key = None
    best_count = 0
    for key, count in counts.items():
        if count > best_count:
            best_key = key
            best_count = count
    return best_key, best_count


def _archive_path_for_now(base_path: Path) -> Path:
    # Implements archive path for now behavior for this module.
    timestamp_label = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return base_path.with_name(f"{base_path.stem}_archive_{timestamp_label}{base_path.suffix}")


def _format_agent_error_digest(
    observation_error_code: Any,
    reasoning_error_code: Any,
) -> str:
    # Formats agent error digest into concise operator-facing text.
    observation = observation_error_code if isinstance(observation_error_code, str) and observation_error_code.strip() else "none"
    reasoning = reasoning_error_code if isinstance(reasoning_error_code, str) and reasoning_error_code.strip() else "none"
    return f"obs({observation}),rea({reasoning})"


def _derive_handoff_quality(
    handoff_validation_status: Any,
    handoff_validation_error_code: Any,
) -> str:
    # Implements derive handoff quality behavior for this module.
    if isinstance(handoff_validation_status, str) and handoff_validation_status.strip().lower() == "error":
        return "degraded"
    if isinstance(handoff_validation_error_code, str) and handoff_validation_error_code.strip():
        return "degraded"
    return "healthy"


def _coerce_nonempty_str(value: Any) -> str | None:
    # Implements coerce nonempty str behavior for this module.
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_str_list(value: Any, max_items: int = MAX_EVIDENCE_STR_LIST_ITEMS) -> list[str]:
    # Implements coerce str list behavior for this module.
    if not isinstance(value, list):
        return []

    result: list[str] = []
    for item in value:
        normalized = _coerce_nonempty_str(item)
        if normalized is None:
            continue
        result.append(normalized)
        if len(result) >= max_items:
            break

    return result


def _append_unique_nonempty(target: list[str], value: str | None) -> None:
    if not isinstance(value, str):
        return
    normalized = value.strip()
    if not normalized:
        return
    if normalized not in target:
        target.append(normalized)


def _derive_conflicts(
    *,
    observation_status: Any,
    reasoning_status: Any,
    observation_error_code: Any,
    reasoning_error_code: Any,
    handoff_validation_status: Any,
    handoff_validation_error_code: Any,
) -> list[str]:
    conflicts: list[str] = []

    obs_status = _coerce_nonempty_str(observation_status)
    rea_status = _coerce_nonempty_str(reasoning_status)
    obs_error = _coerce_nonempty_str(observation_error_code)
    rea_error = _coerce_nonempty_str(reasoning_error_code)
    handoff_status = _coerce_nonempty_str(handoff_validation_status)
    handoff_error = _coerce_nonempty_str(handoff_validation_error_code)

    if handoff_status is not None and handoff_status.lower() == "error":
        _append_unique_nonempty(conflicts, "handoff_validation_failed")
    if handoff_error is not None:
        _append_unique_nonempty(conflicts, f"handoff_validation_error:{handoff_error}")

    if obs_status is not None and rea_status is not None and obs_status != rea_status:
        _append_unique_nonempty(
            conflicts,
            f"agent_status_mismatch:obs={obs_status},rea={rea_status}",
        )

    if obs_error is not None and rea_error is not None and obs_error != rea_error:
        _append_unique_nonempty(
            conflicts,
            f"agent_error_code_divergence:obs={obs_error},rea={rea_error}",
        )

    return conflicts


def _derive_limitations(
    *,
    cycle_error_code: Any,
    bounded_out: bool,
    summary_uncertainties: list[str],
    assessment_uncertainties: list[str],
    handoff_validation_error_code: Any,
) -> list[str]:
    limitations: list[str] = []

    cycle_error = _coerce_nonempty_str(cycle_error_code)
    handoff_error = _coerce_nonempty_str(handoff_validation_error_code)

    if cycle_error is not None:
        _append_unique_nonempty(limitations, f"cycle_error:{cycle_error}")
    if bounded_out:
        _append_unique_nonempty(limitations, "bounded_out")
    if handoff_error is not None:
        _append_unique_nonempty(limitations, f"handoff_validation_error:{handoff_error}")

    for uncertainty in summary_uncertainties:
        _append_unique_nonempty(limitations, f"summary_uncertainty:{uncertainty}")
    for uncertainty in assessment_uncertainties:
        _append_unique_nonempty(limitations, f"assessment_uncertainty:{uncertainty}")

    return limitations


def _extract_ground_position(value: Any) -> dict[str, Any] | None:
    # Extracts ground position from structured payloads.
    if not isinstance(value, dict):
        return None
    return {
        "x": value.get("x"),
        "z": value.get("z"),
    }


def _extract_evidence_trace(summary: Any) -> dict[str, Any]:
    # Extracts evidence trace from structured payloads.
    if not isinstance(summary, dict):
        return {
            "latest_track_snapshot": {"available": False},
            "tracking_agent_context": {"available": False},
            "coordinate_context": {"available": False},
        }

    latest_track_snapshot = summary.get("latest_track_snapshot")
    tracking_agent_context = summary.get("tracking_agent_context")
    coordinate_context = summary.get("coordinate_context")

    if not isinstance(latest_track_snapshot, dict):
        latest_track_snapshot = {"available": False}
    if not isinstance(tracking_agent_context, dict):
        tracking_agent_context = {"available": False}
    if not isinstance(coordinate_context, dict):
        coordinate_context = {"available": False}

    normalized_snapshot = {
        "available": bool(latest_track_snapshot.get("available")),
        "track_id": _coerce_nonempty_str(latest_track_snapshot.get("track_id")),
        "timestamp": latest_track_snapshot.get("timestamp"),
        "ground_position": _extract_ground_position(latest_track_snapshot.get("ground_position")),
        "ground_coordinate_unit": _coerce_nonempty_str(latest_track_snapshot.get("ground_coordinate_unit")),
        "position_kind": _coerce_nonempty_str(latest_track_snapshot.get("position_kind")),
        "relative_depth": latest_track_snapshot.get("relative_depth"),
        "confidence": latest_track_snapshot.get("confidence"),
        "quality_flags": _coerce_str_list(latest_track_snapshot.get("quality_flags")),
    }

    raw_selected_updates = tracking_agent_context.get("selected_updates")
    normalized_selected_updates: list[dict[str, Any]] = []
    if isinstance(raw_selected_updates, list):
        for item in raw_selected_updates:
            if not isinstance(item, dict):
                continue

            normalized_selected_updates.append(
                {
                    "detection_id": _coerce_nonempty_str(item.get("detection_id")),
                    "resolved_track_id": _coerce_nonempty_str(item.get("resolved_track_id")),
                    "accepted": item.get("accepted"),
                    "rejection_reasons": _coerce_str_list(item.get("rejection_reasons")),
                    "uncertainties": _coerce_str_list(item.get("uncertainties")),
                    "supporting_evidence": _coerce_str_list(item.get("supporting_evidence")),
                }
            )
            if len(normalized_selected_updates) >= MAX_EVIDENCE_SELECTED_UPDATES:
                break

    normalized_tracking_context = {
        "available": bool(tracking_agent_context.get("available")),
        "source_agent": _coerce_nonempty_str(tracking_agent_context.get("source_agent")),
        "vlm_model": _coerce_nonempty_str(tracking_agent_context.get("vlm_model")),
        "timestamp": tracking_agent_context.get("timestamp"),
        "proposal_latency_ms": tracking_agent_context.get("proposal_latency_ms"),
        "uncertainties": _coerce_str_list(tracking_agent_context.get("uncertainties")),
        "selected_updates": normalized_selected_updates,
    }

    normalized_coordinate_context = {
        "available": bool(coordinate_context.get("available")),
        "coordinate_system": _coerce_nonempty_str(coordinate_context.get("coordinate_system")),
        "calibration_status": _coerce_nonempty_str(coordinate_context.get("calibration_status")),
        "camera_status": _coerce_nonempty_str(coordinate_context.get("camera_status")),
        "detector_status": _coerce_nonempty_str(coordinate_context.get("detector_status")),
        "last_observation_timestamp": coordinate_context.get("last_observation_timestamp"),
    }

    return {
        "latest_track_snapshot": normalized_snapshot,
        "tracking_agent_context": normalized_tracking_context,
        "coordinate_context": normalized_coordinate_context,
    }


def _initialize_reasoning_cycle(max_tracks: int) -> dict[str, Any]:
    if max_tracks <= 0:
        raise ValueError("max_tracks must be greater than zero")

    radar_state = dashboard_reasoning._load_radar_state()
    if radar_state is None:
        return {
            "available": False,
            "response": {
                "available": False,
                "error": "radar_state_unavailable",
                "event_log_path": str(_agent_events_path()),
            },
        }

    cycle_started_at = time.time()
    cycle_id = f"cycle-{uuid.uuid4().hex}"
    track_ids = _extract_track_ids(radar_state)[:max_tracks]

    return {
        "available": True,
        "cycle_started_at": cycle_started_at,
        "cycle_id": cycle_id,
        "track_ids": track_ids,
    }


def _observation_prediction_supported(summary_payload: dict[str, Any] | None) -> bool:
    if not isinstance(summary_payload, dict):
        return False
    sample_count = summary_payload.get("sample_count")
    speed = summary_payload.get("speed")
    if not isinstance(sample_count, int) or sample_count < 2:
        return False
    if not isinstance(speed, (int, float)):
        return False
    return True


def _infer_observation_decision_from_summary(
    *,
    summary_payload: dict[str, Any] | None,
    request_evidence_on_uncertainty: bool,
) -> dict[str, Any]:
    if not isinstance(summary_payload, dict):
        return {
            "action": OBSERVATION_ACTION_INCONCLUSIVE,
            "reason": "summary_unavailable",
            "uncertainty": ["summary_unavailable"],
            "tool_requests": [],
        }

    needs_more_observations = bool(summary_payload.get("needs_more_observations", False))
    track_id = _coerce_nonempty_str(summary_payload.get("track_id"))
    if needs_more_observations and request_evidence_on_uncertainty and track_id is not None:
        return {
            "action": OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
            "reason": "observation_evidence_insufficient",
            "uncertainty": _coerce_str_list(summary_payload.get("uncertainties")),
            "tool_requests": [
                {
                    "request_id": None,
                    "tool_name": OBSERVATION_TOOL_GET_TRACK_EVIDENCE_HISTORY,
                    "arguments": {
                        "track_id": track_id,
                        "max_messages": 5,
                        "max_history_records": 12,
                        "max_scan_records": 120,
                    },
                }
            ],
        }

    if needs_more_observations:
        if _observation_prediction_supported(summary_payload):
            return {
                "action": OBSERVATION_ACTION_PREDICT_BOUNDED,
                "reason": "bounded_prediction_from_deterministic_history",
                "uncertainty": _coerce_str_list(summary_payload.get("uncertainties")),
                "tool_requests": [],
            }
        return {
            "action": OBSERVATION_ACTION_INCONCLUSIVE,
            "reason": "insufficient_evidence",
            "uncertainty": _coerce_str_list(summary_payload.get("uncertainties")),
            "tool_requests": [],
        }

    return {
        "action": OBSERVATION_ACTION_ASSESS_NOW,
        "reason": "evidence_sufficient",
        "uncertainty": _coerce_str_list(summary_payload.get("uncertainties")),
        "tool_requests": [],
    }


def _invalid_explicit_decision(*, summary_payload: dict[str, Any], validation_error: str) -> dict[str, Any]:
    return {
        "action": OBSERVATION_ACTION_INCONCLUSIVE,
        "reason": "invalid_explicit_decision",
        "uncertainty": _coerce_str_list(summary_payload.get("uncertainties")),
        "tool_requests": [],
        "decision_source": "invalid_explicit_decision",
        "validation_error": validation_error,
        "phase": "terminal_invalid_explicit_decision",
    }


def _resolve_observation_decision(
    *,
    summary_payload: dict[str, Any] | None,
    request_evidence_on_uncertainty: bool,
) -> dict[str, Any]:
    if not isinstance(summary_payload, dict):
        decision = _infer_observation_decision_from_summary(
            summary_payload=summary_payload,
            request_evidence_on_uncertainty=request_evidence_on_uncertainty,
        )
        decision["decision_source"] = "llm_summary_inference"
        return decision

    validation_error = _coerce_nonempty_str(summary_payload.get("observation_decision_validation_error"))
    decision_present = bool(summary_payload.get("observation_decision_present", False)) or (
        "observation_decision" in summary_payload
    )

    if validation_error is not None:
        return _invalid_explicit_decision(summary_payload=summary_payload, validation_error=validation_error)

    explicit_decision = summary_payload.get("observation_decision")
    if decision_present:
        if not isinstance(explicit_decision, dict):
            return _invalid_explicit_decision(
                summary_payload=summary_payload,
                validation_error="decision_payload_missing",
            )
        normalized = dict(explicit_decision)
        action = _coerce_nonempty_str(normalized.get("action"))
        if action in OBSERVATION_DECISION_REQUEST_ACTIONS and not request_evidence_on_uncertainty:
            return _invalid_explicit_decision(
                summary_payload=summary_payload,
                validation_error="decision_request_not_allowed_in_follow_up",
            )
        normalized["decision_source"] = "llm_explicit"
        return normalized

    decision = _infer_observation_decision_from_summary(
        summary_payload=summary_payload,
        request_evidence_on_uncertainty=request_evidence_on_uncertainty,
    )
    decision["decision_source"] = "llm_summary_inference"
    return decision


def _build_observation_tool_request_decision(
    *,
    decision: dict[str, Any],
    track_id: str,
    investigation_id: str,
    request_id: str,
    correlation_id: str | None,
) -> dict[str, Any]:
    tool_requests = decision.get("tool_requests")
    tool_request = tool_requests[0] if isinstance(tool_requests, list) and tool_requests else None
    tool_name = None
    arguments: dict[str, Any] = {}
    if isinstance(tool_request, dict):
        tool_name = _coerce_nonempty_str(tool_request.get("tool_name"))
        if isinstance(tool_request.get("arguments"), dict):
            arguments = dict(tool_request.get("arguments"))
    resolved_track_id = _coerce_nonempty_str(track_id)
    if resolved_track_id is not None:
        arguments["track_id"] = resolved_track_id
    if tool_name is None:
        action = _coerce_nonempty_str(decision.get("action"))
        tool_name = OBSERVATION_REQUEST_ACTION_TO_TOOL_NAME.get(str(action))

    validation = validate_track_evidence_history_arguments(
        arguments,
        require_track_id=False,
        validate_track_id_value=False,
    )

    return {
        "action": OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
        "investigation_id": investigation_id,
        "correlation_id": correlation_id,
        "iteration_index": 1,
        "tool_requests": [
            {
                "request_id": request_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        ],
        "argument_validation": validation,
    }


def _extract_observation_tool_result_for_pending(track_id: str) -> dict[str, Any] | None:
    pending = _PENDING_OBSERVATION_TOOL_FOLLOWUPS.get(track_id)
    if not isinstance(pending, dict):
        return None
    investigation_id = _coerce_nonempty_str(pending.get("investigation_id"))
    request_id = _coerce_nonempty_str(pending.get("request_id"))
    tool_name = _coerce_nonempty_str(pending.get("tool_name"))
    if investigation_id is None or request_id is None or tool_name is None:
        return None
    transport = _observation_ipc_transport()
    lookup = transport.lookup_results_for_request(
        investigation_id=investigation_id,
        request_id=request_id,
        tool_name=tool_name,
    )
    results = lookup.get("results") if isinstance(lookup, dict) else None
    if not isinstance(results, list):
        return None

    pending_created_at = pending.get("created_at")
    min_finished_at = float(pending_created_at) if isinstance(pending_created_at, (int, float)) else None
    for item in results:
        if not isinstance(item, dict):
            continue
        status = _coerce_nonempty_str(item.get("status"))
        if status not in _OBSERVATION_FOLLOWUP_RESULT_ALLOWED_STATUSES:
            continue
        finished_at = item.get("finished_at")
        if min_finished_at is not None and isinstance(finished_at, (int, float)) and float(finished_at) < min_finished_at:
            continue
        return dict(item)
    return None


def _observation_follow_up_input(
    *,
    summary_payload: dict[str, Any],
    tool_result_event: dict[str, Any],
) -> dict[str, Any]:
    result_payload = tool_result_event.get("result_payload")
    if not isinstance(result_payload, dict):
        result_payload = {}
    return {
        "track_id": summary_payload.get("track_id"),
        "created_at": summary_payload.get("created_at"),
        "summary_measurements": {
            "velocity_x": summary_payload.get("velocity_x"),
            "velocity_z": summary_payload.get("velocity_z"),
            "speed": summary_payload.get("speed"),
            "confidence": summary_payload.get("confidence"),
            "sample_count": summary_payload.get("sample_count"),
        },
        "quality_uncertainties": _coerce_str_list(summary_payload.get("uncertainties")),
        "supporting_observation_ids": _coerce_str_list(summary_payload.get("supporting_observation_ids")),
        "evidence_tool_result": {
            "status": tool_result_event.get("status"),
            "error_code": tool_result_event.get("error_code"),
            "error": tool_result_event.get("error"),
            "payload": result_payload,
        },
    }



def _evaluate_track_for_cycle(
    track_id: str,
    *,
    request_evidence_on_uncertainty: bool,
    correlation_id: str | None = None,
    cycle_id: str | None = None,
) -> dict[str, Any]:
    context_previous = latency_instrumentation.set_context(
        correlation_id=correlation_id,
        cycle_id=cycle_id,
        track_id=track_id,
    )
    started = time.perf_counter()
    try:
        result = dashboard_reasoning.simulate_reasoning_assessment(
            track_id,
            request_evidence_on_uncertainty=request_evidence_on_uncertainty,
            correlation_id=correlation_id,
            cycle_id=cycle_id,
        )
    finally:
        latency_ms = (time.perf_counter() - started) * 1000.0
        latency_instrumentation.emit_event(
            "track_processing_completed",
            stage="track_processing",
            duration_ms=round(latency_ms, 3),
            track_id=track_id,
        )
        latency_instrumentation.restore_context(context_previous)

    summary_payload = result.get("summary") if isinstance(result.get("summary"), dict) else None
    observation_decision = _resolve_observation_decision(
        summary_payload=summary_payload,
        request_evidence_on_uncertainty=request_evidence_on_uncertainty and _observation_tool_requests_enabled(),
    )
    observation_tool_request: dict[str, Any] | None = None
    observation_tool_result: dict[str, Any] | None = None

    action = _coerce_nonempty_str(observation_decision.get("action"))
    if action in OBSERVATION_DECISION_REQUEST_ACTIONS and _observation_tool_requests_enabled():
        pending = _PENDING_OBSERVATION_TOOL_FOLLOWUPS.get(track_id)
        if isinstance(pending, dict):
            pending_created_at = pending.get("created_at")
            pending_age_seconds = (
                max(0.0, time.time() - float(pending_created_at))
                if isinstance(pending_created_at, (int, float))
                else None
            )
            if (
                isinstance(pending_age_seconds, float)
                and pending_age_seconds > OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS
            ):
                _PENDING_OBSERVATION_TOOL_FOLLOWUPS.pop(track_id, None)
                if _observation_prediction_supported(summary_payload):
                    observation_decision = {
                        "action": OBSERVATION_ACTION_PREDICT_BOUNDED,
                        "reason": "tool_result_timeout",
                        "uncertainty": _coerce_str_list(summary_payload.get("uncertainties") if isinstance(summary_payload, dict) else []),
                        "decision_source": observation_decision.get("decision_source", "llm_summary_inference"),
                        "phase": "tool_result_timeout",
                    }
                else:
                    observation_decision = {
                        "action": OBSERVATION_ACTION_INCONCLUSIVE,
                        "reason": "tool_result_timeout",
                        "uncertainty": _coerce_str_list(summary_payload.get("uncertainties") if isinstance(summary_payload, dict) else []),
                        "decision_source": observation_decision.get("decision_source", "llm_summary_inference"),
                        "phase": "tool_result_timeout",
                    }
            else:
                correlated_result = _extract_observation_tool_result_for_pending(track_id)
                if isinstance(correlated_result, dict):
                    _PENDING_OBSERVATION_TOOL_FOLLOWUPS.pop(track_id, None)
                    observation_tool_result = {
                        "status": correlated_result.get("status"),
                        "error_code": correlated_result.get("error_code"),
                        "error": correlated_result.get("error"),
                        "idempotency_key": correlated_result.get("idempotency_key"),
                        "result_payload": correlated_result.get("result_payload"),
                    }

                    if isinstance(summary_payload, dict):
                        follow_up_payload = _observation_follow_up_input(
                            summary_payload=summary_payload,
                            tool_result_event=correlated_result,
                        )
                        follow_up_observation_payload, follow_up_observation_meta = (
                            dashboard_reasoning._call_observation_agent_with_retry(follow_up_payload)
                        )
                        result_agents = result.get("agents") if isinstance(result.get("agents"), dict) else {}
                        if not isinstance(result_agents, dict):
                            result_agents = {}
                        result_agents["observation"] = follow_up_observation_meta
                        result["agents"] = result_agents

                        if isinstance(follow_up_observation_payload, dict):
                            valid_observation_ids = set(
                                _coerce_str_list(summary_payload.get("supporting_observation_ids"))
                            )
                            tool_payload = correlated_result.get("result_payload")
                            if isinstance(tool_payload, dict):
                                messages = tool_payload.get("messages")
                                if isinstance(messages, list):
                                    for item in messages:
                                        if not isinstance(item, dict):
                                            continue
                                        bounded_history = item.get("bounded_history")
                                        if not isinstance(bounded_history, list):
                                            continue
                                        for record in bounded_history:
                                            if not isinstance(record, dict):
                                                continue
                                            observation_id = _coerce_nonempty_str(record.get("observation_id"))
                                            if observation_id is not None:
                                                valid_observation_ids.add(observation_id)

                            normalized_summary = dashboard_reasoning._validate_observation_llm_payload(
                                follow_up_observation_payload,
                                fallback_summary=summary_payload,
                                valid_observation_ids=valid_observation_ids,
                            )
                            result["summary"] = normalized_summary
                            summary_payload = normalized_summary

                    observation_decision = _resolve_observation_decision(
                        summary_payload=summary_payload,
                        request_evidence_on_uncertainty=False,
                    )
                    if observation_decision.get("decision_source") == "invalid_explicit_decision":
                        observation_decision["phase"] = "terminal_invalid_explicit_decision"
                    else:
                        observation_decision["phase"] = "follow_up_terminal"
                else:
                    observation_decision["phase"] = "awaiting_tool_result"
        else:
            transport = _observation_ipc_transport()
            resolved_track_id = _coerce_nonempty_str(track_id)
            investigation_id = _build_investigation_id(
                cycle_id=cycle_id or "cycle-unknown",
                track_id=resolved_track_id or "track-unknown",
            )
            request_id = f"obs-req-{uuid.uuid4().hex[:16]}"
            tool_name = OBSERVATION_REQUEST_ACTION_TO_TOOL_NAME.get(
                OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                OBSERVATION_TOOL_GET_TRACK_EVIDENCE_HISTORY,
            )
            request_decision = _build_observation_tool_request_decision(
                decision=observation_decision,
                track_id=track_id,
                investigation_id=investigation_id,
                request_id=request_id,
                correlation_id=correlation_id,
            )
            argument_validation = request_decision.get("argument_validation")
            validation_errors = (
                list(argument_validation.get("errors", []))
                if isinstance(argument_validation, dict)
                else []
            )
            if validation_errors:
                diagnostics = build_track_evidence_history_argument_diagnostics(
                    validation=argument_validation if isinstance(argument_validation, dict) else {},
                    request_id=request_id,
                    investigation_id=investigation_id,
                    correlation_id=correlation_id,
                )
                observation_tool_request = {
                    "submitted": False,
                    "rejected": True,
                    "duplicate": False,
                    "reason": "invalid_tool_arguments_pre_dispatch",
                    "investigation_id": investigation_id,
                    "request_id": request_id,
                    "tool_name": tool_name,
                    "arguments": (
                        request_decision.get("tool_requests", [{}])[0].get("arguments")
                        if isinstance(request_decision.get("tool_requests"), list)
                        else {}
                    ),
                    **diagnostics,
                }
                if _observation_prediction_supported(summary_payload):
                    observation_decision = {
                        "action": OBSERVATION_ACTION_PREDICT_BOUNDED,
                        "reason": "invalid_tool_arguments_pre_dispatch",
                        "uncertainty": _coerce_str_list(summary_payload.get("uncertainties") if isinstance(summary_payload, dict) else []),
                        "decision_source": observation_decision.get("decision_source", "llm_summary_inference"),
                        "phase": "tool_request_pre_dispatch_rejected",
                        "error_code": "invalid_tool_arguments",
                        "validation_errors": validation_errors,
                        "unknown_arguments": diagnostics.get("unknown_arguments", []),
                    }
                else:
                    observation_decision = {
                        "action": OBSERVATION_ACTION_INCONCLUSIVE,
                        "reason": "invalid_tool_arguments_pre_dispatch",
                        "uncertainty": _coerce_str_list(summary_payload.get("uncertainties") if isinstance(summary_payload, dict) else []),
                        "decision_source": observation_decision.get("decision_source", "llm_summary_inference"),
                        "phase": "tool_request_pre_dispatch_rejected",
                        "error_code": "invalid_tool_arguments",
                        "validation_errors": validation_errors,
                        "unknown_arguments": diagnostics.get("unknown_arguments", []),
                    }
                result["observation_tool_request_diagnostics"] = diagnostics
                result["observation_tool_request_validation_errors"] = validation_errors
                result["observation_tool_request_original_arguments"] = diagnostics.get("original_requested_arguments", {})
            else:
                submit_response = transport.dashboard_submit_decision_request(
                    decision=request_decision,
                    llm_correlation_id=correlation_id,
                )
                observation_tool_request = {
                    "submitted": bool(submit_response.get("accepted", False)),
                    "rejected": bool(submit_response.get("rejected", False)),
                    "duplicate": bool(submit_response.get("duplicate", False)),
                    "reason": submit_response.get("reason"),
                    "investigation_id": investigation_id,
                    "request_id": request_id,
                    "tool_name": tool_name,
                    "arguments": (
                        request_decision.get("tool_requests", [{}])[0].get("arguments")
                        if isinstance(request_decision.get("tool_requests"), list)
                        else {}
                    ),
                }
                if bool(submit_response.get("accepted", False)):
                    _PENDING_OBSERVATION_TOOL_FOLLOWUPS[track_id] = {
                        "investigation_id": investigation_id,
                        "request_id": request_id,
                        "tool_name": tool_name,
                        "created_at": time.time(),
                    }
                    observation_decision["phase"] = "tool_request_submitted"
                else:
                    if _observation_prediction_supported(summary_payload):
                        observation_decision = {
                            "action": OBSERVATION_ACTION_PREDICT_BOUNDED,
                            "reason": "tool_request_not_accepted",
                            "uncertainty": _coerce_str_list(summary_payload.get("uncertainties") if isinstance(summary_payload, dict) else []),
                            "decision_source": observation_decision.get("decision_source", "llm_summary_inference"),
                            "phase": "tool_request_rejected",
                        }
                    else:
                        observation_decision = {
                            "action": OBSERVATION_ACTION_INCONCLUSIVE,
                            "reason": "tool_request_not_accepted",
                            "uncertainty": _coerce_str_list(summary_payload.get("uncertainties") if isinstance(summary_payload, dict) else []),
                            "decision_source": observation_decision.get("decision_source", "llm_summary_inference"),
                            "phase": "tool_request_rejected",
                        }

    result["observation_decision"] = observation_decision
    if observation_tool_request is not None:
        result["observation_tool_request"] = observation_tool_request
    if observation_tool_result is not None:
        result["observation_tool_result"] = observation_tool_result

    available = bool(result.get("available"))
    status = "ok" if available else "error"

    assessment = result.get("assessment")
    assessment_label = None
    assessment_uncertainties: list[str] = []
    if isinstance(assessment, dict):
        label = assessment.get("assessment_label")
        if isinstance(label, str):
            assessment_label = label
        assessment_uncertainties = _coerce_str_list(assessment.get("uncertainties"))

    summary_payload = result.get("summary")
    summary_uncertainties = (
        _coerce_str_list(summary_payload.get("uncertainties"))
        if isinstance(summary_payload, dict)
        else []
    )

    agents_payload = result.get("agents")
    observation_agent = agents_payload.get("observation") if isinstance(agents_payload, dict) else None
    reasoning_agent = agents_payload.get("reasoning") if isinstance(agents_payload, dict) else None
    observation_error_code = (
        observation_agent.get("error_code")
        if isinstance(observation_agent, dict)
        else None
    )
    reasoning_error_code = (
        reasoning_agent.get("error_code")
        if isinstance(reasoning_agent, dict)
        else None
    )
    handoff_validation = result.get("handoff_validation")
    handoff_validation_status = (
        handoff_validation.get("status")
        if isinstance(handoff_validation, dict)
        else None
    )
    handoff_validation_error_code = (
        handoff_validation.get("error_code")
        if isinstance(handoff_validation, dict)
        else None
    )
    handoff_validation_error = (
        handoff_validation.get("error")
        if isinstance(handoff_validation, dict)
        else None
    )
    handoff_quality = _derive_handoff_quality(
        handoff_validation_status,
        handoff_validation_error_code,
    )
    cycle_error_code = None
    raw_cycle_error = result.get("error")
    if isinstance(raw_cycle_error, str) and raw_cycle_error.strip():
        cycle_error_code = raw_cycle_error.strip()

    bounded_out = bool(
        result.get("evidence_request") is None
        and bool(result.get("summary", {}).get("needs_more_observations", False))
        and request_evidence_on_uncertainty
    )

    conflicts = _derive_conflicts(
        observation_status=observation_agent.get("status") if isinstance(observation_agent, dict) else None,
        reasoning_status=reasoning_agent.get("status") if isinstance(reasoning_agent, dict) else None,
        observation_error_code=observation_error_code,
        reasoning_error_code=reasoning_error_code,
        handoff_validation_status=handoff_validation_status,
        handoff_validation_error_code=handoff_validation_error_code,
    )

    limitations = _derive_limitations(
        cycle_error_code=cycle_error_code,
        bounded_out=bounded_out,
        summary_uncertainties=summary_uncertainties,
        assessment_uncertainties=assessment_uncertainties,
        handoff_validation_error_code=handoff_validation_error_code,
    )

    return {
        "result": result,
        "latency_ms": latency_ms,
        "available": available,
        "status": status,
        "assessment_label": assessment_label,
        "observation_agent": observation_agent,
        "reasoning_agent": reasoning_agent,
        "observation_error_code": observation_error_code,
        "reasoning_error_code": reasoning_error_code,
        "handoff_validation_status": handoff_validation_status,
        "handoff_validation_error_code": handoff_validation_error_code,
        "handoff_validation_error": handoff_validation_error,
        "handoff_quality": handoff_quality,
        "bounded_out": bounded_out,
        "conflicts": conflicts,
        "limitations": limitations,
        "cycle_error_code": cycle_error_code,
    }


def _build_reasoning_cycle_event(
    *,
    cycle_id: str,
    cycle_started_at: float,
    index: int,
    track_id: str,
    status: str,
    latency_ms: float,
    assessment_label: Any,
    result: dict[str, Any],
    bounded_out: bool,
    conflicts: list[str],
    limitations: list[str],
    observation_agent: Any,
    observation_error_code: Any,
    reasoning_agent: Any,
    reasoning_error_code: Any,
    handoff_validation_status: Any,
    handoff_validation_error_code: Any,
    handoff_validation_error: Any,
    handoff_quality: str,
    cycle_error_code: Any,
    event_origin: str,
    correlation_id: str | None = None,
    cycle_correlation_id: str | None = None,
) -> dict[str, Any]:
    evidence_requested = result.get("evidence_request") is not None
    observation_decision = result.get("observation_decision") if isinstance(result.get("observation_decision"), dict) else {}
    observation_tool_request = result.get("observation_tool_request") if isinstance(result.get("observation_tool_request"), dict) else None
    observation_tool_result = result.get("observation_tool_result") if isinstance(result.get("observation_tool_result"), dict) else None
    investigation_id = (
        _build_investigation_id(cycle_id=cycle_id, track_id=track_id)
        if evidence_requested
        else None
    )
    investigation_state = "awaiting_evidence" if evidence_requested else None
    return {
        "event_type": "reasoning_cycle",
        "event_origin": event_origin,
        "cycle_id": cycle_id,
        "task_id": f"{cycle_id}:track-{index}",
        "parent_task_id": None,
        "timestamp": cycle_started_at,
        "track_id": track_id,
        "status": status,
        "latency_ms": round(latency_ms, 3),
        "assessment_label": assessment_label,
        "evidence_requested": evidence_requested,
        "bounded_out": bounded_out,
        "conflicts": conflicts,
        "limitations": limitations,
        "observation_agent_model": (
            observation_agent.get("model")
            if isinstance(observation_agent, dict)
            else None
        ),
        "observation_agent_status": (
            observation_agent.get("status")
            if isinstance(observation_agent, dict)
            else None
        ),
        "observation_agent_latency_ms": (
            observation_agent.get("latency_ms")
            if isinstance(observation_agent, dict)
            else None
        ),
        "observation_agent_error_code": (
            observation_error_code
        ),
        "observation_agent_attempts": (
            observation_agent.get("attempts")
            if isinstance(observation_agent, dict)
            else None
        ),
        "observation_agent_retry_count": (
            observation_agent.get("retry_count")
            if isinstance(observation_agent, dict)
            else None
        ),
        "reasoning_agent_model": (
            reasoning_agent.get("model")
            if isinstance(reasoning_agent, dict)
            else None
        ),
        "reasoning_agent_status": (
            reasoning_agent.get("status")
            if isinstance(reasoning_agent, dict)
            else None
        ),
        "reasoning_agent_latency_ms": (
            reasoning_agent.get("latency_ms")
            if isinstance(reasoning_agent, dict)
            else None
        ),
        "reasoning_agent_error_code": (
            reasoning_error_code
        ),
        "reasoning_agent_attempts": (
            reasoning_agent.get("attempts")
            if isinstance(reasoning_agent, dict)
            else None
        ),
        "reasoning_agent_retry_count": (
            reasoning_agent.get("retry_count")
            if isinstance(reasoning_agent, dict)
            else None
        ),
        "handoff_validation_status": handoff_validation_status,
        "handoff_validation_error_code": handoff_validation_error_code,
        "handoff_validation_error": handoff_validation_error,
        "handoff_quality": handoff_quality,
        "agent_error_digest": _format_agent_error_digest(
            observation_error_code,
            reasoning_error_code,
        ),
        "cycle_error_code": cycle_error_code,
        "evidence_trace": _extract_evidence_trace(result.get("summary")),
        "observation_decision_action": observation_decision.get("action"),
        "observation_decision_reason": observation_decision.get("reason"),
        "observation_decision_phase": observation_decision.get("phase"),
        "observation_decision_uncertainty": _coerce_str_list(observation_decision.get("uncertainty")),
        "observation_tool_request": observation_tool_request,
        "observation_tool_result": observation_tool_result,
        "error": result.get("error"),
        "correlation_id": correlation_id,
        "cycle_correlation_id": cycle_correlation_id,
        "investigation_id": investigation_id,
        "investigation_state": (
            investigation_state
            if investigation_state in VALID_INVESTIGATION_STATES
            else None
        ),
    }


def _attach_evidence_followup_links(
    event: dict[str, Any],
    evidence_request_payload: Any,
) -> None:
    if isinstance(evidence_request_payload, dict):
        request_index = evidence_request_payload.get("request_index")
        if isinstance(request_index, int) and request_index > 0:
            followup_task_id = f"{event['task_id']}:evidence-{request_index}"
            event["evidence_followup_task_id"] = followup_task_id
            event["evidence_followup_parent_task_id"] = event["task_id"]
            if request_index > 1:
                event["evidence_followup_previous_task_id"] = (
                    f"{event['task_id']}:evidence-{request_index - 1}"
                )


def _persist_event_for_cycle(event: dict[str, Any]) -> str | None:
    publish_started = time.perf_counter()
    try:
        _append_event(event)
        latency_instrumentation.emit_event(
            "final_assessment_published",
            stage="final_assessment_publication",
            duration_ms=round((time.perf_counter() - publish_started) * 1000.0, 3),
            track_id=event.get("track_id"),
            cycle_id=event.get("cycle_id"),
            correlation_id=event.get("correlation_id"),
            publication_status="ok",
        )
    except OSError as error:
        latency_instrumentation.emit_event(
            "final_assessment_published",
            stage="final_assessment_publication",
            duration_ms=round((time.perf_counter() - publish_started) * 1000.0, 3),
            track_id=event.get("track_id"),
            cycle_id=event.get("cycle_id"),
            correlation_id=event.get("correlation_id"),
            publication_status="error",
            error=f"{type(error).__name__}:{error}",
        )
        return f"{type(error).__name__}:{error}"
    return None


def _build_reasoning_cycle_response(
    *,
    cycle_id: str,
    cycle_started_at: float,
    track_ids: list[str],
    success_count: int,
    error_count: int,
    request_evidence_on_uncertainty: bool,
    event_log_error_count: int,
    event_log_last_error: str | None,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "available": True,
        "event_log_path": str(_agent_events_path()),
        "event_log_writable": event_log_error_count == 0,
        "event_log_error_count": event_log_error_count,
        "event_log_last_error": event_log_last_error,
        "cycle": {
            "cycle_id": cycle_id,
            "started_at": cycle_started_at,
            "evaluated_tracks": len(track_ids),
            "success_count": success_count,
            "error_count": error_count,
            "request_evidence_on_uncertainty": request_evidence_on_uncertainty,
        },
        "events": events,
    }


@mcp.tool()
def run_reasoning_cycle_and_log(
    request_evidence_on_uncertainty: bool = True,
    max_tracks: int = 10,
    event_origin: str = "production_monitor",
) -> dict[str, Any]:
    # Runs reasoning cycle and log workflow steps and returns status outcomes.
    """Run a bounded reasoning cycle for current tracks and append structured events."""
    cycle_started_perf = time.perf_counter()
    cycle_context = _initialize_reasoning_cycle(max_tracks)
    cycle_correlation_id = latency_instrumentation.new_correlation_id("cycle")
    latency_instrumentation.emit_event(
        "reasoning_cycle_started",
        stage="orchestration_cycle",
        duration_ms=0.0,
        cycle_id=cycle_context.get("cycle_id") if isinstance(cycle_context, dict) else None,
        correlation_id=cycle_correlation_id,
        request_evidence_on_uncertainty=request_evidence_on_uncertainty,
        max_tracks=max_tracks,
    )
    if not bool(cycle_context.get("available")):
        unavailable_reason = cycle_context.get("response", {}).get("error") if isinstance(cycle_context, dict) else "cycle_unavailable"
        cycle_status = "skipped"
        if unavailable_reason in {"radar_state_stale"}:
            cycle_status = "failed"
        latency_instrumentation.emit_event(
            "reasoning_cycle_completed",
            stage="orchestration_cycle",
            duration_ms=round((time.perf_counter() - cycle_started_perf) * 1000.0, 3),
            cycle_id=cycle_context.get("cycle_id") if isinstance(cycle_context, dict) else None,
            correlation_id=cycle_correlation_id,
            cycle_status=cycle_status,
            completed_tracks=0,
            failed_tracks=0,
            skipped_tracks=0,
            timed_out_tracks=0,
            stale_radar_tracks=1 if unavailable_reason == "radar_state_stale" else 0,
            unavailable_reason=unavailable_reason,
        )
        return cycle_context["response"]

    cycle_started_at = cycle_context["cycle_started_at"]
    cycle_id = cycle_context["cycle_id"]
    track_ids = cycle_context["track_ids"]
    normalized_event_origin = (
        event_origin.strip()
        if isinstance(event_origin, str) and event_origin.strip() in {"startup_activation", "production_monitor"}
        else "unknown"
    )

    events: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0
    timed_out_track_count = 0
    skipped_track_count = 0
    stale_radar_track_count = 0
    event_log_error_count = 0
    event_log_last_error: str | None = None

    for index, track_id in enumerate(track_ids, start=1):
        track_correlation_id = latency_instrumentation.new_correlation_id("track")
        track_evaluation = _evaluate_track_for_cycle(
            track_id,
            request_evidence_on_uncertainty=request_evidence_on_uncertainty,
            correlation_id=track_correlation_id,
            cycle_id=cycle_id,
        )
        result = track_evaluation["result"]
        latency_ms = track_evaluation["latency_ms"]

        available = track_evaluation["available"]
        status = track_evaluation["status"]
        if available:
            success_count += 1
        else:
            error_count += 1

        assessment_label = track_evaluation["assessment_label"]
        observation_agent = track_evaluation["observation_agent"]
        reasoning_agent = track_evaluation["reasoning_agent"]
        observation_error_code = track_evaluation["observation_error_code"]
        reasoning_error_code = track_evaluation["reasoning_error_code"]
        handoff_validation_status = track_evaluation["handoff_validation_status"]
        handoff_validation_error_code = track_evaluation["handoff_validation_error_code"]
        handoff_validation_error = track_evaluation["handoff_validation_error"]
        handoff_quality = track_evaluation["handoff_quality"]
        bounded_out = track_evaluation["bounded_out"]
        conflicts = track_evaluation["conflicts"]
        limitations = track_evaluation["limitations"]
        cycle_error_code = track_evaluation["cycle_error_code"]
        if observation_error_code == "timeout" or reasoning_error_code == "timeout":
            timed_out_track_count += 1
        if cycle_error_code in {"radar_state_unavailable", "radar_state_stale"}:
            skipped_track_count += 1
        if cycle_error_code == "radar_state_stale":
            stale_radar_track_count += 1

        event = _build_reasoning_cycle_event(
            cycle_id=cycle_id,
            cycle_started_at=cycle_started_at,
            index=index,
            track_id=track_id,
            status=status,
            latency_ms=latency_ms,
            assessment_label=assessment_label,
            result=result,
            bounded_out=bounded_out,
            conflicts=conflicts,
            limitations=limitations,
            observation_agent=observation_agent,
            observation_error_code=observation_error_code,
            reasoning_agent=reasoning_agent,
            reasoning_error_code=reasoning_error_code,
            handoff_validation_status=handoff_validation_status,
            handoff_validation_error_code=handoff_validation_error_code,
            handoff_validation_error=handoff_validation_error,
            handoff_quality=handoff_quality,
            cycle_error_code=cycle_error_code,
            event_origin=normalized_event_origin,
            correlation_id=track_correlation_id,
            cycle_correlation_id=cycle_correlation_id,
        )

        _attach_evidence_followup_links(event, result.get("evidence_request"))

        persistence_error = _persist_event_for_cycle(event)
        if persistence_error is not None:
            event_log_error_count += 1
            event_log_last_error = persistence_error
        events.append(event)

    cycle_status = "completed"
    if error_count > 0 and success_count == 0:
        cycle_status = "failed"
    if timed_out_track_count > 0:
        cycle_status = "timed_out"
    if len(track_ids) == 0:
        cycle_status = "skipped"
    latency_instrumentation.emit_event(
        "reasoning_cycle_completed",
        stage="orchestration_cycle",
        duration_ms=round((time.perf_counter() - cycle_started_perf) * 1000.0, 3),
        cycle_id=cycle_id,
        correlation_id=cycle_correlation_id,
        cycle_status=cycle_status,
        completed_tracks=success_count,
        failed_tracks=error_count,
        skipped_tracks=skipped_track_count,
        timed_out_tracks=timed_out_track_count,
        stale_radar_tracks=stale_radar_track_count,
    )

    return _build_reasoning_cycle_response(
        cycle_id=cycle_id,
        cycle_started_at=cycle_started_at,
        track_ids=track_ids,
        success_count=success_count,
        error_count=error_count,
        request_evidence_on_uncertainty=request_evidence_on_uncertainty,
        event_log_error_count=event_log_error_count,
        event_log_last_error=event_log_last_error,
        events=events,
    )


@mcp.tool()
def get_recent_agent_events(limit: int = 20) -> dict[str, Any]:
    # Gets recent agent events values for downstream consumers.
    """Return recent structured agent events from captures/agent_events.jsonl."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    event_path = _agent_events_path()
    if not event_path.is_file():
        return {
            "available": False,
            "event_log_path": str(event_path),
            "error": "agent_events_missing",
            "events": [],
        }

    events, dropped_lines = _load_events(limit, event_path)

    return {
        "available": True,
        "event_log_path": str(event_path),
        "requested_limit": limit,
        "returned_events": len(events),
        "dropped_lines": dropped_lines,
        "events": events,
    }


@mcp.tool()
def summarize_agent_event_health(window_size: int = 100) -> dict[str, Any]:
    # Implements summarize agent event health behavior for this module.
    """Summarize recent agent event health from captures/agent_events.jsonl."""
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")

    event_path = _agent_events_path()
    if not event_path.is_file():
        return {
            "available": False,
            "event_log_path": str(event_path),
            "error": "agent_events_missing",
        }

    events, dropped_lines = _load_events(window_size, event_path)
    if not events:
        return {
            "available": False,
            "event_log_path": str(event_path),
            "error": "no_valid_events",
            "dropped_lines": dropped_lines,
        }

    ok_count = 0
    error_count = 0
    bounded_out_count = 0
    latency_values: list[float] = []
    timestamps: list[float] = []
    assessment_counts: dict[str, int] = {}
    observation_error_code_counts: dict[str, int] = {}
    reasoning_error_code_counts: dict[str, int] = {}
    cycle_error_code_counts: dict[str, int] = {}
    handoff_validation_status_counts: dict[str, int] = {}
    handoff_validation_error_code_counts: dict[str, int] = {}
    handoff_quality_counts: dict[str, int] = {}
    conflict_event_count = 0
    limitation_event_count = 0
    conflict_item_count = 0
    limitation_item_count = 0

    for event in events:
        status = event.get("status")
        if status == "ok":
            ok_count += 1
        elif status == "error":
            error_count += 1

        if bool(event.get("bounded_out")):
            bounded_out_count += 1

        latency_ms = event.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            latency_values.append(float(latency_ms))

        timestamp = event.get("timestamp")
        if isinstance(timestamp, (int, float)):
            timestamps.append(float(timestamp))

        label = event.get("assessment_label")
        if isinstance(label, str) and label.strip():
            assessment_counts[label] = assessment_counts.get(label, 0) + 1

        observation_error_code = event.get("observation_agent_error_code")
        if isinstance(observation_error_code, str) and observation_error_code.strip():
            observation_error_code_counts[observation_error_code] = (
                observation_error_code_counts.get(observation_error_code, 0) + 1
            )

        reasoning_error_code = event.get("reasoning_agent_error_code")
        if isinstance(reasoning_error_code, str) and reasoning_error_code.strip():
            reasoning_error_code_counts[reasoning_error_code] = (
                reasoning_error_code_counts.get(reasoning_error_code, 0) + 1
            )

        cycle_error_code = event.get("cycle_error_code")
        if isinstance(cycle_error_code, str) and cycle_error_code.strip():
            cycle_error_code_counts[cycle_error_code] = (
                cycle_error_code_counts.get(cycle_error_code, 0) + 1
            )
        else:
            fallback_error = event.get("error")
            if isinstance(fallback_error, str) and fallback_error.strip():
                cycle_error_code_counts[fallback_error] = (
                    cycle_error_code_counts.get(fallback_error, 0) + 1
                )

        handoff_validation_status = event.get("handoff_validation_status")
        if isinstance(handoff_validation_status, str) and handoff_validation_status.strip():
            handoff_validation_status_counts[handoff_validation_status] = (
                handoff_validation_status_counts.get(handoff_validation_status, 0) + 1
            )

        handoff_validation_error_code = event.get("handoff_validation_error_code")
        if isinstance(handoff_validation_error_code, str) and handoff_validation_error_code.strip():
            handoff_validation_error_code_counts[handoff_validation_error_code] = (
                handoff_validation_error_code_counts.get(handoff_validation_error_code, 0) + 1
            )

        handoff_quality = event.get("handoff_quality")
        if not isinstance(handoff_quality, str) or not handoff_quality.strip():
            handoff_quality = _derive_handoff_quality(
                handoff_validation_status,
                handoff_validation_error_code,
            )
        handoff_quality_counts[handoff_quality] = handoff_quality_counts.get(handoff_quality, 0) + 1

        conflicts = event.get("conflicts")
        if isinstance(conflicts, list):
            normalized_conflicts = [
                item for item in conflicts if isinstance(item, str) and item.strip()
            ]
            if normalized_conflicts:
                conflict_event_count += 1
                conflict_item_count += len(normalized_conflicts)

        limitations = event.get("limitations")
        if isinstance(limitations, list):
            normalized_limitations = [
                item for item in limitations if isinstance(item, str) and item.strip()
            ]
            if normalized_limitations:
                limitation_event_count += 1
                limitation_item_count += len(normalized_limitations)

    total_events = len(events)
    success_rate = ok_count / total_events
    error_rate = error_count / total_events
    bounded_out_rate = bounded_out_count / total_events

    observation_top_code, observation_top_count = _top_count_entry(observation_error_code_counts)
    reasoning_top_code, reasoning_top_count = _top_count_entry(reasoning_error_code_counts)
    cycle_top_code, cycle_top_count = _top_count_entry(cycle_error_code_counts)
    handoff_top_code, handoff_top_count = _top_count_entry(handoff_validation_error_code_counts)

    observation_top_rate = (
        observation_top_count / total_events if observation_top_code is not None else 0.0
    )
    reasoning_top_rate = (
        reasoning_top_count / total_events if reasoning_top_code is not None else 0.0
    )
    cycle_top_rate = (
        cycle_top_count / total_events if cycle_top_code is not None else 0.0
    )
    handoff_top_rate = (
        handoff_top_count / total_events if handoff_top_code is not None else 0.0
    )
    handoff_degraded_count = handoff_quality_counts.get("degraded", 0)
    handoff_degraded_rate = handoff_degraded_count / total_events
    conflict_event_rate = conflict_event_count / total_events
    limitation_event_rate = limitation_event_count / total_events

    latency_summary = {
        "count": len(latency_values),
        "p50_ms": _percentile(latency_values, 0.5),
        "p95_ms": _percentile(latency_values, 0.95),
        "max_ms": max(latency_values) if latency_values else None,
    }

    time_window = {
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }

    latest_timestamp = time_window["last_timestamp"]
    freshness_age_seconds: float | None = None
    freshness_is_fresh = False
    if isinstance(latest_timestamp, (int, float)):
        freshness_age_seconds = max(0.0, time.time() - float(latest_timestamp))
        freshness_is_fresh = freshness_age_seconds <= AGENT_EVENT_HEALTH_FRESHNESS_SECONDS

    return {
        "available": True,
        "event_log_path": str(_agent_events_path()),
        "window_size": window_size,
        "returned_events": total_events,
        "dropped_lines": dropped_lines,
        "status": {
            "ok_count": ok_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "bounded_out_count": bounded_out_count,
            "bounded_out_rate": bounded_out_rate,
        },
        "latency": latency_summary,
        "assessment_label_counts": assessment_counts,
        "agent_error_code_counts": {
            "observation": observation_error_code_counts,
            "reasoning": reasoning_error_code_counts,
            "cycle": cycle_error_code_counts,
        },
        "handoff_validation": {
            "status_counts": handoff_validation_status_counts,
            "error_code_counts": handoff_validation_error_code_counts,
            "quality_counts": handoff_quality_counts,
            "degraded_count": handoff_degraded_count,
            "degraded_rate": handoff_degraded_rate,
        },
        "collaboration_signals": {
            "conflict_event_count": conflict_event_count,
            "conflict_event_rate": conflict_event_rate,
            "conflict_item_count": conflict_item_count,
            "limitation_event_count": limitation_event_count,
            "limitation_event_rate": limitation_event_rate,
            "limitation_item_count": limitation_item_count,
        },
        "agent_error_trends": {
            "observation": {
                "top_error_code": observation_top_code,
                "top_error_count": observation_top_count,
                "top_error_rate": observation_top_rate,
            },
            "reasoning": {
                "top_error_code": reasoning_top_code,
                "top_error_count": reasoning_top_count,
                "top_error_rate": reasoning_top_rate,
            },
            "cycle": {
                "top_error_code": cycle_top_code,
                "top_error_count": cycle_top_count,
                "top_error_rate": cycle_top_rate,
            },
            "handoff_validation": {
                "top_error_code": handoff_top_code,
                "top_error_count": handoff_top_count,
                "top_error_rate": handoff_top_rate,
            },
        },
        "freshness": {
            "is_fresh": freshness_is_fresh,
            "age_seconds": freshness_age_seconds,
            "threshold_seconds": AGENT_EVENT_HEALTH_FRESHNESS_SECONDS,
        },
        "time_window": time_window,
    }


@mcp.tool()
def prune_agent_event_log(
    max_events_keep: int = 5000,
    archive_removed_events: bool = True,
) -> dict[str, Any]:
    # Implements prune agent event log behavior for this module.
    """Prune old agent event log lines and optionally archive removed lines."""
    if max_events_keep < 0:
        raise ValueError("max_events_keep must be zero or greater")

    event_path = _agent_events_path()
    if not event_path.is_file():
        return {
            "available": False,
            "event_log_path": str(_agent_events_path()),
            "error": "agent_events_missing",
        }

    lines = event_path.read_text(encoding="utf-8").splitlines()
    total_before = len(lines)

    if total_before <= max_events_keep:
        return {
            "available": True,
            "event_log_path": str(_agent_events_path()),
            "total_lines_before": total_before,
            "total_lines_after": total_before,
            "removed_lines": 0,
            "archive_path": None,
            "pruned": False,
        }

    cut_index = total_before - max_events_keep
    removed_lines = lines[:cut_index]
    kept_lines = lines[cut_index:]

    archive_path: Path | None = None
    if archive_removed_events and removed_lines:
        archive_path = _archive_path_for_now(event_path)
        archive_path.write_text("\n".join(removed_lines) + "\n", encoding="utf-8")

    if kept_lines:
        event_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    else:
        event_path.write_text("", encoding="utf-8")

    return {
        "available": True,
        "event_log_path": str(_agent_events_path()),
        "total_lines_before": total_before,
        "total_lines_after": len(kept_lines),
        "removed_lines": len(removed_lines),
        "archive_path": None if archive_path is None else str(archive_path),
        "pruned": True,
    }
