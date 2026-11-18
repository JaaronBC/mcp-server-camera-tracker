from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from agents.dashboard.config import CAPTURES_DIR
from agents.tracking.calibration import GroundPlaneCalibration, load_ground_plane_calibration
from agents.coordinator import AgentCoordinator
from agents.tracking.agent_tracking import (
    AsyncVisualTrackingRuntime,
    LLMVisualTrackingAgent,
    _validate_tracking_decision_payload,
    encode_frame_as_data_url,
)
from agents.tracking.evidence_packaging import TrackingEvidencePackager
from agents.tracking.estimate_proposals import TrackingEstimateProposalEngine
from agents.tracking.evidence_store import TrackingEvidenceStore
from agents.tracking.frame_transformer import build_tracking_frame_outputs
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer
from agents.tracking.runtime_support import (
    apply_tracking_agent_metric_update_to_state,
    apply_tracking_observation_update,
    commit_live_tracking_frame_state,
    summarize_tracking_agent_observation_metrics,
    submit_tracking_runtime_candidates,
)
from agents.tracking.identity_registry import PersistentIdentityRegistry
from agents.tracking.state_publisher import (
    build_tracking_frame_message,
    publish_tracking_frame_message,
    write_json_payload,
)
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport
from .identity_reassociation_consumer import IdentityReassociationIntentConsumer

from .config import (
    GROUND_CALIBRATION_PATH,
    LATEST_TRACKING_ESTIMATE_DIAGNOSTICS_PATH,
    LATEST_TRACKING_FRAME_PATH,
    LATEST_TRACKING_PREDICTION_DETECTION_ADVISORIES_PATH,
    LATEST_OPERATOR_TRACKING_FEEDBACK_PATH,
    TRACKING_FEEDBACK_CONSUMER_DECISIONS_PATH,
    TRACKING_DECISION_IPC_ENABLED,
    TRACKING_DECISION_TOOL_REQUESTS_PATH,
    TRACKING_DECISION_TOOL_RESULTS_PATH,
    TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH,
    LATEST_TRACKING_DECISION_REQUEST_INDEX_PATH,
    LATEST_TRACKING_DECISION_RESULT_INDEX_PATH,
    TRACKING_DECISION_IPC_MAX_RESULTS_PER_POLL,
    MAX_GROUND_X,
    MAX_GROUND_Z,
    TRACKING_AGENT_REQUEST_TIMEOUT_SECONDS,
    TRACKING_MAX_EFFECTIVE_HORIZON_SECONDS,
    RADAR_TRAIL_LENGTH,
    TRACKING_MISSING_TRACK_GRACE_SECONDS,
    UDP_TRACKING_HOST,
    UDP_TRACKING_PORT,
    IDENTITY_REGISTRY_ENABLED,
    IDENTITY_REGISTRY_HISTORY_LIMIT,
    IDENTITY_REGISTRY_TEMPORARY_LOSS_SECONDS,
    IDENTITY_REGISTRY_RETIREMENT_SECONDS,
    IDENTITY_REGISTRY_MAX_RETIRED,
    IDENTITY_SNAPSHOT_BRIDGE_ENABLED,
    TRACKING_IDENTITY_REASSOCIATION_ENABLED,
    TRACKING_IDENTITY_REASSOCIATION_MAX_INTENTS_PER_POLL,
    TRACKING_IDENTITY_REASSOCIATION_MAX_ACK_ENTRIES,
    TRACKING_IDENTITY_REASSOCIATION_POLL_INTERVAL_SECONDS,
    LATEST_TRACKING_IDENTITY_REASSOCIATION_INTENTS_PATH,
    LATEST_TRACKING_IDENTITY_REASSOCIATION_ACK_LEDGER_PATH,
    TRACKING_STABLE_IDENTITY_VISUALIZATION_ENABLED,
)
from .state import SharedDashboardState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOLLOW_UP_INVESTIGATION_PREFIX = "followup_investigation_id:"
FOLLOW_UP_REQUEST_PREFIX = "followup_request_id:"
FOLLOW_UP_RESULT_KEY_PREFIX = "followup_result_key:"
FOLLOW_UP_RESULT_STATUS_PREFIX = "followup_result_status:"
UNSUCCESSFUL_DECISION_RESULT_STATUSES = {
    "timeout",
    "unavailable",
    "stale",
    "incompatible",
    "expired",
    "terminal",
    "rejected",
}
FOLLOW_UP_TIMEOUT_GRACE_SECONDS = max(5.0, float(TRACKING_AGENT_REQUEST_TIMEOUT_SECONDS) * 2.0)


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _extract_follow_up_metadata_from_observation(observation: Any) -> dict[str, str | None]:
    uncertainties = getattr(observation, "uncertainties", None)
    if not isinstance(uncertainties, list):
        uncertainties = []

    investigation_id = None
    request_id = None
    result_key = None
    result_status = None

    for item in uncertainties:
        if not isinstance(item, str):
            continue
        if item.startswith(FOLLOW_UP_INVESTIGATION_PREFIX):
            investigation_id = _nonempty_str(item[len(FOLLOW_UP_INVESTIGATION_PREFIX):])
        elif item.startswith(FOLLOW_UP_REQUEST_PREFIX):
            request_id = _nonempty_str(item[len(FOLLOW_UP_REQUEST_PREFIX):])
        elif item.startswith(FOLLOW_UP_RESULT_KEY_PREFIX):
            result_key = _nonempty_str(item[len(FOLLOW_UP_RESULT_KEY_PREFIX):])
        elif item.startswith(FOLLOW_UP_RESULT_STATUS_PREFIX):
            result_status = _nonempty_str(item[len(FOLLOW_UP_RESULT_STATUS_PREFIX):])

    return {
        "investigation_id": investigation_id,
        "request_id": request_id,
        "result_key": result_key,
        "result_status": result_status,
    }


def _build_inconclusive_follow_up_decision(
    *,
    investigation_id: str,
    reason: str,
    correlation_id: str | None,
    request_id: str | None,
    result_key: str | None,
    initial_decision: dict[str, Any] | None,
    evidence_references: dict[str, Any] | None,
    result_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    decision: dict[str, Any] = {
        "action": "inconclusive",
        "investigation_id": investigation_id,
        "reason": reason,
        "follow_up": {
            "status": "failed",
            "request_id": request_id,
            "result_key": result_key,
        },
    }
    if correlation_id is not None:
        decision["correlation_id"] = correlation_id
    if isinstance(initial_decision, dict):
        decision["initial_decision"] = dict(initial_decision)
    if isinstance(evidence_references, dict):
        decision["evidence_references"] = dict(evidence_references)
    if isinstance(result_payload, dict):
        decision["mcp_result"] = dict(result_payload)
    return decision


def _append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _persist_terminal_follow_up_decision_event(
    *,
    now: float,
    observation: Any,
    follow_up_meta: dict[str, str | None],
    decision_payload: dict[str, Any] | None,
    validation_errors: list[str],
    status: str,
    reason: str,
    investigation_state: dict[str, Any] | None,
) -> None:
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "event_type": "tracking_terminal_follow_up_decision",
        "timestamp": float(now),
        "status": status,
        "reason": reason,
        "observation_id": _nonempty_str(getattr(observation, "observation_id", None)),
        "llm_correlation_id": _nonempty_str(getattr(observation, "llm_correlation_id", None)),
        "investigation_id": _nonempty_str(follow_up_meta.get("investigation_id")),
        "request_id": _nonempty_str(follow_up_meta.get("request_id")),
        "result_key": _nonempty_str(follow_up_meta.get("result_key")),
        "result_status": _nonempty_str(follow_up_meta.get("result_status")),
        "validation": {
            "valid": len(validation_errors) == 0,
            "errors": list(validation_errors),
        },
    }

    if isinstance(decision_payload, dict):
        record["decision"] = dict(decision_payload)
        record["action"] = _nonempty_str(decision_payload.get("action"))
        record["decision_reason"] = _nonempty_str(decision_payload.get("reason"))
        uncertainty = decision_payload.get("uncertainty")
        if isinstance(uncertainty, list):
            record["decision_uncertainty"] = [
                str(item) for item in uncertainty if isinstance(item, (str, int, float, bool)) or item is None
            ]
        provenance = decision_payload.get("provenance")
        if isinstance(provenance, dict):
            attempt_id = _nonempty_str(provenance.get("attempt_id"))
            if attempt_id is not None:
                record["llm_attempt_id"] = attempt_id
        evidence_references = decision_payload.get("evidence_references")
        if isinstance(evidence_references, dict):
            record["evidence_references"] = dict(evidence_references)

    if isinstance(investigation_state, dict):
        if "evidence_references" not in record and isinstance(investigation_state.get("evidence_references"), dict):
            record["evidence_references"] = dict(investigation_state.get("evidence_references"))
        if record.get("request_id") is None:
            record["request_id"] = _nonempty_str(investigation_state.get("request_id"))
        if record.get("result_key") is None:
            record["result_key"] = _nonempty_str(investigation_state.get("result_key"))
        if record.get("investigation_id") is None:
            record["investigation_id"] = _nonempty_str(investigation_state.get("investigation_id"))
        if record.get("llm_correlation_id") is None:
            record["llm_correlation_id"] = _nonempty_str(investigation_state.get("correlation_id"))

    _append_jsonl_record(TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH, record)


def _snapshot_constraints_satisfied(
    *,
    tool_name: str,
    result_payload: dict[str, Any],
    initial_decision: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    if tool_name != "get_identity_snapshot":
        return True, None

    if bool(result_payload.get("stale_snapshot", False)):
        return False, "identity_snapshot_stale"

    compatibility = result_payload.get("compatibility")
    if isinstance(compatibility, dict) and compatibility.get("compatible") is False:
        return False, "snapshot_incompatible_with_minimum_requirements"

    constraints = None
    if isinstance(initial_decision, dict):
        constraints = initial_decision.get("snapshot_constraints")
    if not isinstance(constraints, dict):
        return True, None

    min_generation = constraints.get("min_source_generation")
    min_version = constraints.get("min_source_version")
    source_generation = result_payload.get("source_generation")
    source_version = result_payload.get("source_version")

    if isinstance(min_generation, int):
        if not isinstance(source_generation, int) or source_generation < min_generation:
            return False, "snapshot_generation_mismatch"

    if isinstance(min_version, int):
        if not isinstance(source_version, int) or source_version < min_version:
            return False, "snapshot_version_mismatch"

    return True, None


def _pending_investigation_follow_up_context_locked(
    state: SharedDashboardState,
    *,
    now: float,
) -> tuple[str | None, dict[str, Any] | None]:
    selected_id: str | None = None
    selected_state: dict[str, Any] | None = None

    for investigation_id, investigation_state in state.tracking_decision_investigations.items():
        if not isinstance(investigation_state, dict):
            continue
        if bool(investigation_state.get("terminal", False)):
            continue

        result_status = _nonempty_str(investigation_state.get("result_status"))
        follow_up_completed = bool(investigation_state.get("follow_up_completed", False))
        follow_up_submitted = bool(investigation_state.get("follow_up_submitted", False))

        if result_status is None or follow_up_completed or follow_up_submitted:
            continue

        selected_id = _nonempty_str(investigation_id)
        selected_state = investigation_state
        break

    if selected_id is None or not isinstance(selected_state, dict):
        return None, None

    result_payload = selected_state.get("result_payload")
    if not isinstance(result_payload, dict):
        result_payload = {}

    return selected_id, {
        "investigation_id": selected_id,
        "request_id": _nonempty_str(selected_state.get("request_id")),
        "idempotency_key": _nonempty_str(selected_state.get("result_key")),
        "result_status": _nonempty_str(selected_state.get("result_status")),
        "tool_name": _nonempty_str(selected_state.get("tool_name")),
        "correlation_id": _nonempty_str(selected_state.get("correlation_id")),
        "initial_decision": (
            dict(selected_state.get("initial_decision"))
            if isinstance(selected_state.get("initial_decision"), dict)
            else {}
        ),
        "evidence_references": (
            dict(selected_state.get("evidence_references"))
            if isinstance(selected_state.get("evidence_references"), dict)
            else {}
        ),
        "mcp_result": dict(result_payload),
    }


def _enforce_investigation_follow_up_timeout_locked(
    state: SharedDashboardState,
    *,
    now: float,
) -> None:
    for investigation_id, investigation_state in state.tracking_decision_investigations.items():
        if not isinstance(investigation_state, dict):
            continue
        if bool(investigation_state.get("terminal", False)):
            continue

        result_status = _nonempty_str(investigation_state.get("result_status"))
        follow_up_completed = bool(investigation_state.get("follow_up_completed", False))
        if follow_up_completed:
            continue

        updated_at = investigation_state.get("updated_at")
        if not isinstance(updated_at, (int, float)):
            continue

        if (float(now) - float(updated_at)) < FOLLOW_UP_TIMEOUT_GRACE_SECONDS:
            continue

        timeout_reason = "follow_up_timeout"
        if result_status is None:
            timeout_reason = "mcp_result_timeout"

        investigation_state["terminal"] = True
        investigation_state["follow_up_completed"] = True
        investigation_state["final_decision"] = _build_inconclusive_follow_up_decision(
            investigation_id=_nonempty_str(investigation_id) or str(investigation_id),
            reason=timeout_reason,
            correlation_id=_nonempty_str(investigation_state.get("correlation_id")),
            request_id=_nonempty_str(investigation_state.get("request_id")),
            result_key=_nonempty_str(investigation_state.get("result_key")),
            initial_decision=(
                dict(investigation_state.get("initial_decision"))
                if isinstance(investigation_state.get("initial_decision"), dict)
                else None
            ),
            evidence_references=(
                dict(investigation_state.get("evidence_references"))
                if isinstance(investigation_state.get("evidence_references"), dict)
                else None
            ),
            result_payload=(
                dict(investigation_state.get("result_payload"))
                if isinstance(investigation_state.get("result_payload"), dict)
                else None
            ),
        )
        investigation_state["updated_at"] = now


def _active_nonterminal_investigation_locked(state: SharedDashboardState) -> tuple[str | None, dict[str, Any] | None]:
    for investigation_id, investigation_state in state.tracking_decision_investigations.items():
        if not isinstance(investigation_state, dict):
            continue
        if bool(investigation_state.get("terminal", False)):
            continue

        selected_id = _nonempty_str(investigation_id)
        if selected_id is not None:
            return selected_id, investigation_state
    return None, None


def resolve_model_path() -> str:
    candidates = [
        Path(__file__).resolve().parent / "yolo11n.pt",
        PROJECT_ROOT / "yolo11n.pt",
        Path("yolov8n.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "yolov8n.pt"


def build_tracking_message(timestamp: float, detections: list[dict]) -> dict:
    return build_tracking_frame_message(timestamp, detections)


def _capture_frame_or_mark_error(state: SharedDashboardState, camera) -> tuple[bool, object | None]:
    ok, frame = camera.read()
    if not ok:
        with state.lock:
            state.camera_status = "frame_error"
        time.sleep(0.1)
        return False, None

    with state.lock:
        state.camera_status = "connected"
    return True, frame


def _run_detection_or_mark_error(state: SharedDashboardState, model: YOLO, frame):
    try:
        result = model.track(
            source=frame,
            tracker="bytetrack.yaml",
            classes=[0],
            persist=True,
            verbose=False,
            conf=0.65,
        )[0]
    except Exception as error:
        with state.lock:
            state.camera_status = "frame_error"
            state.detector_status = "error"
            state.input_error = f"{type(error).__name__}: {error}"
        time.sleep(0.15)
        return None, True

    with state.lock:
        state.detector_status = "running"
    return result, False


def _apply_tracking_agent_runtime_cycle(
    *,
    state: SharedDashboardState,
    tracking_runtime: AsyncVisualTrackingRuntime,
    now: float,
    frame,
    tracks,
    tracking_candidates,
    selected_event_reasons,
) -> object | None:
    did_submit = False
    suppress_generic_submit_for_cycle = False
    with state.lock:
        track_trails_snapshot = dict(state.track_trails)
    tracking_agent_observation = tracking_runtime.poll_latest_observation()
    if tracking_agent_observation is not None:
        follow_up_terminalization_blocked = False
        follow_up_meta = _extract_follow_up_metadata_from_observation(tracking_agent_observation)
        follow_up_investigation_id = _nonempty_str(follow_up_meta.get("investigation_id"))
        follow_up_request_id = _nonempty_str(follow_up_meta.get("request_id"))
        follow_up_result_key = _nonempty_str(follow_up_meta.get("result_key"))

        should_apply_updates = True
        if follow_up_investigation_id is not None:
            with state.lock:
                investigation_state = state.tracking_decision_investigations.get(follow_up_investigation_id)

            if not isinstance(investigation_state, dict):
                should_apply_updates = False
                follow_up_terminalization_blocked = True
            else:
                expected_request_id = _nonempty_str(investigation_state.get("request_id"))
                expected_result_key = _nonempty_str(investigation_state.get("result_key"))
                is_terminal = bool(investigation_state.get("terminal", False))

                if is_terminal:
                    should_apply_updates = False
                    follow_up_terminalization_blocked = True
                elif (
                    follow_up_request_id is not None
                    and expected_request_id is not None
                    and follow_up_request_id != expected_request_id
                ):
                    should_apply_updates = False
                    follow_up_terminalization_blocked = True
                elif (
                    follow_up_result_key is not None
                    and expected_result_key is not None
                    and follow_up_result_key != expected_result_key
                ):
                    should_apply_updates = False
                    follow_up_terminalization_blocked = True

            if should_apply_updates:
                decision_payload = getattr(tracking_agent_observation, "decision", None)
                if isinstance(decision_payload, dict):
                    action = _nonempty_str(decision_payload.get("action"))
                    has_tool_requests = isinstance(decision_payload.get("tool_requests"), list) and bool(
                        decision_payload.get("tool_requests")
                    )
                    if action in {"request_identity_snapshot", "request_additional_tracking_evidence"} or has_tool_requests:
                        should_apply_updates = False
        else:
            decision_payload = getattr(tracking_agent_observation, "decision", None)
            if isinstance(decision_payload, dict):
                action = _nonempty_str(decision_payload.get("action"))
                has_tool_requests = isinstance(decision_payload.get("tool_requests"), list) and bool(
                    decision_payload.get("tool_requests")
                )
                if action in {"request_identity_snapshot", "request_additional_tracking_evidence"} or has_tool_requests:
                    suppress_generic_submit_for_cycle = True

        update_summary = {
            "accepted": 0,
            "rejected": 0,
            "unmatched": 0,
            "fallback_used": 0,
            "llm_applied": 0,
        }
        if should_apply_updates:
            update_summary = apply_tracking_observation_update(
                tracks,
                tracking_agent_observation,
                track_trails=track_trails_snapshot,
            )
            metric_update = summarize_tracking_agent_observation_metrics(
                tracking_agent_observation,
                update_summary,
            )
        else:
            metric_update = summarize_tracking_agent_observation_metrics(
                tracking_agent_observation,
                update_summary,
            )

        if follow_up_investigation_id is not None and not follow_up_terminalization_blocked:
            with state.lock:
                investigation_state = state.tracking_decision_investigations.get(follow_up_investigation_id)
                if isinstance(investigation_state, dict):
                    final_decision_payload = (
                        dict(getattr(tracking_agent_observation, "decision", {}))
                        if isinstance(getattr(tracking_agent_observation, "decision", None), dict)
                        else None
                    )
                    if isinstance(final_decision_payload, dict):
                        action = _nonempty_str(final_decision_payload.get("action"))
                        has_tool_requests = isinstance(final_decision_payload.get("tool_requests"), list) and bool(
                            final_decision_payload.get("tool_requests")
                        )
                        if action in {"request_identity_snapshot", "request_additional_tracking_evidence"} or has_tool_requests:
                            final_decision_payload = _build_inconclusive_follow_up_decision(
                                investigation_id=follow_up_investigation_id,
                                reason="follow_up_tool_request_not_allowed",
                                correlation_id=_nonempty_str(investigation_state.get("correlation_id")),
                                request_id=follow_up_request_id,
                                result_key=follow_up_result_key,
                                initial_decision=(
                                    dict(investigation_state.get("initial_decision"))
                                    if isinstance(investigation_state.get("initial_decision"), dict)
                                    else None
                                ),
                                evidence_references=(
                                    dict(investigation_state.get("evidence_references"))
                                    if isinstance(investigation_state.get("evidence_references"), dict)
                                    else None
                                ),
                                result_payload=(
                                    dict(investigation_state.get("result_payload"))
                                    if isinstance(investigation_state.get("result_payload"), dict)
                                    else None
                                ),
                            )
                            should_apply_updates = False

                    if final_decision_payload is None:
                        if update_summary.get("llm_applied", 0) > 0:
                            final_decision_payload = {
                                "action": "propose_identity_association",
                                "investigation_id": follow_up_investigation_id,
                                "follow_up": {
                                    "status": "completed",
                                    "request_id": follow_up_request_id,
                                    "result_key": follow_up_result_key,
                                },
                            }
                            correlation_id = _nonempty_str(investigation_state.get("correlation_id"))
                            if correlation_id is not None:
                                final_decision_payload["correlation_id"] = correlation_id
                        else:
                            final_decision_payload = _build_inconclusive_follow_up_decision(
                                investigation_id=follow_up_investigation_id,
                                reason="follow_up_no_supported_association",
                                correlation_id=_nonempty_str(investigation_state.get("correlation_id")),
                                request_id=follow_up_request_id,
                                result_key=follow_up_result_key,
                                initial_decision=(
                                    dict(investigation_state.get("initial_decision"))
                                    if isinstance(investigation_state.get("initial_decision"), dict)
                                    else None
                                ),
                                evidence_references=(
                                    dict(investigation_state.get("evidence_references"))
                                    if isinstance(investigation_state.get("evidence_references"), dict)
                                    else None
                                ),
                                result_payload=(
                                    dict(investigation_state.get("result_payload"))
                                    if isinstance(investigation_state.get("result_payload"), dict)
                                    else None
                                ),
                            )

                    validated_terminal_decision, terminal_decision_errors = _validate_tracking_decision_payload(
                        final_decision_payload
                    )
                    terminal_action = (
                        _nonempty_str(validated_terminal_decision.get("action"))
                        if isinstance(validated_terminal_decision, dict)
                        else None
                    )
                    if terminal_action not in {"inconclusive", "propose_identity_association"}:
                        final_decision_payload = _build_inconclusive_follow_up_decision(
                            investigation_id=follow_up_investigation_id,
                            reason=(
                                "follow_up_invalid_terminal_decision"
                                if terminal_decision_errors
                                else "follow_up_invalid_terminal_action"
                            ),
                            correlation_id=_nonempty_str(investigation_state.get("correlation_id")),
                            request_id=follow_up_request_id,
                            result_key=follow_up_result_key,
                            initial_decision=(
                                dict(investigation_state.get("initial_decision"))
                                if isinstance(investigation_state.get("initial_decision"), dict)
                                else None
                            ),
                            evidence_references=(
                                dict(investigation_state.get("evidence_references"))
                                if isinstance(investigation_state.get("evidence_references"), dict)
                                else None
                            ),
                            result_payload=(
                                dict(investigation_state.get("result_payload"))
                                if isinstance(investigation_state.get("result_payload"), dict)
                                else None
                            ),
                        )
                    else:
                        final_decision_payload = validated_terminal_decision

                    investigation_state["terminal"] = True
                    investigation_state["follow_up_completed"] = True
                    investigation_state["follow_up_submitted"] = True
                    investigation_state["final_decision"] = final_decision_payload
                    investigation_state["updated_at"] = now

        apply_tracking_agent_metric_update_to_state(
            state=state,
            metric_update=metric_update,
        )

        _submit_tracking_decision_request_if_enabled(
            state=state,
            tracking_agent_observation=tracking_agent_observation,
            now=now,
        )

    with state.lock:
        _enforce_investigation_follow_up_timeout_locked(state, now=now)
        active_investigation_id, active_investigation_state = _active_nonterminal_investigation_locked(state)
        pending_investigation_id, pending_follow_up_context = _pending_investigation_follow_up_context_locked(
            state,
            now=now,
        )

    if isinstance(pending_follow_up_context, dict):
        did_submit = submit_tracking_runtime_candidates(
            runtime=tracking_runtime,
            now=now,
            frame=frame,
            detections=tracking_candidates,
            event_reasons=["tracking_decision_follow_up"],
            follow_up_context=pending_follow_up_context,
            encode_frame=encode_frame_as_data_url,
            request_timeout_seconds=TRACKING_AGENT_REQUEST_TIMEOUT_SECONDS,
        )
        if did_submit and pending_investigation_id is not None:
            with state.lock:
                investigation_state = state.tracking_decision_investigations.get(pending_investigation_id)
                if isinstance(investigation_state, dict):
                    investigation_state["follow_up_submitted"] = True
                    investigation_state["follow_up_submit_pending"] = False
                    investigation_state["iterations"] = int(investigation_state.get("iterations", 1)) + 1
                    result_key = _nonempty_str(investigation_state.get("result_key"))
                    if result_key is not None:
                        state.tracking_decision_followup_result_keys.add(result_key)
                    investigation_state["updated_at"] = now
    elif active_investigation_id is not None and isinstance(active_investigation_state, dict):
        did_submit = False
    elif suppress_generic_submit_for_cycle:
        did_submit = False
    else:
        did_submit = submit_tracking_runtime_candidates(
            runtime=tracking_runtime,
            now=now,
            frame=frame,
            detections=tracking_candidates,
            event_reasons=selected_event_reasons,
            encode_frame=encode_frame_as_data_url,
            request_timeout_seconds=TRACKING_AGENT_REQUEST_TIMEOUT_SECONDS,
        )
    if did_submit:
        with state.lock:
            state.tracking_agent_submissions += 1

    return tracking_agent_observation


def _publish_tracking_outputs(
    *,
    state: SharedDashboardState,
    result,
    now: float,
    tracks,
    tracking_candidates,
    tracking_agent_observation,
    trail_length: int,
    detections_for_file: list[dict],
    udp_socket,
    stable_identity_visualization_enabled: bool,
    display_identity_by_track: dict[str, dict[str, Any]] | None = None,
) -> None:
    if stable_identity_visualization_enabled:
        annotated = result.plot(labels=False)
    else:
        annotated = result.plot()
    _apply_stable_identity_webcam_overlays(
        frame=annotated,
        tracking_candidates=tracking_candidates,
        stable_identity_visualization_enabled=stable_identity_visualization_enabled,
        display_identity_by_track=display_identity_by_track,
        result_names=getattr(result, "names", None),
    )
    commit_live_tracking_frame_state(
        state=state,
        now=now,
        annotated_frame=annotated,
        tracks=tracks,
        tracking_agent_observation=tracking_agent_observation,
        trail_length=trail_length,
        display_identity_by_track=display_identity_by_track,
    )

    publish_tracking_frame_message(
        tracking_frame_path=LATEST_TRACKING_FRAME_PATH,
        timestamp=now,
        detections=detections_for_file,
        udp_socket=udp_socket,
        udp_host=UDP_TRACKING_HOST,
        udp_port=UDP_TRACKING_PORT,
    )


def _build_stable_identity_display_map(
    *,
    registry: PersistentIdentityRegistry | None,
    tracking_candidates,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if registry is None:
        return {}, {
            "enabled": True,
            "available": False,
            "reason": "registry_unavailable",
            "mapped_count": 0,
            "conflict_count": 0,
            "fallback_count": len(tracking_candidates),
        }

    visible_tracks = set()
    for candidate in tracking_candidates:
        track_id = getattr(candidate, "yolo_track_id", None)
        if isinstance(track_id, str) and track_id.strip():
            visible_tracks.add(track_id.strip())

    display_map: dict[str, dict[str, Any]] = {}
    provisional_count = 0
    non_visible_status_count = 0

    for track_id in sorted(visible_tracks):
        app_identity_id = registry.get_identity_for_yolo_track(track_id)
        if not isinstance(app_identity_id, str) or not app_identity_id:
            continue

        record = registry.get_record(app_identity_id)
        if record is None:
            continue
        if record.status != "visible":
            non_visible_status_count += 1
            continue

        provisional = bool(record.provisional)
        if provisional:
            provisional_count += 1
        display_label = f"{app_identity_id} (P)" if provisional else app_identity_id
        display_map[track_id] = {
            "display_label": display_label,
            "app_identity_id": app_identity_id,
            "status": record.status,
            "provisional": provisional,
        }

    owners: dict[str, str] = {}
    conflicting_app_ids: set[str] = set()
    for yolo_track_id, metadata in display_map.items():
        app_identity_id = metadata.get("app_identity_id")
        if not isinstance(app_identity_id, str) or not app_identity_id:
            continue
        prior_owner = owners.get(app_identity_id)
        if prior_owner is None:
            owners[app_identity_id] = yolo_track_id
            continue
        if prior_owner != yolo_track_id:
            conflicting_app_ids.add(app_identity_id)

    if conflicting_app_ids:
        for yolo_track_id in list(display_map.keys()):
            app_identity_id = display_map[yolo_track_id].get("app_identity_id")
            if app_identity_id in conflicting_app_ids:
                display_map.pop(yolo_track_id, None)

    summary = {
        "enabled": True,
        "available": True,
        "mapped_count": len(display_map),
        "visible_track_count": len(visible_tracks),
        "provisional_count": provisional_count,
        "conflict_count": len(conflicting_app_ids),
        "non_visible_status_count": non_visible_status_count,
        "fallback_count": max(0, len(visible_tracks) - len(display_map)),
    }
    return display_map, summary


def _update_stable_identity_display_mapping_if_enabled(
    *,
    state: SharedDashboardState,
    tracking_candidates,
) -> dict[str, dict[str, Any]]:
    with state.lock:
        enabled = bool(state.tracking_stable_identity_visualization_enabled)
        registry_enabled = bool(state.tracking_identity_registry_enabled)
        registry = state.tracking_identity_registry

    if not enabled:
        with state.lock:
            state.latest_tracking_display_identity_by_track = {}
            state.latest_tracking_display_identity_summary = {
                "enabled": False,
                "available": False,
                "reason": "feature_disabled",
            }
        return {}

    if not registry_enabled:
        with state.lock:
            state.latest_tracking_display_identity_by_track = {}
            state.latest_tracking_display_identity_summary = {
                "enabled": True,
                "available": False,
                "reason": "identity_registry_disabled",
            }
        return {}

    display_map, summary = _build_stable_identity_display_map(
        registry=registry,
        tracking_candidates=tracking_candidates,
    )
    with state.lock:
        state.latest_tracking_display_identity_by_track = display_map
        state.latest_tracking_display_identity_summary = summary
    return display_map


def _apply_stable_identity_webcam_overlays(
    *,
    frame,
    tracking_candidates,
    stable_identity_visualization_enabled: bool,
    display_identity_by_track: dict[str, dict[str, Any]] | None,
    result_names: dict[int, str] | None = None,
) -> None:
    if not stable_identity_visualization_enabled:
        return

    annotator = Annotator(frame, line_width=None, font_size=None, pil=False)
    class_index_by_name: dict[str, int] = {}
    if isinstance(result_names, dict):
        for class_index, class_name in result_names.items():
            if isinstance(class_index, int) and isinstance(class_name, str):
                class_index_by_name[class_name] = class_index

    for candidate in tracking_candidates:
        track_id = getattr(candidate, "yolo_track_id", None)
        if not isinstance(track_id, str):
            continue
        metadata = (
            display_identity_by_track.get(track_id)
            if isinstance(display_identity_by_track, dict)
            else None
        )

        display_label: str
        if isinstance(metadata, dict):
            mapped_label = metadata.get("display_label")
            display_label = (
                mapped_label
                if isinstance(mapped_label, str) and mapped_label
                else track_id
            )
        else:
            display_label = track_id

        class_name_raw = getattr(candidate, "label", None)
        class_name = class_name_raw.strip() if isinstance(class_name_raw, str) and class_name_raw.strip() else "object"
        class_index = class_index_by_name.get(class_name, 0)

        confidence_value = getattr(candidate, "detection_confidence", None)
        confidence_text = (
            f"{float(confidence_value):.2f}"
            if isinstance(confidence_value, (int, float))
            else None
        )

        has_mapped_identity = isinstance(metadata, dict)
        identity_label = display_label if has_mapped_identity else f"id:{track_id}"

        label_parts = [class_name, identity_label]
        if isinstance(confidence_text, str):
            label_parts.append(confidence_text)
        rendered_label = " ".join(label_parts)

        pixel_bbox = getattr(candidate, "pixel_bbox", None)
        if pixel_bbox is None:
            continue

        frame_height, frame_width = frame.shape[:2]
        x1 = int(max(0.0, float(pixel_bbox.x1)))
        y1 = int(max(0.0, float(pixel_bbox.y1)))
        x1 = min(max(0, x1), max(0, frame_width - 1))
        y_text = max(2, y1 - 2)
        y_text = min(y_text, max(0, frame_height - 1))

        label_color = colors(class_index, True)
        annotator.text(
            (x1, y_text),
            rendered_label,
            txt_color=(255, 255, 255),
            anchor="bottom",
            box_color=label_color,
        )


def _update_identity_registry_if_enabled(
    *,
    state: SharedDashboardState,
    now: float,
    tracking_candidates,
) -> None:
    with state.lock:
        registry_enabled = bool(state.tracking_identity_registry_enabled)
        registry = state.tracking_identity_registry

    if not registry_enabled or registry is None:
        return

    try:
        summary = registry.process_detections(
            now=now,
            detections=tracking_candidates,
        )
    except Exception:
        summary = {
            "enabled": True,
            "error": "identity_registry_update_failed",
        }

    with state.lock:
        state.latest_tracking_identity_registry_summary = summary
        if isinstance(summary, dict):
            generation = summary.get("generation")
            if isinstance(generation, int):
                state.tracking_identity_registry_generation = generation
            version = summary.get("version")
            if isinstance(version, int):
                state.tracking_identity_registry_version = version


def _build_prediction_detection_advisories(
    *,
    state: SharedDashboardState,
    now: float,
    tracking_candidates,
) -> None:
    with state.lock:
        registry_enabled = bool(state.tracking_identity_registry_enabled)
        registry = state.tracking_identity_registry
        calibration_status = str(state.calibration_status)
        prediction_map = dict(state.latest_tracking_prediction_evidence_by_identity)
        max_advisories = max(1, int(state.tracking_prediction_detection_advisory_max_records))

    if not registry_enabled or registry is None:
        with state.lock:
            state.latest_tracking_prediction_detection_advisories = []
            state.latest_tracking_prediction_detection_advisory_summary = {
                "updated_at": float(now),
                "available": False,
                "reason": "identity_registry_unavailable",
                "advisory_count": 0,
            }
        try:
            write_json_payload(
                LATEST_TRACKING_PREDICTION_DETECTION_ADVISORIES_PATH,
                {
                    "schema_version": "1.0",
                    "updated_at": float(now),
                    "available": False,
                    "summary": {
                        "updated_at": float(now),
                        "available": False,
                        "reason": "identity_registry_unavailable",
                        "advisory_count": 0,
                    },
                    "advisories": [],
                },
            )
        except OSError:
            pass
        return

    snapshot = registry.snapshot()
    snapshot_generation = snapshot.get("generation") if isinstance(snapshot, dict) else None
    snapshot_version = snapshot.get("version") if isinstance(snapshot, dict) else None

    max_effective_horizon_seconds = min(
        10.0,
        max(0.1, float(TRACKING_MAX_EFFECTIVE_HORIZON_SECONDS)),
    )
    pixel_threshold = 120.0
    ground_threshold = 2.5
    pixel_ambiguity = 25.0
    ground_ambiguity = 0.35

    candidate_track_owner: dict[str, str | None] = {}
    detection_candidates: list[tuple[str, Any]] = []
    for candidate in tracking_candidates:
        track_id = _nonempty_str(getattr(candidate, "yolo_track_id", None))
        if track_id is None:
            continue
        detection_candidates.append((track_id, candidate))
        owner = registry.get_identity_for_yolo_track(track_id)
        candidate_track_owner[track_id] = owner if isinstance(owner, str) and owner else None

    links: list[dict[str, Any]] = []
    rejected_count = 0
    stale_or_invalid_count = 0

    for app_identity_id, record in prediction_map.items():
        if not isinstance(app_identity_id, str) or not app_identity_id:
            rejected_count += 1
            continue
        if not isinstance(record, dict):
            rejected_count += 1
            continue

        identity_record = registry.get_record(app_identity_id)
        if identity_record is None or identity_record.status != "temporarily_lost":
            stale_or_invalid_count += 1
            continue

        source_generation = record.get("source_identity_registry_generation")
        source_version = record.get("source_identity_registry_version")
        expires_at = record.get("expires_at")
        predicted_for = record.get("predicted_for_timestamp")
        estimated_at = record.get("source_estimated_at")
        source_track_id = _nonempty_str(record.get("source_yolo_track_id"))
        coordinate_system = _nonempty_str(record.get("coordinate_system"))

        if (
            not isinstance(expires_at, (int, float))
            or not isinstance(predicted_for, (int, float))
            or not isinstance(estimated_at, (int, float))
            or source_track_id is None
            or coordinate_system is None
        ):
            stale_or_invalid_count += 1
            continue
        if float(expires_at) <= float(now):
            stale_or_invalid_count += 1
            continue
        if abs(float(now) - float(predicted_for)) > max_effective_horizon_seconds:
            stale_or_invalid_count += 1
            continue
        if float(predicted_for) < float(estimated_at):
            stale_or_invalid_count += 1
            continue
        if (
            isinstance(source_generation, int)
            and isinstance(snapshot_generation, int)
            and source_generation > snapshot_generation
        ):
            stale_or_invalid_count += 1
            continue
        if (
            isinstance(source_version, int)
            and isinstance(snapshot_version, int)
            and source_version > snapshot_version
        ):
            stale_or_invalid_count += 1
            continue

        expected_ground = record.get("estimated_ground_position")
        expected_pixel = record.get("estimated_pixel_position")
        expected_unit = _nonempty_str(record.get("coordinate_unit"))

        for candidate_track_id, candidate in detection_candidates:
            if candidate_track_id == source_track_id:
                continue

            owner = candidate_track_owner.get(candidate_track_id)
            if owner == app_identity_id:
                continue

            match_metric = None
            match_distance = None
            match_threshold = None

            if coordinate_system == "calibrated_ground_plane":
                if calibration_status != "loaded":
                    continue
                candidate_kind = _nonempty_str(getattr(candidate, "position_kind", None))
                candidate_unit = _nonempty_str(getattr(candidate, "ground_coordinate_unit", None))
                candidate_ground_x = getattr(candidate, "ground_x", None)
                candidate_ground_z = getattr(candidate, "ground_z", None)
                if (
                    candidate_kind != "calibrated_ground_plane"
                    or candidate_unit is None
                    or expected_unit is None
                    or candidate_unit != expected_unit
                    or not isinstance(expected_ground, dict)
                    or not isinstance(candidate_ground_x, (int, float))
                    or not isinstance(candidate_ground_z, (int, float))
                    or not isinstance(expected_ground.get("x"), (int, float))
                    or not isinstance(expected_ground.get("z"), (int, float))
                ):
                    continue
                dx = float(candidate_ground_x) - float(expected_ground.get("x"))
                dz = float(candidate_ground_z) - float(expected_ground.get("z"))
                match_distance = (dx * dx + dz * dz) ** 0.5
                match_threshold = ground_threshold
                match_metric = "ground_distance"
            else:
                candidate_pixel_x = getattr(candidate, "pixel_center_x", None)
                candidate_pixel_y = getattr(candidate, "pixel_center_y", None)
                if (
                    not isinstance(expected_pixel, dict)
                    or not isinstance(candidate_pixel_x, (int, float))
                    or not isinstance(candidate_pixel_y, (int, float))
                    or not isinstance(expected_pixel.get("x"), (int, float))
                    or not isinstance(expected_pixel.get("y"), (int, float))
                ):
                    continue
                dx = float(candidate_pixel_x) - float(expected_pixel.get("x"))
                dy = float(candidate_pixel_y) - float(expected_pixel.get("y"))
                match_distance = (dx * dx + dy * dy) ** 0.5
                match_threshold = pixel_threshold
                match_metric = "pixel_distance"

            if (
                match_metric is None
                or not isinstance(match_distance, float)
                or not isinstance(match_threshold, float)
                or match_distance > match_threshold
            ):
                continue

            normalized = match_distance / match_threshold if match_threshold > 0 else 1.0
            links.append(
                {
                    "app_identity_id": app_identity_id,
                    "candidate_yolo_track_id": candidate_track_id,
                    "source_yolo_track_id": source_track_id,
                    "estimate_id": record.get("estimate_id"),
                    "predicted_for_timestamp": float(predicted_for),
                    "expires_at": float(expires_at),
                    "source_estimated_at": float(estimated_at),
                    "source_identity_registry_generation": source_generation,
                    "source_identity_registry_version": source_version,
                    "identity_status": identity_record.status,
                    "match_metric": match_metric,
                    "match_distance": match_distance,
                    "match_threshold": match_threshold,
                    "normalized_score": normalized,
                    "coordinate_system": coordinate_system,
                    "ground_coordinate_mode": record.get("ground_coordinate_mode"),
                    "coordinate_unit": expected_unit,
                    "classification": "prediction_detection_advisory",
                    "source_tracking_message_ids": record.get("source_tracking_message_ids", []),
                    "source_verification_observation_id": record.get("source_verification_observation_id"),
                    "source_verification_message_id": record.get("source_verification_message_id"),
                }
            )

    links_by_identity: dict[str, list[dict[str, Any]]] = {}
    links_by_track: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_identity.setdefault(link["app_identity_id"], []).append(link)
        links_by_track.setdefault(link["candidate_yolo_track_id"], []).append(link)

    ambiguous_identity_ids: set[str] = set()
    ambiguous_track_ids: set[str] = set()

    for identity_id, grouped in links_by_identity.items():
        grouped.sort(key=lambda item: float(item.get("normalized_score", 1.0)))
        if len(grouped) >= 2:
            first = grouped[0]
            second = grouped[1]
            margin = (
                ground_ambiguity
                if first.get("match_metric") == "ground_distance"
                else pixel_ambiguity
            )
            if (
                isinstance(first.get("match_distance"), (int, float))
                and isinstance(second.get("match_distance"), (int, float))
                and (float(second.get("match_distance")) - float(first.get("match_distance"))) <= margin
            ):
                ambiguous_identity_ids.add(identity_id)

    for track_id, grouped in links_by_track.items():
        grouped.sort(key=lambda item: float(item.get("normalized_score", 1.0)))
        if len(grouped) >= 2:
            first = grouped[0]
            second = grouped[1]
            margin = (
                ground_ambiguity
                if first.get("match_metric") == "ground_distance"
                else pixel_ambiguity
            )
            if (
                isinstance(first.get("match_distance"), (int, float))
                and isinstance(second.get("match_distance"), (int, float))
                and (float(second.get("match_distance")) - float(first.get("match_distance"))) <= margin
            ):
                ambiguous_track_ids.add(track_id)

    matched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    used_identity_ids: set[str] = set()
    used_track_ids: set[str] = set()

    for link in sorted(links, key=lambda item: float(item.get("normalized_score", 1.0))):
        identity_id = link["app_identity_id"]
        track_id = link["candidate_yolo_track_id"]

        if identity_id in ambiguous_identity_ids or track_id in ambiguous_track_ids:
            ambiguous.append(
                {
                    **link,
                    "decision": "ambiguous",
                    "reason": (
                        "identity_has_competing_candidates"
                        if identity_id in ambiguous_identity_ids
                        else "track_has_competing_identities"
                    ),
                    "advisory_created_at": float(now),
                }
            )
            continue

        if identity_id in used_identity_ids or track_id in used_track_ids:
            continue

        used_identity_ids.add(identity_id)
        used_track_ids.add(track_id)
        matched.append(
            {
                **link,
                "decision": "matched",
                "reason": None,
                "advisory_created_at": float(now),
            }
        )

    advisories = [*matched, *ambiguous]
    advisories.sort(
        key=lambda item: (
            0 if item.get("decision") == "matched" else 1,
            float(item.get("normalized_score", 1.0)),
        )
    )
    if len(advisories) > max_advisories:
        advisories = advisories[:max_advisories]

    with state.lock:
        state.latest_tracking_prediction_detection_advisories = advisories
        state.latest_tracking_prediction_detection_advisory_summary = {
            "updated_at": float(now),
            "available": True,
            "candidate_detection_count": len(detection_candidates),
            "eligible_prediction_count": len(prediction_map),
            "link_count": len(links),
            "matched_count": len(matched),
            "ambiguous_count": len(ambiguous),
            "rejected_count": rejected_count,
            "stale_or_invalid_count": stale_or_invalid_count,
            "max_records": max_advisories,
            "identity_keying": "app_identity_id",
        }
    try:
        write_json_payload(
            LATEST_TRACKING_PREDICTION_DETECTION_ADVISORIES_PATH,
            {
                "schema_version": "1.0",
                "updated_at": float(now),
                "available": True,
                "summary": {
                    "updated_at": float(now),
                    "available": True,
                    "candidate_detection_count": len(detection_candidates),
                    "eligible_prediction_count": len(prediction_map),
                    "link_count": len(links),
                    "matched_count": len(matched),
                    "ambiguous_count": len(ambiguous),
                    "rejected_count": rejected_count,
                    "stale_or_invalid_count": stale_or_invalid_count,
                    "max_records": max_advisories,
                    "identity_keying": "app_identity_id",
                },
                "advisories": advisories,
            },
        )
    except OSError:
        pass


def _package_tracking_evidence_if_enabled(
    *,
    state: SharedDashboardState,
    frame,
    now: float,
    tracking_candidates,
    tracking_agent_observation,
) -> None:
    evidence_store = state.tracking_evidence_store
    evidence_packager = state.tracking_evidence_packager
    if evidence_store is None or evidence_packager is None:
        return

    frame_height, frame_width = frame.shape[:2]
    frame_evidence = evidence_store.register_frame(
        frame=frame,
        timestamp=now,
        width=frame_width,
        height=frame_height,
    )

    updates_by_detection: dict[str, dict] = {}
    if tracking_agent_observation is not None:
        updates = tracking_agent_observation.to_dict().get("updates")
        if isinstance(updates, list):
            for item in updates:
                if not isinstance(item, dict):
                    continue
                detection_id = item.get("detection_id")
                if isinstance(detection_id, str):
                    updates_by_detection[detection_id] = item

    for candidate in tracking_candidates:
        update_payload = updates_by_detection.get(candidate.detection_id)
        packaged = evidence_packager.package_detection(
            detection=candidate,
            timestamp=now,
            frame_evidence=frame_evidence,
            update_payload=update_payload,
        )
        if bool(packaged.validation.get("valid", False)):
            with state.lock:
                state.latest_tracking_evidence_message_by_track[packaged.envelope.track_id] = packaged.envelope.message_id


def _consume_identity_reassociation_if_enabled(
    *,
    state: SharedDashboardState,
    now: float,
    tracking_candidates,
) -> None:
    with state.lock:
        consumer = state.tracking_identity_reassociation_consumer
        enabled = bool(state.tracking_identity_reassociation_enabled)
        registry = state.tracking_identity_registry

    if not enabled or consumer is None:
        return

    try:
        summary = consumer.poll_and_apply(
            registry=registry,
            tracking_candidates=tracking_candidates,
            now_timestamp=now,
        )
    except Exception:
        summary = {
            "enabled": True,
            "skipped": True,
            "reason": "reassociation_consumer_failed",
            "error": "reassociation_consumer_failed",
        }

    with state.lock:
        state.latest_tracking_identity_reassociation_summary = summary


def _submit_tracking_decision_request_if_enabled(
    *,
    state: SharedDashboardState,
    tracking_agent_observation,
    now: float,
) -> None:
    with state.lock:
        transport = state.tracking_decision_ipc_transport
        enabled = bool(state.tracking_decision_ipc_enabled)

    if not enabled or transport is None or tracking_agent_observation is None:
        return

    decision_payload = getattr(tracking_agent_observation, "decision", None)
    if not isinstance(decision_payload, dict):
        return

    follow_up_meta = _extract_follow_up_metadata_from_observation(tracking_agent_observation)
    follow_up_investigation_id = _nonempty_str(follow_up_meta.get("investigation_id"))
    is_follow_up_observation = follow_up_investigation_id is not None

    if is_follow_up_observation:
        with state.lock:
            investigation_state = state.tracking_decision_investigations.get(follow_up_investigation_id)

        if not isinstance(investigation_state, dict):
            _persist_terminal_follow_up_decision_event(
                now=now,
                observation=tracking_agent_observation,
                follow_up_meta=follow_up_meta,
                decision_payload=(dict(decision_payload) if isinstance(decision_payload, dict) else None),
                validation_errors=[],
                status="rejected",
                reason="unknown_follow_up_investigation",
                investigation_state=None,
            )
            return

        if bool(investigation_state.get("terminal", False)) and bool(
            investigation_state.get("terminal_follow_up_persisted", False)
        ):
            _persist_terminal_follow_up_decision_event(
                now=now,
                observation=tracking_agent_observation,
                follow_up_meta=follow_up_meta,
                decision_payload=(dict(decision_payload) if isinstance(decision_payload, dict) else None),
                validation_errors=[],
                status="rejected",
                reason="late_follow_up_after_investigation_closed",
                investigation_state=investigation_state,
            )
            return

        expected_request_id = _nonempty_str(investigation_state.get("request_id"))
        expected_result_key = _nonempty_str(investigation_state.get("result_key"))
        follow_up_request_id = _nonempty_str(follow_up_meta.get("request_id"))
        follow_up_result_key = _nonempty_str(follow_up_meta.get("result_key"))
        if (
            (expected_request_id is not None and follow_up_request_id is not None and expected_request_id != follow_up_request_id)
            or (expected_result_key is not None and follow_up_result_key is not None and expected_result_key != follow_up_result_key)
        ):
            _persist_terminal_follow_up_decision_event(
                now=now,
                observation=tracking_agent_observation,
                follow_up_meta=follow_up_meta,
                decision_payload=(dict(decision_payload) if isinstance(decision_payload, dict) else None),
                validation_errors=[],
                status="rejected",
                reason="follow_up_context_mismatch",
                investigation_state=investigation_state,
            )
            return

        terminal_candidate = (
            dict(decision_payload)
            if isinstance(decision_payload, dict)
            else (
                dict(investigation_state.get("final_decision"))
                if isinstance(investigation_state.get("final_decision"), dict)
                else None
            )
        )
        validated_terminal_decision, terminal_decision_errors = _validate_tracking_decision_payload(terminal_candidate)
        terminal_action = (
            _nonempty_str(validated_terminal_decision.get("action"))
            if isinstance(validated_terminal_decision, dict)
            else None
        )
        if terminal_action not in {"inconclusive", "propose_identity_association"}:
            fallback_inconclusive = _build_inconclusive_follow_up_decision(
                investigation_id=follow_up_investigation_id,
                reason=(
                    "follow_up_invalid_terminal_decision"
                    if terminal_decision_errors
                    else "follow_up_invalid_terminal_action"
                ),
                correlation_id=_nonempty_str(investigation_state.get("correlation_id")),
                request_id=follow_up_request_id,
                result_key=follow_up_result_key,
                initial_decision=(
                    dict(investigation_state.get("initial_decision"))
                    if isinstance(investigation_state.get("initial_decision"), dict)
                    else None
                ),
                evidence_references=(
                    dict(investigation_state.get("evidence_references"))
                    if isinstance(investigation_state.get("evidence_references"), dict)
                    else None
                ),
                result_payload=(
                    dict(investigation_state.get("result_payload"))
                    if isinstance(investigation_state.get("result_payload"), dict)
                    else None
                ),
            )
            with state.lock:
                active_state = state.tracking_decision_investigations.get(follow_up_investigation_id)
                if isinstance(active_state, dict):
                    active_state["terminal"] = True
                    active_state["follow_up_completed"] = True
                    active_state["final_decision"] = fallback_inconclusive
                    active_state["updated_at"] = now
                    active_state["terminal_follow_up_persisted"] = True
            _persist_terminal_follow_up_decision_event(
                now=now,
                observation=tracking_agent_observation,
                follow_up_meta=follow_up_meta,
                decision_payload=fallback_inconclusive,
                validation_errors=(
                    terminal_decision_errors
                    if isinstance(terminal_decision_errors, list)
                    else ["follow_up_invalid_terminal_decision"]
                ),
                status="accepted",
                reason="terminal_follow_up_normalized_to_inconclusive",
                investigation_state=investigation_state,
            )
            return

        with state.lock:
            active_state = state.tracking_decision_investigations.get(follow_up_investigation_id)
            if isinstance(active_state, dict):
                active_state["terminal"] = True
                active_state["follow_up_completed"] = True
                active_state["final_decision"] = dict(validated_terminal_decision)
                active_state["updated_at"] = now
                active_state["terminal_follow_up_persisted"] = True
        _persist_terminal_follow_up_decision_event(
            now=now,
            observation=tracking_agent_observation,
            follow_up_meta=follow_up_meta,
            decision_payload=dict(validated_terminal_decision),
            validation_errors=[],
            status="accepted",
            reason="terminal_follow_up_persisted",
            investigation_state=investigation_state,
        )
        return

    if not is_follow_up_observation:
        investigation_id = _nonempty_str(decision_payload.get("investigation_id"))
        with state.lock:
            active_investigation_id, _ = _active_nonterminal_investigation_locked(state)
        if (
            active_investigation_id is not None
            and investigation_id is not None
            and investigation_id != active_investigation_id
        ):
            with state.lock:
                state.latest_tracking_decision_ipc_summary = {
                    "request": {
                        "accepted": False,
                        "duplicate": False,
                        "rejected": True,
                        "reason": "non_selected_investigation_rejected",
                        "selected_investigation_id": active_investigation_id,
                        "rejected_investigation_id": investigation_id,
                    },
                    "results": None,
                    "updated_at": now,
                }
            return

    try:
        summary = transport.dashboard_submit_decision_request(
            decision=decision_payload,
            llm_correlation_id=getattr(tracking_agent_observation, "llm_correlation_id", None),
            now_timestamp=now,
        )
    except Exception:
        summary = {
            "accepted": False,
            "duplicate": False,
            "rejected": True,
            "reason": "decision_ipc_request_submit_failed",
        }

    if bool(summary.get("accepted", False)) or bool(summary.get("duplicate", False)):
        investigation_id = _nonempty_str(decision_payload.get("investigation_id"))
        tool_requests = decision_payload.get("tool_requests")
        tool_request = tool_requests[0] if isinstance(tool_requests, list) and tool_requests else None
        request_id = _nonempty_str(tool_request.get("request_id")) if isinstance(tool_request, dict) else None
        tool_name = _nonempty_str(tool_request.get("tool_name")) if isinstance(tool_request, dict) else None
        if investigation_id is not None and request_id is not None and tool_name is not None and not is_follow_up_observation:
            with state.lock:
                existing = state.tracking_decision_investigations.get(investigation_id)
                investigation_state = existing if isinstance(existing, dict) else {}
                investigation_state.update(
                    {
                        "investigation_id": investigation_id,
                        "request_id": request_id,
                        "tool_name": tool_name,
                        "correlation_id": _nonempty_str(decision_payload.get("correlation_id"))
                        or _nonempty_str(getattr(tracking_agent_observation, "llm_correlation_id", None)),
                        "initial_decision": dict(decision_payload),
                        "evidence_references": (
                            dict(decision_payload.get("evidence_references"))
                            if isinstance(decision_payload.get("evidence_references"), dict)
                            else {}
                        ),
                        "tool_calls": 1,
                        "iterations": 1,
                        "terminal": False,
                        "follow_up_submitted": False,
                        "follow_up_completed": False,
                        "updated_at": now,
                    }
                )
                state.tracking_decision_investigations[investigation_id] = investigation_state

    with state.lock:
        state.latest_tracking_decision_ipc_summary = {
            "request": summary,
            "results": None,
            "updated_at": now,
        }


def _consume_tracking_decision_results_if_enabled(
    *,
    state: SharedDashboardState,
    tracking_runtime: AsyncVisualTrackingRuntime,
    frame,
    tracking_candidates,
    now: float,
) -> None:
    with state.lock:
        transport = state.tracking_decision_ipc_transport
        enabled = bool(state.tracking_decision_ipc_enabled)

    if not enabled or transport is None:
        return

    try:
        results_summary = transport.dashboard_poll_results(now_timestamp=now)
    except Exception:
        results_summary = {
            "available": False,
            "result_count": 0,
            "results": [],
            "error": "decision_ipc_result_poll_failed",
        }

    processed_count = 0
    duplicate_count = 0
    mismatch_count = 0
    follow_up_submitted_count = 0
    failed_count = 0
    terminal_count = 0

    result_items = results_summary.get("results") if isinstance(results_summary, dict) else None
    if not isinstance(result_items, list):
        result_items = []

    for result_event in result_items:
        if not isinstance(result_event, dict):
            continue

        idempotency_key = _nonempty_str(result_event.get("idempotency_key"))
        investigation_id = _nonempty_str(result_event.get("investigation_id"))
        request_id = _nonempty_str(result_event.get("request_id"))
        tool_name = _nonempty_str(result_event.get("tool_name"))
        status = _nonempty_str(result_event.get("status")) or "rejected"
        result_payload = result_event.get("result_payload")
        if not isinstance(result_payload, dict):
            result_payload = {}

        with state.lock:
            if idempotency_key is not None and idempotency_key in state.tracking_decision_consumed_result_keys:
                duplicate_count += 1
                continue

            investigation_state = (
                state.tracking_decision_investigations.get(investigation_id)
                if investigation_id is not None
                else None
            )
            if not isinstance(investigation_state, dict):
                mismatch_count += 1
                if idempotency_key is not None:
                    state.tracking_decision_consumed_result_keys.add(idempotency_key)
                continue

            expected_request_id = _nonempty_str(investigation_state.get("request_id"))
            expected_tool_name = _nonempty_str(investigation_state.get("tool_name"))
            if (
                expected_request_id is None
                or request_id is None
                or expected_request_id != request_id
                or expected_tool_name is None
                or tool_name is None
                or expected_tool_name != tool_name
            ):
                mismatch_count += 1
                if idempotency_key is not None:
                    state.tracking_decision_consumed_result_keys.add(idempotency_key)
                continue

            if idempotency_key is not None:
                state.tracking_decision_consumed_result_keys.add(idempotency_key)

            if bool(investigation_state.get("terminal", False)):
                terminal_count += 1
                continue

            if idempotency_key is not None and idempotency_key in state.tracking_decision_followup_result_keys:
                duplicate_count += 1
                continue

            investigation_state["result_status"] = status
            investigation_state["result_payload"] = dict(result_payload)
            investigation_state["result_key"] = idempotency_key
            investigation_state["updated_at"] = now

        if status in UNSUCCESSFUL_DECISION_RESULT_STATUSES:
            with state.lock:
                active_state = state.tracking_decision_investigations.get(investigation_id)
                if isinstance(active_state, dict):
                    active_state["terminal"] = True
                    active_state["follow_up_completed"] = True
                    active_state["final_decision"] = _build_inconclusive_follow_up_decision(
                        investigation_id=investigation_id,
                        reason=f"tool_result_{status}",
                        correlation_id=_nonempty_str(active_state.get("correlation_id")),
                        request_id=request_id,
                        result_key=idempotency_key,
                        initial_decision=(
                            dict(active_state.get("initial_decision"))
                            if isinstance(active_state.get("initial_decision"), dict)
                            else None
                        ),
                        evidence_references=(
                            dict(active_state.get("evidence_references"))
                            if isinstance(active_state.get("evidence_references"), dict)
                            else None
                        ),
                        result_payload=dict(result_payload),
                    )
                    active_state["updated_at"] = now
            failed_count += 1
            processed_count += 1
            continue

        with state.lock:
            active_state = state.tracking_decision_investigations.get(investigation_id)
            if not isinstance(active_state, dict):
                mismatch_count += 1
                continue
            if bool(active_state.get("follow_up_submitted", False)):
                duplicate_count += 1
                continue

            already_used_tool_calls = int(active_state.get("tool_calls", 0))
            already_used_iterations = int(active_state.get("iterations", 1))
            if already_used_tool_calls > 1 or already_used_iterations >= 2:
                active_state["terminal"] = True
                active_state["follow_up_completed"] = True
                active_state["final_decision"] = _build_inconclusive_follow_up_decision(
                    investigation_id=investigation_id,
                    reason="follow_up_iteration_or_tool_limit_reached",
                    correlation_id=_nonempty_str(active_state.get("correlation_id")),
                    request_id=request_id,
                    result_key=idempotency_key,
                    initial_decision=(
                        dict(active_state.get("initial_decision"))
                        if isinstance(active_state.get("initial_decision"), dict)
                        else None
                    ),
                    evidence_references=(
                        dict(active_state.get("evidence_references"))
                        if isinstance(active_state.get("evidence_references"), dict)
                        else None
                    ),
                    result_payload=dict(result_payload),
                )
                active_state["updated_at"] = now
                failed_count += 1
                processed_count += 1
                continue

            constraints_ok, constraint_reason = _snapshot_constraints_satisfied(
                tool_name=tool_name or "",
                result_payload=result_payload,
                initial_decision=(
                    dict(active_state.get("initial_decision"))
                    if isinstance(active_state.get("initial_decision"), dict)
                    else None
                ),
            )
            if not constraints_ok:
                active_state["terminal"] = True
                active_state["follow_up_completed"] = True
                active_state["final_decision"] = _build_inconclusive_follow_up_decision(
                    investigation_id=investigation_id,
                    reason=constraint_reason or "snapshot_constraints_not_satisfied",
                    correlation_id=_nonempty_str(active_state.get("correlation_id")),
                    request_id=request_id,
                    result_key=idempotency_key,
                    initial_decision=(
                        dict(active_state.get("initial_decision"))
                        if isinstance(active_state.get("initial_decision"), dict)
                        else None
                    ),
                    evidence_references=(
                        dict(active_state.get("evidence_references"))
                        if isinstance(active_state.get("evidence_references"), dict)
                        else None
                    ),
                    result_payload=dict(result_payload),
                )
                active_state["updated_at"] = now
                failed_count += 1
                processed_count += 1
                continue

            follow_up_context = {
                "investigation_id": investigation_id,
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "result_status": status,
                "tool_name": tool_name,
                "correlation_id": _nonempty_str(active_state.get("correlation_id")),
                "initial_decision": (
                    dict(active_state.get("initial_decision"))
                    if isinstance(active_state.get("initial_decision"), dict)
                    else {}
                ),
                "evidence_references": (
                    dict(active_state.get("evidence_references"))
                    if isinstance(active_state.get("evidence_references"), dict)
                    else {}
                ),
                "mcp_result": dict(result_payload),
            }

        did_submit = submit_tracking_runtime_candidates(
            runtime=tracking_runtime,
            now=now,
            frame=frame,
            detections=tracking_candidates,
            event_reasons=["tracking_decision_follow_up"],
            follow_up_context=follow_up_context,
            encode_frame=encode_frame_as_data_url,
            request_timeout_seconds=TRACKING_AGENT_REQUEST_TIMEOUT_SECONDS,
        )

        with state.lock:
            active_state = state.tracking_decision_investigations.get(investigation_id)
            if isinstance(active_state, dict):
                if did_submit:
                    active_state["follow_up_submitted"] = True
                    active_state["iterations"] = int(active_state.get("iterations", 1)) + 1
                    active_state["updated_at"] = now
                else:
                    active_state["follow_up_submit_pending"] = True
                    active_state["updated_at"] = now

        if did_submit:
            if idempotency_key is not None:
                with state.lock:
                    state.tracking_decision_followup_result_keys.add(idempotency_key)
            follow_up_submitted_count += 1
        processed_count += 1

    with state.lock:
        prior = state.latest_tracking_decision_ipc_summary
        prior_request = prior.get("request") if isinstance(prior, dict) else None
        state.latest_tracking_decision_ipc_summary = {
            "request": prior_request,
            "results": {
                **(results_summary if isinstance(results_summary, dict) else {}),
                "processed_count": processed_count,
                "duplicate_count": duplicate_count,
                "mismatch_count": mismatch_count,
                "failed_count": failed_count,
                "terminal_count": terminal_count,
                "follow_up_submitted_count": follow_up_submitted_count,
            },
            "updated_at": now,
        }


def _consume_tracking_observation_feedback(
    *,
    state: SharedDashboardState,
    now: float,
    tracks: dict,
) -> None:
    with state.lock:
        consumer = state.tracking_feedback_consumer
        current_track_ids = set(tracks.keys())
        bounded_grace_seconds = min(
            10.0,
            max(0.1, float(getattr(state, "tracking_missing_track_grace_seconds", 1.5))),
        )
        state.tracking_missing_track_grace_seconds = bounded_grace_seconds

        lost_message_by_track = state.latest_tracking_lost_evidence_message_by_track
        lost_expires_by_track = state.latest_tracking_lost_track_expires_at_by_track

        for track_id in list(lost_expires_by_track.keys()):
            expires_at = lost_expires_by_track.get(track_id)
            if track_id in current_track_ids or not isinstance(expires_at, (int, float)) or float(expires_at) <= float(now):
                lost_expires_by_track.pop(track_id, None)
                lost_message_by_track.pop(track_id, None)

        active_message_by_track = state.latest_tracking_evidence_message_by_track
        for track_id in list(active_message_by_track.keys()):
            if track_id in current_track_ids:
                lost_expires_by_track.pop(track_id, None)
                lost_message_by_track.pop(track_id, None)
                continue
            message_id = active_message_by_track.pop(track_id, None)
            if not isinstance(message_id, str) or not message_id:
                continue
            lost_message_by_track[track_id] = message_id
            lost_expires_by_track[track_id] = float(now) + bounded_grace_seconds

        temporarily_lost_track_ids = {
            track_id
            for track_id, expires_at in lost_expires_by_track.items()
            if isinstance(expires_at, (int, float)) and float(expires_at) > float(now)
        }

        latest_message_id_by_track = dict(active_message_by_track)
        for track_id in temporarily_lost_track_ids:
            message_id = lost_message_by_track.get(track_id)
            if isinstance(message_id, str) and message_id:
                latest_message_id_by_track[track_id] = message_id

    if consumer is None:
        return

    try:
        feedback_summary = consumer.poll(
            current_track_ids=current_track_ids,
            temporarily_lost_track_ids=temporarily_lost_track_ids,
            latest_message_id_by_track=latest_message_id_by_track,
            now_timestamp=now,
        )
    except Exception:
        feedback_summary = {
            "accepted": [],
            "accepted_count": 0,
            "rejected_count": 0,
            "error": "feedback_poll_failed",
        }

    _append_tracking_feedback_consumer_decisions(
        feedback_summary=feedback_summary,
        now=now,
    )

    with state.lock:
        state.latest_tracking_observation_feedback = feedback_summary


def _generate_tracking_estimate_proposals(
    *,
    state: SharedDashboardState,
    now: float,
    tracks: dict,
) -> None:
    with state.lock:
        estimate_engine = state.tracking_estimate_engine
        feedback_summary = state.latest_tracking_observation_feedback
        tracks_snapshot = dict(tracks)

    if estimate_engine is None:
        return

    accepted_feedback = []
    if isinstance(feedback_summary, dict):
        accepted_payload = feedback_summary.get("accepted")
        if isinstance(accepted_payload, list):
            accepted_feedback = accepted_payload

    try:
        proposal_summary = estimate_engine.generate(
            accepted_feedback_records=accepted_feedback,
            current_tracks=tracks_snapshot,
            now_timestamp=now,
        )
    except Exception:
        proposal_summary = {
            "accepted": [],
            "accepted_count": 0,
            "rejected": [],
            "rejected_count": 0,
            "generated_at": now,
            "error": "estimate_generation_failed",
        }

    with state.lock:
        state.latest_tracking_estimate_proposals = proposal_summary


def _track_prediction_ordering_key(record: dict[str, Any]) -> tuple[float, float, float]:
    predicted_for = record.get("predicted_for_timestamp")
    estimated_at = record.get("source_estimated_at")
    promoted_at = record.get("promoted_at")
    predicted_for_value = float(predicted_for) if isinstance(predicted_for, (int, float)) else 0.0
    estimated_at_value = float(estimated_at) if isinstance(estimated_at, (int, float)) else 0.0
    promoted_at_value = float(promoted_at) if isinstance(promoted_at, (int, float)) else 0.0
    return (predicted_for_value, estimated_at_value, promoted_at_value)


def _is_valid_estimate_position_payload(proposal: dict[str, Any]) -> bool:
    estimated_pixel = proposal.get("estimated_pixel_center")
    if isinstance(estimated_pixel, dict):
        pixel_x = estimated_pixel.get("x")
        pixel_y = estimated_pixel.get("y")
        if isinstance(pixel_x, (int, float)) and isinstance(pixel_y, (int, float)):
            return True

    estimated_ground = proposal.get("estimated_ground_position")
    if isinstance(estimated_ground, dict):
        ground_x = estimated_ground.get("x")
        ground_z = estimated_ground.get("z")
        ground_unit = estimated_ground.get("unit")
        if (
            isinstance(ground_x, (int, float))
            and isinstance(ground_z, (int, float))
            and isinstance(ground_unit, str)
            and ground_unit.strip()
        ):
            return True

    return False


def _build_feedback_index_by_observation_id(feedback_summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(feedback_summary, dict):
        return {}
    accepted = feedback_summary.get("accepted")
    if not isinstance(accepted, list):
        return {}

    index: dict[str, dict[str, Any]] = {}
    for item in accepted:
        if not isinstance(item, dict):
            continue
        observation_id = item.get("verification_observation_id")
        if not isinstance(observation_id, str) or not observation_id.strip():
            continue
        index[observation_id] = item
    return index


def _promote_and_prune_tracking_prediction_evidence(
    *,
    state: SharedDashboardState,
    now: float,
    tracks: dict[str, Any],
) -> None:
    with state.lock:
        proposal_summary = state.latest_tracking_estimate_proposals
        feedback_summary = state.latest_tracking_observation_feedback
        registry_enabled = bool(state.tracking_identity_registry_enabled)
        registry = state.tracking_identity_registry
        registry_generation = state.tracking_identity_registry_generation
        registry_version = state.tracking_identity_registry_version
        calibration_status = str(state.calibration_status)
        mapping_mode = str(state.mapping_mode)

        prediction_map = dict(state.latest_tracking_prediction_evidence_by_identity)
        unresolved_records = list(state.latest_tracking_prediction_evidence_unresolved)
        last_calibration_status = state.tracking_prediction_last_calibration_status
        last_mapping_mode = state.tracking_prediction_last_mapping_mode
        max_records = max(1, int(state.tracking_prediction_evidence_max_records))
        max_unresolved = max(1, int(state.tracking_prediction_unresolved_max_records))

    invalidated_by_calibration_change = 0
    if (
        last_calibration_status is not None
        and last_mapping_mode is not None
        and (last_calibration_status != calibration_status or last_mapping_mode != mapping_mode)
    ):
        invalidated_by_calibration_change = len(prediction_map) + len(unresolved_records)
        prediction_map = {}
        unresolved_records = []

    pruned_expired = 0
    for identity_id in list(prediction_map.keys()):
        record = prediction_map.get(identity_id)
        if not isinstance(record, dict):
            prediction_map.pop(identity_id, None)
            pruned_expired += 1
            continue
        expires_at = record.get("expires_at")
        evidence_expires_at = record.get("source_feedback_expires_at")
        if (
            not isinstance(expires_at, (int, float))
            or float(expires_at) <= float(now)
            or (isinstance(evidence_expires_at, (int, float)) and float(evidence_expires_at) <= float(now))
        ):
            prediction_map.pop(identity_id, None)
            pruned_expired += 1

    pruned_unresolved_expired = 0
    bounded_unresolved: list[dict[str, Any]] = []
    for record in unresolved_records:
        if not isinstance(record, dict):
            pruned_unresolved_expired += 1
            continue
        expires_at = record.get("expires_at")
        evidence_expires_at = record.get("source_feedback_expires_at")
        if (
            not isinstance(expires_at, (int, float))
            or float(expires_at) <= float(now)
            or (isinstance(evidence_expires_at, (int, float)) and float(evidence_expires_at) <= float(now))
        ):
            pruned_unresolved_expired += 1
            continue
        bounded_unresolved.append(record)
    unresolved_records = bounded_unresolved

    observed_track_ids = {
        track_id
        for track_id in tracks.keys()
        if isinstance(track_id, str) and track_id.strip()
    }

    invalidated_on_reappearance = 0
    if observed_track_ids:
        for identity_id in list(prediction_map.keys()):
            record = prediction_map.get(identity_id)
            if not isinstance(record, dict):
                continue
            source_track_id = record.get("source_yolo_track_id")
            if isinstance(source_track_id, str) and source_track_id in observed_track_ids:
                prediction_map.pop(identity_id, None)
                invalidated_on_reappearance += 1

        unresolved_remaining: list[dict[str, Any]] = []
        for record in unresolved_records:
            source_track_id = record.get("source_yolo_track_id") if isinstance(record, dict) else None
            if isinstance(source_track_id, str) and source_track_id in observed_track_ids:
                invalidated_on_reappearance += 1
                continue
            unresolved_remaining.append(record)
        unresolved_records = unresolved_remaining

    invalidated_identity_state = 0
    if registry_enabled and registry is not None:
        observed_identity_ids: set[str] = set()
        for track_id in observed_track_ids:
            app_identity_id = registry.get_identity_for_yolo_track(track_id)
            if isinstance(app_identity_id, str) and app_identity_id:
                observed_identity_ids.add(app_identity_id)

        for identity_id in list(prediction_map.keys()):
            record = prediction_map.get(identity_id)
            if not isinstance(record, dict):
                continue

            if identity_id in observed_identity_ids:
                prediction_map.pop(identity_id, None)
                invalidated_identity_state += 1
                continue

            record_snapshot = registry.get_record(identity_id)
            if record_snapshot is None:
                prediction_map.pop(identity_id, None)
                invalidated_identity_state += 1
                continue

            if record_snapshot.status == "retired":
                prediction_map.pop(identity_id, None)
                invalidated_identity_state += 1
                continue

            source_track_id = record.get("source_yolo_track_id")
            if isinstance(source_track_id, str) and source_track_id:
                if record_snapshot.yolo_track_id != source_track_id:
                    prediction_map.pop(identity_id, None)
                    invalidated_identity_state += 1
                    continue

    feedback_by_observation_id = _build_feedback_index_by_observation_id(feedback_summary)
    accepted_items = proposal_summary.get("accepted") if isinstance(proposal_summary, dict) else None
    promoted_count = 0
    unresolved_promoted = 0
    rejected_malformed = 0
    rejected_stale_or_invalid = 0
    rejected_out_of_order = 0

    if isinstance(accepted_items, list):
        for item in accepted_items:
            if not isinstance(item, dict):
                rejected_malformed += 1
                continue

            proposal = item.get("proposal")
            validation = item.get("validation")
            if not isinstance(proposal, dict) or not isinstance(validation, dict):
                rejected_malformed += 1
                continue

            if not bool(validation.get("valid", False)):
                rejected_stale_or_invalid += 1
                continue

            track_id = proposal.get("track_id")
            estimate_id = proposal.get("estimate_id")
            expires_at = proposal.get("expires_at")
            predicted_for = proposal.get("predicted_for_timestamp")
            estimated_at = proposal.get("estimated_at")
            coordinate_system = proposal.get("coordinate_system")
            ground_coordinate_mode = proposal.get("ground_coordinate_mode")

            if (
                not isinstance(track_id, str)
                or not track_id
                or not isinstance(estimate_id, str)
                or not estimate_id
                or not isinstance(expires_at, (int, float))
                or float(expires_at) <= float(now)
                or not isinstance(predicted_for, (int, float))
                or not isinstance(estimated_at, (int, float))
                or not isinstance(coordinate_system, str)
                or not coordinate_system
                or not isinstance(ground_coordinate_mode, str)
                or not ground_coordinate_mode
                or not _is_valid_estimate_position_payload(proposal)
            ):
                rejected_stale_or_invalid += 1
                continue

            feedback_observation_id = proposal.get("source_verification_observation_id")
            feedback_record = (
                feedback_by_observation_id.get(feedback_observation_id)
                if isinstance(feedback_observation_id, str)
                else None
            )

            evidence_expires_at = None
            verification_timestamp = None
            visibility_status = None
            uncertainty = proposal.get("uncertainty")
            if isinstance(feedback_record, dict):
                feedback_expires = feedback_record.get("expires_at")
                if isinstance(feedback_expires, (int, float)):
                    evidence_expires_at = float(feedback_expires)
                feedback_verification_timestamp = feedback_record.get("verification_timestamp")
                if isinstance(feedback_verification_timestamp, (int, float)):
                    verification_timestamp = float(feedback_verification_timestamp)
                details = feedback_record.get("details")
                if isinstance(details, dict):
                    status_value = details.get("visibility_status")
                    if isinstance(status_value, str) and status_value:
                        visibility_status = status_value
                    feedback_uncertainty = details.get("uncertainty")
                    if isinstance(feedback_uncertainty, list):
                        uncertainty = [str(value) for value in feedback_uncertainty if isinstance(value, str)]

            if isinstance(evidence_expires_at, float) and evidence_expires_at <= float(now):
                rejected_stale_or_invalid += 1
                continue

            if visibility_status in {"occluded", "partially_occluded", "not_visible"}:
                classification = "occlusion_candidate"
            else:
                classification = "predicted_estimate"

            record_payload = {
                "app_identity_id": None,
                "source_yolo_track_id": track_id,
                "estimate_id": estimate_id,
                "source_generated_at": proposal_summary.get("generated_at") if isinstance(proposal_summary, dict) else float(now),
                "source_identity_registry_generation": (
                    int(registry_generation)
                    if isinstance(registry_generation, int)
                    else None
                ),
                "source_identity_registry_version": (
                    int(registry_version)
                    if isinstance(registry_version, int)
                    else None
                ),
                "source_estimated_at": float(estimated_at),
                "source_verification_timestamp": verification_timestamp,
                "predicted_for_timestamp": float(predicted_for),
                "expires_at": float(expires_at),
                "coordinate_system": coordinate_system,
                "ground_coordinate_mode": ground_coordinate_mode,
                "coordinate_unit": (
                    proposal.get("estimated_ground_position", {}).get("unit")
                    if isinstance(proposal.get("estimated_ground_position"), dict)
                    else None
                ),
                "estimated_pixel_position": (
                    proposal.get("estimated_pixel_center")
                    if isinstance(proposal.get("estimated_pixel_center"), dict)
                    else None
                ),
                "estimated_ground_position": (
                    proposal.get("estimated_ground_position")
                    if isinstance(proposal.get("estimated_ground_position"), dict)
                    else None
                ),
                "uncertainty": uncertainty if isinstance(uncertainty, list) else [],
                "classification": classification,
                "source_tracking_message_ids": (
                    proposal.get("source_tracking_message_ids")
                    if isinstance(proposal.get("source_tracking_message_ids"), list)
                    else []
                ),
                "source_verification_observation_id": proposal.get("source_verification_observation_id"),
                "source_verification_message_id": proposal.get("source_verification_message_id"),
                "validation": {
                    "valid": True,
                    "freshness_status": validation.get("freshness_status"),
                    "plausibility_status": validation.get("plausibility_status"),
                    "warnings": validation.get("warnings", []),
                },
                "source_feedback_expires_at": evidence_expires_at,
                "promoted_at": float(now),
            }

            app_identity_id = None
            if registry_enabled and registry is not None:
                resolved_identity_id = registry.get_identity_for_yolo_track(track_id)
                if isinstance(resolved_identity_id, str) and resolved_identity_id:
                    resolved_record = registry.get_record(resolved_identity_id)
                    if resolved_record is not None and resolved_record.status != "retired":
                        app_identity_id = resolved_identity_id

            if app_identity_id is None:
                unresolved_record = dict(record_payload)
                unresolved_record["diagnostic_reason"] = (
                    "identity_registry_unavailable"
                    if not registry_enabled or registry is None
                    else "identity_resolution_unavailable"
                )
                unresolved_records.append(unresolved_record)
                unresolved_promoted += 1
                continue

            record_payload["app_identity_id"] = app_identity_id
            existing = prediction_map.get(app_identity_id)
            if isinstance(existing, dict):
                if _track_prediction_ordering_key(record_payload) <= _track_prediction_ordering_key(existing):
                    rejected_out_of_order += 1
                    continue

            prediction_map[app_identity_id] = record_payload
            promoted_count += 1

    if len(prediction_map) > max_records:
        sorted_items = sorted(
            prediction_map.items(),
            key=lambda item: _track_prediction_ordering_key(item[1]),
        )
        overflow = len(sorted_items) - max_records
        for identity_id, _ in sorted_items[:overflow]:
            prediction_map.pop(identity_id, None)

    if len(unresolved_records) > max_unresolved:
        unresolved_records = unresolved_records[-max_unresolved:]

    with state.lock:
        state.latest_tracking_prediction_evidence_by_identity = prediction_map
        state.latest_tracking_prediction_evidence_unresolved = unresolved_records
        state.latest_tracking_prediction_evidence_summary = {
            "updated_at": float(now),
            "store_count": len(prediction_map),
            "unresolved_count": len(unresolved_records),
            "promoted_count": promoted_count,
            "unresolved_promoted_count": unresolved_promoted,
            "rejected_malformed_count": rejected_malformed,
            "rejected_stale_or_invalid_count": rejected_stale_or_invalid,
            "rejected_out_of_order_count": rejected_out_of_order,
            "pruned_expired_count": pruned_expired,
            "pruned_unresolved_expired_count": pruned_unresolved_expired,
            "invalidated_on_reappearance_count": invalidated_on_reappearance,
            "invalidated_identity_state_count": invalidated_identity_state,
            "invalidated_calibration_change_count": invalidated_by_calibration_change,
            "max_records": max_records,
            "max_unresolved_records": max_unresolved,
            "identity_keying": "app_identity_id",
        }
        state.tracking_prediction_last_calibration_status = calibration_status
        state.tracking_prediction_last_mapping_mode = mapping_mode


def _build_tracking_estimate_diagnostic_payload(
    *,
    proposal_summary: dict[str, Any] | None,
    now: float,
    max_current_valid_estimates: int = 32,
) -> dict[str, Any]:
    current_valid_estimates: list[dict[str, Any]] = []
    rejected_or_invalid_count = 0
    expired_count = 0

    accepted_payload = None
    if isinstance(proposal_summary, dict):
        accepted_payload = proposal_summary.get("accepted")

    accepted_items = accepted_payload if isinstance(accepted_payload, list) else []
    for item in accepted_items:
        if not isinstance(item, dict):
            rejected_or_invalid_count += 1
            continue

        proposal = item.get("proposal")
        validation = item.get("validation")
        if not isinstance(proposal, dict) or not isinstance(validation, dict):
            rejected_or_invalid_count += 1
            continue

        if not bool(validation.get("valid", False)):
            rejected_or_invalid_count += 1
            continue

        expires_at = proposal.get("expires_at")
        if not isinstance(expires_at, (int, float)) or float(expires_at) <= float(now):
            expired_count += 1
            continue

        estimated_pixel_position = proposal.get("estimated_pixel_center")
        if not isinstance(estimated_pixel_position, dict):
            estimated_pixel_position = None

        estimated_ground_position = proposal.get("estimated_ground_position")
        if (
            proposal.get("coordinate_system") != "calibrated_ground_plane"
            or not isinstance(estimated_ground_position, dict)
        ):
            estimated_ground_position = None

        source_tracking_message_ids = proposal.get("source_tracking_message_ids")
        if not isinstance(source_tracking_message_ids, list):
            source_tracking_message_ids = []

        current_valid_estimates.append(
            {
                "track_id": proposal.get("track_id"),
                "estimate_id": proposal.get("estimate_id"),
                "estimated_pixel_position": estimated_pixel_position,
                "estimated_ground_position": estimated_ground_position,
                "estimated_at": proposal.get("estimated_at"),
                "predicted_for_timestamp": proposal.get("predicted_for_timestamp"),
                "expires_at": proposal.get("expires_at"),
                "source_tracking_evidence_ids": source_tracking_message_ids,
                "source_verification_observation_id": proposal.get("source_verification_observation_id"),
                "source_verification_message_id": proposal.get("source_verification_message_id"),
                "coordinator_validation": {
                    "valid": True,
                    "freshness_status": validation.get("freshness_status"),
                    "plausibility_status": validation.get("plausibility_status"),
                    "errors": validation.get("errors", []),
                    "warnings": validation.get("warnings", []),
                },
                "classification": "predicted_estimate",
            }
        )

    bounded_estimates = current_valid_estimates[-max(1, int(max_current_valid_estimates)) :]
    return {
        "schema_version": "1.0",
        "generated_at": float(now),
        "classification": "predicted_estimate_diagnostics",
        "current_valid_estimates": bounded_estimates,
        "current_valid_count": len(bounded_estimates),
        "expired_count": expired_count,
        "rejected_or_invalid_count": rejected_or_invalid_count,
    }


def _publish_tracking_estimate_diagnostics(
    *,
    state: SharedDashboardState,
    now: float,
    diagnostics_path: Path = LATEST_TRACKING_ESTIMATE_DIAGNOSTICS_PATH,
) -> None:
    with state.lock:
        proposal_summary = state.latest_tracking_estimate_proposals

    payload = _build_tracking_estimate_diagnostic_payload(
        proposal_summary=proposal_summary,
        now=now,
    )
    write_json_payload(diagnostics_path, payload)


def _resolve_tracking_feedback_run_id() -> str | None:
    value = os.getenv("PHASE16E_RUN_ID", "").strip()
    return value if value else None


def _append_tracking_feedback_consumer_decisions(
    *,
    feedback_summary: dict[str, Any],
    now: float,
    diagnostics_path: Path = TRACKING_FEEDBACK_CONSUMER_DECISIONS_PATH,
) -> None:
    evaluated_payload = feedback_summary.get("evaluated_records") if isinstance(feedback_summary, dict) else None
    evaluated_records = [item for item in evaluated_payload if isinstance(item, dict)] if isinstance(evaluated_payload, list) else []

    outcome_payload = feedback_summary.get("consumer_outcome") if isinstance(feedback_summary, dict) else None
    consumer_outcome = outcome_payload if isinstance(outcome_payload, dict) else {}
    poll_error = feedback_summary.get("error") if isinstance(feedback_summary, dict) else None
    run_id = _resolve_tracking_feedback_run_id()

    rows: list[dict[str, Any]] = []
    if evaluated_records:
        for item in evaluated_records:
            rows.append(
                {
                    "schema_version": "1.0",
                    "logged_at": float(now),
                    "run_id": run_id,
                    "feedback_id": item.get("feedback_id"),
                    "verification_request_id": item.get("verification_request_id"),
                    "track_id": item.get("track_id"),
                    "evidence_message_id": item.get("evidence_message_id"),
                    "verification_observation_id": item.get("verification_observation_id"),
                    "decision": item.get("decision"),
                    "reason": item.get("reason"),
                    "consumer_outcome": consumer_outcome,
                    "poll_error": poll_error,
                }
            )
    elif isinstance(poll_error, str) and poll_error:
        rows.append(
            {
                "schema_version": "1.0",
                "logged_at": float(now),
                "run_id": run_id,
                "feedback_id": None,
                "verification_request_id": None,
                "track_id": None,
                "evidence_message_id": None,
                "verification_observation_id": None,
                "decision": "error",
                "reason": poll_error,
                "consumer_outcome": consumer_outcome,
                "poll_error": poll_error,
            }
        )

    if not rows:
        return

    try:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        return


def _initialize_tracking_runtime(
    state: SharedDashboardState,
    video_path: str | None = None,
) -> tuple[YOLO, object, GroundPlaneCalibration | None, socket.socket, AsyncVisualTrackingRuntime] | None:
    configured_video_path = video_path if video_path is not None else state.video_path

    source: str | int = (
        str(Path(configured_video_path).expanduser())
        if configured_video_path is not None
        else 0
    )

    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        camera.release()
        with state.lock:
            state.video_path = configured_video_path
            state.camera_status = "unavailable"
            state.detector_status = "idle"
            input_name = (
                f"video file '{source}'"
                if configured_video_path is not None
                else "webcam (camera index 0)"
            )
            state.input_error = f"Could not open {input_name}."
        return None

    model_path = resolve_model_path()
    try:
        model = YOLO(model_path)
    except Exception as error:
        camera.release()
        with state.lock:
            state.video_path = configured_video_path
            state.detector_status = "error"
            state.camera_status = "error"
            state.input_error = f"{type(error).__name__}: {error}"
        return None

    with state.lock:
        state.video_path = configured_video_path
        state.camera_status = "connected"
        state.detector_status = "running"
        state.input_error = None

    calibration: GroundPlaneCalibration | None = None
    try:
        calibration = load_ground_plane_calibration(GROUND_CALIBRATION_PATH)
        calibration_status = "loaded" if calibration is not None else "missing"
    except Exception:
        calibration = None
        calibration_status = "invalid"

    with state.lock:
        state.calibration_status = calibration_status

    udp_socket: socket.socket | None = None
    tracking_runtime: AsyncVisualTrackingRuntime | None = None
    evidence_store: TrackingEvidenceStore | None = None
    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tracking_coordinator = AgentCoordinator()
        tracking_agent = LLMVisualTrackingAgent()
        tracking_runtime = AsyncVisualTrackingRuntime(
            coordinator=tracking_coordinator,
            agent=tracking_agent,
            min_invoke_interval_seconds=0.75,
        )
        evidence_store = TrackingEvidenceStore(
            root_dir=CAPTURES_DIR / "tracking_evidence",
            max_age_seconds=60.0,
            max_frame_records=240,
            max_detection_records=6000,
            max_envelope_records=12000,
        )
        evidence_packager = TrackingEvidencePackager(
            coordinator=tracking_coordinator,
            evidence_store=evidence_store,
            max_history_per_track=20,
        )
        feedback_consumer = TrackingObservationFeedbackConsumer(
            artifact_path=LATEST_OPERATOR_TRACKING_FEEDBACK_PATH,
            max_seen_feedback_ids=1024,
            max_artifact_bytes=512_000,
        )
        estimate_engine = TrackingEstimateProposalEngine(
            coordinator=tracking_coordinator,
            evidence_root_dir=CAPTURES_DIR / "tracking_evidence",
            prediction_horizon_seconds=0.5,
            proposal_ttl_seconds=1.0,
            max_feedback_items=32,
            max_history_points=12,
            max_verification_age_seconds=5.0,
            max_evidence_age_seconds=5.0,
            max_effective_horizon_seconds=min(
                10.0,
                max(0.1, TRACKING_MAX_EFFECTIVE_HORIZON_SECONDS),
            ),
        )
        identity_registry = None
        if IDENTITY_REGISTRY_ENABLED:
            identity_registry = PersistentIdentityRegistry(
                max_history_per_identity=IDENTITY_REGISTRY_HISTORY_LIMIT,
                temporary_loss_seconds=IDENTITY_REGISTRY_TEMPORARY_LOSS_SECONDS,
                retirement_seconds=IDENTITY_REGISTRY_RETIREMENT_SECONDS,
                max_retired_records=IDENTITY_REGISTRY_MAX_RETIRED,
            )
        identity_reassociation_consumer = IdentityReassociationIntentConsumer(
            intents_path=LATEST_TRACKING_IDENTITY_REASSOCIATION_INTENTS_PATH,
            ack_ledger_path=LATEST_TRACKING_IDENTITY_REASSOCIATION_ACK_LEDGER_PATH,
            evidence_root_dir=CAPTURES_DIR / "tracking_evidence",
            enabled=TRACKING_IDENTITY_REASSOCIATION_ENABLED,
            max_intents_per_poll=TRACKING_IDENTITY_REASSOCIATION_MAX_INTENTS_PER_POLL,
            max_ack_entries=TRACKING_IDENTITY_REASSOCIATION_MAX_ACK_ENTRIES,
            min_poll_interval_seconds=TRACKING_IDENTITY_REASSOCIATION_POLL_INTERVAL_SECONDS,
        )
        decision_transport = TrackingDecisionIpcTransport(
            requests_path=TRACKING_DECISION_TOOL_REQUESTS_PATH,
            results_path=TRACKING_DECISION_TOOL_RESULTS_PATH,
            request_index_path=LATEST_TRACKING_DECISION_REQUEST_INDEX_PATH,
            result_index_path=LATEST_TRACKING_DECISION_RESULT_INDEX_PATH,
            max_dashboard_results_per_poll=TRACKING_DECISION_IPC_MAX_RESULTS_PER_POLL,
            writer_role="dashboard",
        )
        with state.lock:
            state.tracking_agent_model = tracking_agent.model
            state.tracking_evidence_store = evidence_store
            state.tracking_evidence_packager = evidence_packager
            state.tracking_feedback_consumer = feedback_consumer
            state.tracking_estimate_engine = estimate_engine
            state.tracking_identity_registry_enabled = IDENTITY_REGISTRY_ENABLED
            state.tracking_identity_registry = identity_registry
            state.tracking_identity_registry_generation = 0
            state.tracking_identity_registry_version = 0
            state.latest_tracking_identity_registry_summary = None
            state.tracking_identity_snapshot_bridge_enabled = IDENTITY_SNAPSHOT_BRIDGE_ENABLED
            state.latest_tracking_identity_snapshot_metadata = None
            state.latest_tracking_identity_snapshot_source_version = None
            state.latest_tracking_identity_snapshot_published_at = 0.0
            state.tracking_identity_reassociation_enabled = TRACKING_IDENTITY_REASSOCIATION_ENABLED
            state.tracking_identity_reassociation_consumer = identity_reassociation_consumer
            state.latest_tracking_identity_reassociation_summary = None
            state.tracking_stable_identity_visualization_enabled = (
                TRACKING_STABLE_IDENTITY_VISUALIZATION_ENABLED
            )
            state.latest_tracking_display_identity_by_track = {}
            state.latest_tracking_display_identity_summary = None
            state.tracking_decision_ipc_enabled = TRACKING_DECISION_IPC_ENABLED
            state.tracking_decision_ipc_transport = decision_transport
            state.latest_tracking_decision_ipc_summary = None
            state.tracking_decision_investigations = {}
            state.tracking_decision_consumed_result_keys = set()
            state.tracking_decision_followup_result_keys = set()
            state.tracking_missing_track_grace_seconds = min(
                10.0,
                max(0.1, TRACKING_MISSING_TRACK_GRACE_SECONDS),
            )
    except Exception:
        if evidence_store is not None:
            evidence_store.shutdown()
        if tracking_runtime is not None:
            tracking_runtime.shutdown()
        if udp_socket is not None:
            udp_socket.close()
        camera.release()
        raise

    return model, camera, calibration, udp_socket, tracking_runtime


def run_live_tracking_loop(
    state: SharedDashboardState,
    video_path: str | None = None,
) -> None:
    initialized = _initialize_tracking_runtime(state, video_path=video_path)
    if initialized is None:
        return

    with state.lock:
        configured_video_path = state.video_path
    source: str | int = (
        str(Path(configured_video_path).expanduser())
        if configured_video_path is not None
        else 0
    )
    is_video_file = configured_video_path is not None

    model, camera, calibration, udp_socket, tracking_runtime = initialized

    try:
        while True:
            with state.lock:
                if not state.running:
                    break

            if is_video_file:
                ok, frame = camera.read()
                if not ok:
                    frame_count = camera.get(cv2.CAP_PROP_FRAME_COUNT)
                    frame_position = camera.get(cv2.CAP_PROP_POS_FRAMES)
                    is_video_end = frame_count > 0 and frame_position >= frame_count
                    with state.lock:
                        if is_video_end:
                            state.camera_status = "stopped"
                            state.detector_status = "stopped"
                            state.input_error = None
                        else:
                            state.camera_status = "frame_error"
                            state.detector_status = "error"
                            state.input_error = f"Unable to read a frame from {source!r}."
                    break
            else:
                ok, frame = _capture_frame_or_mark_error(state, camera)
                if not ok:
                    continue
            assert frame is not None

            now = time.time()
            with state.lock:
                state.camera_status = "connected"
                state.detector_status = "running"
                state.input_error = None
                state.frame_height, state.frame_width = frame.shape[:2]
                state.last_live_frame_time = now

            detection_result = _run_detection_or_mark_error(state, model, frame)
            if isinstance(detection_result, tuple) and len(detection_result) == 2:
                result, should_break = detection_result
            else:
                result, should_break = detection_result, False
            if result is None:
                if should_break:
                    break
                continue

            now = time.time()
            with state.lock:
                track_trails_snapshot = dict(state.track_trails)
            tracks, detections_for_file, tracking_candidates, selected_event_reasons = build_tracking_frame_outputs(
                boxes=result.boxes,
                model_names=model.names,
                now=now,
                frame_shape=frame.shape,
                calibration=calibration,
                track_trails=track_trails_snapshot,
                max_ground_x=MAX_GROUND_X,
                max_ground_z=MAX_GROUND_Z,
            )

            _build_prediction_detection_advisories(
                state=state,
                now=now,
                tracking_candidates=tracking_candidates,
            )

            _consume_identity_reassociation_if_enabled(
                state=state,
                now=now,
                tracking_candidates=tracking_candidates,
            )

            _update_identity_registry_if_enabled(
                state=state,
                now=now,
                tracking_candidates=tracking_candidates,
            )

            display_identity_by_track = _update_stable_identity_display_mapping_if_enabled(
                state=state,
                tracking_candidates=tracking_candidates,
            )

            _consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=tracking_runtime,
                frame=frame,
                tracking_candidates=tracking_candidates,
                now=now,
            )

            tracking_agent_observation = _apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=tracking_runtime,
                now=now,
                frame=frame,
                tracks=tracks,
                tracking_candidates=tracking_candidates,
                selected_event_reasons=selected_event_reasons,
            )

            _package_tracking_evidence_if_enabled(
                state=state,
                frame=frame,
                now=now,
                tracking_candidates=tracking_candidates,
                tracking_agent_observation=tracking_agent_observation,
            )

            _consume_tracking_observation_feedback(
                state=state,
                now=now,
                tracks=tracks,
            )

            _generate_tracking_estimate_proposals(
                state=state,
                now=now,
                tracks=tracks,
            )

            _promote_and_prune_tracking_prediction_evidence(
                state=state,
                now=now,
                tracks=tracks,
            )

            _publish_tracking_estimate_diagnostics(
                state=state,
                now=now,
            )

            _publish_tracking_outputs(
                state=state,
                result=result,
                now=now,
                tracks=tracks,
                tracking_candidates=tracking_candidates,
                tracking_agent_observation=tracking_agent_observation,
                trail_length=RADAR_TRAIL_LENGTH,
                detections_for_file=detections_for_file,
                udp_socket=udp_socket,
                stable_identity_visualization_enabled=bool(
                    TRACKING_STABLE_IDENTITY_VISUALIZATION_ENABLED
                ),
                display_identity_by_track=display_identity_by_track,
            )
    except Exception as error:
        with state.lock:
            state.camera_status = "frame_error"
            state.detector_status = "error"
            state.input_error = f"{type(error).__name__}: {error}"
    finally:
        with state.lock:
            if state.detector_status != "error":
                state.camera_status = "stopped"
                state.detector_status = "stopped"
            evidence_store = state.tracking_evidence_store
            state.tracking_evidence_store = None
            state.tracking_evidence_packager = None
            state.tracking_feedback_consumer = None
            state.tracking_estimate_engine = None
            state.latest_tracking_evidence_message_by_track = {}
            state.latest_tracking_lost_evidence_message_by_track = {}
            state.latest_tracking_lost_track_expires_at_by_track = {}
            state.latest_tracking_estimate_proposals = None
            state.latest_tracking_prediction_evidence_by_identity = {}
            state.latest_tracking_prediction_evidence_unresolved = []
            state.latest_tracking_prediction_evidence_summary = None
            state.tracking_prediction_last_calibration_status = None
            state.tracking_prediction_last_mapping_mode = None
            state.latest_tracking_prediction_detection_advisories = []
            state.latest_tracking_prediction_detection_advisory_summary = None
            state.tracking_identity_registry_enabled = False
            state.tracking_identity_registry = None
            state.tracking_identity_registry_generation = 0
            state.tracking_identity_registry_version = 0
            state.latest_tracking_identity_registry_summary = None
            state.tracking_identity_snapshot_bridge_enabled = False
            state.latest_tracking_identity_snapshot_metadata = None
            state.latest_tracking_identity_snapshot_source_version = None
            state.latest_tracking_identity_snapshot_published_at = 0.0
            state.tracking_identity_reassociation_enabled = False
            state.tracking_identity_reassociation_consumer = None
            state.latest_tracking_identity_reassociation_summary = None
            state.tracking_stable_identity_visualization_enabled = False
            state.latest_tracking_display_identity_by_track = {}
            state.latest_tracking_display_identity_summary = None
            state.tracking_decision_ipc_enabled = False
            state.tracking_decision_ipc_transport = None
            state.latest_tracking_decision_ipc_summary = None
            state.tracking_decision_investigations = {}
            state.tracking_decision_consumed_result_keys = set()
            state.tracking_decision_followup_result_keys = set()
        if evidence_store is not None:
            evidence_store.shutdown()
        try:
            write_json_payload(
                LATEST_TRACKING_ESTIMATE_DIAGNOSTICS_PATH,
                _build_tracking_estimate_diagnostic_payload(
                    proposal_summary=None,
                    now=time.time(),
                ),
            )
            write_json_payload(
                LATEST_TRACKING_PREDICTION_DETECTION_ADVISORIES_PATH,
                {
                    "schema_version": "1.0",
                    "updated_at": float(time.time()),
                    "available": False,
                    "summary": {
                        "updated_at": float(time.time()),
                        "available": False,
                        "reason": "tracking_runtime_stopped",
                        "advisory_count": 0,
                    },
                    "advisories": [],
                },
            )
        except OSError:
            pass
        udp_socket.close()
        tracking_runtime.shutdown()
        camera.release()
