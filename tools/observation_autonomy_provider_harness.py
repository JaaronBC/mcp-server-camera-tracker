from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import main
from agents.observation import agent_observation
from agents.observation.tools.track_evidence_history_contract import (
    build_track_evidence_history_argument_diagnostics,
    validate_track_evidence_history_arguments,
)
from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport
from agents.orchestration.tools import ops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "observation_autonomy" / "single_investigation_evidence_gap.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "captures" / "experiments" / "observation_autonomy_provider_harness"
HARNESS_SECOND_PASS_TERMINAL_SYSTEM_PROMPT = (
    "You are the Observation/Behavior Agent in a terminal follow-up pass. "
    "Evidence retrieval is complete and no additional tool requests are permitted. "
    "Return a terminal assessment only as strict JSON with keys: summary, measured_facts, interpretations, hypotheses, "
    "supporting_observation_ids, conflicting_evidence, uncertainties, needs_more_observations, and optional decision. "
    "If decision is provided, action must be one of: assess_now, predict_bounded, inconclusive. "
    "If decision.action is inconclusive, decision.reason must be meaningful and evidence-specific (not blank, generic, or placeholder). "
    "Do not use request_additional_tracking_evidence and do not include tool_requests for any tool call. "
    "Preserve deterministic measurements and uncertainty reporting; do not invent measurements. "
    "Return JSON only, with no Markdown or unnecessary explanation."
)
HARNESS_FIRST_PASS_CONCISE_SYSTEM_PROMPT = (
    "You are the Observation/Behavior Agent. Return strict JSON only with keys: summary, measured_facts, interpretations, hypotheses, "
    "supporting_observation_ids, conflicting_evidence, uncertainties, needs_more_observations, and optional decision. "
    "Keep each field concise and evidence-grounded; do not add narrative outside JSON. "
    "If decision is provided, it must be an object with keys: action, reason, uncertainty, tool_requests. "
    "If decision is provided, action must be one of: assess_now, request_additional_tracking_evidence, predict_bounded, inconclusive. "
    "For request_additional_tracking_evidence, tool_requests must contain exactly one request for get_track_evidence_history with a valid arguments object. "
    "For request_additional_tracking_evidence, only tool get_track_evidence_history is allowed. "
    "Model-selectable arguments are only max_messages, max_history_records, and max_scan_records (positive integers). "
    "Do not include unsupported arguments such as since_timestamp. "
    "track_id and correlation metadata are authoritative runtime context and are set by the system."
)
HARNESS_SECOND_PASS_TERMINAL_CONCISE_SYSTEM_PROMPT = (
    "You are the Observation/Behavior Agent in a terminal follow-up pass. "
    "Evidence retrieval is complete and no additional tool requests are permitted. "
    "Return strict JSON only with keys: summary, measured_facts, interpretations, hypotheses, "
    "supporting_observation_ids, conflicting_evidence, uncertainties, needs_more_observations, and optional decision. "
    "Keep output concise and evidence-grounded; avoid long narrative text. "
    "If decision is provided, action must be one of: assess_now, predict_bounded, inconclusive. "
    "If decision.action is inconclusive, decision.reason must be meaningful and evidence-specific (not blank, generic, or placeholder). "
    "Do not use request_additional_tracking_evidence and do not include tool_requests for any tool call. "
    "Preserve deterministic measurements and uncertainty reporting; do not invent measurements. "
    "Return JSON only, with no Markdown or unnecessary explanation."
)


@dataclass(frozen=True, slots=True)
class HarnessBudgets:
    provider_total_limit: int = 2
    provider_observation_text_limit: int = 2
    provider_tracking_vlm_limit: int = 0
    provider_reasoning_text_limit: int = 0
    provider_observation_visual_limit: int = 0
    max_observation_llm_calls: int = 2
    max_mcp_evidence_requests: int = 1
    wall_clock_budget_seconds: float = 45.0
    scheduler_analysis_interval_seconds: float = 0.01


@dataclass(frozen=True, slots=True)
class HarnessPaths:
    run_dir: Path
    report_path: Path
    requests_path: Path
    results_path: Path
    request_index_path: Path
    result_index_path: Path
    budget_state_path: Path
    budget_lock_path: Path
    budget_log_path: Path


def _build_paths(*, run_dir: Path) -> HarnessPaths:
    run_dir.mkdir(parents=True, exist_ok=True)
    return HarnessPaths(
        run_dir=run_dir,
        report_path=run_dir / "observation_autonomy_harness_report.json",
        requests_path=run_dir / "tracking_decision_tool_requests.jsonl",
        results_path=run_dir / "tracking_decision_tool_results.jsonl",
        request_index_path=run_dir / "latest_tracking_decision_request_index.json",
        result_index_path=run_dir / "latest_tracking_decision_result_index.json",
        budget_state_path=run_dir / "provider_call_budget_state.json",
        budget_lock_path=run_dir / "provider_call_budget_state.lock",
        budget_log_path=run_dir / "provider_call_budget_events.jsonl",
    )


def _has_existing_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    try:
        next(run_dir.iterdir())
    except StopIteration:
        return False
    return True


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture must be an object")
    return payload


def _bounded_float(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _budget_env(paths: HarnessPaths, budgets: HarnessBudgets) -> dict[str, str]:
    timeout_seconds = _bounded_float(budgets.wall_clock_budget_seconds / 2.0, default=30.0, minimum=5.0, maximum=300.0)
    return {
        "OBSERVATION_AGENT_MAX_OUTPUT_TOKENS": "1024",
        "PROVIDER_CALL_BUDGET_ENABLED": "1",
        "PROVIDER_CALL_BUDGET_TOTAL_LIMIT": str(budgets.provider_total_limit),
        "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT": str(budgets.provider_tracking_vlm_limit),
        "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT": str(budgets.provider_observation_text_limit),
        "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT": str(budgets.provider_reasoning_text_limit),
        "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT": str(budgets.provider_observation_visual_limit),
        "PROVIDER_CALL_BUDGET_STATE_PATH": str(paths.budget_state_path),
        "PROVIDER_CALL_BUDGET_LOCK_PATH": str(paths.budget_lock_path),
        "PROVIDER_CALL_BUDGET_LOG_PATH": str(paths.budget_log_path),
        "PHASE2D_TRACKING_DECISION_IPC_ENABLED": "1",
        "OBSERVATION_TOOL_REQUESTS_ENABLED": "1",
        "OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS": str(timeout_seconds),
        "PHASE2D_TRACKING_DECISION_TOOL_REQUESTS_PATH": str(paths.requests_path),
        "PHASE2D_TRACKING_DECISION_TOOL_RESULTS_PATH": str(paths.results_path),
        "PHASE2D_TRACKING_DECISION_REQUEST_INDEX_PATH": str(paths.request_index_path),
        "PHASE2D_TRACKING_DECISION_RESULT_INDEX_PATH": str(paths.result_index_path),
    }


def _read_budget_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "totals": {
                "attempted": 0,
                "per_role_attempted": {
                    "tracking_vlm": 0,
                    "observation_text": 0,
                    "reasoning_text": 0,
                    "observation_visual": 0,
                },
            }
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("budget state payload invalid")
    return payload


def _read_budget_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw_text.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _extract_attempt_count(state_payload: dict[str, Any]) -> int:
    totals = state_payload.get("totals") if isinstance(state_payload, dict) else None
    attempted = totals.get("attempted") if isinstance(totals, dict) else None
    if isinstance(attempted, int) and attempted >= 0:
        return attempted
    return 0


def _classify_pass_failure(*, phase: str, llm_meta: dict[str, Any] | None) -> str:
    phase_prefix = "first_pass" if phase == "pass1" else "second_pass"
    error_code = ""
    if isinstance(llm_meta, dict):
        raw_error_code = llm_meta.get("error_code")
        if isinstance(raw_error_code, str):
            error_code = raw_error_code.strip().lower()
    if error_code == "output_truncated":
        return f"{phase_prefix}_output_truncated"
    if error_code == "budget_exhausted":
        return f"{phase_prefix}_budget_exhausted"
    return f"{phase_prefix}_provider_error"


def _build_fallback_summary(fixture: dict[str, Any]) -> dict[str, Any]:
    measurements = fixture.get("summary_measurements")
    if not isinstance(measurements, dict):
        raise ValueError("fixture.summary_measurements must be an object")
    track_id = fixture.get("track_id")
    if not isinstance(track_id, str) or not track_id.strip():
        raise ValueError("fixture.track_id must be non-empty string")

    supporting_ids = fixture.get("supporting_observation_ids")
    if not isinstance(supporting_ids, list):
        supporting_ids = []

    return {
        "track_id": track_id,
        "created_at": fixture.get("created_at", time.time()),
        "velocity_x": measurements.get("velocity_x"),
        "velocity_z": measurements.get("velocity_z"),
        "speed": measurements.get("speed"),
        "confidence": measurements.get("confidence"),
        "sample_count": measurements.get("sample_count"),
        "supporting_observation_ids": [item for item in supporting_ids if isinstance(item, str)],
        "uncertainties": [item for item in fixture.get("quality_uncertainties", []) if isinstance(item, str)],
        "needs_more_observations": bool(fixture.get("expected_needs_more_observations", True)),
        "summary": "Deterministic fallback summary.",
        "measured_facts": [],
        "interpretations": [],
        "hypotheses": [],
        "conflicting_evidence": [],
    }


def _build_observation_user_payload(fixture: dict[str, Any], fallback_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_id": fallback_summary.get("track_id"),
        "created_at": fallback_summary.get("created_at"),
        "summary_measurements": {
            "velocity_x": fallback_summary.get("velocity_x"),
            "velocity_z": fallback_summary.get("velocity_z"),
            "speed": fallback_summary.get("speed"),
            "confidence": fallback_summary.get("confidence"),
            "sample_count": fallback_summary.get("sample_count"),
        },
        "quality_uncertainties": list(fallback_summary.get("uncertainties", [])),
        "history_records": copy.deepcopy(fixture.get("history_records", [])),
        "latest_track_snapshot": copy.deepcopy(fixture.get("latest_track_snapshot", {})),
        "tracking_agent_context": copy.deepcopy(fixture.get("tracking_agent_context", {})),
        "coordinate_context": copy.deepcopy(fixture.get("coordinate_context", {})),
    }


class _MockEvidenceSession:
    def __init__(self, *, fixture_result_payload: dict[str, Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fixture_payload = copy.deepcopy(fixture_result_payload)

    async def call_tool(self, tool_name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append({"tool_name": tool_name, "arguments": dict(arguments)})
        if tool_name != "get_track_evidence_history":
            payload = {
                "available": False,
                "error": "unexpected_tool_name",
                "tool_name": tool_name,
            }
            return SimpleNamespace(content=[SimpleNamespace(json=payload)])
        return SimpleNamespace(content=[SimpleNamespace(json=copy.deepcopy(self._fixture_payload))])


async def _run_scheduler_window(
    *,
    decision_cycle_fn: Callable[[], Awaitable[dict[str, Any]]],
    analysis_interval_seconds: float,
) -> dict[str, Any]:
    tracking_iterations = 0

    async def run_tracking_cycle(_session: object, _address: list[object]) -> None:
        nonlocal tracking_iterations
        tracking_iterations += 1
        if tracking_iterations >= 6:
            raise KeyboardInterrupt()

    async def run_agent_ops_pipeline_fn(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "agent_ops_enabled": False,
            "agent_ops_summary_enabled": False,
            "depth_summary_enabled": False,
            "completed_cycle_increment": 0,
        }

    with_keyboard_interrupt = False
    try:
        await runtime_flow.monitor_movement(
            session=SimpleNamespace(),
            tracking_address=("127.0.0.1", 5005),
            analysis_interval_seconds=analysis_interval_seconds,
            enable_agent_ops_heartbeat=False,
            agent_ops_interval_seconds=999.0,
            agent_ops_max_tracks=1,
            agent_ops_summary_every_cycles=10,
            agent_ops_summary_window_size=10,
            handoff_degraded_warn_rate=0.1,
            handoff_degraded_critical_rate=0.3,
            depth_coverage_summary_enabled=False,
            depth_coverage_summary_every_cycles=10,
            depth_coverage_sample_limit=10,
            run_tracking_cycle_fn=run_tracking_cycle,
            run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
            emit=lambda _message: None,
            decision_ipc_fixture_cycle_fn=decision_cycle_fn,
        )
    except KeyboardInterrupt:
        with_keyboard_interrupt = True

    return {
        "tracking_iterations": tracking_iterations,
        "stopped_by_keyboard_interrupt": with_keyboard_interrupt,
    }


def _build_request_decision(*, investigation_id: str, request_id: str, track_id: str, correlation_id: str) -> dict[str, Any]:
    return _build_request_decision_from_observation_decision(
        investigation_id=investigation_id,
        request_id=request_id,
        track_id=track_id,
        correlation_id=correlation_id,
        observation_decision=None,
    )


def _build_request_decision_from_observation_decision(
    *,
    investigation_id: str,
    request_id: str,
    track_id: str,
    correlation_id: str,
    observation_decision: dict[str, Any] | None,
) -> dict[str, Any]:
    tool_name = "get_track_evidence_history"
    arguments: dict[str, Any] = {
        "track_id": track_id,
        "max_messages": 5,
        "max_history_records": 12,
        "max_scan_records": 120,
    }

    if isinstance(observation_decision, dict):
        tool_requests = observation_decision.get("tool_requests")
        tool_request = tool_requests[0] if isinstance(tool_requests, list) and tool_requests else None
        if isinstance(tool_request, dict):
            selected_tool_name = tool_request.get("tool_name")
            selected_arguments = tool_request.get("arguments")
            if isinstance(selected_tool_name, str) and selected_tool_name.strip():
                tool_name = selected_tool_name.strip()
            if isinstance(selected_arguments, dict):
                arguments.update(dict(selected_arguments))

    arguments["track_id"] = track_id
    argument_validation = validate_track_evidence_history_arguments(
        arguments,
        require_track_id=False,
        validate_track_id_value=False,
    )

    return {
        "action": "request_additional_tracking_evidence",
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
        "argument_validation": argument_validation,
    }


def _resolve_observation_decision(
    *,
    summary_payload: dict[str, Any],
    request_evidence_on_uncertainty: bool,
) -> dict[str, Any]:
    decision = ops._resolve_observation_decision(
        summary_payload=summary_payload,
        request_evidence_on_uncertainty=request_evidence_on_uncertainty,
    )
    if not isinstance(decision, dict):
        decision = {
            "action": ops.OBSERVATION_ACTION_INCONCLUSIVE,
            "reason": "decision_unavailable",
            "decision_source": "llm_summary_inference",
            "tool_requests": [],
            "uncertainty": [],
        }

    action = decision.get("action")
    if action == ops.OBSERVATION_ACTION_PREDICT_BOUNDED and not ops._observation_prediction_supported(summary_payload):
        downgraded = dict(decision)
        downgraded["action"] = ops.OBSERVATION_ACTION_INCONCLUSIVE
        downgraded["reason"] = "predict_bounded_not_supported_by_deterministic_evidence"
        downgraded["predict_bounded_downgraded"] = True
        downgraded["tool_requests"] = []
        return downgraded
    return decision


def _classify_evidence_retrieval(correlated_result: dict[str, Any] | None, request_submitted: bool) -> str:
    if not request_submitted:
        return "not_requested"
    if not isinstance(correlated_result, dict):
        return "missing"

    status = correlated_result.get("status")
    error_code = correlated_result.get("error_code")
    payload = correlated_result.get("result_payload")
    payload_available = isinstance(payload, dict) and bool(payload.get("available", False))

    if status == "ok" and payload_available:
        return "retrieved"
    if status == "timeout" or error_code in {"tool_timeout", "request_expired"}:
        return "timeout"
    return "failed"


def _classify_investigation_outcome(
    *,
    evidence_retrieval_status: str,
    final_decision: dict[str, Any],
    validation: dict[str, Any],
) -> str:
    final_action = final_decision.get("action") if isinstance(final_decision, dict) else None
    justified_inconclusive = bool(
        validation.get("allows_inconclusive")
        and final_action in {
            ops.OBSERVATION_ACTION_INCONCLUSIVE,
            ops.OBSERVATION_ACTION_PREDICT_BOUNDED,
        }
    )
    if evidence_retrieval_status == "timeout":
        if justified_inconclusive:
            return "tool_timeout_inconclusive"
        return "failed_timeout"
    if evidence_retrieval_status == "missing":
        if justified_inconclusive:
            return "missing_evidence_inconclusive"
        return "failed_evidence_retrieval"
    if evidence_retrieval_status == "failed":
        if justified_inconclusive:
            return "tool_failure_inconclusive"
        return "failed_evidence_retrieval"
    if evidence_retrieval_status == "retrieved" and not justified_inconclusive:
        return "successful_evidence_based_assessment"
    if justified_inconclusive:
        return "justified_inconclusive"
    return "no_evidence_request"


def _correlated_result_observation_ids(correlated_result: dict[str, Any] | None) -> set[str]:
    if not isinstance(correlated_result, dict):
        return set()
    payload = correlated_result.get("result_payload")
    if not isinstance(payload, dict):
        return set()

    observation_ids: set[str] = set()

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            bounded_history = message.get("bounded_history")
            if not isinstance(bounded_history, list):
                continue
            for record in bounded_history:
                if not isinstance(record, dict):
                    continue
                observation_id = record.get("observation_id")
                if isinstance(observation_id, str) and observation_id:
                    observation_ids.add(observation_id)

    history_records = payload.get("history_records")
    if isinstance(history_records, list):
        for record in history_records:
            if not isinstance(record, dict):
                continue
            observation_id = record.get("observation_id")
            if isinstance(observation_id, str) and observation_id:
                observation_ids.add(observation_id)

    return observation_ids


def _classify_terminal_decision_provenance(
    *,
    first_pass_decision_source: str,
    evidence_retrieval_status: str,
    final_decision: dict[str, Any],
    final_summary: dict[str, Any],
    correlated_result: dict[str, Any] | None,
) -> str:
    final_decision_source = str(final_decision.get("decision_source") or "llm_summary_inference")
    final_phase = str(final_decision.get("phase") or "")

    if final_decision_source == "invalid_explicit_decision" or final_phase == "terminal_invalid_explicit_decision":
        return "invalid_explicit_decision"

    if first_pass_decision_source != "llm_explicit" or final_decision_source != "llm_explicit":
        return "orchestration_fallback"

    reason = final_decision.get("reason")
    normalized_reason = reason.strip() if isinstance(reason, str) else ""
    if normalized_reason in {"", "decision_reason_missing", "decision_unavailable", "decision_invalid"}:
        return "explicit_terminal_missing_reason"

    final_action = str(final_decision.get("action") or "")
    terminal_action_valid = final_action in {
        ops.OBSERVATION_ACTION_ASSESS_NOW,
        ops.OBSERVATION_ACTION_PREDICT_BOUNDED,
        ops.OBSERVATION_ACTION_INCONCLUSIVE,
    }
    evidence_retrieved = evidence_retrieval_status == "retrieved"

    supporting_observation_ids = {
        item
        for item in final_summary.get("supporting_observation_ids", [])
        if isinstance(item, str) and item
    }
    correlated_observation_ids = _correlated_result_observation_ids(correlated_result)
    supporting_ids_aligned_with_correlated_evidence = bool(
        correlated_observation_ids
        and supporting_observation_ids
        and supporting_observation_ids.intersection(correlated_observation_ids)
    )

    if terminal_action_valid and evidence_retrieved and supporting_ids_aligned_with_correlated_evidence:
        return "explicit_terminal_with_evidence_support"
    return "explicit_terminal_insufficient_evidence"


def _compact_correlated_evidence_tool_result(
    correlated_result: dict[str, Any],
    *,
    authoritative_track_id: str,
) -> dict[str, Any]:
    payload = correlated_result.get("result_payload")
    if not isinstance(payload, dict):
        payload = {}

    compact_messages: list[dict[str, Any]] = []
    supporting_observation_ids: list[str] = []
    seen_supporting_ids: set[str] = set()

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages[:6]:
            if not isinstance(message, dict):
                continue
            compact_history: list[dict[str, Any]] = []
            bounded_history = message.get("bounded_history")
            if isinstance(bounded_history, list):
                for record in bounded_history[:24]:
                    if not isinstance(record, dict):
                        continue
                    compact_record = {
                        "observation_id": record.get("observation_id"),
                        "timestamp": record.get("timestamp"),
                        "x": record.get("x"),
                        "z": record.get("z"),
                        "velocity_x": record.get("velocity_x"),
                        "velocity_z": record.get("velocity_z"),
                        "speed": record.get("speed"),
                        "confidence": record.get("confidence"),
                        "occluded": record.get("occluded"),
                        "quality_flags": record.get("quality_flags"),
                    }
                    compact_history.append(compact_record)
                    observation_id = compact_record.get("observation_id")
                    if isinstance(observation_id, str) and observation_id and observation_id not in seen_supporting_ids:
                        seen_supporting_ids.add(observation_id)
                        supporting_observation_ids.append(observation_id)
            compact_messages.append(
                {
                    "message_id": message.get("message_id"),
                    "source": message.get("source"),
                    "bounded_history": compact_history,
                }
            )

    compact_payload: dict[str, Any] = {
        "available": bool(payload.get("available", False)),
        "track_id": (
            payload.get("track_id")
            if isinstance(payload.get("track_id"), str) and payload.get("track_id")
            else authoritative_track_id
        ),
        "messages": compact_messages,
        "supporting_observation_ids": supporting_observation_ids,
        "quality_flags": payload.get("quality_flags"),
        "state": payload.get("state"),
        "errors": payload.get("errors"),
    }

    if not compact_messages:
        history_records = payload.get("history_records")
        if isinstance(history_records, list):
            compact_payload["history_records"] = [
                {
                    "observation_id": record.get("observation_id"),
                    "timestamp": record.get("timestamp"),
                    "x": record.get("x"),
                    "z": record.get("z"),
                    "velocity_x": record.get("velocity_x"),
                    "velocity_z": record.get("velocity_z"),
                    "speed": record.get("speed"),
                    "confidence": record.get("confidence"),
                    "occluded": record.get("occluded"),
                    "quality_flags": record.get("quality_flags"),
                }
                for record in history_records[:24]
                if isinstance(record, dict)
            ]

    return {
        "status": correlated_result.get("status"),
        "error_code": correlated_result.get("error_code"),
        "error": correlated_result.get("error"),
        "payload": compact_payload,
    }


def run_single_investigation(
    *,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    run_dir: Path | None = None,
    preflight_only: bool = True,
    concise_prompts: bool = False,
    budgets: HarnessBudgets = HarnessBudgets(),
) -> dict[str, Any]:
    started_at = time.time()
    selected_run_dir = run_dir or (DEFAULT_RUNS_DIR / f"run_{int(started_at)}_{uuid.uuid4().hex[:8]}")

    if not preflight_only and _has_existing_artifacts(selected_run_dir):
        return {
            "mode": "live_provider",
            "live_mode_enabled": True,
            "prompt_mode": "concise_harness" if concise_prompts else "baseline",
            "fixture_path": str(fixture_path),
            "run_dir": str(selected_run_dir),
            "budgets": {
                "provider_total_limit": budgets.provider_total_limit,
                "provider_observation_text_limit": budgets.provider_observation_text_limit,
                "provider_tracking_vlm_limit": budgets.provider_tracking_vlm_limit,
                "provider_reasoning_text_limit": budgets.provider_reasoning_text_limit,
                "provider_observation_visual_limit": budgets.provider_observation_visual_limit,
                "max_observation_llm_calls": budgets.max_observation_llm_calls,
                "max_mcp_evidence_requests": budgets.max_mcp_evidence_requests,
                "wall_clock_budget_seconds": budgets.wall_clock_budget_seconds,
            },
            "investigation": {
                "timing": {
                    "started_at": started_at,
                    "completed_at": time.time(),
                    "elapsed_seconds": max(0.0, time.time() - started_at),
                }
            },
            "safety": {
                "camera_interaction": "disabled",
                "tracking_identity_mutation": "disabled",
                "mcp_tool_scope": ["get_track_evidence_history"],
                "ipc_scope": "harness_run_dir_only",
            },
            "status": "error",
            "error": "RuntimeError:live_run_dir_not_empty",
            "outcome": "failed_runtime_error",
        }

    paths = _build_paths(run_dir=selected_run_dir)
    fixture = _load_fixture(fixture_path)
    budget_env = _budget_env(paths, budgets)

    original_environ = dict(os.environ)
    os.environ.update(budget_env)

    report: dict[str, Any] = {
        "mode": "preflight" if preflight_only else "live_provider",
        "live_mode_enabled": not preflight_only,
        "prompt_mode": "concise_harness" if concise_prompts else "baseline",
        "fixture_path": str(fixture_path),
        "run_dir": str(paths.run_dir),
        "budgets": {
            "provider_total_limit": budgets.provider_total_limit,
            "provider_observation_text_limit": budgets.provider_observation_text_limit,
            "provider_tracking_vlm_limit": budgets.provider_tracking_vlm_limit,
            "provider_reasoning_text_limit": budgets.provider_reasoning_text_limit,
            "provider_observation_visual_limit": budgets.provider_observation_visual_limit,
            "max_observation_llm_calls": budgets.max_observation_llm_calls,
            "max_mcp_evidence_requests": budgets.max_mcp_evidence_requests,
            "wall_clock_budget_seconds": budgets.wall_clock_budget_seconds,
        },
        "investigation": {},
        "safety": {
            "camera_interaction": "disabled",
            "tracking_identity_mutation": "disabled",
            "mcp_tool_scope": ["get_track_evidence_history"],
            "ipc_scope": "harness_run_dir_only",
        },
    }

    llm_call_count = 0
    llm_inputs: list[dict[str, Any]] = []
    llm_decisions: list[dict[str, Any]] = []
    correlation_id: str | None = None
    investigation_id: str | None = None
    request_id: str | None = None
    request_submission: dict[str, Any] | None = None
    request_decision: dict[str, Any] | None = None
    scheduler_summary: dict[str, Any] | None = None
    correlated_result: dict[str, Any] | None = None
    second_meta: dict[str, Any] | None = None
    evidence_session: _MockEvidenceSession | None = None
    first_decision: dict[str, Any] | None = None
    pass1_meta: dict[str, Any] | None = None
    pass2_meta: dict[str, Any] | None = None

    report["investigation"] = {
        "correlation_id": None,
        "investigation_id": None,
        "request_id": None,
        "request_submitted": False,
        "request_submission": None,
        "request_decision": None,
        "first_pass_decision_source": "llm_summary_inference",
        "first_decision": None,
        "final_decision": None,
        "final_summary": None,
        "correlated_result": None,
        "evidence_retrieval_status": "not_requested",
        "second_pass": {
            "attempted": False,
            "status": "not_attempted",
            "error": None,
            "error_code": None,
            "terminal_prompt_used": False,
            "received_correlated_evidence": False,
            "skipped": False,
            "skip_reason": None,
        },
        "phase_failure": {
            "code": None,
            "phase": None,
        },
        "pass_attempts": {
            "pass1": {
                "attempted": False,
                "provider_attempts": 0,
                "llm_attempts": 0,
                "retry_count": 0,
                "latency_ms": None,
                "status": None,
                "error": None,
                "error_code": None,
                "finish_reason": None,
                "truncated": False,
                "provider_events": [],
            },
            "pass2": {
                "attempted": False,
                "provider_attempts": 0,
                "llm_attempts": 0,
                "retry_count": 0,
                "latency_ms": None,
                "status": None,
                "error": None,
                "error_code": None,
                "finish_reason": None,
                "truncated": False,
                "provider_events": [],
            },
        },
        "observation_llm_call_count": 0,
        "mcp_evidence_call_count": 0,
        "provider_call_attempted_total": 0,
        "provider_call_attempted_by_role": {},
        "scheduler": None,
        "llm_inputs": llm_inputs,
        "llm_decisions": llm_decisions,
        "validation": {
            "deterministic_measurements_protected": False,
            "deterministic_integrity": {
                "velocity_x_unchanged": False,
                "velocity_z_unchanged": False,
                "speed_unchanged": False,
                "sample_count_unchanged": False,
            },
            "allows_inconclusive": False,
            "follow_up_received_correlated_evidence": False,
        },
        "timing": {
            "started_at": started_at,
            "completed_at": None,
            "elapsed_seconds": 0.0,
        },
    }

    try:
        fallback_summary = _build_fallback_summary(fixture)
        user_payload = _build_observation_user_payload(fixture, fallback_summary)

        responses = fixture.get("preflight_observation_responses") if isinstance(fixture.get("preflight_observation_responses"), list) else []

        async def _preflight_invoke_fn(**kwargs: Any) -> dict[str, Any]:
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count > len(responses):
                raise RuntimeError("preflight_response_sequence_exhausted")
            return copy.deepcopy(responses[llm_call_count - 1])

        def _call_observation(
            payload: dict[str, Any],
            *,
            system_prompt: str | None = None,
        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            llm_inputs.append(copy.deepcopy(payload))
            if preflight_only:
                return agent_observation.call_observation_agent_with_retry(
                    payload,
                    invoke_fn=_preflight_invoke_fn,
                    retry_limit=0,
                    system_prompt=system_prompt,
                )
            return agent_observation.call_observation_agent_with_retry(
                payload,
                retry_limit=0,
                system_prompt=system_prompt,
            )

        pass1_events_before = len(_read_budget_events(paths.budget_log_path))
        pass1_attempts_before = _extract_attempt_count(_read_budget_state(paths.budget_state_path))
        first_payload, first_meta = _call_observation(
            user_payload,
            system_prompt=HARNESS_FIRST_PASS_CONCISE_SYSTEM_PROMPT if concise_prompts else None,
        )
        pass1_meta = first_meta if isinstance(first_meta, dict) else None
        pass1_events_after = _read_budget_events(paths.budget_log_path)
        pass1_attempts_after = _extract_attempt_count(_read_budget_state(paths.budget_state_path))
        pass1_new_events = pass1_events_after[pass1_events_before:]
        pass1_provider_attempts = max(0, pass1_attempts_after - pass1_attempts_before)
        pass1_telemetry = report["investigation"]["pass_attempts"]["pass1"]
        pass1_telemetry["attempted"] = True
        pass1_telemetry["provider_attempts"] = pass1_provider_attempts
        pass1_telemetry["llm_attempts"] = int(first_meta.get("attempts", 0)) if isinstance(first_meta, dict) else 0
        pass1_telemetry["retry_count"] = int(first_meta.get("retry_count", 0)) if isinstance(first_meta, dict) else 0
        pass1_telemetry["latency_ms"] = first_meta.get("latency_ms") if isinstance(first_meta, dict) else None
        pass1_telemetry["status"] = first_meta.get("status") if isinstance(first_meta, dict) else None
        pass1_telemetry["error"] = first_meta.get("error") if isinstance(first_meta, dict) else None
        pass1_telemetry["error_code"] = first_meta.get("error_code") if isinstance(first_meta, dict) else None
        pass1_telemetry["finish_reason"] = first_meta.get("finish_reason") if isinstance(first_meta, dict) else None
        pass1_telemetry["truncated"] = bool(isinstance(first_meta, dict) and first_meta.get("error_code") == "output_truncated")
        pass1_telemetry["provider_events"] = pass1_new_events

        if not isinstance(first_payload, dict):
            phase_failure_code = _classify_pass_failure(phase="pass1", llm_meta=first_meta if isinstance(first_meta, dict) else None)
            report["investigation"]["phase_failure"] = {
                "code": phase_failure_code,
                "phase": "pass1",
            }
            report["investigation"]["second_pass"]["attempted"] = False
            report["investigation"]["second_pass"]["status"] = "skipped"
            report["investigation"]["second_pass"]["skipped"] = True
            report["investigation"]["second_pass"]["skip_reason"] = phase_failure_code
            report["investigation"]["second_pass"]["error"] = None
            report["investigation"]["second_pass"]["error_code"] = None
            report["investigation"]["request_submitted"] = False
            report["investigation"]["evidence_retrieval_status"] = "not_requested"
            report["investigation"]["first_pass_decision_source"] = "llm_summary_inference"

            elapsed = time.time() - started_at
            budget_state = _read_budget_state(paths.budget_state_path)
            per_role_attempted = (
                budget_state.get("totals", {}).get("per_role_attempted", {})
                if isinstance(budget_state.get("totals"), dict)
                else {}
            )
            provider_attempted_total = (
                budget_state.get("totals", {}).get("attempted", 0)
                if isinstance(budget_state.get("totals"), dict)
                else 0
            )
            report["investigation"]["observation_llm_call_count"] = llm_call_count
            report["investigation"]["mcp_evidence_call_count"] = 0
            report["investigation"]["provider_call_attempted_total"] = provider_attempted_total
            report["investigation"]["provider_call_attempted_by_role"] = per_role_attempted
            report["investigation"]["timing"] = {
                "started_at": started_at,
                "completed_at": time.time(),
                "elapsed_seconds": elapsed,
            }
            report["status"] = "error"
            report["error"] = (
                first_meta.get("error")
                if isinstance(first_meta, dict) and isinstance(first_meta.get("error"), str)
                else "RuntimeError:first_observation_payload_missing"
            )
            report["outcome"] = "failed_runtime_error"
            paths.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report

        valid_ids = set(item for item in fallback_summary.get("supporting_observation_ids", []) if isinstance(item, str))
        first_summary = agent_observation.validate_observation_llm_payload(
            first_payload,
            fallback_summary=fallback_summary,
            valid_observation_ids=valid_ids,
        )
        llm_decisions.append({"phase": "pass1", "meta": first_meta, "summary": copy.deepcopy(first_summary)})

        first_decision = _resolve_observation_decision(
            summary_payload=first_summary,
            request_evidence_on_uncertainty=True,
        )
        report["investigation"]["first_pass_decision_source"] = first_decision.get("decision_source", "llm_summary_inference")
        report["investigation"]["first_decision"] = first_decision
        report["investigation"]["final_summary"] = first_summary

        correlation_id = f"obs-corr-{uuid.uuid4().hex[:12]}"
        investigation_id = f"obs-investigation-{uuid.uuid4().hex[:12]}"
        request_id = f"obs-request-{uuid.uuid4().hex[:12]}"
        report["investigation"]["correlation_id"] = correlation_id
        report["investigation"]["investigation_id"] = investigation_id
        report["investigation"]["request_id"] = request_id

        dashboard_transport = TrackingDecisionIpcTransport(
            requests_path=paths.requests_path,
            results_path=paths.results_path,
            request_index_path=paths.request_index_path,
            result_index_path=paths.result_index_path,
            writer_role="dashboard",
        )
        orchestration_transport = TrackingDecisionIpcTransport(
            requests_path=paths.requests_path,
            results_path=paths.results_path,
            request_index_path=paths.request_index_path,
            result_index_path=paths.result_index_path,
            writer_role="orchestration",
        )
        evidence_session = _MockEvidenceSession(
            fixture_result_payload=copy.deepcopy(fixture.get("mcp_evidence_result", {})),
        )

        second_payload = None
        final_summary = first_summary
        final_decision = first_decision

        if first_decision.get("action") == ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE:
            request_decision = _build_request_decision_from_observation_decision(
                investigation_id=investigation_id,
                request_id=request_id,
                track_id=str(fallback_summary.get("track_id")),
                correlation_id=correlation_id,
                observation_decision=first_decision,
            )
            report["investigation"]["request_decision"] = request_decision
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
                request_submission = {
                    "accepted": False,
                    "duplicate": False,
                    "rejected": True,
                    "reason": "invalid_tool_arguments_pre_dispatch",
                    **diagnostics,
                }
                report["investigation"]["request_submission"] = request_submission
                report["investigation"]["request_submitted"] = False
                report["investigation"]["observation_tool_request_diagnostics"] = diagnostics
                report["investigation"]["observation_tool_request_validation_errors"] = validation_errors
                report["investigation"]["observation_tool_request_original_arguments"] = diagnostics.get("original_requested_arguments", {})
                correlated_result = {
                    "status": "rejected",
                    "error_code": "invalid_tool_arguments",
                    "error": "invalid_tool_arguments",
                    "result_payload": {
                        "available": False,
                        "errors": validation_errors,
                        "unknown_arguments": diagnostics.get("unknown_arguments", []),
                        "original_requested_arguments": diagnostics.get("original_requested_arguments", {}),
                    },
                    "request_id": request_id,
                    "investigation_id": investigation_id,
                    "tool_name": "get_track_evidence_history",
                }
                report["investigation"]["correlated_result"] = correlated_result
                if ops._observation_prediction_supported(first_summary):
                    final_decision = {
                        "action": ops.OBSERVATION_ACTION_PREDICT_BOUNDED,
                        "reason": "invalid_tool_arguments_pre_dispatch",
                        "uncertainty": list(first_summary.get("uncertainties", [])),
                        "tool_requests": [],
                        "decision_source": first_decision.get("decision_source", "llm_summary_inference"),
                        "phase": "tool_request_pre_dispatch_rejected",
                        "error_code": "invalid_tool_arguments",
                        "validation_errors": validation_errors,
                        "unknown_arguments": diagnostics.get("unknown_arguments", []),
                    }
                else:
                    final_decision = {
                        "action": ops.OBSERVATION_ACTION_INCONCLUSIVE,
                        "reason": "invalid_tool_arguments_pre_dispatch",
                        "uncertainty": list(first_summary.get("uncertainties", [])),
                        "tool_requests": [],
                        "decision_source": first_decision.get("decision_source", "llm_summary_inference"),
                        "phase": "tool_request_pre_dispatch_rejected",
                        "error_code": "invalid_tool_arguments",
                        "validation_errors": validation_errors,
                        "unknown_arguments": diagnostics.get("unknown_arguments", []),
                    }
                final_summary = first_summary
            else:
                request_submission = dashboard_transport.dashboard_submit_decision_request(
                    decision=request_decision,
                    llm_correlation_id=correlation_id,
                )
                report["investigation"]["request_submission"] = request_submission
                report["investigation"]["request_submitted"] = bool(request_submission.get("accepted", False))
                if not bool(request_submission.get("accepted", False)):
                    raise RuntimeError(f"request_not_accepted:{request_submission}")

                async def _decision_cycle() -> dict[str, Any]:
                    return await main._run_tracking_decision_ipc_mcp_cycle(
                        session=evidence_session,
                        transport=orchestration_transport,
                        max_requests_per_cycle=8,
                        tool_timeout_seconds=5.0,
                    )

                scheduler_summary = asyncio.run(
                    _run_scheduler_window(
                        decision_cycle_fn=_decision_cycle,
                        analysis_interval_seconds=budgets.scheduler_analysis_interval_seconds,
                    )
                )
                report["investigation"]["scheduler"] = scheduler_summary

                lookup = dashboard_transport.lookup_results_for_request(
                    investigation_id=investigation_id,
                    request_id=request_id,
                    tool_name="get_track_evidence_history",
                )
                latest = lookup.get("latest") if isinstance(lookup, dict) else None
                if not isinstance(latest, dict):
                    raise RuntimeError("correlated_result_missing")
                correlated_result = copy.deepcopy(latest)
                report["investigation"]["correlated_result"] = correlated_result

                compact_evidence_tool_result = _compact_correlated_evidence_tool_result(
                    correlated_result,
                    authoritative_track_id=str(fallback_summary.get("track_id")),
                )

                second_input = {
                    "track_id": first_summary.get("track_id"),
                    "created_at": first_summary.get("created_at"),
                    "summary_measurements": {
                        "velocity_x": first_summary.get("velocity_x"),
                        "velocity_z": first_summary.get("velocity_z"),
                        "speed": first_summary.get("speed"),
                        "confidence": first_summary.get("confidence"),
                        "sample_count": first_summary.get("sample_count"),
                    },
                    "quality_uncertainties": first_summary.get("uncertainties", []),
                    "supporting_observation_ids": first_summary.get("supporting_observation_ids", []),
                    "evidence_phase": "complete",
                    "terminal_assessment_required": True,
                    "tool_request_policy": "no_additional_tool_requests_permitted",
                    "evidence_tool_result": compact_evidence_tool_result,
                }
                report["investigation"]["second_pass"] = {
                    "attempted": True,
                    "status": "started",
                    "error": None,
                    "error_code": None,
                    "terminal_prompt_used": True,
                    "received_correlated_evidence": True,
                    "skipped": False,
                    "skip_reason": None,
                }
                pass2_events_before = len(_read_budget_events(paths.budget_log_path))
                pass2_attempts_before = _extract_attempt_count(_read_budget_state(paths.budget_state_path))
                second_payload, second_meta = _call_observation(
                    second_input,
                    system_prompt=(
                        HARNESS_SECOND_PASS_TERMINAL_CONCISE_SYSTEM_PROMPT
                        if concise_prompts
                        else HARNESS_SECOND_PASS_TERMINAL_SYSTEM_PROMPT
                    ),
                )
                pass2_meta = second_meta if isinstance(second_meta, dict) else None
                pass2_events_after = _read_budget_events(paths.budget_log_path)
                pass2_attempts_after = _extract_attempt_count(_read_budget_state(paths.budget_state_path))
                pass2_new_events = pass2_events_after[pass2_events_before:]
                pass2_provider_attempts = max(0, pass2_attempts_after - pass2_attempts_before)
                pass2_telemetry = report["investigation"]["pass_attempts"]["pass2"]
                pass2_telemetry["attempted"] = True
                pass2_telemetry["provider_attempts"] = pass2_provider_attempts
                pass2_telemetry["llm_attempts"] = int(second_meta.get("attempts", 0)) if isinstance(second_meta, dict) else 0
                pass2_telemetry["retry_count"] = int(second_meta.get("retry_count", 0)) if isinstance(second_meta, dict) else 0
                pass2_telemetry["latency_ms"] = second_meta.get("latency_ms") if isinstance(second_meta, dict) else None
                pass2_telemetry["status"] = second_meta.get("status") if isinstance(second_meta, dict) else None
                pass2_telemetry["error"] = second_meta.get("error") if isinstance(second_meta, dict) else None
                pass2_telemetry["error_code"] = second_meta.get("error_code") if isinstance(second_meta, dict) else None
                pass2_telemetry["finish_reason"] = second_meta.get("finish_reason") if isinstance(second_meta, dict) else None
                pass2_telemetry["truncated"] = bool(isinstance(second_meta, dict) and second_meta.get("error_code") == "output_truncated")
                pass2_telemetry["provider_events"] = pass2_new_events
                report["investigation"]["second_pass"]["status"] = str(second_meta.get("status", "unknown")) if isinstance(second_meta, dict) else "unknown"
                report["investigation"]["second_pass"]["error"] = second_meta.get("error") if isinstance(second_meta, dict) else None
                report["investigation"]["second_pass"]["error_code"] = second_meta.get("error_code") if isinstance(second_meta, dict) else None
                if not isinstance(second_payload, dict):
                    report["investigation"]["phase_failure"] = {
                        "code": _classify_pass_failure(phase="pass2", llm_meta=second_meta if isinstance(second_meta, dict) else None),
                        "phase": "pass2",
                    }
                    raise RuntimeError("second_observation_payload_missing")

                evidence_messages = second_input["evidence_tool_result"]["payload"].get("messages", [])
                if isinstance(evidence_messages, list):
                    for item in evidence_messages:
                        if not isinstance(item, dict):
                            continue
                        bounded_history = item.get("bounded_history")
                        if not isinstance(bounded_history, list):
                            continue
                        for record in bounded_history:
                            if isinstance(record, dict) and isinstance(record.get("observation_id"), str):
                                valid_ids.add(record["observation_id"])

                final_summary = agent_observation.validate_observation_llm_payload(
                    second_payload,
                    fallback_summary=first_summary,
                    valid_observation_ids=valid_ids,
                )
                final_decision = _resolve_observation_decision(
                    summary_payload=final_summary,
                    request_evidence_on_uncertainty=False,
                )
                if final_decision.get("decision_source") == "invalid_explicit_decision":
                    final_decision["phase"] = "terminal_invalid_explicit_decision"
                else:
                    final_decision["phase"] = "follow_up_terminal"
                report["investigation"]["second_pass"]["status"] = "ok"
                llm_decisions.append({"phase": "pass2", "meta": second_meta, "summary": copy.deepcopy(final_summary)})

        elapsed = time.time() - started_at
        budget_state = _read_budget_state(paths.budget_state_path)
        per_role_attempted = (
            budget_state.get("totals", {}).get("per_role_attempted", {})
            if isinstance(budget_state.get("totals"), dict)
            else {}
        )
        provider_attempted_total = (
            budget_state.get("totals", {}).get("attempted", 0)
            if isinstance(budget_state.get("totals"), dict)
            else 0
        )
        mcp_call_count = len(evidence_session.calls)

        deterministic_integrity = {
            "velocity_x_unchanged": final_summary.get("velocity_x") == fallback_summary.get("velocity_x"),
            "velocity_z_unchanged": final_summary.get("velocity_z") == fallback_summary.get("velocity_z"),
            "speed_unchanged": final_summary.get("speed") == fallback_summary.get("speed"),
            "sample_count_unchanged": final_summary.get("sample_count") == fallback_summary.get("sample_count"),
        }

        if llm_call_count > budgets.max_observation_llm_calls:
            raise RuntimeError("observation_llm_call_budget_exceeded")
        if mcp_call_count > budgets.max_mcp_evidence_requests:
            raise RuntimeError("mcp_request_budget_exceeded")
        if isinstance(provider_attempted_total, int) and provider_attempted_total > budgets.provider_total_limit:
            raise RuntimeError("provider_call_attempt_budget_exceeded")
        if elapsed > budgets.wall_clock_budget_seconds:
            raise RuntimeError("wall_clock_budget_exceeded")

        report["investigation"]["request_id"] = request_id if isinstance(request_submission, dict) else None
        report["investigation"]["request_submitted"] = bool(request_submission and request_submission.get("accepted", False))
        report["investigation"]["final_decision"] = final_decision
        report["investigation"]["final_summary"] = final_summary
        report["investigation"]["request_submission"] = request_submission
        report["investigation"]["correlated_result"] = correlated_result
        report["investigation"]["evidence_references"] = {
            "message_ids": [
                item.get("message_id")
                for item in (correlated_result or {}).get("result_payload", {}).get("messages", [])
                if isinstance(item, dict)
            ]
        }
        report["investigation"]["observation_llm_call_count"] = llm_call_count
        report["investigation"]["mcp_evidence_call_count"] = mcp_call_count
        report["investigation"]["provider_call_attempted_total"] = provider_attempted_total
        report["investigation"]["provider_call_attempted_by_role"] = per_role_attempted
        report["investigation"]["scheduler"] = scheduler_summary
        report["investigation"]["validation"] = {
            "deterministic_measurements_protected": all(deterministic_integrity.values()),
            "deterministic_integrity": deterministic_integrity,
            "allows_inconclusive": final_decision.get("action") in {
                ops.OBSERVATION_ACTION_INCONCLUSIVE,
                ops.OBSERVATION_ACTION_ASSESS_NOW,
                ops.OBSERVATION_ACTION_PREDICT_BOUNDED,
            },
            "follow_up_received_correlated_evidence": (
                isinstance(correlated_result, dict)
                and llm_call_count >= 2
                and isinstance(llm_inputs[-1], dict)
                and isinstance(llm_inputs[-1].get("evidence_tool_result"), dict)
            ),
        }
        report["investigation"]["timing"] = {
            "started_at": started_at,
            "completed_at": time.time(),
            "elapsed_seconds": elapsed,
        }

        validation_payload = report["investigation"].get("validation", {})
        request_submitted = bool(report["investigation"].get("request_submitted", False))
        evidence_retrieval_status = _classify_evidence_retrieval(correlated_result, request_submitted)
        first_pass_decision_source = str(report["investigation"].get("first_pass_decision_source") or "llm_summary_inference")
        first_action = str(first_decision.get("action") or "")
        final_phase = str(final_decision.get("phase") or "")
        explicit_autonomy_demonstrated = first_pass_decision_source == "llm_explicit"
        explicit_tool_request_validated = bool(
            explicit_autonomy_demonstrated
            and first_action == ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE
            and request_submitted
        )
        explicit_tool_request_dispatched = bool(
            explicit_tool_request_validated
            and mcp_call_count == 1
            and evidence_retrieval_status == "retrieved"
        )
        explicit_tool_request_followed_by_terminal_second_pass = bool(
            explicit_tool_request_dispatched
            and llm_call_count >= 2
            and final_phase in {"follow_up_terminal", "terminal_invalid_explicit_decision"}
            and final_decision.get("action") != ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE
        )
        terminal_decision_provenance = _classify_terminal_decision_provenance(
            first_pass_decision_source=first_pass_decision_source,
            evidence_retrieval_status=evidence_retrieval_status,
            final_decision=final_decision if isinstance(final_decision, dict) else {},
            final_summary=final_summary if isinstance(final_summary, dict) else {},
            correlated_result=correlated_result if isinstance(correlated_result, dict) else None,
        )
        tool_request_autonomy_demonstrated = bool(
            explicit_tool_request_validated
            and explicit_tool_request_dispatched
            and explicit_tool_request_followed_by_terminal_second_pass
        )

        report["investigation"]["evidence_retrieval_status"] = evidence_retrieval_status
        report["investigation"]["explicit_autonomy_demonstrated"] = explicit_autonomy_demonstrated
        report["investigation"]["tool_request_autonomy_demonstrated"] = tool_request_autonomy_demonstrated
        report["investigation"]["successful_explicit_observation_autonomy"] = bool(
            first_pass_decision_source == "llm_explicit"
        )
        report["investigation"]["classification"] = {
            "explicit_tool_request_validated": explicit_tool_request_validated,
            "explicit_tool_request_dispatched": explicit_tool_request_dispatched,
            "explicit_tool_request_followed_by_terminal_second_pass": explicit_tool_request_followed_by_terminal_second_pass,
            "terminal_decision_provenance": terminal_decision_provenance,
            "explicit_non_request_decision": bool(
                explicit_autonomy_demonstrated
                and first_action in {
                    ops.OBSERVATION_ACTION_ASSESS_NOW,
                    ops.OBSERVATION_ACTION_PREDICT_BOUNDED,
                    ops.OBSERVATION_ACTION_INCONCLUSIVE,
                }
            ),
            "predict_bounded_downgraded": bool(
                first_decision.get("predict_bounded_downgraded")
                or final_decision.get("predict_bounded_downgraded")
            ),
        }
        report["investigation"]["outcome"] = _classify_investigation_outcome(
            evidence_retrieval_status=evidence_retrieval_status,
            final_decision=final_decision,
            validation=validation_payload if isinstance(validation_payload, dict) else {},
        )
        report["status"] = "ok"
    except Exception as error:
        budget_state = _read_budget_state(paths.budget_state_path)
        per_role_attempted = (
            budget_state.get("totals", {}).get("per_role_attempted", {})
            if isinstance(budget_state.get("totals"), dict)
            else {}
        )
        provider_attempted_total = (
            budget_state.get("totals", {}).get("attempted", 0)
            if isinstance(budget_state.get("totals"), dict)
            else 0
        )
        report["investigation"]["observation_llm_call_count"] = llm_call_count
        report["investigation"]["mcp_evidence_call_count"] = len(evidence_session.calls) if isinstance(evidence_session, _MockEvidenceSession) else 0
        report["investigation"]["provider_call_attempted_total"] = provider_attempted_total
        report["investigation"]["provider_call_attempted_by_role"] = per_role_attempted
        report["investigation"]["scheduler"] = scheduler_summary
        report["investigation"]["request_submission"] = request_submission
        report["investigation"]["request_decision"] = request_decision
        report["investigation"]["correlated_result"] = correlated_result
        report["investigation"]["first_decision"] = first_decision
        report["investigation"]["first_pass_decision_source"] = (
            first_decision.get("decision_source", "llm_summary_inference")
            if isinstance(first_decision, dict)
            else "llm_summary_inference"
        )
        report["investigation"]["request_submitted"] = bool(request_submission and request_submission.get("accepted", False))
        if isinstance(second_meta, dict):
            report["investigation"]["second_pass"]["attempted"] = True
            report["investigation"]["second_pass"]["status"] = str(second_meta.get("status", "error"))
            report["investigation"]["second_pass"]["error"] = second_meta.get("error")
            report["investigation"]["second_pass"]["error_code"] = second_meta.get("error_code")
            report["investigation"]["pass_attempts"]["pass2"]["attempted"] = True
            report["investigation"]["pass_attempts"]["pass2"]["llm_attempts"] = int(second_meta.get("attempts", 0))
            report["investigation"]["pass_attempts"]["pass2"]["retry_count"] = int(second_meta.get("retry_count", 0))
            report["investigation"]["pass_attempts"]["pass2"]["latency_ms"] = second_meta.get("latency_ms")
            report["investigation"]["pass_attempts"]["pass2"]["status"] = second_meta.get("status")
            report["investigation"]["pass_attempts"]["pass2"]["error"] = second_meta.get("error")
            report["investigation"]["pass_attempts"]["pass2"]["error_code"] = second_meta.get("error_code")
            report["investigation"]["pass_attempts"]["pass2"]["finish_reason"] = second_meta.get("finish_reason")
            report["investigation"]["pass_attempts"]["pass2"]["truncated"] = bool(second_meta.get("error_code") == "output_truncated")
        if isinstance(pass1_meta, dict):
            report["investigation"]["pass_attempts"]["pass1"]["attempted"] = True
            report["investigation"]["pass_attempts"]["pass1"]["llm_attempts"] = int(pass1_meta.get("attempts", 0))
            report["investigation"]["pass_attempts"]["pass1"]["retry_count"] = int(pass1_meta.get("retry_count", 0))
            report["investigation"]["pass_attempts"]["pass1"]["latency_ms"] = pass1_meta.get("latency_ms")
            report["investigation"]["pass_attempts"]["pass1"]["status"] = pass1_meta.get("status")
            report["investigation"]["pass_attempts"]["pass1"]["error"] = pass1_meta.get("error")
            report["investigation"]["pass_attempts"]["pass1"]["error_code"] = pass1_meta.get("error_code")
            report["investigation"]["pass_attempts"]["pass1"]["finish_reason"] = pass1_meta.get("finish_reason")
            report["investigation"]["pass_attempts"]["pass1"]["truncated"] = bool(pass1_meta.get("error_code") == "output_truncated")
        if not isinstance(report["investigation"].get("phase_failure"), dict):
            report["investigation"]["phase_failure"] = {"code": None, "phase": None}
        phase_failure = report["investigation"].get("phase_failure")
        if isinstance(phase_failure, dict) and not isinstance(phase_failure.get("code"), str):
            if isinstance(second_meta, dict) and second_meta.get("status") == "error":
                phase_failure["code"] = _classify_pass_failure(phase="pass2", llm_meta=second_meta)
                phase_failure["phase"] = "pass2"
            elif isinstance(pass1_meta, dict) and pass1_meta.get("status") == "error":
                phase_failure["code"] = _classify_pass_failure(phase="pass1", llm_meta=pass1_meta)
                phase_failure["phase"] = "pass1"
        report["investigation"]["evidence_retrieval_status"] = _classify_evidence_retrieval(
            correlated_result,
            bool(request_submission and request_submission.get("accepted", False)),
        )
        report["investigation"]["timing"] = {
            "started_at": started_at,
            "completed_at": time.time(),
            "elapsed_seconds": max(0.0, time.time() - started_at),
        }
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}:{error}"
        error_text = report["error"].lower()
        if "timeout" in error_text:
            report["outcome"] = "failed_timeout"
        else:
            report["outcome"] = "failed_runtime_error"
    finally:
        os.environ.clear()
        os.environ.update(original_environ)
        paths.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observation autonomy provider harness")
    parser.add_argument("--fixture", default=str(DEFAULT_FIXTURE_PATH), help="Path to single investigation fixture JSON")
    parser.add_argument("--run-dir", default="", help="Optional explicit output run directory")
    parser.add_argument("--live", action="store_true", help="Enable real Observation provider calls")
    parser.add_argument("--concise-prompts", action="store_true", help="Use harness-only concise prompt variants")
    return parser.parse_args(argv)


def main_cli(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fixture_path = Path(args.fixture).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    report = run_single_investigation(
        fixture_path=fixture_path,
        run_dir=run_dir,
        preflight_only=not bool(args.live),
        concise_prompts=bool(args.concise_prompts),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main_cli())
