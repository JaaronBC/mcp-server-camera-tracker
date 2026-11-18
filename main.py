# Runtime entrypoint for initialization/bootstrap and runtime lifecycle wiring.
import argparse
import asyncio
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from agents.orchestration.runtime import app as runtime_app
from agents.orchestration.runtime import config as runtime_config
from agents.orchestration.runtime import core as runtime_agent
from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.identity_reassociation_intents import IdentityReassociationIntentPublisher
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport
from agents.orchestration.runtime.verification_feedback import VerificationFeedbackPublisher
from agents.orchestration.runtime.verification_queue import VerificationQueueManager
from agents.observation.tools.track_evidence_history_contract import (
    validate_track_evidence_history_arguments,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "server.py"
ANALYSIS_INTERVAL_SECONDS = runtime_agent.ANALYSIS_INTERVAL_SECONDS
VERIFICATION_QUEUE_CAPACITY = 32
VERIFICATION_MAX_CONCURRENCY = 2
VERIFICATION_RESULTS_CAPACITY = 512
LATEST_VERIFICATION_QUEUE_RESULTS_PATH = PROJECT_ROOT / "captures" / "latest_verification_queue_results.json"
LATEST_VERIFICATION_QUEUE_SUBMISSIONS_PATH = PROJECT_ROOT / "captures" / "latest_verification_queue_submissions.json"
LATEST_OPERATOR_TRACKING_FEEDBACK_PATH = PROJECT_ROOT / "captures" / "operator_tracking_feedback.json"
LATEST_IDENTITY_REASSOCIATION_INTENTS_PATH = PROJECT_ROOT / "captures" / "tracking_identity_reassociation_intents.json"
LATEST_IDENTITY_REASSOCIATION_ACK_LEDGER_PATH = PROJECT_ROOT / "captures" / "tracking_identity_reassociation_ack_ledger.json"
LATEST_TRACKING_DECISION_TOOL_REQUESTS_PATH = PROJECT_ROOT / "captures" / "tracking_decision_tool_requests.jsonl"
LATEST_TRACKING_DECISION_TOOL_RESULTS_PATH = PROJECT_ROOT / "captures" / "tracking_decision_tool_results.jsonl"
LATEST_TRACKING_DECISION_REQUEST_INDEX_PATH = PROJECT_ROOT / "captures" / "latest_tracking_decision_request_index.json"
LATEST_TRACKING_DECISION_RESULT_INDEX_PATH = PROJECT_ROOT / "captures" / "latest_tracking_decision_result_index.json"
TRACKING_DECISION_TOOL_ALLOWLIST = {
    "get_identity_snapshot",
    "get_track_evidence_history",
}


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_verification_request_timeout_seconds() -> float:
    raw = os.getenv("VERIFICATION_REQUEST_TIMEOUT_SECONDS")
    if raw is None:
        return 8.0
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            "Invalid configuration for VERIFICATION_REQUEST_TIMEOUT_SECONDS: "
            f"expected a positive finite float, got {raw!r}"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Invalid configuration for VERIFICATION_REQUEST_TIMEOUT_SECONDS: "
            f"expected a positive finite float, got {raw!r}"
        )
    return value


VERIFICATION_REQUEST_TIMEOUT_SECONDS = _resolve_verification_request_timeout_seconds()

get_result_text = runtime_agent.get_result_text
get_result_json_object = runtime_agent.get_result_json_object
format_agent_ops_cycle = runtime_agent.format_agent_ops_cycle
format_agent_ops_cycle_evidence = runtime_agent.format_agent_ops_cycle_evidence
format_agent_ops_health_summary = runtime_agent.format_agent_ops_health_summary
_format_agent_error_digest = runtime_agent._format_agent_error_digest
_format_agent_error_trends = runtime_agent._format_agent_error_trends
_format_handoff_validation_trend = runtime_agent._format_handoff_validation_trend
_format_handoff_status_summary = runtime_agent._format_handoff_status_summary
_format_handoff_degraded_rate = runtime_agent._format_handoff_degraded_rate
_classify_handoff_severity = runtime_agent._classify_handoff_severity
_format_handoff_severity = runtime_agent._format_handoff_severity
_is_agent_health_payload_fresh = runtime_agent._is_agent_health_payload_fresh
format_depth_coverage_summary = runtime_agent.format_depth_coverage_summary
format_runtime_config_banner = runtime_agent.format_runtime_config_banner
print_depth_coverage_startup_snapshot = runtime_agent.print_depth_coverage_startup_snapshot
print_pipeline_contract_startup_snapshot = runtime_agent.print_pipeline_contract_startup_snapshot
verify_required_tools = runtime_agent.verify_required_tools
start_live_tracking = runtime_agent.start_live_tracking
start_orchestration_agents = runtime_agent.start_orchestration_agents
run_agent_ops_cycle = runtime_agent.run_agent_ops_cycle
run_agent_ops_health_summary = runtime_agent.run_agent_ops_health_summary
run_depth_coverage_summary = runtime_agent.run_depth_coverage_summary


def _persist_latest_verification_queue_results(results: list[dict[str, Any]]) -> None:
    _persist_json_artifact(LATEST_VERIFICATION_QUEUE_RESULTS_PATH, results)


def _persist_latest_verification_queue_submissions(submissions: list[dict[str, Any]]) -> None:
    _persist_json_artifact(LATEST_VERIFICATION_QUEUE_SUBMISSIONS_PATH, submissions)


def _persist_json_artifact(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _resolve_artifact_path(default_path: Path, env_var_name: str) -> Path:
    raw = os.getenv(env_var_name, "").strip()
    if not raw:
        return default_path
    return Path(raw).expanduser().resolve()


def _resolve_verification_artifact_paths() -> tuple[Path, Path, Path, Path, Path]:
    results_path = _resolve_artifact_path(
        LATEST_VERIFICATION_QUEUE_RESULTS_PATH,
        "PHASE16E_VERIFICATION_RESULTS_PATH",
    )
    submissions_path = _resolve_artifact_path(
        LATEST_VERIFICATION_QUEUE_SUBMISSIONS_PATH,
        "PHASE16E_VERIFICATION_SUBMISSIONS_PATH",
    )
    feedback_path = _resolve_artifact_path(
        LATEST_OPERATOR_TRACKING_FEEDBACK_PATH,
        "PHASE16E_OPERATOR_FEEDBACK_PATH",
    )
    identity_intents_path = _resolve_artifact_path(
        LATEST_IDENTITY_REASSOCIATION_INTENTS_PATH,
        "PHASE2D_IDENTITY_REASSOCIATION_INTENTS_PATH",
    )
    identity_intents_ack_ledger_path = _resolve_artifact_path(
        LATEST_IDENTITY_REASSOCIATION_ACK_LEDGER_PATH,
        "PHASE2D_IDENTITY_REASSOCIATION_ACK_LEDGER_PATH",
    )
    return results_path, submissions_path, feedback_path, identity_intents_path, identity_intents_ack_ledger_path


def _resolve_tracking_decision_ipc_artifact_paths() -> tuple[Path, Path, Path, Path]:
    requests_path = _resolve_artifact_path(
        LATEST_TRACKING_DECISION_TOOL_REQUESTS_PATH,
        "PHASE2D_TRACKING_DECISION_TOOL_REQUESTS_PATH",
    )
    results_path = _resolve_artifact_path(
        LATEST_TRACKING_DECISION_TOOL_RESULTS_PATH,
        "PHASE2D_TRACKING_DECISION_TOOL_RESULTS_PATH",
    )
    request_index_path = _resolve_artifact_path(
        LATEST_TRACKING_DECISION_REQUEST_INDEX_PATH,
        "PHASE2D_TRACKING_DECISION_REQUEST_INDEX_PATH",
    )
    result_index_path = _resolve_artifact_path(
        LATEST_TRACKING_DECISION_RESULT_INDEX_PATH,
        "PHASE2D_TRACKING_DECISION_RESULT_INDEX_PATH",
    )
    return requests_path, results_path, request_index_path, result_index_path


def _resolve_tracking_decision_ipc_max_requests_per_cycle(default: int = 8) -> int:
    raw = os.getenv("PHASE2D_TRACKING_DECISION_IPC_MAX_REQUESTS_PER_CYCLE", str(default)).strip()
    try:
        parsed = int(float(raw))
    except ValueError:
        parsed = default
    return max(1, min(64, parsed))


def _resolve_tracking_decision_ipc_tool_timeout_seconds(default: float = 1.5) -> float:
    raw = os.getenv("PHASE2D_TRACKING_DECISION_IPC_TOOL_TIMEOUT_SECONDS", str(default)).strip()
    try:
        parsed = float(raw)
    except ValueError:
        parsed = default
    if not math.isfinite(parsed) or parsed <= 0:
        parsed = default
    return max(0.1, min(30.0, parsed))


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _extract_mcp_result_payload(result: Any) -> dict[str, Any] | None:
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        return None
    for item in content:
        for key in ("json", "data", "value"):
            value = getattr(item, key, None)
            if isinstance(value, dict):
                return value
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        raw_text = text.strip()
        if not raw_text:
            continue
        try:
            decoded = json.loads(raw_text)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def _validate_identity_snapshot_arguments(arguments: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_keys = {
        "app_identity_id",
        "detection_reference",
        "include_history",
        "max_history_entries",
        "max_candidates",
        "min_source_generation",
        "min_source_version",
        "stale_after_seconds",
        "investigation_id",
        "correlation_id",
    }
    unknown = [key for key in arguments.keys() if key not in allowed_keys]
    if unknown:
        errors.append("identity_snapshot_unknown_arguments")

    app_identity_id = _nonempty_str(arguments.get("app_identity_id"))
    detection_reference = _nonempty_str(arguments.get("detection_reference"))
    if int(app_identity_id is not None) + int(detection_reference is not None) != 1:
        errors.append("identity_snapshot_selector_required")

    include_history = arguments.get("include_history")
    if include_history is not None and not isinstance(include_history, bool):
        errors.append("identity_snapshot_include_history_invalid")

    for key in ("max_history_entries", "max_candidates"):
        value = arguments.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value <= 0:
            errors.append(f"identity_snapshot_{key}_invalid")

    for key in ("min_source_generation", "min_source_version"):
        value = arguments.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value < 0:
            errors.append(f"identity_snapshot_{key}_invalid")

    stale_after_seconds = arguments.get("stale_after_seconds")
    if stale_after_seconds is not None and (
        not isinstance(stale_after_seconds, (int, float)) or float(stale_after_seconds) <= 0
    ):
        errors.append("identity_snapshot_stale_after_seconds_invalid")

    for key in ("investigation_id", "correlation_id"):
        value = arguments.get(key)
        if value is None:
            continue
        if _nonempty_str(value) is None:
            errors.append(f"identity_snapshot_{key}_invalid")

    return sorted(set(errors))


def _validate_track_evidence_history_arguments(arguments: dict[str, Any]) -> list[str]:
    validation = validate_track_evidence_history_arguments(
        arguments,
        require_track_id=True,
        validate_track_id_value=True,
    )
    errors = validation.get("errors")
    return list(errors) if isinstance(errors, list) else []


def _validate_tracking_decision_request_for_dispatch(request_event: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tool_name = _nonempty_str(request_event.get("tool_name"))
    arguments = request_event.get("arguments")

    if tool_name not in TRACKING_DECISION_TOOL_ALLOWLIST:
        errors.append("tool_name_not_allowed")
    if not isinstance(arguments, dict):
        errors.append("arguments_invalid")
        arguments = {}

    if tool_name == "get_identity_snapshot":
        errors.extend(_validate_identity_snapshot_arguments(arguments))
    elif tool_name == "get_track_evidence_history":
        errors.extend(_validate_track_evidence_history_arguments(arguments))

    request_id = _nonempty_str(request_event.get("request_id"))
    investigation_id = _nonempty_str(request_event.get("investigation_id"))
    iteration_index = request_event.get("iteration_index")
    if request_id is None:
        errors.append("request_id_required")
    if investigation_id is None:
        errors.append("investigation_id_required")
    if not isinstance(iteration_index, int) or iteration_index < 1 or iteration_index > 2:
        errors.append("iteration_index_invalid")

    return sorted(set(errors))


def _classify_tracking_decision_tool_result_status(
    *,
    tool_name: str,
    payload: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    if tool_name == "get_identity_snapshot":
        available = bool(payload.get("available", False))
        state = _nonempty_str(payload.get("state"))
        stale_snapshot = bool(payload.get("stale_snapshot", False))
        if available:
            return "ok", None, None
        if state == "incompatible_snapshot":
            return "incompatible", "snapshot_incompatible", "snapshot_incompatible_with_minimum_requirements"
        if state == "stale" or stale_snapshot:
            return "stale", "snapshot_stale", "identity_snapshot_stale"
        return "unavailable", _nonempty_str(payload.get("error")) or "identity_snapshot_unavailable", "identity_snapshot_unavailable"

    if tool_name == "get_track_evidence_history":
        available = bool(payload.get("available", False))
        if available:
            return "ok", None, None
        return "unavailable", _nonempty_str(payload.get("error")) or "track_evidence_unavailable", "track_evidence_unavailable"

    return "rejected", "tool_name_not_allowed", "tool_name_not_allowed"


async def _run_tracking_decision_ipc_mcp_cycle(
    *,
    session: ClientSession,
    transport: TrackingDecisionIpcTransport,
    max_requests_per_cycle: int,
    tool_timeout_seconds: float,
) -> dict[str, Any]:
    pending = transport.orchestration_poll_pending_requests(
        max_requests=max_requests_per_cycle,
    )
    requests = pending.get("pending_requests")
    if not isinstance(requests, list):
        requests = []

    published_count = 0
    duplicate_count = 0
    rejected_count = 0
    timeout_count = 0
    processed_investigations: set[str] = set()

    def _persist_tool_budget_exhaustion_if_needed(
        *,
        investigation_id: str | None,
        reason: str,
    ) -> None:
        if investigation_id is None:
            return
        try:
            transport.mark_tool_budget_exhausted_investigation(
                investigation_id=investigation_id,
                reason=reason,
            )
        except Exception:
            pass

    for request in requests:
        request_id = _nonempty_str(request.get("request_id"))
        investigation_id = _nonempty_str(request.get("investigation_id"))
        tool_name = _nonempty_str(request.get("tool_name"))

        validation = transport.validate_pending_request_for_dispatch(request)
        validation_errors = list(validation.get("errors", []))
        if validation_errors:
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="rejected",
                result_payload={"available": False, "errors": validation_errors},
                error_code="invalid_request_event",
                error="invalid_request_event",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            elif bool(publish.get("published", False)):
                rejected_count += 1
            continue

        request_errors = _validate_tracking_decision_request_for_dispatch(request)
        if request_errors:
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="rejected",
                result_payload={"available": False, "errors": request_errors},
                error_code="invalid_tool_arguments",
                error="invalid_tool_arguments",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            elif bool(publish.get("published", False)):
                rejected_count += 1
            continue

        if investigation_id is not None and investigation_id in processed_investigations:
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="rejected",
                result_payload={
                    "available": False,
                    "error": "max_one_tool_call_per_investigation_per_cycle",
                },
                error_code="per_investigation_call_limit",
                error="max_one_tool_call_per_investigation_per_cycle",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            elif bool(publish.get("published", False)):
                rejected_count += 1
            continue

        if investigation_id is not None:
            processed_investigations.add(investigation_id)

        if request_id is None or tool_name is None:
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="rejected",
                result_payload={"available": False, "error": "invalid_request_identity"},
                error_code="invalid_request_identity",
                error="invalid_request_identity",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            elif bool(publish.get("published", False)):
                rejected_count += 1
            continue

        expires_at = request.get("expires_at")
        if isinstance(expires_at, (int, float)) and float(expires_at) <= time.time():
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="expired",
                result_payload={"available": False, "error": "request_expired_before_dispatch"},
                error_code="request_expired",
                error="request_expired_before_dispatch",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            elif bool(publish.get("published", False)):
                rejected_count += 1
            continue

        started_at = time.time()
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=dict(request.get("arguments") or {})),
                timeout=tool_timeout_seconds,
            )
        except asyncio.TimeoutError:
            timeout_count += 1
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="timeout",
                result_payload={
                    "available": False,
                    "error": "tool_timeout",
                    "timeout_seconds": tool_timeout_seconds,
                },
                error_code="tool_timeout",
                error="tool_timeout",
                now_timestamp=time.time(),
            )
            _persist_tool_budget_exhaustion_if_needed(
                investigation_id=investigation_id,
                reason="tool_timeout",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            continue
        except Exception as error:
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="unavailable",
                result_payload={
                    "available": False,
                    "error": "tool_execution_failed",
                    "exception": type(error).__name__,
                },
                error_code="tool_execution_failed",
                error="tool_execution_failed",
                now_timestamp=time.time(),
            )
            _persist_tool_budget_exhaustion_if_needed(
                investigation_id=investigation_id,
                reason="tool_execution_failed",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            elif bool(publish.get("published", False)):
                rejected_count += 1
            continue

        payload = _extract_mcp_result_payload(result)
        finished_at = time.time()
        if not isinstance(payload, dict):
            publish = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="rejected",
                result_payload={"available": False, "error": "malformed_tool_response"},
                error_code="malformed_tool_response",
                error="malformed_tool_response",
                now_timestamp=finished_at,
            )
            _persist_tool_budget_exhaustion_if_needed(
                investigation_id=investigation_id,
                reason="malformed_tool_response",
            )
            if bool(publish.get("duplicate", False)):
                duplicate_count += 1
            elif bool(publish.get("published", False)):
                rejected_count += 1
            continue

        status, error_code, error_message = _classify_tracking_decision_tool_result_status(
            tool_name=tool_name,
            payload=payload,
        )
        publish = transport.orchestration_publish_fixture_result(
            request_event=request,
            status=status,
            result_payload={
                **payload,
                "request_id": request_id,
                "tool_name": tool_name,
                "started_at": started_at,
                "finished_at": finished_at,
                "execution_mode": "bounded_mcp_dispatch",
            },
            error_code=error_code,
            error=error_message,
            now_timestamp=finished_at,
        )
        if bool(publish.get("duplicate", False)):
            duplicate_count += 1
        elif bool(publish.get("published", False)):
            published_count += 1
            terminal_reason = error_code or status
            _persist_tool_budget_exhaustion_if_needed(
                investigation_id=investigation_id,
                reason=terminal_reason,
            )

    return {
        "pending_count": int(pending.get("pending_count", 0)),
        "published_count": published_count,
        "duplicate_count": duplicate_count,
        "expired_count": int(pending.get("expired_count", 0)),
        "terminal_rejections": int(pending.get("terminal_rejections", 0)),
        "timeout_count": timeout_count,
        "rejected_count": rejected_count,
        "execution": "bounded_mcp_dispatch",
        "tool_timeout_seconds": tool_timeout_seconds,
    }

def load_config(args: list[str] | None = None) -> runtime_app.RuntimeConfig:
    """Build runtime configuration from CLI args and environment defaults."""
    raw_args = args if args is not None else sys.argv[1:]
    options = parse_cli_args(raw_args)
    return runtime_app.RuntimeConfig(
        show_startup_banner=not options.quiet_startup_banner,
        show_startup_depth_snapshot=not options.no_startup_depth_snapshot,
        show_startup_pipeline_contract_snapshot=True,
        enable_agent_ops_heartbeat=not options.disable_agent_ops_heartbeat,
        agent_ops_interval_seconds=options.agent_ops_interval_seconds,
        agent_ops_max_tracks=options.agent_ops_max_tracks,
        agent_ops_summary_every_cycles=options.agent_ops_summary_every_cycles,
        agent_ops_summary_window_size=options.agent_ops_summary_window_size,
        handoff_degraded_warn_rate=options.handoff_degraded_warn_rate,
        handoff_degraded_critical_rate=options.handoff_degraded_critical_rate,
        depth_coverage_summary_enabled=not options.disable_depth_coverage_summary,
        depth_coverage_summary_every_cycles=options.depth_coverage_summary_every_cycles,
        depth_coverage_sample_limit=options.depth_coverage_sample_limit,
        video_path=options.video_path,
    )


async def initialize_runtime_services(session: ClientSession, config: runtime_app.RuntimeConfig) -> None:
    """Run startup checks and activate required runtime services/tools."""
    async def start_tracking(session: ClientSession) -> None:
        await start_live_tracking(session, video_path=config.video_path)

    await runtime_flow.start_runtime_services(
        session,
        config,
        format_runtime_config_banner_fn=format_runtime_config_banner,
        print_depth_coverage_startup_snapshot_fn=print_depth_coverage_startup_snapshot,
        print_pipeline_contract_startup_snapshot_fn=print_pipeline_contract_startup_snapshot,
        verify_required_tools_fn=verify_required_tools,
        start_live_tracking_fn=start_tracking,
        start_orchestration_agents_fn=start_orchestration_agents,
        emit=lambda message: print(message, flush=True),
    )


async def run_operator_loop(session: ClientSession, config: runtime_app.RuntimeConfig) -> None:
    """Execute the long-running operator monitoring loop."""
    await runtime_flow.run_operator_loop(
        session,
        config,
        monitor_movement_fn=monitor_movement,
    )


def _resolve_handoff_severity_thresholds_from_env() -> tuple[float, float]:
    """Resolve handoff degradation thresholds from environment variables."""
    return runtime_config.resolve_handoff_severity_thresholds(
        os.getenv("HANDOFF_DEGRADED_WARN_RATE"),
        os.getenv("HANDOFF_DEGRADED_CRITICAL_RATE"),
        default_warn_rate=runtime_agent.HANDOFF_DEGRADED_WARN_RATE,
        default_critical_rate=runtime_agent.HANDOFF_DEGRADED_CRITICAL_RATE,
    )


def parse_cli_args(args: list[str]) -> argparse.Namespace:
    """Parse and validate CLI controls for orchestration runtime behavior."""
    env_warn_rate, env_critical_rate = _resolve_handoff_severity_thresholds_from_env()
    return runtime_config.parse_cli_args(
        args,
        description="Run object behavior MCP agent",
        agent_ops_interval_seconds_default=runtime_agent.AGENT_OPS_INTERVAL_SECONDS,
        agent_ops_max_tracks_default=runtime_agent.AGENT_OPS_MAX_TRACKS,
        agent_ops_summary_every_cycles_default=runtime_agent.AGENT_OPS_SUMMARY_EVERY_CYCLES,
        agent_ops_summary_window_size_default=runtime_agent.AGENT_OPS_SUMMARY_WINDOW_SIZE,
        depth_coverage_summary_every_cycles_default=runtime_agent.DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
        depth_coverage_sample_limit_default=runtime_agent.DEPTH_COVERAGE_SAMPLE_LIMIT,
        handoff_degraded_warn_rate_default=env_warn_rate,
        handoff_degraded_critical_rate_default=env_critical_rate,
    )


async def _run_agent_ops_pipeline(
    session: ClientSession,
    *,
    cycle_number: int,
    max_tracks: int,
    agent_ops_summary_enabled: bool,
    agent_ops_summary_every_cycles: int,
    agent_ops_summary_window_size: int,
    handoff_degraded_warn_rate: float,
    handoff_degraded_critical_rate: float,
    depth_summary_enabled: bool,
    depth_coverage_summary_every_cycles: int,
    depth_coverage_sample_limit: int,
) -> dict[str, Any]:
    """Run one agent-ops heartbeat cycle plus optional summary follow-ups."""
    return await runtime_agent.run_agent_ops_pipeline(
        session,
        cycle_number=cycle_number,
        max_tracks=max_tracks,
        agent_ops_summary_enabled=agent_ops_summary_enabled,
        agent_ops_summary_every_cycles=agent_ops_summary_every_cycles,
        agent_ops_summary_window_size=agent_ops_summary_window_size,
        handoff_degraded_warn_rate=handoff_degraded_warn_rate,
        handoff_degraded_critical_rate=handoff_degraded_critical_rate,
        depth_summary_enabled=depth_summary_enabled,
        depth_coverage_summary_every_cycles=depth_coverage_summary_every_cycles,
        depth_coverage_sample_limit=depth_coverage_sample_limit,
        run_cycle_fn=run_agent_ops_cycle,
        run_health_summary_fn=run_agent_ops_health_summary,
        run_depth_summary_fn=run_depth_coverage_summary,
    )


async def monitor_movement(
    session: ClientSession,
    enable_agent_ops_heartbeat: bool = True,
    agent_ops_interval_seconds: float = runtime_agent.AGENT_OPS_INTERVAL_SECONDS,
    agent_ops_max_tracks: int = runtime_agent.AGENT_OPS_MAX_TRACKS,
    agent_ops_summary_every_cycles: int = runtime_agent.AGENT_OPS_SUMMARY_EVERY_CYCLES,
    agent_ops_summary_window_size: int = runtime_agent.AGENT_OPS_SUMMARY_WINDOW_SIZE,
    handoff_degraded_warn_rate: float = runtime_agent.HANDOFF_DEGRADED_WARN_RATE,
    handoff_degraded_critical_rate: float = runtime_agent.HANDOFF_DEGRADED_CRITICAL_RATE,
    depth_coverage_summary_enabled: bool = True,
    depth_coverage_summary_every_cycles: int = runtime_agent.DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
    depth_coverage_sample_limit: int = runtime_agent.DEPTH_COVERAGE_SAMPLE_LIMIT,
    enable_visual_verification_queue: bool = True,
    verification_request_source_fn: Any | None = None,
) -> None:
    """Run movement analysis loop and trigger periodic agent-ops heartbeats."""
    monitor_impl = runtime_agent.make_monitor_movement(
        run_agent_ops_pipeline_fn=_run_agent_ops_pipeline,
        analysis_interval_seconds=ANALYSIS_INTERVAL_SECONDS,
    )
    run_id = os.getenv("PHASE16E_RUN_ID", "").strip() or None
    results_path, submissions_path, feedback_path, identity_intents_path, identity_intents_ack_ledger_path = _resolve_verification_artifact_paths()

    def persist_results(results: list[dict[str, Any]]) -> None:
        _persist_json_artifact(results_path, results)

    def persist_submissions(submissions: list[dict[str, Any]]) -> None:
        _persist_json_artifact(submissions_path, submissions)

    verification_manager: VerificationQueueManager | None = None
    verification_feedback_publisher: VerificationFeedbackPublisher | None = None
    identity_reassociation_intent_publisher: IdentityReassociationIntentPublisher | None = None
    tracking_decision_ipc_transport: TrackingDecisionIpcTransport | None = None
    identity_reassociation_intent_enabled = _env_flag_enabled(
        "IDENTITY_REASSOCIATION_INTENT_ENABLED",
        False,
    )
    tracking_decision_ipc_enabled = _env_flag_enabled(
        "PHASE2D_TRACKING_DECISION_IPC_ENABLED",
        False,
    )
    tracking_decision_ipc_max_requests_per_cycle = _resolve_tracking_decision_ipc_max_requests_per_cycle(8)
    tracking_decision_ipc_tool_timeout_seconds = _resolve_tracking_decision_ipc_tool_timeout_seconds(1.5)
    if enable_visual_verification_queue:
        verification_manager = VerificationQueueManager(
            evidence_root_dir=PROJECT_ROOT / "captures" / "tracking_evidence",
            run_id=run_id,
            queue_capacity=VERIFICATION_QUEUE_CAPACITY,
            max_concurrency=VERIFICATION_MAX_CONCURRENCY,
            request_timeout_seconds=VERIFICATION_REQUEST_TIMEOUT_SECONDS,
            result_registry_capacity=VERIFICATION_RESULTS_CAPACITY,
        )
        verification_feedback_publisher = VerificationFeedbackPublisher(
            artifact_path=feedback_path,
            max_records=256,
            record_ttl_seconds=45.0,
        )
        if identity_reassociation_intent_enabled:
            identity_reassociation_intent_publisher = IdentityReassociationIntentPublisher(
                artifact_path=identity_intents_path,
                ack_ledger_path=identity_intents_ack_ledger_path,
                evidence_root_dir=PROJECT_ROOT / "captures" / "tracking_evidence",
                max_intents=256,
                max_decisions=1024,
                intent_ttl_seconds=45.0,
            )
        await verification_manager.start(session=session)

    if tracking_decision_ipc_enabled:
        (
            tracking_decision_requests_path,
            tracking_decision_results_path,
            tracking_decision_request_index_path,
            tracking_decision_result_index_path,
        ) = _resolve_tracking_decision_ipc_artifact_paths()
        tracking_decision_ipc_transport = TrackingDecisionIpcTransport(
            requests_path=tracking_decision_requests_path,
            results_path=tracking_decision_results_path,
            request_index_path=tracking_decision_request_index_path,
            result_index_path=tracking_decision_result_index_path,
            writer_role="orchestration",
        )

    def publish_verification_outputs(recent_results: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if verification_feedback_publisher is not None:
            summary["feedback"] = verification_feedback_publisher.publish_from_results(recent_results)
        if identity_reassociation_intent_publisher is not None:
            summary["identity_reassociation_intents"] = identity_reassociation_intent_publisher.publish_from_results(
                recent_results
            )
        return summary

    async def run_tracking_decision_ipc_mcp_cycle() -> dict[str, Any]:
        if tracking_decision_ipc_transport is None:
            return {
                "pending_count": 0,
                "published_count": 0,
                "duplicate_count": 0,
                "expired_count": 0,
                "terminal_rejections": 0,
            }
        return await _run_tracking_decision_ipc_mcp_cycle(
            session=session,
            transport=tracking_decision_ipc_transport,
            max_requests_per_cycle=tracking_decision_ipc_max_requests_per_cycle,
            tool_timeout_seconds=tracking_decision_ipc_tool_timeout_seconds,
        )

    try:
        monitor_kwargs: dict[str, Any] = {
            "enable_agent_ops_heartbeat": enable_agent_ops_heartbeat,
            "agent_ops_interval_seconds": agent_ops_interval_seconds,
            "agent_ops_max_tracks": agent_ops_max_tracks,
            "agent_ops_summary_every_cycles": agent_ops_summary_every_cycles,
            "agent_ops_summary_window_size": agent_ops_summary_window_size,
            "handoff_degraded_warn_rate": handoff_degraded_warn_rate,
            "handoff_degraded_critical_rate": handoff_degraded_critical_rate,
            "depth_coverage_summary_enabled": depth_coverage_summary_enabled,
            "depth_coverage_summary_every_cycles": depth_coverage_summary_every_cycles,
            "depth_coverage_sample_limit": depth_coverage_sample_limit,
            "verification_manager": verification_manager,
            "verification_request_source_fn": verification_request_source_fn,
            "verification_results_persist_fn": (
                None
                if verification_manager is None
                else persist_results
            ),
            "verification_submissions_persist_fn": (
                None
                if verification_manager is None
                else persist_submissions
            ),
            "verification_feedback_publish_fn": (
                None
                if verification_feedback_publisher is None and identity_reassociation_intent_publisher is None
                else publish_verification_outputs
            ),
        }
        if "decision_ipc_fixture_cycle_fn" in inspect.signature(monitor_impl).parameters:
            monitor_kwargs["decision_ipc_fixture_cycle_fn"] = (
                run_tracking_decision_ipc_mcp_cycle
                if tracking_decision_ipc_enabled
                else None
            )
        await monitor_impl(session, **monitor_kwargs)
    finally:
        if verification_manager is not None:
            await verification_manager.stop()


async def run_agent(
    show_startup_banner: bool = True,
    show_startup_depth_snapshot: bool = True,
    show_startup_pipeline_contract_snapshot: bool = True,
    enable_agent_ops_heartbeat: bool = True,
    agent_ops_interval_seconds: float = runtime_agent.AGENT_OPS_INTERVAL_SECONDS,
    agent_ops_max_tracks: int = runtime_agent.AGENT_OPS_MAX_TRACKS,
    agent_ops_summary_every_cycles: int = runtime_agent.AGENT_OPS_SUMMARY_EVERY_CYCLES,
    agent_ops_summary_window_size: int = runtime_agent.AGENT_OPS_SUMMARY_WINDOW_SIZE,
    handoff_degraded_warn_rate: float = runtime_agent.HANDOFF_DEGRADED_WARN_RATE,
    handoff_degraded_critical_rate: float = runtime_agent.HANDOFF_DEGRADED_CRITICAL_RATE,
    depth_coverage_summary_enabled: bool = True,
    depth_coverage_summary_every_cycles: int = runtime_agent.DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
    depth_coverage_sample_limit: int = runtime_agent.DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> None:
    """Public runtime entry for tests/scripts using explicit startup options."""
    await runtime_agent.run_agent(
        run_with_runtime_config_fn=run_system_with_config,
        show_startup_banner=show_startup_banner,
        show_startup_depth_snapshot=show_startup_depth_snapshot,
        show_startup_pipeline_contract_snapshot=show_startup_pipeline_contract_snapshot,
        enable_agent_ops_heartbeat=enable_agent_ops_heartbeat,
        agent_ops_interval_seconds=agent_ops_interval_seconds,
        agent_ops_max_tracks=agent_ops_max_tracks,
        agent_ops_summary_every_cycles=agent_ops_summary_every_cycles,
        agent_ops_summary_window_size=agent_ops_summary_window_size,
        handoff_degraded_warn_rate=handoff_degraded_warn_rate,
        handoff_degraded_critical_rate=handoff_degraded_critical_rate,
        depth_coverage_summary_enabled=depth_coverage_summary_enabled,
        depth_coverage_summary_every_cycles=depth_coverage_summary_every_cycles,
        depth_coverage_sample_limit=depth_coverage_sample_limit,
    )


async def run_system_with_config(config: runtime_app.RuntimeConfig) -> None:
    """Connect MCP runtime, wire agents/coordinator, run, and ensure cleanup."""
    logger = runtime_app.configure_logging(config)

    services = await runtime_app.initialize_services(
        config,
        logger,
        server_path=SERVER_PATH,
        python_executable=sys.executable,
        stdio_client_fn=stdio_client,
        client_session_cls=ClientSession,
        startup_runner=initialize_runtime_services,
    )

    coordinator = runtime_app.RuntimeLoopCoordinator(
        tracking_agent=services.tracking_agent,
        observation_agent=services.observation_agent,
        reasoning_agent=services.reasoning_agent,
        services=services,
        run_loop=run_operator_loop,
    )

    operator = runtime_app.OrchestrationAgent(
        coordinator=coordinator,
        observation_source=services.observation_source,
    )

    try:
        await services.start()
        await operator.run()
    finally:
        operator_stop_error: Exception | None = None
        services_stop_error: Exception | None = None

        try:
            await operator.stop()
        except Exception as error:  # pragma: no cover - exercised via lifecycle tests
            operator_stop_error = error

        try:
            await services.stop()
        except Exception as error:  # pragma: no cover - exercised via lifecycle tests
            services_stop_error = error

        if operator_stop_error is not None:
            raise operator_stop_error
        if services_stop_error is not None:
            raise services_stop_error


async def main(args: list[str] | None = None) -> None:
    """Resolve configuration and start the orchestrated runtime system."""
    config = load_config(args)
    await run_system_with_config(config)


def cli_main(args: list[str] | None = None) -> None:
    """Synchronous CLI wrapper for async runtime startup and shutdown handling."""
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nStopping movement analysis.", flush=True)
    except Exception as error:
        print(
            f"Startup failed ({type(error).__name__}): {error}",
            flush=True,
        )
        raise SystemExit(1) from error

def user_main() -> None:
    """Choose webcam or video-file input before starting the runtime."""
    args = sys.argv[1:]
    if "--video-path" in args:
        cli_main(args)
        return

    print("Type 0 for webcam, 1 for video:")
    try:
        choice = input().strip()
    except EOFError:
        print(
            "Error: No input available. Use --video-path <path> or run "
            "interactively to choose webcam or video."
        )
        sys.exit(2)

    if choice == "1":
        print("Enter the path to the video file:")
        try:
            video_path = input().strip()
        except EOFError:
            print("Error: No video path was provided.")
            sys.exit(2)

        video_file = Path(video_path).expanduser()
        if not video_file.is_file():
            print(f"Error: File '{video_path}' does not exist.")
            sys.exit(1)
        cli_main(["--video-path", str(video_file), *args])
    elif choice == "0":
        cli_main(args)
    else:
        print(f"Error: Invalid input choice '{choice}'. Enter 0 or 1.")
        sys.exit(2)


if __name__ == "__main__":
    user_main()