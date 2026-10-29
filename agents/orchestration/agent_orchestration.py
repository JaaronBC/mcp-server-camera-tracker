# LLM-oriented orchestration helpers for startup activation and periodic agent-ops cycles.
from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from mcp import ClientSession

AGENT_OPS_CYCLE_TOOL = "run_reasoning_cycle_and_log"
AGENT_OPS_HEALTH_SUMMARY_TOOL = "summarize_agent_event_health"
DEPTH_COVERAGE_SUMMARY_TOOL = "dashboard_depth_coverage_metrics"
PIPELINE_CONTRACT_VERIFY_TOOL = "verify_dashboard_pipeline_contract"
DASHBOARD_HEALTH_STATUS_TOOL = "dashboard_health_status"


def get_result_text(result: Any) -> str:
    # Gets result text values for downstream consumers.
    return "\n".join(
        item.text
        for item in (result.content or [])
        if hasattr(item, "text") and isinstance(item.text, str)
    )


def get_result_json_object(result: Any) -> dict[str, Any] | None:
    # Gets result json object values for downstream consumers.
    for item in (result.content or []):
        for attribute in ("json", "data", "value"):
            payload = getattr(item, attribute, None)
            if isinstance(payload, dict):
                return payload

        if hasattr(item, "text") and isinstance(item.text, str):
            text = item.text.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

    return None


def _emit_message(message: str, emit: Callable[[str], None] | None) -> None:
    # Emits user-facing runtime messages through provided sink.
    if emit is None:
        print(message, flush=True)
    else:
        emit(message)


async def run_agent_ops_cycle(
    session: ClientSession,
    *,
    request_evidence_on_uncertainty: bool = True,
    max_tracks: int = 10,
    tool_name: str = AGENT_OPS_CYCLE_TOOL,
    result_text_fn: Callable[[Any], str] = get_result_text,
    result_json_fn: Callable[[Any], dict[str, Any] | None] = get_result_json_object,
    format_cycle_fn: Callable[[dict[str, Any]], str] | None = None,
    format_evidence_fn: Callable[[dict[str, Any]], list[str]] | None = None,
    emit: Callable[[str], None] | None = None,
) -> bool:
    # Runs one observation/reasoning cycle and reports availability status.
    try:
        result = await session.call_tool(
            tool_name,
            arguments={
                "request_evidence_on_uncertainty": request_evidence_on_uncertainty,
                "max_tracks": max_tracks,
            },
        )
    except Exception as error:
        message = str(error).lower()
        _emit_message(
            f"Agent ops cycle failed ({type(error).__name__}): {error}",
            emit,
        )
        if "not registered" in message or "unknown tool" in message or "not found" in message:
            _emit_message("Disabling agent ops heartbeat: tool unavailable.", emit)
            return False
        return True

    message = result_text_fn(result)
    if getattr(result, "is_error", False):
        _emit_message(f"Agent ops cycle returned error: {message}", emit)
        lowered = (message or "").lower()
        if "not registered" in lowered or "unknown tool" in lowered or "not found" in lowered:
            _emit_message("Disabling agent ops heartbeat: tool unavailable.", emit)
            return False
        return True

    payload = result_json_fn(result)
    if payload is None:
        if message:
            _emit_message(f"Agent ops cycle: {message}", emit)
        else:
            _emit_message("Agent ops cycle completed with no structured payload.", emit)
        return True

    if format_cycle_fn is None:
        _emit_message("Agent ops cycle completed with structured payload.", emit)
    else:
        _emit_message(format_cycle_fn(payload), emit)

    if format_evidence_fn is not None:
        for evidence_line in format_evidence_fn(payload):
            _emit_message(evidence_line, emit)
    return True


async def run_agent_ops_health_summary(
    session: ClientSession,
    *,
    window_size: int = 100,
    handoff_warn_rate: float = 0.1,
    handoff_critical_rate: float = 0.3,
    tool_name: str = AGENT_OPS_HEALTH_SUMMARY_TOOL,
    result_text_fn: Callable[[Any], str] = get_result_text,
    result_json_fn: Callable[[Any], dict[str, Any] | None] = get_result_json_object,
    format_health_fn: Callable[[dict[str, Any], float, float], str] | None = None,
    emit: Callable[[str], None] | None = None,
) -> bool:
    # Runs one heartbeat-health summary and reports availability status.
    try:
        result = await session.call_tool(
            tool_name,
            arguments={"window_size": window_size},
        )
    except Exception as error:
        message = str(error).lower()
        _emit_message(
            f"Agent ops health summary failed ({type(error).__name__}): {error}",
            emit,
        )
        if "not registered" in message or "unknown tool" in message or "not found" in message:
            _emit_message("Disabling agent ops health summary: tool unavailable.", emit)
            return False
        return True

    message = result_text_fn(result)
    if getattr(result, "is_error", False):
        _emit_message(f"Agent ops health summary returned error: {message}", emit)
        lowered = (message or "").lower()
        if "not registered" in lowered or "unknown tool" in lowered or "not found" in lowered:
            _emit_message("Disabling agent ops health summary: tool unavailable.", emit)
            return False
        return True

    payload = result_json_fn(result)
    if payload is None:
        if message:
            _emit_message(f"Agent ops health summary: {message}", emit)
        else:
            _emit_message("Agent ops health summary completed with no structured payload.", emit)
        return True

    if format_health_fn is None:
        _emit_message("Agent ops health summary completed.", emit)
    else:
        _emit_message(
            format_health_fn(
                payload,
                handoff_warn_rate,
                handoff_critical_rate,
            ),
            emit,
        )
    return True


async def run_depth_coverage_summary(
    session: ClientSession,
    *,
    sample_limit: int = 50,
    tool_name: str = DEPTH_COVERAGE_SUMMARY_TOOL,
    result_text_fn: Callable[[Any], str] = get_result_text,
    result_json_fn: Callable[[Any], dict[str, Any] | None] = get_result_json_object,
    format_depth_fn: Callable[[dict[str, Any]], str] | None = None,
    emit: Callable[[str], None] | None = None,
) -> bool:
    # Runs one depth-coverage summary and reports availability status.
    try:
        result = await session.call_tool(
            tool_name,
            arguments={"sample_limit": sample_limit},
        )
    except Exception as error:
        message = str(error).lower()
        _emit_message(
            f"Depth coverage summary failed ({type(error).__name__}): {error}",
            emit,
        )
        if "not registered" in message or "unknown tool" in message or "not found" in message:
            _emit_message("Disabling depth coverage summary: tool unavailable.", emit)
            return False
        return True

    message = result_text_fn(result)
    if getattr(result, "is_error", False):
        _emit_message(f"Depth coverage summary returned error: {message}", emit)
        lowered = (message or "").lower()
        if "not registered" in lowered or "unknown tool" in lowered or "not found" in lowered:
            _emit_message("Disabling depth coverage summary: tool unavailable.", emit)
            return False
        return True

    payload = result_json_fn(result)
    if payload is None:
        if message:
            _emit_message(f"Depth coverage summary: {message}", emit)
        else:
            _emit_message("Depth coverage summary completed with no structured payload.", emit)
        return True

    if format_depth_fn is None:
        _emit_message("Depth coverage summary completed.", emit)
    else:
        _emit_message(format_depth_fn(payload), emit)
    return True


async def run_agent_ops_pipeline(
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
    run_cycle_fn: Callable[..., Awaitable[bool]],
    run_health_summary_fn: Callable[..., Awaitable[bool]],
    run_depth_summary_fn: Callable[..., Awaitable[bool]],
) -> dict[str, Any]:
    # Runs one full periodic orchestration pass and tracks enabled state transitions.
    result = {
        "agent_ops_enabled": True,
        "agent_ops_summary_enabled": agent_ops_summary_enabled,
        "depth_summary_enabled": depth_summary_enabled,
        "completed_cycle_increment": 0,
    }

    agent_ops_enabled = await run_cycle_fn(
        session,
        max_tracks=max_tracks,
    )
    result["agent_ops_enabled"] = agent_ops_enabled
    if not agent_ops_enabled:
        return result

    result["completed_cycle_increment"] = 1

    if (
        agent_ops_summary_enabled
        and agent_ops_summary_every_cycles > 0
        and cycle_number % agent_ops_summary_every_cycles == 0
    ):
        result["agent_ops_summary_enabled"] = await run_health_summary_fn(
            session,
            window_size=agent_ops_summary_window_size,
            handoff_warn_rate=handoff_degraded_warn_rate,
            handoff_critical_rate=handoff_degraded_critical_rate,
        )

    if (
        depth_summary_enabled
        and depth_coverage_summary_every_cycles > 0
        and cycle_number % depth_coverage_summary_every_cycles == 0
    ):
        result["depth_summary_enabled"] = await run_depth_summary_fn(
            session,
            sample_limit=depth_coverage_sample_limit,
        )

    return result


async def start_orchestration_agents(
    session: ClientSession,
    *,
    enable_agent_ops_heartbeat: bool,
    agent_ops_max_tracks: int,
    run_cycle_fn: Callable[..., Awaitable[bool]],
    emit: Callable[[str], None] | None = None,
) -> bool:
    # Performs startup activation pass to initialize observation and reasoning agents.
    if not enable_agent_ops_heartbeat:
        return False

    _emit_message(
        "Starting orchestration agents (observation + reasoning) with startup cycle...",
        emit,
    )
    return await run_cycle_fn(
        session,
        max_tracks=agent_ops_max_tracks,
    )


async def start_orchestration_agents_with_activation_checks(
    session: ClientSession,
    *,
    enable_agent_ops_heartbeat: bool = True,
    agent_ops_max_tracks: int = 10,
    max_attempts: int = 5,
    retry_seconds: float = 0.5,
    dashboard_health_status_tool: str = DASHBOARD_HEALTH_STATUS_TOOL,
    agent_ops_cycle_tool: str = AGENT_OPS_CYCLE_TOOL,
    result_text_fn: Callable[[Any], str] = get_result_text,
    result_json_fn: Callable[[Any], dict[str, Any] | None] = get_result_json_object,
    format_cycle_fn: Callable[[dict[str, Any]], str] | None = None,
    format_evidence_fn: Callable[[dict[str, Any]], list[str]] | None = None,
    emit: Callable[[str], None] | None = None,
) -> bool:
    # Performs startup activation checks for all collaborating agents.
    if not enable_agent_ops_heartbeat:
        return False

    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")
    if retry_seconds < 0:
        raise ValueError("retry_seconds must be zero or greater")

    _emit_message(
        "Starting orchestration agents with strict startup activation checks...",
        emit,
    )

    for attempt in range(1, max_attempts + 1):
        tracking_probe_ok = False
        try:
            health_result = await session.call_tool(
                dashboard_health_status_tool,
                arguments={},
            )
            health_payload = result_json_fn(health_result)
            if isinstance(health_payload, dict):
                available = bool(health_payload.get("available", False))
                track_count = health_payload.get("track_count")
                tracking_probe_ok = available and isinstance(track_count, int) and track_count > 0
        except Exception:
            tracking_probe_ok = False

        try:
            cycle_result = await session.call_tool(
                agent_ops_cycle_tool,
                arguments={
                    "request_evidence_on_uncertainty": True,
                    "max_tracks": agent_ops_max_tracks,
                },
            )
        except Exception as error:
            message = str(error).lower()
            _emit_message(
                f"Startup activation cycle failed ({type(error).__name__}): {error}",
                emit,
            )
            if "not registered" in message or "unknown tool" in message or "not found" in message:
                _emit_message("Startup activation aborted: agent-ops tool unavailable.", emit)
                return False
            if attempt < max_attempts and retry_seconds > 0:
                await asyncio.sleep(retry_seconds)
            continue

        cycle_message = result_text_fn(cycle_result)
        if getattr(cycle_result, "is_error", False):
            _emit_message(
                f"Startup activation cycle returned error: {cycle_message}",
                emit,
            )
            lowered = (cycle_message or "").lower()
            if "not registered" in lowered or "unknown tool" in lowered or "not found" in lowered:
                _emit_message("Startup activation aborted: agent-ops tool unavailable.", emit)
                return False
            if attempt < max_attempts and retry_seconds > 0:
                await asyncio.sleep(retry_seconds)
            continue

        payload = result_json_fn(cycle_result)
        if not isinstance(payload, dict):
            if cycle_message:
                _emit_message(f"Startup activation cycle: {cycle_message}", emit)
            else:
                _emit_message("Startup activation cycle returned no structured payload.", emit)
            if attempt < max_attempts and retry_seconds > 0:
                await asyncio.sleep(retry_seconds)
            continue

        if format_cycle_fn is not None:
            _emit_message(format_cycle_fn(payload), emit)
        if format_evidence_fn is not None:
            for evidence_line in format_evidence_fn(payload):
                _emit_message(evidence_line, emit)

        cycle = payload.get("cycle")
        evaluated_tracks = cycle.get("evaluated_tracks", 0) if isinstance(cycle, dict) else 0
        events = payload.get("events") if isinstance(payload.get("events"), list) else []
        observation_ok = any(
            isinstance(event, dict)
            and (
                event.get("observation_agent_status") in {"ok", "error", "skipped"}
                or isinstance(event.get("observation_agent_model"), str)
            )
            for event in events
        )
        reasoning_ok = any(
            isinstance(event, dict)
            and (
                event.get("reasoning_agent_status") in {"ok", "error", "skipped"}
                or isinstance(event.get("reasoning_agent_model"), str)
            )
            for event in events
        )
        orchestration_ok = isinstance(cycle, dict)
        tracking_ok = bool(
            tracking_probe_ok
            or (isinstance(evaluated_tracks, int) and evaluated_tracks > 0)
        )

        _emit_message(
            "Startup agent activation: "
            f"attempt={attempt}/{max_attempts} | "
            f"tracking={'ok' if tracking_ok else 'pending'} | "
            f"orchestration={'ok' if orchestration_ok else 'pending'} | "
            f"observation={'ok' if observation_ok else 'pending'} | "
            f"reasoning={'ok' if reasoning_ok else 'pending'}",
            emit,
        )

        if tracking_ok and orchestration_ok and observation_ok and reasoning_ok:
            _emit_message("All four agents are active and working together.", emit)
            return True

        if attempt < max_attempts and retry_seconds > 0:
            await asyncio.sleep(retry_seconds)

    _emit_message(
        "Startup activation incomplete: could not confirm all four agents within retry budget.",
        emit,
    )
    return False


async def print_depth_coverage_startup_snapshot(
    session: ClientSession,
    *,
    sample_limit: int = 50,
    tool_name: str = DEPTH_COVERAGE_SUMMARY_TOOL,
    result_text_fn: Callable[[Any], str] = get_result_text,
    result_json_fn: Callable[[Any], dict[str, Any] | None] = get_result_json_object,
    format_depth_fn: Callable[[dict[str, Any]], str] | None = None,
    emit: Callable[[str], None] | None = None,
) -> None:
    # Prints one startup depth-coverage snapshot for runtime observability.
    try:
        result = await session.call_tool(
            tool_name,
            arguments={"sample_limit": sample_limit},
        )
    except Exception as error:
        _emit_message(
            f"Depth coverage snapshot unavailable ({type(error).__name__}): {error}",
            emit,
        )
        return

    message = result_text_fn(result)
    if getattr(result, "is_error", False):
        _emit_message(
            f"Depth coverage snapshot unavailable: {message or 'unknown tool error'}",
            emit,
        )
        return

    payload = result_json_fn(result)
    if payload is None:
        if message:
            _emit_message(f"Depth coverage snapshot: {message}", emit)
        else:
            _emit_message("Depth coverage snapshot unavailable: no structured data.", emit)
        return

    if format_depth_fn is None:
        _emit_message("Startup depth coverage snapshot available.", emit)
    else:
        _emit_message(f"Startup {format_depth_fn(payload)}", emit)


async def print_pipeline_contract_startup_snapshot(
    session: ClientSession,
    *,
    max_observation_age_seconds: float = 3.0,
    require_live_pipeline: bool = False,
    tool_name: str = PIPELINE_CONTRACT_VERIFY_TOOL,
    result_text_fn: Callable[[Any], str] = get_result_text,
    result_json_fn: Callable[[Any], dict[str, Any] | None] = get_result_json_object,
    emit: Callable[[str], None] | None = None,
) -> None:
    # Prints one startup pipeline-contract verification snapshot for observability.
    try:
        result = await session.call_tool(
            tool_name,
            arguments={
                "max_observation_age_seconds": max_observation_age_seconds,
                "require_live_pipeline": require_live_pipeline,
            },
        )
    except Exception as error:
        _emit_message(
            f"Pipeline contract snapshot unavailable ({type(error).__name__}): {error}",
            emit,
        )
        return

    message = result_text_fn(result)
    if getattr(result, "is_error", False):
        _emit_message(
            f"Pipeline contract snapshot unavailable: {message or 'unknown tool error'}",
            emit,
        )
        return

    payload = result_json_fn(result)
    if payload is None:
        if message:
            _emit_message(f"Pipeline contract snapshot: {message}", emit)
        else:
            _emit_message("Pipeline contract snapshot unavailable: no structured data.", emit)
        return

    ready = bool(payload.get("ready"))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    error_count = summary.get("error_count", 0)
    warn_count = summary.get("warn_count", 0)
    status_text = "ready" if ready else "degraded"
    _emit_message(
        "Startup pipeline contract: "
        f"status={status_text} | errors={error_count} | warnings={warn_count}",
        emit,
    )
