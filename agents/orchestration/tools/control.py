# Handles dashboardControl behavior.
from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from mcp import ClientSession

from .. import agent_orchestration as orchestrationAgent
from .calibration import (
    set_ground_plane_calibration,
    validate_ground_plane_calibration,
)
from .health import dashboard_health_status, wait_for_dashboard_ready
from ..process import (
    dashboard_process_status,
    start_dashboard_process,
    stop_dashboard_process,
)
from ...registry import mcp
from ...tracking import agent_runtime as trackingAgent


async def verify_required_tools_runtime(
    session: ClientSession,
    *,
    live_tracking_tool: str,
    detect_objects_tool: str,
) -> None:
    await trackingAgent.verify_required_tools(
        session,
        live_tracking_tool=live_tracking_tool,
        detect_objects_tool=detect_objects_tool,
    )


async def start_live_tracking_runtime(
    session: ClientSession,
    *,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    emit: Callable[[str], None],
) -> None:
    await trackingAgent.start_live_tracking(
        session,
        tool_name=tool_name,
        result_text_fn=result_text_fn,
        emit=emit,
    )


async def run_agent_ops_cycle_runtime(
    session: ClientSession,
    *,
    request_evidence_on_uncertainty: bool,
    max_tracks: int,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_cycle_fn: Callable[[dict[str, Any]], str],
    format_evidence_fn: Callable[[dict[str, Any]], list[str]],
    emit: Callable[[str], None],
) -> bool:
    return await orchestrationAgent.run_agent_ops_cycle(
        session,
        request_evidence_on_uncertainty=request_evidence_on_uncertainty,
        max_tracks=max_tracks,
        tool_name=tool_name,
        result_text_fn=result_text_fn,
        result_json_fn=result_json_fn,
        format_cycle_fn=format_cycle_fn,
        format_evidence_fn=format_evidence_fn,
        emit=emit,
    )


async def run_agent_ops_health_summary_runtime(
    session: ClientSession,
    *,
    window_size: int,
    handoff_warn_rate: float,
    handoff_critical_rate: float,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_health_fn: Callable[[dict[str, Any], float, float], str],
    emit: Callable[[str], None],
) -> bool:
    return await orchestrationAgent.run_agent_ops_health_summary(
        session,
        window_size=window_size,
        handoff_warn_rate=handoff_warn_rate,
        handoff_critical_rate=handoff_critical_rate,
        tool_name=tool_name,
        result_text_fn=result_text_fn,
        result_json_fn=result_json_fn,
        format_health_fn=format_health_fn,
        emit=emit,
    )


async def run_depth_coverage_summary_runtime(
    session: ClientSession,
    *,
    sample_limit: int,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_depth_fn: Callable[[dict[str, Any]], str],
    emit: Callable[[str], None],
) -> bool:
    return await orchestrationAgent.run_depth_coverage_summary(
        session,
        sample_limit=sample_limit,
        tool_name=tool_name,
        result_text_fn=result_text_fn,
        result_json_fn=result_json_fn,
        format_depth_fn=format_depth_fn,
        emit=emit,
    )


async def run_agent_ops_pipeline_runtime(
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
    return await orchestrationAgent.run_agent_ops_pipeline(
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
        run_cycle_fn=run_cycle_fn,
        run_health_summary_fn=run_health_summary_fn,
        run_depth_summary_fn=run_depth_summary_fn,
    )


async def start_orchestration_agents_with_activation_checks_runtime(
    session: ClientSession,
    *,
    enable_agent_ops_heartbeat: bool,
    agent_ops_max_tracks: int,
    max_attempts: int,
    retry_seconds: float,
    dashboard_health_status_tool: str,
    agent_ops_cycle_tool: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_cycle_fn: Callable[[dict[str, Any]], str],
    format_evidence_fn: Callable[[dict[str, Any]], list[str]],
    emit: Callable[[str], None],
) -> bool:
    return await orchestrationAgent.start_orchestration_agents_with_activation_checks(
        session,
        enable_agent_ops_heartbeat=enable_agent_ops_heartbeat,
        agent_ops_max_tracks=agent_ops_max_tracks,
        max_attempts=max_attempts,
        retry_seconds=retry_seconds,
        dashboard_health_status_tool=dashboard_health_status_tool,
        agent_ops_cycle_tool=agent_ops_cycle_tool,
        result_text_fn=result_text_fn,
        result_json_fn=result_json_fn,
        format_cycle_fn=format_cycle_fn,
        format_evidence_fn=format_evidence_fn,
        emit=emit,
    )


async def print_depth_coverage_startup_snapshot_runtime(
    session: ClientSession,
    *,
    sample_limit: int,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_depth_fn: Callable[[dict[str, Any]], str],
    emit: Callable[[str], None],
) -> None:
    await orchestrationAgent.print_depth_coverage_startup_snapshot(
        session,
        sample_limit=sample_limit,
        tool_name=tool_name,
        result_text_fn=result_text_fn,
        result_json_fn=result_json_fn,
        format_depth_fn=format_depth_fn,
        emit=emit,
    )


async def print_pipeline_contract_startup_snapshot_runtime(
    session: ClientSession,
    *,
    max_observation_age_seconds: float,
    require_live_pipeline: bool,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    emit: Callable[[str], None],
) -> None:
    await orchestrationAgent.print_pipeline_contract_startup_snapshot(
        session,
        max_observation_age_seconds=max_observation_age_seconds,
        require_live_pipeline=require_live_pipeline,
        tool_name=tool_name,
        result_text_fn=result_text_fn,
        result_json_fn=result_json_fn,
        emit=emit,
    )


@mcp.tool()
def start_dashboard_demo() -> str:
    # Performs start dashboard demo logic.
    """Start the dashboard demo in the background and return immediately."""
    _, message = start_dashboard_process()
    return message


@mcp.tool()
def stop_dashboard_demo(force: bool = False) -> str:
    # Performs stop dashboard demo logic.
    """Stop the running dashboard demo process."""
    return stop_dashboard_process(force=force)


@mcp.tool()
def dashboard_demo_status() -> dict:
    # Performs dashboard demo status logic.
    """Return current dashboard process status including PID and exit code."""
    return dashboard_process_status()


@mcp.tool()
def start_dashboard_and_wait_ready(
    timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    # Performs start dashboard and wait ready logic.
    """Start dashboard demo and block until observations are fresh or timeout is reached."""
    started, message = start_dashboard_process()
    readiness = wait_for_dashboard_ready(
        timeout_seconds=timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    return {
        "started": started,
        "start_message": message,
        "readiness": readiness,
    }


@mcp.tool()
def stop_dashboard_and_confirm(
    force: bool = False,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.1,
) -> dict:
    # Performs stop dashboard and confirm logic.
    """Stop dashboard demo and poll until process is confirmed stopped or timeout occurs."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero")

    stop_message = stop_dashboard_process(force=force)
    deadline = time.time() + timeout_seconds
    attempts = 0
    last_status = dashboard_process_status()

    while True:
        attempts += 1
        last_status = dashboard_process_status()
        if not last_status.get("running", False):
            return {
                "stopped": True,
                "stop_message": stop_message,
                "attempts": attempts,
                "status": last_status,
            }

        now = time.time()
        if now >= deadline:
            return {
                "stopped": False,
                "stop_message": stop_message,
                "attempts": attempts,
                "timeout_seconds": timeout_seconds,
                "status": last_status,
            }

        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


@mcp.tool()
def restart_dashboard_and_wait_ready(
    force_stop: bool = False,
    stop_timeout_seconds: float = 5.0,
    start_timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    # Performs restart dashboard and wait ready logic.
    """Restart dashboard process and wait until fresh observations are available or timeout."""
    stop_result = stop_dashboard_and_confirm(
        force=force_stop,
        timeout_seconds=stop_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    if not stop_result.get("stopped", False):
        return {
            "restarted": False,
            "stop": stop_result,
            "start": None,
            "readiness": None,
        }

    start_result = start_dashboard_and_wait_ready(
        timeout_seconds=start_timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    readiness = start_result.get("readiness")
    ready = isinstance(readiness, dict) and readiness.get("ready", False)
    return {
        "restarted": bool(ready),
        "stop": stop_result,
        "start": {
            "started": start_result.get("started", False),
            "start_message": start_result.get("start_message"),
        },
        "readiness": readiness,
    }


@mcp.tool()
def dashboard_lifecycle_status(
    max_observation_age_seconds: float = 3.0,
) -> dict:
    # Performs dashboard lifecycle status logic.
    """Return merged lifecycle status across process, health, and calibration checks."""
    if max_observation_age_seconds < 0:
        raise ValueError("max_observation_age_seconds must be non-negative")

    process = dashboard_process_status()
    health = dashboard_health_status()
    calibration = validate_ground_plane_calibration()

    running = bool(process.get("running", False))
    available = bool(health.get("available", False))
    age = health.get("last_observation_age_seconds")
    fresh = isinstance(age, (int, float)) and age <= max_observation_age_seconds

    ready = running and available and fresh

    return {
        "ready": ready,
        "running": running,
        "health_available": available,
        "observations_fresh": bool(fresh),
        "max_observation_age_seconds": max_observation_age_seconds,
        "process": process,
        "health": health,
        "calibration": calibration,
    }


@mcp.tool()
def dashboard_bootstrap(
    calibration: dict[str, Any] | None = None,
    restart_if_running: bool = False,
    force_stop: bool = False,
    stop_timeout_seconds: float = 5.0,
    start_timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    # Performs dashboard bootstrap logic.
    """Optionally apply calibration, start/restart dashboard, wait readiness, and return lifecycle status."""
    calibration_result: dict[str, Any] | None = None
    if calibration is not None:
        calibration_result = set_ground_plane_calibration(calibration)

    if restart_if_running:
        orchestration = restart_dashboard_and_wait_ready(
            force_stop=force_stop,
            stop_timeout_seconds=stop_timeout_seconds,
            start_timeout_seconds=start_timeout_seconds,
            max_observation_age_seconds=max_observation_age_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        ready = bool(orchestration.get("restarted", False))
    else:
        orchestration = start_dashboard_and_wait_ready(
            timeout_seconds=start_timeout_seconds,
            max_observation_age_seconds=max_observation_age_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        readiness = orchestration.get("readiness")
        ready = isinstance(readiness, dict) and bool(readiness.get("ready", False))

    lifecycle = dashboard_lifecycle_status(
        max_observation_age_seconds=max_observation_age_seconds,
    )

    return {
        "bootstrapped": bool(ready),
        "restart_if_running": restart_if_running,
        "calibration_result": calibration_result,
        "orchestration": orchestration,
        "lifecycle": lifecycle,
    }


@mcp.tool()
def start_orchestration_demo() -> str:
    # Performs start orchestration demo logic.
    return start_dashboard_demo()


@mcp.tool()
def stop_orchestration_demo(force: bool = False) -> str:
    # Performs stop orchestration demo logic.
    return stop_dashboard_demo(force=force)


@mcp.tool()
def orchestration_demo_status() -> dict:
    # Performs orchestration demo status logic.
    return dashboard_demo_status()


@mcp.tool()
def start_orchestration_and_wait_ready(
    timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    # Performs start orchestration and wait ready logic.
    return start_dashboard_and_wait_ready(
        timeout_seconds=timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


@mcp.tool()
def stop_orchestration_and_confirm(
    force: bool = False,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.1,
) -> dict:
    # Performs stop orchestration and confirm logic.
    return stop_dashboard_and_confirm(
        force=force,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


@mcp.tool()
def restart_orchestration_and_wait_ready(
    force_stop: bool = False,
    stop_timeout_seconds: float = 5.0,
    start_timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    # Performs restart orchestration and wait ready logic.
    return restart_dashboard_and_wait_ready(
        force_stop=force_stop,
        stop_timeout_seconds=stop_timeout_seconds,
        start_timeout_seconds=start_timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


@mcp.tool()
def orchestration_lifecycle_status(
    max_observation_age_seconds: float = 3.0,
) -> dict:
    # Performs orchestration lifecycle status logic.
    return dashboard_lifecycle_status(
        max_observation_age_seconds=max_observation_age_seconds,
    )


@mcp.tool()
def orchestration_bootstrap(
    calibration: dict[str, Any] | None = None,
    restart_if_running: bool = False,
    force_stop: bool = False,
    stop_timeout_seconds: float = 5.0,
    start_timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    # Performs orchestration bootstrap logic.
    return dashboard_bootstrap(
        calibration=calibration,
        restart_if_running=restart_if_running,
        force_stop=force_stop,
        stop_timeout_seconds=stop_timeout_seconds,
        start_timeout_seconds=start_timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
