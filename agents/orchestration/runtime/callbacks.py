# Factory helpers that build runtime callback functions for main bootstrap wiring.
from __future__ import annotations

from typing import Any, Awaitable, Callable

from mcp import ClientSession

from agents.orchestration.tools import control as orchestration_control


def make_verify_required_tools(
    *,
    live_tracking_tool_name: str,
    detect_objects_tool_name: str,
) -> Callable[[ClientSession], Awaitable[None]]:
    async def _verify_required_tools(session: ClientSession) -> None:
        await orchestration_control.verify_required_tools_runtime(
            session,
            live_tracking_tool_name=live_tracking_tool_name,
            detect_objects_tool_name=detect_objects_tool_name,
        )

    return _verify_required_tools


def make_start_live_tracking(
    *,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    emit: Callable[[str], None],
) -> Callable[[ClientSession], Awaitable[None]]:
    async def _start_live_tracking(session: ClientSession) -> None:
        await orchestration_control.start_live_tracking_runtime(
            session,
            tool_name=tool_name,
            result_text_fn=result_text_fn,
            emit=emit,
        )

    return _start_live_tracking


def make_run_agent_ops_cycle(
    *,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_cycle_fn: Callable[[dict[str, Any]], str],
    format_evidence_fn: Callable[[dict[str, Any]], list[str]],
    emit: Callable[[str], None],
    default_max_tracks: int,
) -> Callable[[ClientSession, bool, int], Awaitable[bool]]:
    async def _run_agent_ops_cycle(
        session: ClientSession,
        request_evidence_on_uncertainty: bool = True,
        max_tracks: int = default_max_tracks,
    ) -> bool:
        return await orchestration_control.run_agent_ops_cycle_runtime(
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

    return _run_agent_ops_cycle


def make_run_agent_ops_health_summary(
    *,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_health_fn: Callable[[dict[str, Any], float, float], str],
    emit: Callable[[str], None],
    default_window_size: int,
    default_handoff_warn_rate: float,
    default_handoff_critical_rate: float,
) -> Callable[[ClientSession, int, float, float], Awaitable[bool]]:
    async def _run_agent_ops_health_summary(
        session: ClientSession,
        window_size: int = default_window_size,
        handoff_warn_rate: float = default_handoff_warn_rate,
        handoff_critical_rate: float = default_handoff_critical_rate,
    ) -> bool:
        return await orchestration_control.run_agent_ops_health_summary_runtime(
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

    return _run_agent_ops_health_summary


def make_run_depth_coverage_summary(
    *,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_depth_fn: Callable[[dict[str, Any]], str],
    emit: Callable[[str], None],
    default_sample_limit: int,
) -> Callable[[ClientSession, int], Awaitable[bool]]:
    async def _run_depth_coverage_summary(
        session: ClientSession,
        sample_limit: int = default_sample_limit,
    ) -> bool:
        return await orchestration_control.run_depth_coverage_summary_runtime(
            session,
            sample_limit=sample_limit,
            tool_name=tool_name,
            result_text_fn=result_text_fn,
            result_json_fn=result_json_fn,
            format_depth_fn=format_depth_fn,
            emit=emit,
        )

    return _run_depth_coverage_summary


def make_start_orchestration_agents(
    *,
    dashboard_health_status_tool: str,
    agent_ops_cycle_tool: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_cycle_fn: Callable[[dict[str, Any]], str],
    format_evidence_fn: Callable[[dict[str, Any]], list[str]],
    emit: Callable[[str], None],
    default_enable_agent_ops_heartbeat: bool,
    default_agent_ops_max_tracks: int,
    default_max_attempts: int,
    default_retry_seconds: float,
) -> Callable[[ClientSession, bool, int, int, float], Awaitable[bool]]:
    async def _start_orchestration_agents(
        session: ClientSession,
        enable_agent_ops_heartbeat: bool = default_enable_agent_ops_heartbeat,
        agent_ops_max_tracks: int = default_agent_ops_max_tracks,
        max_attempts: int = default_max_attempts,
        retry_seconds: float = default_retry_seconds,
    ) -> bool:
        return await orchestration_control.start_orchestration_agents_with_activation_checks_runtime(
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

    return _start_orchestration_agents


def make_print_depth_coverage_startup_snapshot(
    *,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    format_depth_fn: Callable[[dict[str, Any]], str],
    emit: Callable[[str], None],
    default_sample_limit: int,
) -> Callable[[ClientSession, int], Awaitable[None]]:
    async def _print_depth_coverage_startup_snapshot(
        session: ClientSession,
        sample_limit: int = default_sample_limit,
    ) -> None:
        await orchestration_control.print_depth_coverage_startup_snapshot_runtime(
            session,
            sample_limit=sample_limit,
            tool_name=tool_name,
            result_text_fn=result_text_fn,
            result_json_fn=result_json_fn,
            format_depth_fn=format_depth_fn,
            emit=emit,
        )

    return _print_depth_coverage_startup_snapshot


def make_print_pipeline_contract_startup_snapshot(
    *,
    tool_name: str,
    result_text_fn: Callable[[Any], str],
    result_json_fn: Callable[[Any], dict[str, Any] | None],
    emit: Callable[[str], None],
    default_max_observation_age_seconds: float,
    default_require_live_pipeline: bool,
) -> Callable[[ClientSession, float, bool], Awaitable[None]]:
    async def _print_pipeline_contract_startup_snapshot(
        session: ClientSession,
        max_observation_age_seconds: float = default_max_observation_age_seconds,
        require_live_pipeline: bool = default_require_live_pipeline,
    ) -> None:
        await orchestration_control.print_pipeline_contract_startup_snapshot_runtime(
            session,
            max_observation_age_seconds=max_observation_age_seconds,
            require_live_pipeline=require_live_pipeline,
            tool_name=tool_name,
            result_text_fn=result_text_fn,
            result_json_fn=result_json_fn,
            emit=emit,
        )

    return _print_pipeline_contract_startup_snapshot


def make_run_agent_ops_pipeline(
    *,
    run_pipeline_fn: Callable[..., Awaitable[dict[str, Any]]],
    run_cycle_fn: Callable[..., Awaitable[bool]],
    run_health_summary_fn: Callable[..., Awaitable[bool]],
    run_depth_summary_fn: Callable[..., Awaitable[bool]],
) -> Callable[..., Awaitable[dict[str, Any]]]:
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
        return await run_pipeline_fn(
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

    return _run_agent_ops_pipeline


def make_monitor_movement(
    *,
    monitor_movement_fn: Callable[..., Awaitable[None]],
    run_tracking_cycle_fn: Callable[..., Awaitable[None]],
    tracking_address: tuple[str, int],
    analysis_interval_seconds: float,
    detect_objects_tool_name: str,
    result_text_fn: Callable[[Any], str],
    run_agent_ops_pipeline_fn: Callable[..., Awaitable[dict[str, Any]]],
    emit: Callable[[str], None],
    default_agent_ops_interval_seconds: float,
    default_agent_ops_max_tracks: int,
    default_agent_ops_summary_every_cycles: int,
    default_agent_ops_summary_window_size: int,
    default_handoff_degraded_warn_rate: float,
    default_handoff_degraded_critical_rate: float,
    default_depth_coverage_summary_every_cycles: int,
    default_depth_coverage_sample_limit: int,
) -> Callable[..., Awaitable[None]]:
    async def _monitor_movement(
        session: ClientSession,
        enable_agent_ops_heartbeat: bool = True,
        agent_ops_interval_seconds: float = default_agent_ops_interval_seconds,
        agent_ops_max_tracks: int = default_agent_ops_max_tracks,
        agent_ops_summary_every_cycles: int = default_agent_ops_summary_every_cycles,
        agent_ops_summary_window_size: int = default_agent_ops_summary_window_size,
        handoff_degraded_warn_rate: float = default_handoff_degraded_warn_rate,
        handoff_degraded_critical_rate: float = default_handoff_degraded_critical_rate,
        depth_coverage_summary_enabled: bool = True,
        depth_coverage_summary_every_cycles: int = default_depth_coverage_summary_every_cycles,
        depth_coverage_sample_limit: int = default_depth_coverage_sample_limit,
    ) -> None:
        async def _run_tracking_cycle(session_handle: ClientSession, address: list[Any]) -> None:
            await run_tracking_cycle_fn(
                session_handle,
                address=address,
                detect_objects_tool_name=detect_objects_tool_name,
                result_text_fn=result_text_fn,
                emit=emit,
            )

        await monitor_movement_fn(
            session,
            tracking_address=tracking_address,
            analysis_interval_seconds=analysis_interval_seconds,
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
            run_tracking_cycle_fn=_run_tracking_cycle,
            run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
            emit=emit,
        )

    return _monitor_movement
