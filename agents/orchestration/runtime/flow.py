# Shared runtime flow helpers to keep `main.py` focused on bootstrap/wiring.
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from mcp import ClientSession

from . import app as runtimeApp


async def start_runtime_services(
    session: ClientSession,
    config: runtimeApp.RuntimeConfig,
    *,
    format_runtime_config_banner_fn: Callable[..., str],
    print_depth_coverage_startup_snapshot_fn: Callable[..., Awaitable[None]],
    print_pipeline_contract_startup_snapshot_fn: Callable[..., Awaitable[None]],
    verify_required_tools_fn: Callable[[ClientSession], Awaitable[None]],
    start_live_tracking_fn: Callable[[ClientSession], Awaitable[None]],
    start_orchestration_agents_fn: Callable[..., Awaitable[bool]],
    emit: Callable[[str], None],
) -> None:
    if config.show_startup_banner:
        emit("Initializing MCP pipeline and connecting to server.py...")
        emit(
            format_runtime_config_banner_fn(
                enable_agent_ops_heartbeat=config.enable_agent_ops_heartbeat,
                agent_ops_interval_seconds=config.agent_ops_interval_seconds,
                agent_ops_max_tracks=config.agent_ops_max_tracks,
                agent_ops_summary_every_cycles=config.agent_ops_summary_every_cycles,
                agent_ops_summary_window_size=config.agent_ops_summary_window_size,
                handoff_degraded_warn_rate=config.handoff_degraded_warn_rate,
                handoff_degraded_critical_rate=config.handoff_degraded_critical_rate,
                depth_coverage_summary_enabled=config.depth_coverage_summary_enabled,
                depth_coverage_summary_every_cycles=config.depth_coverage_summary_every_cycles,
                depth_coverage_sample_limit=config.depth_coverage_sample_limit,
            )
        )

    if config.show_startup_depth_snapshot:
        await print_depth_coverage_startup_snapshot_fn(
            session,
            sample_limit=config.depth_coverage_sample_limit,
        )
    if config.show_startup_pipeline_contract_snapshot:
        await print_pipeline_contract_startup_snapshot_fn(
            session,
            max_observation_age_seconds=3.0,
            require_live_pipeline=False,
        )

    await verify_required_tools_fn(session)
    await start_live_tracking_fn(session)

    startup_agent_ops_enabled = await start_orchestration_agents_fn(
        session,
        enable_agent_ops_heartbeat=config.enable_agent_ops_heartbeat,
        agent_ops_max_tracks=config.agent_ops_max_tracks,
    )
    if config.enable_agent_ops_heartbeat and not startup_agent_ops_enabled:
        emit(
            "Startup activation pending: proceeding with runtime and continuing "
            "agent warm-up in the background heartbeat."
        )


async def run_operator_loop(
    session: ClientSession,
    config: runtimeApp.RuntimeConfig,
    *,
    monitor_movement_fn: Callable[..., Awaitable[None]],
) -> None:
    await monitor_movement_fn(
        session,
        enable_agent_ops_heartbeat=config.enable_agent_ops_heartbeat,
        agent_ops_interval_seconds=config.agent_ops_interval_seconds,
        agent_ops_max_tracks=config.agent_ops_max_tracks,
        agent_ops_summary_every_cycles=config.agent_ops_summary_every_cycles,
        agent_ops_summary_window_size=config.agent_ops_summary_window_size,
        handoff_degraded_warn_rate=config.handoff_degraded_warn_rate,
        handoff_degraded_critical_rate=config.handoff_degraded_critical_rate,
        depth_coverage_summary_enabled=config.depth_coverage_summary_enabled,
        depth_coverage_summary_every_cycles=config.depth_coverage_summary_every_cycles,
        depth_coverage_sample_limit=config.depth_coverage_sample_limit,
    )


async def monitor_movement(
    session: ClientSession,
    *,
    tracking_address: tuple[str, int],
    analysis_interval_seconds: float,
    enable_agent_ops_heartbeat: bool,
    agent_ops_interval_seconds: float,
    agent_ops_max_tracks: int,
    agent_ops_summary_every_cycles: int,
    agent_ops_summary_window_size: int,
    handoff_degraded_warn_rate: float,
    handoff_degraded_critical_rate: float,
    depth_coverage_summary_enabled: bool,
    depth_coverage_summary_every_cycles: int,
    depth_coverage_sample_limit: int,
    run_tracking_cycle_fn: Callable[[ClientSession, list[Any]], Awaitable[None]],
    run_agent_ops_pipeline_fn: Callable[..., Awaitable[dict[str, Any]]],
    emit: Callable[[str], None],
) -> None:
    address = [tracking_address[0], tracking_address[1]]

    emit("Automatic movement analysis started. Press Ctrl+C to stop.")

    loop = asyncio.get_running_loop()
    agent_ops_enabled = enable_agent_ops_heartbeat
    last_agent_ops_cycle_at = -agent_ops_interval_seconds
    agent_ops_summary_enabled = enable_agent_ops_heartbeat and agent_ops_summary_every_cycles > 0
    depth_summary_enabled = (
        enable_agent_ops_heartbeat
        and depth_coverage_summary_enabled
        and depth_coverage_summary_every_cycles > 0
    )
    completed_agent_ops_cycles = 0
    agent_ops_cycle_task: asyncio.Task[dict[str, Any]] | None = None

    try:
        while True:
            iteration_started_at = loop.time()

            if agent_ops_cycle_task is not None and agent_ops_cycle_task.done():
                try:
                    pipeline_result = agent_ops_cycle_task.result()
                except Exception as error:
                    emit(
                        f"Agent ops cycle task failed ({type(error).__name__}): {error}"
                    )
                else:
                    agent_ops_enabled = bool(
                        pipeline_result.get("agent_ops_enabled", agent_ops_enabled)
                    )
                    agent_ops_summary_enabled = bool(
                        pipeline_result.get(
                            "agent_ops_summary_enabled",
                            agent_ops_summary_enabled,
                        )
                    )
                    depth_summary_enabled = bool(
                        pipeline_result.get("depth_summary_enabled", depth_summary_enabled)
                    )
                    completed_agent_ops_cycles += int(
                        pipeline_result.get("completed_cycle_increment", 0)
                    )
                agent_ops_cycle_task = None

            try:
                await run_tracking_cycle_fn(session, address)
            except Exception as error:
                emit(
                    "Movement analysis tool call failed "
                    f"({type(error).__name__}): {error}"
                )

            now = loop.time()
            if agent_ops_enabled and (now - last_agent_ops_cycle_at) >= agent_ops_interval_seconds:
                if agent_ops_cycle_task is not None and not agent_ops_cycle_task.done():
                    emit("Skipping agent ops trigger: prior cycle still running.")
                else:
                    next_cycle_number = completed_agent_ops_cycles + 1
                    agent_ops_cycle_task = asyncio.create_task(
                        run_agent_ops_pipeline_fn(
                            session,
                            cycle_number=next_cycle_number,
                            max_tracks=agent_ops_max_tracks,
                            agent_ops_summary_enabled=agent_ops_summary_enabled,
                            agent_ops_summary_every_cycles=agent_ops_summary_every_cycles,
                            agent_ops_summary_window_size=agent_ops_summary_window_size,
                            handoff_degraded_warn_rate=handoff_degraded_warn_rate,
                            handoff_degraded_critical_rate=handoff_degraded_critical_rate,
                            depth_summary_enabled=depth_summary_enabled,
                            depth_coverage_summary_every_cycles=depth_coverage_summary_every_cycles,
                            depth_coverage_sample_limit=depth_coverage_sample_limit,
                        )
                    )
                    last_agent_ops_cycle_at = loop.time()

            elapsed = loop.time() - iteration_started_at
            await asyncio.sleep(max(0, analysis_interval_seconds - elapsed))
    finally:
        if agent_ops_cycle_task is not None and not agent_ops_cycle_task.done():
            agent_ops_cycle_task.cancel()
            try:
                await agent_ops_cycle_task
            except asyncio.CancelledError:
                pass