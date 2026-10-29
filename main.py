# Runtime entrypoint for initialization/bootstrap and runtime lifecycle wiring.
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from agents.orchestration.runtime import app as runtime_app
from agents.orchestration.runtime import config as runtime_config
from agents.orchestration.runtime import core as runtime_agent
from agents.orchestration.runtime import flow as runtime_flow

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "server.py"
ANALYSIS_INTERVAL_SECONDS = runtime_agent.ANALYSIS_INTERVAL_SECONDS

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
    )


async def initialize_runtime_services(session: ClientSession, config: runtime_app.RuntimeConfig) -> None:
    """Run startup checks and activate required runtime services/tools."""
    await runtime_flow.start_runtime_services(
        session,
        config,
        format_runtime_config_banner_fn=format_runtime_config_banner,
        print_depth_coverage_startup_snapshot_fn=print_depth_coverage_startup_snapshot,
        print_pipeline_contract_startup_snapshot_fn=print_pipeline_contract_startup_snapshot,
        verify_required_tools_fn=verify_required_tools,
        start_live_tracking_fn=start_live_tracking,
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
) -> None:
    """Run movement analysis loop and trigger periodic agent-ops heartbeats."""
    monitor_impl = runtime_agent.make_monitor_movement(
        run_agent_ops_pipeline_fn=_run_agent_ops_pipeline,
        analysis_interval_seconds=ANALYSIS_INTERVAL_SECONDS,
    )
    await monitor_impl(
        session,
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

    coordinator = runtime_app.AgentCoordinator(
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
        await operator.stop()
        await services.stop()


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


if __name__ == "__main__":
    cli_main()