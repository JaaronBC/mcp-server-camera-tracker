# Runtime agent operations and callback wiring, extracted from main bootstrap.
from __future__ import annotations

from typing import Any, Awaitable, Callable

from mcp import ClientSession

from agents.tracking import agent_runtime as trackingAgent

from agents.orchestration.tools import control as orchestrationControl
from agents.orchestration import formatting as orchestrationFormatting

from . import app as runtimeApp
from . import callbacks as orchestrationCallbacks
from . import config as runtimeConfig
from . import flow as runtimeFlow

TRACKING_ADDRESS = ("127.0.0.1", 5005)
LIVE_TRACKING_TOOL = "live_tracking"
DETECT_OBJECTS_TOOL = "detect_objects"
AGENT_OPS_CYCLE_TOOL = "run_reasoning_cycle_and_log"
AGENT_OPS_HEALTH_SUMMARY_TOOL = "summarize_agent_event_health"
DEPTH_COVERAGE_SUMMARY_TOOL = "dashboard_depth_coverage_metrics"
PIPELINE_CONTRACT_VERIFY_TOOL = "verify_dashboard_pipeline_contract"
DASHBOARD_HEALTH_STATUS_TOOL = "dashboard_health_status"
ANALYSIS_INTERVAL_SECONDS = 3
AGENT_OPS_INTERVAL_SECONDS = 10
AGENT_OPS_MAX_TRACKS = 10
AGENT_OPS_SUMMARY_EVERY_CYCLES = 5
AGENT_OPS_SUMMARY_WINDOW_SIZE = 100
DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES = 5
DEPTH_COVERAGE_SAMPLE_LIMIT = 50
HANDOFF_DEGRADED_WARN_RATE = 0.1
HANDOFF_DEGRADED_CRITICAL_RATE = 0.3
AGENT_STARTUP_ACTIVATION_MAX_ATTEMPTS = 5
AGENT_STARTUP_ACTIVATION_RETRY_SECONDS = 0.5


def get_result_text(result: Any) -> str:
    return trackingAgent.get_result_text(result)


def get_result_json_object(result: Any) -> dict[str, Any] | None:
    return trackingAgent.get_result_json_object(result)


format_agent_ops_cycle = orchestrationFormatting.format_agent_ops_cycle
format_agent_ops_cycle_evidence = orchestrationFormatting.format_agent_ops_cycle_evidence
format_agent_ops_health_summary = orchestrationFormatting.format_agent_ops_health_summary
_format_agent_error_digest = orchestrationFormatting._format_agent_error_digest
_format_agent_error_trends = orchestrationFormatting._format_agent_error_trends
_format_handoff_validation_trend = orchestrationFormatting._format_handoff_validation_trend
_format_handoff_status_summary = orchestrationFormatting._format_handoff_status_summary
_format_handoff_degraded_rate = orchestrationFormatting._format_handoff_degraded_rate
_format_collaboration_signals = orchestrationFormatting._format_collaboration_signals
_classify_handoff_severity = orchestrationFormatting._classify_handoff_severity
_format_handoff_severity = orchestrationFormatting._format_handoff_severity
_is_agent_health_payload_fresh = orchestrationFormatting._is_agent_health_payload_fresh
format_depth_coverage_summary = orchestrationFormatting.format_depth_coverage_summary


def format_runtime_config_banner(
    enable_agent_ops_heartbeat: bool,
    agent_ops_interval_seconds: float,
    agent_ops_max_tracks: int,
    agent_ops_summary_every_cycles: int,
    agent_ops_summary_window_size: int,
    handoff_degraded_warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    handoff_degraded_critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
    depth_coverage_summary_enabled: bool = True,
    depth_coverage_summary_every_cycles: int = DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
    depth_coverage_sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> str:
    return runtimeConfig.format_runtime_config_banner(
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


verify_required_tools = orchestrationCallbacks.make_verify_required_tools(
    live_tracking_tool=LIVE_TRACKING_TOOL,
    detect_objects_tool=DETECT_OBJECTS_TOOL,
)


start_live_tracking = orchestrationCallbacks.make_start_live_tracking(
    tool_name=LIVE_TRACKING_TOOL,
    result_text_fn=get_result_text,
    emit=lambda message: print(message, flush=True),
)


run_agent_ops_cycle = orchestrationCallbacks.make_run_agent_ops_cycle(
    tool_name=AGENT_OPS_CYCLE_TOOL,
    result_text_fn=get_result_text,
    result_json_fn=get_result_json_object,
    format_cycle_fn=format_agent_ops_cycle,
    format_evidence_fn=format_agent_ops_cycle_evidence,
    emit=lambda message: print(message, flush=True),
    default_max_tracks=AGENT_OPS_MAX_TRACKS,
)


run_agent_ops_health_summary = orchestrationCallbacks.make_run_agent_ops_health_summary(
    tool_name=AGENT_OPS_HEALTH_SUMMARY_TOOL,
    result_text_fn=get_result_text,
    result_json_fn=get_result_json_object,
    format_health_fn=lambda payload, warn_rate, critical_rate: format_agent_ops_health_summary(
        payload,
        _handoff_warn_rate=warn_rate,
        _handoff_critical_rate=critical_rate,
    ),
    emit=lambda message: print(message, flush=True),
    default_window_size=AGENT_OPS_SUMMARY_WINDOW_SIZE,
    default_handoff_warn_rate=HANDOFF_DEGRADED_WARN_RATE,
    default_handoff_critical_rate=HANDOFF_DEGRADED_CRITICAL_RATE,
)


run_depth_coverage_summary = orchestrationCallbacks.make_run_depth_coverage_summary(
    tool_name=DEPTH_COVERAGE_SUMMARY_TOOL,
    result_text_fn=get_result_text,
    result_json_fn=get_result_json_object,
    format_depth_fn=format_depth_coverage_summary,
    emit=lambda message: print(message, flush=True),
    default_sample_limit=DEPTH_COVERAGE_SAMPLE_LIMIT,
)


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
    return await orchestrationControl.run_agent_ops_pipeline_runtime(
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


async def _run_agent_ops_pipeline_bound(
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
    return await run_agent_ops_pipeline(
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


start_orchestration_agents = orchestrationCallbacks.make_start_orchestration_agents(
    dashboard_health_status_tool=DASHBOARD_HEALTH_STATUS_TOOL,
    agent_ops_cycle_tool=AGENT_OPS_CYCLE_TOOL,
    result_text_fn=get_result_text,
    result_json_fn=get_result_json_object,
    format_cycle_fn=format_agent_ops_cycle,
    format_evidence_fn=format_agent_ops_cycle_evidence,
    emit=lambda message: print(message, flush=True),
    default_enable_agent_ops_heartbeat=True,
    default_agent_ops_max_tracks=AGENT_OPS_MAX_TRACKS,
    default_max_attempts=AGENT_STARTUP_ACTIVATION_MAX_ATTEMPTS,
    default_retry_seconds=AGENT_STARTUP_ACTIVATION_RETRY_SECONDS,
)


print_depth_coverage_startup_snapshot = orchestrationCallbacks.make_print_depth_coverage_startup_snapshot(
    tool_name=DEPTH_COVERAGE_SUMMARY_TOOL,
    result_text_fn=get_result_text,
    result_json_fn=get_result_json_object,
    format_depth_fn=format_depth_coverage_summary,
    emit=lambda message: print(message, flush=True),
    default_sample_limit=DEPTH_COVERAGE_SAMPLE_LIMIT,
)


print_pipeline_contract_startup_snapshot = orchestrationCallbacks.make_print_pipeline_contract_startup_snapshot(
    tool_name=PIPELINE_CONTRACT_VERIFY_TOOL,
    result_text_fn=get_result_text,
    result_json_fn=get_result_json_object,
    emit=lambda message: print(message, flush=True),
    default_max_observation_age_seconds=3.0,
    default_require_live_pipeline=False,
)


def make_monitor_movement(
    *,
    run_agent_ops_pipeline_fn: Callable[..., Awaitable[dict[str, Any]]],
    analysis_interval_seconds: float = ANALYSIS_INTERVAL_SECONDS,
) -> Callable[..., Awaitable[None]]:
    return orchestrationCallbacks.make_monitor_movement(
        monitor_movement_fn=runtimeFlow.monitor_movement,
        run_tracking_cycle_fn=trackingAgent.run_tracking_cycle,
        tracking_address=TRACKING_ADDRESS,
        analysis_interval_seconds=analysis_interval_seconds,
        detect_objects_tool=DETECT_OBJECTS_TOOL,
        result_text_fn=get_result_text,
        run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
        emit=lambda message: print(message, flush=True),
        default_agent_ops_interval_seconds=AGENT_OPS_INTERVAL_SECONDS,
        default_agent_ops_max_tracks=AGENT_OPS_MAX_TRACKS,
        default_agent_ops_summary_every_cycles=AGENT_OPS_SUMMARY_EVERY_CYCLES,
        default_agent_ops_summary_window_size=AGENT_OPS_SUMMARY_WINDOW_SIZE,
        default_handoff_degraded_warn_rate=HANDOFF_DEGRADED_WARN_RATE,
        default_handoff_degraded_critical_rate=HANDOFF_DEGRADED_CRITICAL_RATE,
        default_depth_coverage_summary_every_cycles=DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
        default_depth_coverage_sample_limit=DEPTH_COVERAGE_SAMPLE_LIMIT,
    )


async def run_monitoring_loop(
    session: ClientSession,
    config: runtimeApp.RuntimeConfig,
) -> None:
    monitor_impl = make_monitor_movement(
        run_agent_ops_pipeline_fn=_run_agent_ops_pipeline_bound,
        analysis_interval_seconds=ANALYSIS_INTERVAL_SECONDS,
    )
    await monitor_impl(
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


async def run_agent(
    *,
    run_with_runtime_config_fn: Callable[[runtimeApp.RuntimeConfig], Awaitable[None]],
    show_startup_banner: bool = True,
    show_startup_depth_snapshot: bool = True,
    show_startup_pipeline_contract_snapshot: bool = True,
    enable_agent_ops_heartbeat: bool = True,
    agent_ops_interval_seconds: float = AGENT_OPS_INTERVAL_SECONDS,
    agent_ops_max_tracks: int = AGENT_OPS_MAX_TRACKS,
    agent_ops_summary_every_cycles: int = AGENT_OPS_SUMMARY_EVERY_CYCLES,
    agent_ops_summary_window_size: int = AGENT_OPS_SUMMARY_WINDOW_SIZE,
    handoff_degraded_warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    handoff_degraded_critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
    depth_coverage_summary_enabled: bool = True,
    depth_coverage_summary_every_cycles: int = DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
    depth_coverage_sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> None:
    config = runtimeApp.RuntimeConfig(
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
    await run_with_runtime_config_fn(config)
