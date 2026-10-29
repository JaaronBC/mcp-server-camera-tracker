# Runtime entrypoint that orchestrates MCP tools, periodic agent-ops heartbeats, and movement summaries.
# main.py
import os
import asyncio
import json
import argparse
import time
from pathlib import Path
from typing import Any
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "server.py"

TRACKING_ADDRESS = ("127.0.0.1", 5005)
LIVE_TRACKING_TOOL = "live_tracking"
DETECT_OBJECTS_TOOL = "detect_objects"
AGENT_OPS_CYCLE_TOOL = "run_reasoning_cycle_and_log"
AGENT_OPS_HEALTH_SUMMARY_TOOL = "summarize_agent_event_health"
DEPTH_COVERAGE_SUMMARY_TOOL = "dashboard_depth_coverage_metrics"
PIPELINE_CONTRACT_VERIFY_TOOL = "verify_dashboard_pipeline_contract"
ANALYSIS_INTERVAL_SECONDS = 3
AGENT_OPS_INTERVAL_SECONDS = 10
AGENT_OPS_MAX_TRACKS = 10
AGENT_OPS_SUMMARY_EVERY_CYCLES = 5
AGENT_OPS_SUMMARY_WINDOW_SIZE = 100
DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES = 5
DEPTH_COVERAGE_SAMPLE_LIMIT = 50
AGENT_OPS_HEALTH_FRESHNESS_SECONDS = 30.0
HANDOFF_DEGRADED_WARN_RATE = 0.1
HANDOFF_DEGRADED_CRITICAL_RATE = 0.3


def _positive_float(value: str) -> float:
    # Implements positive float behavior for this module.
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    # Implements positive int behavior for this module.
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    # Implements non negative int behavior for this module.
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _unit_interval_float(value: str) -> float:
    # Implements unit interval float behavior for this module.
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _resolve_handoff_severity_thresholds_from_env() -> tuple[float, float]:
    # Resolves handoff severity thresholds from env from configuration with safe defaults.
    warn_raw = os.getenv("HANDOFF_DEGRADED_WARN_RATE")
    critical_raw = os.getenv("HANDOFF_DEGRADED_CRITICAL_RATE")

    warn_rate = HANDOFF_DEGRADED_WARN_RATE
    critical_rate = HANDOFF_DEGRADED_CRITICAL_RATE

    if isinstance(warn_raw, str) and warn_raw.strip():
        try:
            warn_candidate = float(warn_raw)
        except ValueError:
            warn_candidate = HANDOFF_DEGRADED_WARN_RATE
        if 0 <= warn_candidate <= 1:
            warn_rate = warn_candidate

    if isinstance(critical_raw, str) and critical_raw.strip():
        try:
            critical_candidate = float(critical_raw)
        except ValueError:
            critical_candidate = HANDOFF_DEGRADED_CRITICAL_RATE
        if 0 <= critical_candidate <= 1:
            critical_rate = critical_candidate

    if critical_rate < warn_rate:
        return HANDOFF_DEGRADED_WARN_RATE, HANDOFF_DEGRADED_CRITICAL_RATE

    return warn_rate, critical_rate


def parse_cli_args(args: list[str]) -> argparse.Namespace:
    # Parses CLI runtime controls for heartbeat cadence, summaries, and startup toggles.
    env_warn_rate, env_critical_rate = _resolve_handoff_severity_thresholds_from_env()
    parser = argparse.ArgumentParser(description="Run object behavior MCP agent")
    parser.add_argument(
        "--disable-agent-ops-heartbeat",
        action="store_true",
        help="Disable periodic agent-ops reasoning cycle during monitoring",
    )
    parser.add_argument(
        "--agent-ops-interval-seconds",
        type=_positive_float,
        default=AGENT_OPS_INTERVAL_SECONDS,
        help="Interval in seconds between agent-ops heartbeat cycles (default: 10)",
    )
    parser.add_argument(
        "--agent-ops-max-tracks",
        type=_positive_int,
        default=AGENT_OPS_MAX_TRACKS,
        help="Maximum tracks evaluated per agent-ops cycle (default: 10)",
    )
    parser.add_argument(
        "--agent-ops-summary-every-cycles",
        type=_non_negative_int,
        default=AGENT_OPS_SUMMARY_EVERY_CYCLES,
        help=(
            "Print heartbeat health summary every N cycles (default: 5, "
            "0 disables periodic summary output)"
        ),
    )
    parser.add_argument(
        "--agent-ops-summary-window-size",
        type=_positive_int,
        default=AGENT_OPS_SUMMARY_WINDOW_SIZE,
        help="Event window size for heartbeat health summary (default: 100)",
    )
    parser.add_argument(
        "--disable-depth-coverage-summary",
        action="store_true",
        help="Disable periodic relative-depth coverage summary output",
    )
    parser.add_argument(
        "--depth-coverage-summary-every-cycles",
        type=_non_negative_int,
        default=DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
        help=(
            "Print depth-coverage summary every N heartbeat cycles "
            "(default: 5, 0 disables periodic summary output)"
        ),
    )
    parser.add_argument(
        "--depth-coverage-sample-limit",
        type=_positive_int,
        default=DEPTH_COVERAGE_SAMPLE_LIMIT,
        help="Sample window for depth-coverage trend metrics (default: 50)",
    )
    parser.add_argument(
        "--no-startup-depth-snapshot",
        action="store_true",
        help="Skip one-time startup depth-coverage baseline snapshot output",
    )
    parser.add_argument(
        "--quiet-startup-banner",
        action="store_true",
        help="Suppress startup banner prints in main runtime",
    )
    parser.add_argument(
        "--handoff-degraded-warn-rate",
        type=_unit_interval_float,
        default=env_warn_rate,
        help=(
            "Warn threshold for handoff degraded rate in [0,1] "
            f"(default: {env_warn_rate})"
        ),
    )
    parser.add_argument(
        "--handoff-degraded-critical-rate",
        type=_unit_interval_float,
        default=env_critical_rate,
        help=(
            "Critical threshold for handoff degraded rate in [0,1] "
            f"(default: {env_critical_rate})"
        ),
    )

    options = parser.parse_args(args)
    if options.handoff_degraded_critical_rate < options.handoff_degraded_warn_rate:
        parser.error("--handoff-degraded-critical-rate must be greater than or equal to --handoff-degraded-warn-rate")
    return options

async def verify_required_tools(session: ClientSession) -> None:
    # Verifies required tools contract/availability requirements.
    """Fail before starting the camera if the server lacks required tools."""
    tool_list = await session.list_tools()
    available = {tool.name for tool in tool_list.tools}
    required = {LIVE_TRACKING_TOOL, DETECT_OBJECTS_TOOL}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            f"Required MCP tools are not registered: {missing}. "
            f"Available tools: {sorted(available)}"
        )

async def start_live_tracking(session: ClientSession) -> None:
    # Implements start live tracking behavior for this module.
    """Start the webcam tracker through the MCP server."""
    result = await session.call_tool(LIVE_TRACKING_TOOL, arguments={})
    message = get_result_text(result)
    if getattr(result, "is_error", False):
        raise RuntimeError(f"Could not start live tracking: {message}")
    print(
        message
        or "Live tracking started. Press 'q' in the webcam window to stop.",
        flush=True,
    )

def get_result_text(result: Any) -> str:
    # Gets result text values for downstream consumers.
    """Join text blocks from an MCP tool result."""
    return "\n".join(
        item.text
        for item in (result.content or [])
        if hasattr(item, "text") and isinstance(item.text, str)
    )


def get_result_json_object(result: Any) -> dict[str, Any] | None:
    # Gets result json object values for downstream consumers.
    """Extract a JSON object payload from MCP tool result content."""
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


def format_agent_ops_cycle(payload: dict[str, Any]) -> str:
    # Formats agent ops cycle into concise operator-facing text.
    """Create concise heartbeat status text from agent-ops cycle payload."""
    cycle = payload.get("cycle")
    if not isinstance(cycle, dict):
        return "Agent ops cycle: unavailable"

    evaluated_tracks = cycle.get("evaluated_tracks", "?")
    success_count = cycle.get("success_count", "?")
    error_count = cycle.get("error_count", "?")
    started_at = cycle.get("started_at", "?")
    event_log_writable = payload.get("event_log_writable")
    event_log_error_count = payload.get("event_log_error_count")

    suffix = ""
    if event_log_writable is False:
        suffix = f" | event_log=degraded(errors={event_log_error_count})"

    return (
        "Agent ops cycle: "
        f"tracks={evaluated_tracks} | ok={success_count} | "
        f"errors={error_count} | started_at={started_at}{suffix}"
    )


def format_agent_ops_cycle_evidence(payload: dict[str, Any]) -> list[str]:
    # Formats agent ops cycle evidence into concise operator-facing text.
    """Create concise per-track evidence trace text from agent-ops cycle payload."""
    events = payload.get("events")
    if not isinstance(events, list):
        return []

    rendered: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue

        evidence_trace = event.get("evidence_trace")
        if not isinstance(evidence_trace, dict):
            continue

        latest_track_snapshot = evidence_trace.get("latest_track_snapshot")
        tracking_agent_context = evidence_trace.get("tracking_agent_context")
        coordinate_context = evidence_trace.get("coordinate_context")
        if not isinstance(latest_track_snapshot, dict):
            latest_track_snapshot = {}
        if not isinstance(tracking_agent_context, dict):
            tracking_agent_context = {}
        if not isinstance(coordinate_context, dict):
            coordinate_context = {}

        track_id = event.get("track_id", "?")

        ground_position = latest_track_snapshot.get("ground_position")
        if isinstance(ground_position, dict):
            ground_x = ground_position.get("x", "?")
            ground_z = ground_position.get("z", "?")
            ground_text = f"({ground_x},{ground_z})"
        else:
            ground_text = "n/a"

        relative_depth = latest_track_snapshot.get("relative_depth")
        depth_text = relative_depth if isinstance(relative_depth, (int, float)) else "n/a"

        vlm_model = tracking_agent_context.get("vlm_model", "n/a")
        selected_updates = tracking_agent_context.get("selected_updates")
        selected_updates_count = len(selected_updates) if isinstance(selected_updates, list) else 0

        coordinate_system = coordinate_context.get("coordinate_system", "n/a")
        calibration_status = coordinate_context.get("calibration_status", "n/a")

        rendered.append(
            "Agent ops evidence: "
            f"track={track_id} | latest_ground={ground_text} | depth={depth_text} | "
            f"tracking_vlm={vlm_model} | selected_updates={selected_updates_count} | "
            f"coord={coordinate_system} | calibration={calibration_status}"
        )

    return rendered


def format_agent_ops_health_summary(
    payload: dict[str, Any],
    _max_age_seconds: float = AGENT_OPS_HEALTH_FRESHNESS_SECONDS,
    _now_timestamp: float | None = None,
    _handoff_warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    _handoff_critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
) -> str:
    # Formats agent ops health summary into concise operator-facing text.
    """Create concise heartbeat health summary text."""
    status = payload.get("status")
    latency = payload.get("latency")
    if not isinstance(status, dict) or not isinstance(latency, dict):
        return "Agent ops health: unavailable"

    success_rate = status.get("success_rate", "?")
    error_rate = status.get("error_rate", "?")
    bounded_out_rate = status.get("bounded_out_rate", "?")
    p50_ms = latency.get("p50_ms", "?")
    p95_ms = latency.get("p95_ms", "?")

    health_is_fresh = _is_agent_health_payload_fresh(
        payload,
        max_age_seconds=_max_age_seconds,
        now_timestamp=_now_timestamp,
    )

    agent_error_digest = None
    agent_error_trends = None
    handoff_validation_trend = None
    handoff_status = None
    handoff_degraded_rate = None
    handoff_severity = None
    if health_is_fresh:
        agent_error_digest = _format_agent_error_digest(payload.get("agent_error_code_counts"))
        agent_error_trends = _format_agent_error_trends(payload.get("agent_error_trends"))
        handoff_validation_trend = _format_handoff_validation_trend(payload.get("agent_error_trends"))
        handoff_status = _format_handoff_status_summary(payload)
        handoff_degraded_rate = _format_handoff_degraded_rate(payload)
        handoff_severity = _format_handoff_severity(
            payload,
            warn_rate=_handoff_warn_rate,
            critical_rate=_handoff_critical_rate,
        )

    suffix_parts: list[str] = []
    if agent_error_digest is not None:
        suffix_parts.append(f"agent_error_digest={agent_error_digest}")
    if agent_error_trends is not None:
        suffix_parts.append(f"agent_error_trends={agent_error_trends}")
    if handoff_validation_trend is not None:
        suffix_parts.append(f"handoff_validation_trend={handoff_validation_trend}")
    if handoff_status is not None:
        suffix_parts.append(f"handoff_status={handoff_status}")
    if handoff_degraded_rate is not None:
        suffix_parts.append(f"handoff_degraded_rate={handoff_degraded_rate}")
    if handoff_severity is not None:
        suffix_parts.append(f"handoff_severity={handoff_severity}")

    suffix = ""
    if suffix_parts:
        suffix = " | " + " | ".join(suffix_parts)

    return (
        "Agent ops health: "
        f"success_rate={success_rate} | error_rate={error_rate} | "
        f"bounded_out_rate={bounded_out_rate} | p50_ms={p50_ms} | p95_ms={p95_ms}"
        f"{suffix}"
    )


def _format_agent_error_digest(agent_error_code_counts: Any) -> str | None:
    # Formats agent error digest into concise operator-facing text.
    if not isinstance(agent_error_code_counts, dict):
        return None

    observation = agent_error_code_counts.get("observation")
    reasoning = agent_error_code_counts.get("reasoning")
    cycle = agent_error_code_counts.get("cycle")
    if not isinstance(observation, dict) and not isinstance(reasoning, dict) and not isinstance(cycle, dict):
        return None

    def _top_code(code_counts: Any) -> str:
        # Implements top code behavior for this module.
        if not isinstance(code_counts, dict):
            return "none"

        best_code = None
        best_count = -1
        for code, count in code_counts.items():
            if not isinstance(code, str) or not code.strip():
                continue
            if not isinstance(count, int):
                continue
            if count > best_count:
                best_code = code
                best_count = count

        if best_code is None:
            return "none"
        return f"{best_code}:{best_count}"

    observation_text = _top_code(observation)
    reasoning_text = _top_code(reasoning)
    cycle_text = _top_code(cycle)
    if isinstance(cycle, dict):
        return f"obs({observation_text}),rea({reasoning_text}),cyc({cycle_text})"
    return f"obs({observation_text}),rea({reasoning_text})"


def _format_agent_error_trends(agent_error_trends: Any) -> str | None:
    # Formats agent error trends into concise operator-facing text.
    if not isinstance(agent_error_trends, dict):
        return None

    observation = agent_error_trends.get("observation")
    reasoning = agent_error_trends.get("reasoning")
    cycle = agent_error_trends.get("cycle")
    if not isinstance(observation, dict) and not isinstance(reasoning, dict) and not isinstance(cycle, dict):
        return None

    def _format_side(value: Any, prefix: str) -> str:
        # Formats side into concise operator-facing text.
        if not isinstance(value, dict):
            return f"{prefix}(none)"

        top_error_code = value.get("top_error_code")
        top_error_count = value.get("top_error_count")
        top_error_rate = value.get("top_error_rate")

        if not isinstance(top_error_code, str) or not top_error_code.strip():
            return f"{prefix}(none)"
        if not isinstance(top_error_count, int):
            return f"{prefix}(none)"
        if not isinstance(top_error_rate, (int, float)):
            return f"{prefix}(none)"

        return f"{prefix}({top_error_code}:{top_error_count}@{top_error_rate:.3f})"

    observation_text = _format_side(observation, "obs")
    reasoning_text = _format_side(reasoning, "rea")
    if isinstance(cycle, dict):
        cycle_text = _format_side(cycle, "cyc")
        return f"{observation_text},{reasoning_text},{cycle_text}"
    return f"{observation_text},{reasoning_text}"


def _format_handoff_validation_trend(agent_error_trends: Any) -> str | None:
    # Formats handoff validation trend into concise operator-facing text.
    if not isinstance(agent_error_trends, dict):
        return None

    handoff = agent_error_trends.get("handoff_validation")
    if not isinstance(handoff, dict):
        return None

    top_error_code = handoff.get("top_error_code")
    top_error_count = handoff.get("top_error_count")
    top_error_rate = handoff.get("top_error_rate")

    if not isinstance(top_error_code, str) or not top_error_code.strip():
        return "handoff(none)"
    if not isinstance(top_error_count, int):
        return "handoff(none)"
    if not isinstance(top_error_rate, (int, float)):
        return "handoff(none)"

    return f"handoff({top_error_code}:{top_error_count}@{top_error_rate:.3f})"


def _format_handoff_status_summary(payload: dict[str, Any]) -> str | None:
    # Formats handoff status summary into concise operator-facing text.
    if not isinstance(payload, dict):
        return None

    handoff_validation = payload.get("handoff_validation")
    if not isinstance(handoff_validation, dict):
        return None

    status_counts = handoff_validation.get("status_counts")
    if not isinstance(status_counts, dict):
        return None

    returned_events = payload.get("returned_events")
    if not isinstance(returned_events, int) or returned_events <= 0:
        return None

    ok_count = status_counts.get("ok")
    error_count = status_counts.get("error")
    if not isinstance(ok_count, int):
        ok_count = 0
    if not isinstance(error_count, int):
        error_count = 0

    ok_rate = ok_count / returned_events
    error_rate = error_count / returned_events
    return f"ok:{ok_count}@{ok_rate:.3f},err:{error_count}@{error_rate:.3f}"


def _format_handoff_degraded_rate(payload: dict[str, Any]) -> str | None:
    # Formats handoff degraded rate into concise operator-facing text.
    if not isinstance(payload, dict):
        return None

    handoff_validation = payload.get("handoff_validation")
    if not isinstance(handoff_validation, dict):
        return None

    degraded_rate = handoff_validation.get("degraded_rate")
    if not isinstance(degraded_rate, (int, float)):
        return None

    return f"{float(degraded_rate):.3f}"


def _classify_handoff_severity(
    degraded_rate: float,
    warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
) -> str:
    # Classifies handoff severity into normalized categories.
    if degraded_rate >= critical_rate:
        return "critical"
    if degraded_rate >= warn_rate:
        return "warn"
    return "healthy"


def _format_handoff_severity(
    payload: dict[str, Any],
    warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
) -> str | None:
    # Formats handoff severity into concise operator-facing text.
    if not isinstance(payload, dict):
        return None

    handoff_validation = payload.get("handoff_validation")
    if not isinstance(handoff_validation, dict):
        return None

    degraded_rate = handoff_validation.get("degraded_rate")
    if not isinstance(degraded_rate, (int, float)):
        return None

    return _classify_handoff_severity(
        float(degraded_rate),
        warn_rate=warn_rate,
        critical_rate=critical_rate,
    )


def _is_agent_health_payload_fresh(
    payload: dict[str, Any],
    max_age_seconds: float = AGENT_OPS_HEALTH_FRESHNESS_SECONDS,
    now_timestamp: float | None = None,
) -> bool:
    # Implements is agent health payload fresh behavior for this module.
    time_window = payload.get("time_window")
    if not isinstance(time_window, dict):
        return False

    last_timestamp = time_window.get("last_timestamp")
    if not isinstance(last_timestamp, (int, float)):
        return False

    current_timestamp = now_timestamp
    if current_timestamp is None:
        current_timestamp = time.time()

    age_seconds = float(current_timestamp) - float(last_timestamp)
    if age_seconds < 0:
        return True
    return age_seconds <= max_age_seconds


def format_depth_coverage_summary(payload: dict[str, Any]) -> str:
    # Formats depth coverage summary into concise operator-facing text.
    """Create concise relative-depth coverage summary text."""
    latest = payload.get("latest_coverage")
    trend = payload.get("trend")
    if not isinstance(latest, dict) or not isinstance(trend, dict):
        return "Depth coverage: unavailable"

    tracks_with_depth = latest.get("tracks_with_depth", "?")
    total_tracks = latest.get("total_tracks", "?")
    coverage_percent = latest.get("coverage_percent", "?")

    direction = trend.get("direction", "unknown")
    delta = trend.get("delta_coverage_percent")
    delta_text = "n/a" if not isinstance(delta, (int, float)) else f"{delta:+.1f}%"

    return (
        "Depth coverage: "
        f"{tracks_with_depth}/{total_tracks} ({coverage_percent}%) | "
        f"trend={direction} ({delta_text})"
    )


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
    # Formats runtime config banner into concise operator-facing text.
    """Create a concise startup banner for runtime configuration."""
    summary_mode = (
        "disabled"
        if agent_ops_summary_every_cycles == 0
        else f"every_{agent_ops_summary_every_cycles}_cycles"
    )
    depth_summary_mode = (
        "disabled"
        if (not depth_coverage_summary_enabled or depth_coverage_summary_every_cycles == 0)
        else f"every_{depth_coverage_summary_every_cycles}_cycles"
    )

    return (
        "Runtime config: "
        f"agent_ops_heartbeat={'on' if enable_agent_ops_heartbeat else 'off'} | "
        f"heartbeat_interval_s={agent_ops_interval_seconds} | "
        f"heartbeat_max_tracks={agent_ops_max_tracks} | "
        f"health_summary={summary_mode} | "
        f"health_summary_window={agent_ops_summary_window_size} | "
        f"handoff_warn_rate={handoff_degraded_warn_rate} | "
        f"handoff_critical_rate={handoff_degraded_critical_rate} | "
        f"depth_summary={depth_summary_mode} | "
        f"depth_sample_limit={depth_coverage_sample_limit}"
    )


async def run_agent_ops_cycle(
    session: ClientSession,
    request_evidence_on_uncertainty: bool = True,
    max_tracks: int = AGENT_OPS_MAX_TRACKS,
) -> bool:
    # Runs agent ops cycle workflow steps and returns status outcomes.
    """Execute one agent-ops cycle; returns False when tool should be disabled."""
    try:
        result = await session.call_tool(
            AGENT_OPS_CYCLE_TOOL,
            arguments={
                "request_evidence_on_uncertainty": request_evidence_on_uncertainty,
                "max_tracks": max_tracks,
            },
        )
    except Exception as error:
        message = str(error).lower()
        print(
            f"Agent ops cycle failed ({type(error).__name__}): {error}",
            flush=True,
        )
        if "not registered" in message or "unknown tool" in message or "not found" in message:
            print("Disabling agent ops heartbeat: tool unavailable.", flush=True)
            return False
        return True

    message = get_result_text(result)
    if getattr(result, "is_error", False):
        print(f"Agent ops cycle returned error: {message}", flush=True)
        lowered = (message or "").lower()
        if "not registered" in lowered or "unknown tool" in lowered or "not found" in lowered:
            print("Disabling agent ops heartbeat: tool unavailable.", flush=True)
            return False
        return True

    payload = get_result_json_object(result)
    if payload is None:
        if message:
            print(f"Agent ops cycle: {message}", flush=True)
        else:
            print("Agent ops cycle completed with no structured payload.", flush=True)
        return True

    print(format_agent_ops_cycle(payload), flush=True)
    for evidence_line in format_agent_ops_cycle_evidence(payload):
        print(evidence_line, flush=True)
    return True


async def run_agent_ops_health_summary(
    session: ClientSession,
    window_size: int = AGENT_OPS_SUMMARY_WINDOW_SIZE,
    handoff_warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    handoff_critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
) -> bool:
    # Runs agent ops health summary workflow steps and returns status outcomes.
    """Execute one heartbeat-health summary call; returns False when tool should be disabled."""
    try:
        result = await session.call_tool(
            AGENT_OPS_HEALTH_SUMMARY_TOOL,
            arguments={"window_size": window_size},
        )
    except Exception as error:
        message = str(error).lower()
        print(
            f"Agent ops health summary failed ({type(error).__name__}): {error}",
            flush=True,
        )
        if "not registered" in message or "unknown tool" in message or "not found" in message:
            print("Disabling agent ops health summary: tool unavailable.", flush=True)
            return False
        return True

    message = get_result_text(result)
    if getattr(result, "is_error", False):
        print(f"Agent ops health summary returned error: {message}", flush=True)
        lowered = (message or "").lower()
        if "not registered" in lowered or "unknown tool" in lowered or "not found" in lowered:
            print("Disabling agent ops health summary: tool unavailable.", flush=True)
            return False
        return True

    payload = get_result_json_object(result)
    if payload is None:
        if message:
            print(f"Agent ops health summary: {message}", flush=True)
        else:
            print("Agent ops health summary completed with no structured payload.", flush=True)
        return True

    print(
        format_agent_ops_health_summary(
            payload,
            _handoff_warn_rate=handoff_warn_rate,
            _handoff_critical_rate=handoff_critical_rate,
        ),
        flush=True,
    )
    return True


async def run_depth_coverage_summary(
    session: ClientSession,
    sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> bool:
    # Runs depth coverage summary workflow steps and returns status outcomes.
    """Execute one depth-coverage summary call; returns False when tool should be disabled."""
    try:
        result = await session.call_tool(
            DEPTH_COVERAGE_SUMMARY_TOOL,
            arguments={"sample_limit": sample_limit},
        )
    except Exception as error:
        message = str(error).lower()
        print(
            f"Depth coverage summary failed ({type(error).__name__}): {error}",
            flush=True,
        )
        if "not registered" in message or "unknown tool" in message or "not found" in message:
            print("Disabling depth coverage summary: tool unavailable.", flush=True)
            return False
        return True

    message = get_result_text(result)
    if getattr(result, "is_error", False):
        print(f"Depth coverage summary returned error: {message}", flush=True)
        lowered = (message or "").lower()
        if "not registered" in lowered or "unknown tool" in lowered or "not found" in lowered:
            print("Disabling depth coverage summary: tool unavailable.", flush=True)
            return False
        return True

    payload = get_result_json_object(result)
    if payload is None:
        if message:
            print(f"Depth coverage summary: {message}", flush=True)
        else:
            print("Depth coverage summary completed with no structured payload.", flush=True)
        return True

    print(format_depth_coverage_summary(payload), flush=True)
    return True


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
    # Runs agent ops pipeline workflow steps and returns status outcomes.
    result = {
        "agent_ops_enabled": True,
        "agent_ops_summary_enabled": agent_ops_summary_enabled,
        "depth_summary_enabled": depth_summary_enabled,
        "completed_cycle_increment": 0,
    }

    agent_ops_enabled = await run_agent_ops_cycle(
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
        result["agent_ops_summary_enabled"] = await run_agent_ops_health_summary(
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
        result["depth_summary_enabled"] = await run_depth_coverage_summary(
            session,
            sample_limit=depth_coverage_sample_limit,
        )

    return result


async def print_depth_coverage_startup_snapshot(
    session: ClientSession,
    sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> None:
    # Prints depth coverage startup snapshot status snapshots for runtime observability.
    """Print one depth-coverage baseline snapshot during startup; never raises."""
    try:
        result = await session.call_tool(
            DEPTH_COVERAGE_SUMMARY_TOOL,
            arguments={"sample_limit": sample_limit},
        )
    except Exception as error:
        print(
            f"Depth coverage snapshot unavailable ({type(error).__name__}): {error}",
            flush=True,
        )
        return

    message = get_result_text(result)
    if getattr(result, "is_error", False):
        print(
            f"Depth coverage snapshot unavailable: {message or 'unknown tool error'}",
            flush=True,
        )
        return

    payload = get_result_json_object(result)
    if payload is None:
        if message:
            print(f"Depth coverage snapshot: {message}", flush=True)
        else:
            print("Depth coverage snapshot unavailable: no structured data.", flush=True)
        return

    print(f"Startup {format_depth_coverage_summary(payload)}", flush=True)


async def print_pipeline_contract_startup_snapshot(
    session: ClientSession,
    max_observation_age_seconds: float = 3.0,
    require_live_pipeline: bool = False,
) -> None:
    # Prints pipeline contract startup snapshot status snapshots for runtime observability.
    """Print one startup pipeline-contract verification snapshot; never raises."""
    try:
        result = await session.call_tool(
            PIPELINE_CONTRACT_VERIFY_TOOL,
            arguments={
                "max_observation_age_seconds": max_observation_age_seconds,
                "require_live_pipeline": require_live_pipeline,
            },
        )
    except Exception as error:
        print(
            f"Pipeline contract snapshot unavailable ({type(error).__name__}): {error}",
            flush=True,
        )
        return

    message = get_result_text(result)
    if getattr(result, "is_error", False):
        print(
            f"Pipeline contract snapshot unavailable: {message or 'unknown tool error'}",
            flush=True,
        )
        return

    payload = get_result_json_object(result)
    if payload is None:
        if message:
            print(f"Pipeline contract snapshot: {message}", flush=True)
        else:
            print("Pipeline contract snapshot unavailable: no structured data.", flush=True)
        return

    ready = bool(payload.get("ready"))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    error_count = summary.get("error_count", 0)
    warn_count = summary.get("warn_count", 0)
    status_text = "ready" if ready else "degraded"
    print(
        "Startup pipeline contract: "
        f"status={status_text} | errors={error_count} | warnings={warn_count}",
        flush=True,
    )


async def monitor_movement(
    session: ClientSession,
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
    # Runs the detect/summarize loop and schedules heartbeat pipelines on cadence.
    """Receive frames from the tracker and print an interpretation."""
    address = [TRACKING_ADDRESS[0], TRACKING_ADDRESS[1]]

    print(
        "Automatic movement analysis started. Press Ctrl+C to stop.",
        flush=True,
    )

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
                    print(
                        f"Agent ops cycle task failed ({type(error).__name__}): {error}",
                        flush=True,
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
                result = await session.call_tool(
                    DETECT_OBJECTS_TOOL,
                    arguments={"address": address},
                )
            except Exception as error:
                print(
                    "Movement analysis tool call failed "
                    f"({type(error).__name__}): {error}",
                    flush=True,
                )
            else:
                frame_data = get_result_text(result)
                if getattr(result, "is_error", False):
                    print(f"Movement analysis failed: {frame_data}", flush=True)
                elif not frame_data:
                    print("Movement analysis returned no frame data.", flush=True)
                else:
                    print(f"\nYOLO frame:\n{frame_data}", flush=True)

                    now = loop.time()
                    if agent_ops_enabled and (now - last_agent_ops_cycle_at) >= agent_ops_interval_seconds:
                        if agent_ops_cycle_task is not None and not agent_ops_cycle_task.done():
                            print(
                                "Skipping agent ops trigger: prior cycle still running.",
                                flush=True,
                            )
                        else:
                            next_cycle_number = completed_agent_ops_cycles + 1
                            agent_ops_cycle_task = asyncio.create_task(
                                _run_agent_ops_pipeline(
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
            await asyncio.sleep(max(0, ANALYSIS_INTERVAL_SECONDS - elapsed))
    finally:
        if agent_ops_cycle_task is not None and not agent_ops_cycle_task.done():
            agent_ops_cycle_task.cancel()
            try:
                await agent_ops_cycle_task
            except asyncio.CancelledError:
                pass


async def run_agent(
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
    # Initializes MCP session, startup snapshots, and movement monitoring runtime.
    """Connect to the MCP server, start tracking, and analyze its feed."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    if show_startup_banner:
        print("Initializing MCP pipeline and connecting to server.py...", flush=True)
        print(
            format_runtime_config_banner(
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
            ),
            flush=True,
        )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            if show_startup_depth_snapshot:
                await print_depth_coverage_startup_snapshot(
                    session,
                    sample_limit=depth_coverage_sample_limit,
                )
            if show_startup_pipeline_contract_snapshot:
                await print_pipeline_contract_startup_snapshot(
                    session,
                    max_observation_age_seconds=3.0,
                    require_live_pipeline=False,
                )
            await verify_required_tools(session)
            await start_live_tracking(session)
            await monitor_movement(
                session,
                enable_agent_ops_heartbeat=enable_agent_ops_heartbeat,
                agent_ops_interval_seconds=agent_ops_interval_seconds,
                agent_ops_max_tracks=agent_ops_max_tracks,
                agent_ops_summary_every_cycles=agent_ops_summary_every_cycles,
                agent_ops_summary_window_size=agent_ops_summary_window_size,
                depth_coverage_summary_enabled=depth_coverage_summary_enabled,
                depth_coverage_summary_every_cycles=depth_coverage_summary_every_cycles,
                depth_coverage_sample_limit=depth_coverage_sample_limit,
            )

def main() -> None:
    # Implements main behavior for this module.
    options = parse_cli_args(sys.argv[1:])
    try:
        asyncio.run(
            run_agent(
                show_startup_banner=not options.quiet_startup_banner,
                show_startup_depth_snapshot=not options.no_startup_depth_snapshot,
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
        )
    except KeyboardInterrupt:
        print("\nStopping movement analysis.", flush=True)

if __name__ == "__main__":
    main()