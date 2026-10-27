# main.py
import os
import asyncio
import json
import argparse
from pathlib import Path
from typing import Any
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent
SKILL_PATH = PROJECT_ROOT / "Skills.md"
SKILL_FALLBACK_PATH = PROJECT_ROOT / "design.md"
SERVER_PATH = PROJECT_ROOT / "server.py"

TRACKING_ADDRESS = ("127.0.0.1", 5005)
LIVE_TRACKING_TOOL = "live_tracking"
DETECT_OBJECTS_TOOL = "detect_objects"
MILESTONE_PROGRESS_TOOL = "milestone_progress_status"
MILESTONE_2_PROGRESS_TOOL = "milestone_2_progress_status"
MILESTONE_3_PROGRESS_TOOL = "milestone_3_progress_status"
MILESTONE_PROGRESS_TOOLS = (
    MILESTONE_PROGRESS_TOOL,
    MILESTONE_2_PROGRESS_TOOL,
    MILESTONE_3_PROGRESS_TOOL,
)
AGENT_OPS_CYCLE_TOOL = "run_reasoning_cycle_and_log"
AGENT_OPS_HEALTH_SUMMARY_TOOL = "summarize_agent_event_health"
DEPTH_COVERAGE_SUMMARY_TOOL = "dashboard_depth_coverage_metrics"
ANALYSIS_INTERVAL_SECONDS = 3
AGENT_OPS_INTERVAL_SECONDS = 10
AGENT_OPS_MAX_TRACKS = 10
AGENT_OPS_SUMMARY_EVERY_CYCLES = 5
AGENT_OPS_SUMMARY_WINDOW_SIZE = 100
DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES = 5
DEPTH_COVERAGE_SAMPLE_LIMIT = 50
OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
OPENAI_MODEL = "openai/gpt-oss-120b"


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def parse_cli_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run object behavior MCP agent")
    parser.add_argument(
        "--no-milestone",
        action="store_true",
        help="Skip milestone snapshot output during startup",
    )
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
    return parser.parse_args(args)

def load_system_skill() -> str:
    """Load instructions used to summarize YOLO tracking frames."""
    selected_path = None
    for candidate in (SKILL_PATH, SKILL_FALLBACK_PATH):
        if candidate.is_file():
            selected_path = candidate
            break

    if selected_path is None:
        raise FileNotFoundError(
            f"System skill not found: checked {SKILL_PATH} and {SKILL_FALLBACK_PATH}"
        )

    system_skill = selected_path.read_text(encoding="utf-8")
    return (
        "You are the Object Behavior Assistant, an agent that interprets "
        "real-time object tracking observations.\n\n"
        "Follow the movement-analysis skill below whenever interpreting "
        "a tracking frame.\n\n"
        f"{system_skill}"
    )


def create_openai_client() -> AsyncOpenAI:
    """Create the configured summarization client from the environment."""
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to the environment or .env."
        )

    base_url = os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL)

    return AsyncOpenAI(base_url=base_url, api_key=api_key)

async def verify_required_tools(session: ClientSession) -> None:
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
    """Join text blocks from an MCP tool result."""
    return "\n".join(
        item.text
        for item in (result.content or [])
        if hasattr(item, "text") and isinstance(item.text, str)
    )


def get_result_json_object(result: Any) -> dict[str, Any] | None:
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


def format_milestone_snapshot(payload: dict[str, Any]) -> str:
    """Create a concise one-line milestone summary for startup logging."""
    phase = str(payload.get("phase", "unknown"))
    status = str(payload.get("status", "unknown"))
    finished = bool(payload.get("milestone_finished", False))
    completion = payload.get("completion_percent", "n/a")
    completed = payload.get("completed_checks", "?")
    total = payload.get("total_checks", "?")
    return (
        f"Milestone snapshot: phase={phase} | status={status} | "
        f"completion={completion}% | checks={completed}/{total} | finished={finished}"
    )


def format_milestone_completion_notice(payload: dict[str, Any]) -> str | None:
    """Return completion notice text when the milestone is finished."""
    if not bool(payload.get("milestone_finished", False)):
        return None
    phase = str(payload.get("phase", "unknown"))
    return f"✅ Milestone finished: {phase}"


async def print_milestone_snapshot(
    session: ClientSession,
    tool_name: str = MILESTONE_PROGRESS_TOOL,
) -> None:
    """Fetch and print milestone progress summary without blocking startup on failure."""
    try:
        result = await session.call_tool(tool_name, arguments={})
    except Exception as error:
        print(
            f"Milestone snapshot unavailable for {tool_name} ({type(error).__name__}): {error}",
            flush=True,
        )
        return

    if getattr(result, "is_error", False):
        message = get_result_text(result) or "unknown tool error"
        print(f"Milestone snapshot unavailable for {tool_name}: {message}", flush=True)
        return

    payload = get_result_json_object(result)
    if payload is None:
        text = get_result_text(result)
        if text:
            print(f"Milestone snapshot ({tool_name}): {text}", flush=True)
        else:
            print(f"Milestone snapshot unavailable for {tool_name}: no structured data.", flush=True)
        return

    print(format_milestone_snapshot(payload), flush=True)
    completion_notice = format_milestone_completion_notice(payload)
    if completion_notice is not None:
        print(completion_notice, flush=True)


async def print_all_milestone_snapshots(session: ClientSession) -> None:
    """Print startup progress snapshots for all registered milestone phases."""
    for tool_name in MILESTONE_PROGRESS_TOOLS:
        await print_milestone_snapshot(session, tool_name=tool_name)


def format_agent_ops_cycle(payload: dict[str, Any]) -> str:
    """Create concise heartbeat status text from agent-ops cycle payload."""
    cycle = payload.get("cycle")
    if not isinstance(cycle, dict):
        return "Agent ops cycle: unavailable"

    evaluated_tracks = cycle.get("evaluated_tracks", "?")
    success_count = cycle.get("success_count", "?")
    error_count = cycle.get("error_count", "?")
    started_at = cycle.get("started_at", "?")

    return (
        "Agent ops cycle: "
        f"tracks={evaluated_tracks} | ok={success_count} | "
        f"errors={error_count} | started_at={started_at}"
    )


def format_agent_ops_health_summary(payload: dict[str, Any]) -> str:
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

    return (
        "Agent ops health: "
        f"success_rate={success_rate} | error_rate={error_rate} | "
        f"bounded_out_rate={bounded_out_rate} | p50_ms={p50_ms} | p95_ms={p95_ms}"
    )


def format_depth_coverage_summary(payload: dict[str, Any]) -> str:
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
    show_milestone_snapshot: bool,
    enable_agent_ops_heartbeat: bool,
    agent_ops_interval_seconds: float,
    agent_ops_max_tracks: int,
    agent_ops_summary_every_cycles: int,
    agent_ops_summary_window_size: int,
    depth_coverage_summary_enabled: bool,
    depth_coverage_summary_every_cycles: int,
    depth_coverage_sample_limit: int,
) -> str:
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
        f"milestone_snapshot={'on' if show_milestone_snapshot else 'off'} | "
        f"agent_ops_heartbeat={'on' if enable_agent_ops_heartbeat else 'off'} | "
        f"heartbeat_interval_s={agent_ops_interval_seconds} | "
        f"heartbeat_max_tracks={agent_ops_max_tracks} | "
        f"health_summary={summary_mode} | "
        f"health_summary_window={agent_ops_summary_window_size} | "
        f"depth_summary={depth_summary_mode} | "
        f"depth_sample_limit={depth_coverage_sample_limit}"
    )


async def run_agent_ops_cycle(
    session: ClientSession,
    request_evidence_on_uncertainty: bool = True,
    max_tracks: int = AGENT_OPS_MAX_TRACKS,
) -> bool:
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
    return True


async def run_agent_ops_health_summary(
    session: ClientSession,
    window_size: int = AGENT_OPS_SUMMARY_WINDOW_SIZE,
) -> bool:
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

    print(format_agent_ops_health_summary(payload), flush=True)
    return True


async def run_depth_coverage_summary(
    session: ClientSession,
    sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> bool:
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


async def print_depth_coverage_startup_snapshot(
    session: ClientSession,
    sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> None:
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


async def summarize_frame(
    client: AsyncOpenAI,
    system_prompt: str,
    frame_data: str,
) -> str:
    """Ask the language model to describe a YOLO frame returned by MCP."""
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Summarize the objects and their positions in this latest "
                    f"YOLO tracking frame:\n{frame_data}"
                ),
            },
        ],
    )

    if not response.choices:
        raise RuntimeError("The language model returned no choices.")

    return response.choices[0].message.content or "(no summary returned)"

async def monitor_movement(
    session: ClientSession,
    client: AsyncOpenAI,
    system_prompt: str,
    enable_agent_ops_heartbeat: bool = True,
    agent_ops_interval_seconds: float = AGENT_OPS_INTERVAL_SECONDS,
    agent_ops_max_tracks: int = AGENT_OPS_MAX_TRACKS,
    agent_ops_summary_every_cycles: int = AGENT_OPS_SUMMARY_EVERY_CYCLES,
    agent_ops_summary_window_size: int = AGENT_OPS_SUMMARY_WINDOW_SIZE,
    depth_coverage_summary_enabled: bool = True,
    depth_coverage_summary_every_cycles: int = DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
    depth_coverage_sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> None:
    """Receive frames from the tracker and print an interpretation."""
    address = [TRACKING_ADDRESS[0], TRACKING_ADDRESS[1]]

    print(
        "Automatic movement analysis started. Press Ctrl+C to stop.",
        flush=True,
    )

    loop = asyncio.get_running_loop()
    summary_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    agent_ops_enabled = enable_agent_ops_heartbeat
    last_agent_ops_cycle_at = -agent_ops_interval_seconds
    agent_ops_summary_enabled = enable_agent_ops_heartbeat and agent_ops_summary_every_cycles > 0
    depth_summary_enabled = (
        enable_agent_ops_heartbeat
        and depth_coverage_summary_enabled
        and depth_coverage_summary_every_cycles > 0
    )
    completed_agent_ops_cycles = 0

    async def summarize_frames() -> None:
        while True:
            frame_data = await summary_queue.get()
            try:
                summary = await summarize_frame(client, system_prompt, frame_data)
            except Exception as error:
                print(
                    f"Movement summary failed ({type(error).__name__}): {error}",
                    flush=True,
                )
            else:
                print(f"\nMovement summary:\n{summary}", flush=True)

    summary_task = asyncio.create_task(summarize_frames())
    try:
        while True:
            iteration_started_at = loop.time()
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
                    if summary_queue.full():
                        summary_queue.get_nowait()
                        print(
                            "Replacing queued movement summary with the latest frame.",
                            flush=True,
                        )
                    summary_queue.put_nowait(frame_data)

                    now = loop.time()
                    if agent_ops_enabled and (now - last_agent_ops_cycle_at) >= agent_ops_interval_seconds:
                        agent_ops_enabled = await run_agent_ops_cycle(
                            session,
                            max_tracks=agent_ops_max_tracks,
                        )
                        if agent_ops_enabled:
                            completed_agent_ops_cycles += 1
                            if (
                                agent_ops_summary_enabled
                                and completed_agent_ops_cycles % agent_ops_summary_every_cycles == 0
                            ):
                                agent_ops_summary_enabled = await run_agent_ops_health_summary(
                                    session,
                                    window_size=agent_ops_summary_window_size,
                                )
                            if (
                                depth_summary_enabled
                                and completed_agent_ops_cycles % depth_coverage_summary_every_cycles == 0
                            ):
                                depth_summary_enabled = await run_depth_coverage_summary(
                                    session,
                                    sample_limit=depth_coverage_sample_limit,
                                )
                        last_agent_ops_cycle_at = loop.time()

            elapsed = loop.time() - iteration_started_at
            await asyncio.sleep(max(0, ANALYSIS_INTERVAL_SECONDS - elapsed))
    finally:
        summary_task.cancel()
        try:
            await summary_task
        except asyncio.CancelledError:
            pass


async def run_agent(
    show_startup_banner: bool = True,
    show_milestone_snapshot: bool = True,
    show_startup_depth_snapshot: bool = True,
    enable_agent_ops_heartbeat: bool = True,
    agent_ops_interval_seconds: float = AGENT_OPS_INTERVAL_SECONDS,
    agent_ops_max_tracks: int = AGENT_OPS_MAX_TRACKS,
    agent_ops_summary_every_cycles: int = AGENT_OPS_SUMMARY_EVERY_CYCLES,
    agent_ops_summary_window_size: int = AGENT_OPS_SUMMARY_WINDOW_SIZE,
    depth_coverage_summary_enabled: bool = True,
    depth_coverage_summary_every_cycles: int = DEPTH_COVERAGE_SUMMARY_EVERY_CYCLES,
    depth_coverage_sample_limit: int = DEPTH_COVERAGE_SAMPLE_LIMIT,
) -> None:
    """Connect to the MCP server, start tracking, and analyze its feed."""
    system_prompt = load_system_skill()
    client = create_openai_client()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    if show_startup_banner:
        print("Initializing MCP pipeline and connecting to server.py...", flush=True)
        print(
            format_runtime_config_banner(
                show_milestone_snapshot=show_milestone_snapshot,
                enable_agent_ops_heartbeat=enable_agent_ops_heartbeat,
                agent_ops_interval_seconds=agent_ops_interval_seconds,
                agent_ops_max_tracks=agent_ops_max_tracks,
                agent_ops_summary_every_cycles=agent_ops_summary_every_cycles,
                agent_ops_summary_window_size=agent_ops_summary_window_size,
                depth_coverage_summary_enabled=depth_coverage_summary_enabled,
                depth_coverage_summary_every_cycles=depth_coverage_summary_every_cycles,
                depth_coverage_sample_limit=depth_coverage_sample_limit,
            ),
            flush=True,
        )

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                if show_milestone_snapshot:
                    await print_all_milestone_snapshots(session)
                if show_startup_depth_snapshot:
                    await print_depth_coverage_startup_snapshot(
                        session,
                        sample_limit=depth_coverage_sample_limit,
                    )
                await verify_required_tools(session)
                await start_live_tracking(session)
                await monitor_movement(
                    session,
                    client,
                    system_prompt,
                    enable_agent_ops_heartbeat=enable_agent_ops_heartbeat,
                    agent_ops_interval_seconds=agent_ops_interval_seconds,
                    agent_ops_max_tracks=agent_ops_max_tracks,
                    agent_ops_summary_every_cycles=agent_ops_summary_every_cycles,
                    agent_ops_summary_window_size=agent_ops_summary_window_size,
                    depth_coverage_summary_enabled=depth_coverage_summary_enabled,
                    depth_coverage_summary_every_cycles=depth_coverage_summary_every_cycles,
                    depth_coverage_sample_limit=depth_coverage_sample_limit,
                )
    finally:
        await client.close()

def main() -> None:
    options = parse_cli_args(sys.argv[1:])
    try:
        asyncio.run(
            run_agent(
                show_startup_banner=not options.quiet_startup_banner,
                show_milestone_snapshot=not options.no_milestone,
                show_startup_depth_snapshot=not options.no_startup_depth_snapshot,
                enable_agent_ops_heartbeat=not options.disable_agent_ops_heartbeat,
                agent_ops_interval_seconds=options.agent_ops_interval_seconds,
                agent_ops_max_tracks=options.agent_ops_max_tracks,
                agent_ops_summary_every_cycles=options.agent_ops_summary_every_cycles,
                agent_ops_summary_window_size=options.agent_ops_summary_window_size,
                depth_coverage_summary_enabled=not options.disable_depth_coverage_summary,
                depth_coverage_summary_every_cycles=options.depth_coverage_summary_every_cycles,
                depth_coverage_sample_limit=options.depth_coverage_sample_limit,
            )
        )
    except KeyboardInterrupt:
        print("\nStopping movement analysis.", flush=True)

if __name__ == "__main__":
    main()