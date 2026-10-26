# main.py
import os
import asyncio
from pathlib import Path
from typing import Any
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path
import tkinter

PROJECT_ROOT = Path(__file__).resolve().parent
SKILL_PATH = PROJECT_ROOT / "Skills.md"
SERVER_PATH = PROJECT_ROOT / "server.py"

TRACKING_ADDRESS = ("127.0.0.1", 5005)
LIVE_TRACKING_TOOL = "live_tracking"
DETECT_OBJECTS_TOOL = "detect_objects"
ANALYSIS_INTERVAL_SECONDS = 3
OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
OPENAI_MODEL = "openai/gpt-oss-120b"

def load_system_skill() -> str:
    """Load instructions used to summarize YOLO tracking frames."""
    if not SKILL_PATH.is_file():
        raise FileNotFoundError(f"System skill not found: {SKILL_PATH}")
    system_skill = SKILL_PATH.read_text(encoding="utf-8")
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

    return AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)

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
        if hasattr(item, "text")
    )

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
) -> None:
    """Receive frames from the tracker and print an interpretation."""
    address = [TRACKING_ADDRESS[0], TRACKING_ADDRESS[1]]

    print(
        "Automatic movement analysis started. Press Ctrl+C to stop.",
        flush=True,
    )

    loop = asyncio.get_running_loop()
    summary_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)

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

            elapsed = loop.time() - iteration_started_at
            await asyncio.sleep(max(0, ANALYSIS_INTERVAL_SECONDS - elapsed))
    finally:
        summary_task.cancel()
        try:
            await summary_task
        except asyncio.CancelledError:
            pass


async def run_agent() -> None:
    """Connect to the MCP server, start tracking, and analyze its feed."""
    system_prompt = load_system_skill()
    client = create_openai_client()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    print("Initializing MCP pipeline and connecting to server.py...", flush=True)

    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await verify_required_tools(session)
                await start_live_tracking(session)
                await monitor_movement(session, client, system_prompt)
    finally:
        await client.close()

def main() -> None:
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nStopping movement analysis.", flush=True)

if __name__ == "__main__":
    main()