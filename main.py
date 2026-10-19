# main.py
import os
import asyncio
import json
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from requests import session
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SKILL_PATH = (
    PROJECT_ROOT
    / "Skills.md"
)

def load_movement_skill() -> str:
    """Load movement-analysis instructions from the skill file."""
    if not SKILL_PATH.is_file():
        raise FileNotFoundError(
            f"Movement skill not found: {SKILL_PATH}"
        )

    return SKILL_PATH.read_text(encoding="utf-8")


movement_skill = load_movement_skill()

SYSTEM_PROMPT = f"""
You are the Object Behavior Assistant, an agent that interprets
real-time object tracking observations.

Follow the movement-analysis skill below whenever interpreting
a movement event or a saved tracking capture.

{movement_skill}
"""


load_dotenv()

# Ensure you have your OpenAI API key set in your environment
# or in a .env file: OPENAI_API_KEY="your-key-here"
openai_client = OpenAI(
    base_url="https://128.171.10.80:9443/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

def parse_events(result):
    """Convert the MCP tool response into a list of event dictionaries."""
    events = []

    for item in result.content or []:
        if not hasattr(item, "text"):
            continue

        try:
            data = json.loads(item.text)
        except json.JSONDecodeError:
            continue

        if isinstance(data, list):
            events.extend(
                event for event in data
                if isinstance(event, dict)
            )
        elif isinstance(data, dict) and data.get("type") == "movement_detected":
            events.append(data)

    return events

async def run_agent():
    # 1. Configure parameters to execute your server.py file via standard I/O
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )
    
    print("Initializing MCP pipeline and connecting to server.py...")
    
    # 2. Establish the MCP Client -> Server connection
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection with the server
            await session.initialize()
            
            # List available tools exposed by the MCP server
            mcp_tools = await session.list_tools()
            
            # 3. Convert MCP tools into OpenAI-compatible schemas
            openai_tools = []

            for tool in mcp_tools.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                })
            

            # Automatically start live tracking on startup.
            live_tracking_tool = "run_live_object_tracking"

            available_tool_names = [
                tool.name for tool in mcp_tools.tools
            ]

            if live_tracking_tool not in available_tool_names:
                print(
                    f"Error: {live_tracking_tool} is not registered. "
                    f"Available tools: {available_tool_names}"
                )
                return

            print("\nStarting live tracking...")

            result = await session.call_tool(
                live_tracking_tool,
                arguments={}
            )

            if getattr(result, "is_error", False):
                print("Failed to start live tracking:", result.content)
                return

            print(
                "\nLive tracking started. "
                "Press 'q' in the webcam window to stop tracking."
            )

            print(
                "\nAutomatic movement monitoring started. "
                "Press Ctrl+C to stop.",
                flush=True,
            )

            analysis_tool = "analyze_object_movement"
            available_tool_names = [
                tool.name for tool in mcp_tools.tools
            ]

            if analysis_tool not in available_tool_names:
                print(
                    f"Error: {analysis_tool} is not registered. "
                    f"Available tools: {available_tool_names}"
                )
                return

            try:
                while True:
                    # Check for new movement events.
                    result = await session.call_tool(
                        "get_pending_movement_events",
                        arguments={},
                    )

                    if getattr(result, "is_error", False):
                        print(
                            "Failed to retrieve movement events:",
                            result.content,
                            flush=True,
                        )
                        await asyncio.sleep(1)
                        continue

                    events = parse_events(result)

                    # Only call the analysis tool when events exist.
                    for event in events:
                        print(
                            "\nMovement detected:",
                            json.dumps(event),
                            flush=True,
                        )

                        # Pass the actual event to the analysis tool.
                        analysis_result = await session.call_tool(
                            analysis_tool,
                            arguments={
                                "tracking_data": json.dumps(event)
                            },
                        )

                        analysis_text = "\n".join(
                            item.text
                            for item in (analysis_result.content or [])
                            if hasattr(item, "text")
                        )

                        if getattr(analysis_result, "is_error", False):
                            print(
                                "Movement analysis failed:",
                                analysis_text,
                                flush=True,
                            )
                            continue

                        # Ask the LLM to summarize this event's analysis.
                        response = await asyncio.to_thread(
                            openai_client.chat.completions.create,
                            model="openai/gpt-oss-120b",
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": 
                                    f"Movement event:\n"
                                    f"{json.dumps(event)}\n\n"
                                    f"Analysis result:\n{analysis_text}\n\n"
                                },
                            ]
                        )

                        print(
                            "\nAgent:",
                            response.choices[0].message.content,
                            flush=True,
                        )

                    # Avoid continuously hammering the MCP server.
                    await asyncio.sleep(1)

            except KeyboardInterrupt:
                print("\nStopping automatic monitoring.", flush=True)

if __name__ == "__main__":
    asyncio.run(run_agent())