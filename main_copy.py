# main.py
import os
import asyncio
import json
import sys
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SKILL_PATH = (
    PROJECT_ROOT
    / "Skills2.md"
)

ADDRESS = ("127.0.0.1", 5005)

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

Follow the movement-analysis skill below.

{movement_skill}
"""


load_dotenv()

# Ensure you have your OpenAI API key set in your environment
# or in a .env file: OPENAI_API_KEY="your-key-here"
openai_client = OpenAI(
    base_url="https://128.171.10.83:9443/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

async def run_agent():
    server_path = PROJECT_ROOT / "server_copy.py"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
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

            tracking_message = "\n".join(
                item.text
                for item in (result.content or [])
                if hasattr(item, "text")
            )

            if getattr(result, "is_error", False):
                print("Failed to start live tracking:", tracking_message)
                return

            print(
                tracking_message or
                "Live tracking started. Press 'q' in the webcam window to stop tracking.",
                flush=True,
            )

            print(
                "\nAutomatic movement analysis started. "
                "Press Ctrl+C to stop.",
                flush=True,
            )

            analysis_tool = "analyze_movement_2"
            if analysis_tool not in available_tool_names:
                print(
                    f"Error: {analysis_tool} is not registered. "
                    f"Available tools: {available_tool_names}"
                )
                return

            analysis_tool_definition = next(
                tool
                for tool in openai_tools
                if tool["function"]["name"] == analysis_tool
            )

            try:
                while True:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Analyze the latest live webcam frame. Call "
                                f"{analysis_tool} exactly once with the address "
                                f"{json.dumps(list(ADDRESS))}, then summarize "
                                "the returned detections."
                            ),
                        },
                    ]

                    response = await asyncio.to_thread(
                        openai_client.chat.completions.create,
                        model="openai/gpt-oss-120b",
                        messages=messages,
                        tools=[analysis_tool_definition],
                        tool_choice={
                            "type": "function",
                            "function": {"name": analysis_tool},
                        },
                    )

                    assistant_message = response.choices[0].message
                    tool_calls = assistant_message.tool_calls or []
                    if not tool_calls:
                        print(
                            "The LLM did not call the movement analysis tool.",
                            flush=True,
                        )
                        await asyncio.sleep(5)
                        continue

                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    })

                    for tool_call in tool_calls:
                        analysis_text = ""
                        if tool_call.function.name != analysis_tool:
                            analysis_text = (
                                f"Unsupported tool call: "
                                f"{tool_call.function.name}"
                            )
                            print(
                                analysis_text,
                                flush=True,
                            )
                        else:
                            try:
                                arguments = json.loads(
                                    tool_call.function.arguments
                                )
                            except json.JSONDecodeError as error:
                                analysis_text = (
                                    f"Invalid LLM tool arguments: {error}"
                                )
                                print(
                                    analysis_text,
                                    flush=True,
                                )
                            else:
                                if not isinstance(arguments, dict):
                                    analysis_text = (
                                        "Invalid LLM tool arguments: "
                                        "expected an object."
                                    )
                                    print(
                                        analysis_text,
                                        flush=True,
                                    )
                                else:
                                    result = await session.call_tool(
                                        analysis_tool,
                                        arguments=arguments,
                                    )
                                    analysis_text = "\n".join(
                                        item.text
                                        for item in (result.content or [])
                                        if hasattr(item, "text")
                                    )

                                    if getattr(result, "is_error", False):
                                        print(
                                            "Movement analysis failed:",
                                            analysis_text,
                                            flush=True,
                                        )
                                    else:
                                        print(
                                            "\nAnalyzeMovement2 information:\n"
                                            f"{analysis_text or '(no text returned)'}",
                                            flush=True,
                                        )

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": analysis_text,
                        })

                    if len(messages) > 2:
                        summary_response = await asyncio.to_thread(
                            openai_client.chat.completions.create,
                            model="openai/gpt-oss-120b",
                            messages=messages,
                        )
                        summary = summary_response.choices[0].message.content
                        if summary:
                            print(summary, flush=True)
                        else:
                            print(
                                "The LLM returned no movement summary.",
                                flush=True,
                            )

                    await asyncio.sleep(5)
            except KeyboardInterrupt:
                print("\nStopping automatic monitoring.", flush=True)

if __name__ == "__main__":
    asyncio.run(run_agent())