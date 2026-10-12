# main.py
import os
import asyncio
import json
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Ensure you have your OpenAI API key set in your environment
# or in a .env file: OPENAI_API_KEY="your-key-here"
openai_client = OpenAI(
    base_url="https://128.171.10.83:9443/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

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

            # 4. Get the user's request
            user_prompt = input("\nWhat cardinal direction is the camera facing (north, east, south, or west)?:")

            messages = [
                {
                    "role": "system",
                    "content": """
                    You are a real-time object-tracking assistant connected to an MCP server.

                    Before interacting with you the user will be asked to provide the cardinal direction of the camera (North, East, South, or West).
                    1. The User will provide the cardinal direction the camera is facing: north, east, south, or west.
                    2. Wait for the user's response before proceeding. If the response is unclear or is not one of the four cardinal directions, ask the user to clarify.
                    3. Store the user's answer as the camera's cardinal direction for the current tracking session.
                    4. After receiving a valid answer, call run_live_object_tracking to start real-time tracking.
                    5. Use actual tool results to describe the tracking status. Do not claim tracking started successfully unless the tool confirms it.
                    6. When discussing camera orientation, make clear that the direction was provided by the user and was not independently measured.
                    7. If the user asks about detected objects, tracking IDs, or movement, base your answer on the actual tracking results available from the MCP tools.

                    Important rules:
                    - Never start real-time tracking before the user provides a valid cardinal direction.
                    - Do not call tools for recorded-video capture or analysis.
                    - If a tool fails, explain the error. Do not repeat the same operation automatically.
                    - Never invent tracking results or claim tracking succeeded without evidence from the tools.
                    - If tracking is already runing, do not start another tracking session unless the user explicitly requests it.
                    - If the user asks to stop tracking, stop it if a supported tool is available. Otherwise, explain how to stop the current tracking process.
                    """,
                },
                {"role": "user", "content": user_prompt},
            ]

            # 5. LLM tool-calling loop
            max_loops = 10

            for loop_number in range(max_loops):
                print(f"\nLoop {loop_number + 1}...")

                response = openai_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )

                response_message = response.choices[0].message

                # Save the assistant message, including tool calls.
                messages.append(
                    response_message.model_dump(exclude_none=True)
                )

                # No tool calls means the model has finished.
                if not response_message.tool_calls:
                    print(f"\nAgent: {response_message.content}")
                    return

                # 6. Execute every requested MCP tool.
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name

                    try:
                        # Parse JSON safely instead of using eval().
                        tool_args = json.loads(
                            tool_call.function.arguments
                        )

                        print(
                            f"\nLLM requested tool: {tool_name}"
                        )
                        print(f"Arguments: {tool_args}")

                        result = await session.call_tool(
                            tool_name,
                            arguments=tool_args,
                        )

                        # Collect any text returned by the MCP tool.
                        tool_output = "\n".join(
                            item.text
                            for item in (result.content or [])
                            if hasattr(item, "text")
                        )
                        
                        # Check the error flag used by this MCP SDK.
                        if getattr(result, "is_error", False):
                            tool_output = "MCP tool error: " + tool_output

                        if not tool_output:
                            tool_output = "MCP tool returned no output."
                        
                        #Stop the model from retrying a failed tool indefinitely.
                        if getattr(result, "is_error", False):
                            tool_output += (
                                " The tool failed, and the model should not "
                                "retry this operation automatically. Explain the error to the user instead."
                            )
                    except Exception as exc:
                        tool_output = (
                            f"Tool {tool_name} failed: {exc}"
                        )

                    print(f"Tool result: {tool_output}")

                    # 7. Give the result back to the LLM.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    })

            print(
                "\nAgent stopped after reaching the maximum "
                "number of tool-calling rounds."
            )


if __name__ == "__main__":
    asyncio.run(run_agent())