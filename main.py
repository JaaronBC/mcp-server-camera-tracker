# main.py
import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Ensure you have your OpenAI API key set in your environment
# or in a .env file: OPENAI_API_KEY="your-key-here"
openai_client = OpenAI(
    base_url="https://128.171.10.80:9443/v1",
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
            
            # 3. Convert MCP tools into OpenAI-compatible function schemas
            openai_tools = []
            for tool in mcp_tools.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })
            
            # 4. Get a prompt from the terminal user
            user_prompt = input("\nAsk the Agent something (e.g., 'What's the weather like in Honolulu?'): ")
            
            messages = [
                {"role": "system", "content": "You are a helpful terminal agent. Use your tools to answer questions accurately."},
                {"role": "user", "content": user_prompt}
            ]
            
            # 5. First LLM Pass: Decide if a tool is needed
            response = openai_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # 6. Execute tool call if requested by the LLM
            if response_message.tool_calls:
                messages.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    # Arguments are provided as a JSON string from the LLM
                    tool_args = eval(tool_call.function.arguments) 
                    
                    print(f"LLM requested tool execution: {tool_name}({tool_args})")
                    
                    # Call the actual Python logic inside server.py across the MCP boundary
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    # Pull text out of the content block
                    tool_output = result.content[0].text 
                    
                    # Append the tool result back into the LLM history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_output
                    })
                
                # 7. Second LLM Pass: Generate final response with tool data loaded
                final_response = openai_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages
                )
                print(f"\nAgent: {final_response.choices[0].message.content}")
            else:
                # If no tool was needed, just print the direct output
                print(f"\nAgent: {response_message.content}")

if __name__ == "__main__":
    asyncio.run(run_agent())
