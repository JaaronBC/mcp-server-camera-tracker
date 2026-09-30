# Dog Behavior MCP Server

An MCP server for recording webcam video, detecting and tracking dogs, and analyzing their horizontal movement.

## Requirements

- Python 3.12 or newer
- Node.js and npm for the MCP Inspector used by `mcp dev`
- `uv` for the development command, or use the installed `mcp` command directly

## Setup (Terminal)

From the project directory, create and activate a virtual environment, then install the Python dependencies:

Windows:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Mac:
python3.13 -m venv .venv
source .venv/bin/activate 
python -m pip install -r requirements.txt

## Setup (LLM)

The LLM we connected to was Claude Desktop. In order to connect, go to the claude_desktop_config.json file. It's accessible through the path: file->settings->Developer->Edit Config

Edit the "mcpServers" list to include the mcp-server-camera-tracker through the following command:
"mcpServers": {
    "mcp-server-camera-tracker": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\path\\to\\file",
        "run",
        "server.py"
      ]
    }
}

## Available tools

- `capture_webcam_video(duration_seconds)`: records a webcam video for 1 to 30 seconds in the `captures` directory.
- `detect_and_track_dogs(video_path)`: detects dogs in a video and writes tracking positions to a neighboring JSON file.
- `analyze_dog_movement(tracking_data)`: classifies each tracked dog's horizontal movement from the tracking JSON file.

## Demonstration Traces

### Trace 1: Capture a webcam video

- User prompt: [add the prompt used to request a recording]
- MCP tool call: `capture_webcam_video(duration_seconds)`
- Input: [add the duration]
- Result: [add the generated video path and a screenshot or transcript excerpt]

### Trace 2: Detect and track dogs

- User prompt: [add the prompt used to analyze a captured video]
- MCP tool call: `detect_and_track_dogs(video_path)`
- Input: [add the video path]
- Result: [add the tracking JSON path and a screenshot or transcript excerpt]

### Trace 3: Analyze dog movement

- User prompt: [add the prompt used to classify movement]
- MCP tool call: `analyze_dog_movement(tracking_data)`
- Input: [add the tracking data path or content]
- Result: [add the movement classification and a screenshot or transcript excerpt]

## MCP Architecture

The MCP server is the local program that exposes this project's webcam, detection, tracking, and movement-analysis tools. It performs the requested work and returns structured results without deciding what the user should ask next. The MCP client is the host integration, such as Claude Desktop. It connects the LLM to the server, discovers the available tools, sends tool requests, and displays the returned results. The LLM interprets the user's natural-language request, selects an appropriate tool, supplies its arguments, and explains the result in conversational language. In a typical trace, the LLM requests a recording through the client, the client forwards that request to the MCP server, and the server returns the output path. The client passes that result back to the LLM, which may then request tracking or movement analysis as a follow-up step.

