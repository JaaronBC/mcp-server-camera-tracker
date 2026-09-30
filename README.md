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

- User prompt: [capture webcam footage using mcp-server-camera-tracker]
- MCP tool call: `capture_webcam_video(duration_seconds)`
- Input: capture webcam footage using mcp-server-camera-tracker for 2 seconds
- Result: 
  The webcam capture worked. There were no errors.
  Result: "Video recorded successfully"
  Duration requested: 2 seconds
  Saved to: captures\webcam_1790752985.mp4

### Trace 2: Detect and track dogs

- User prompt: [detect dogs in video source using mcp-server-camera-tracker]
- MCP tool call: `detect_and_track_dogs(video_path)`
- Input: detect dogs in "local/source/link" using mcp-server-camera-tracker
- Result: 
  Dogs detected: 1 (tracked as ID 1)
  Movement: Stationary
  Horizontal displacement: 6.2 (a very small shift, likely just minor movement or tracking jitter rather than real travel)
  Frames tracked: 214

### Trace 3: Analyze dog movement

- User prompt: [detect dog movement in *json file path* using mcp-server-camera-tracker]
- MCP tool call: `analyze_dog_movement(tracking_data)`
- Input: use mcp-server-camera-tracker to analyze the following *json file path*
- Result: 
  Dog ID: 1
  Movement: Stationary
  Horizontal displacement: 6.2
  Frames tracked: 214

## MCP Architecture

The MCP server is the local program that calls on our yolo api in a multitude of ways. It can use webcam tracking, instance detection, and movement detections all relating to dogs. The server performs the requested work and returns structured results. The entity that decides how to make these structured calls is the LLM.


The MCP client is the host integration, and in this instance we use Claude Desktop. The client connects the LLM to the MCP server, and the LLM checks the available tools, and sends requests using them. The LLM then returns the corresponding results back to the user. In this case, the user can submit webcam access, video source links, and data file source links.


One limitation we observed when testing is that some devices don’t run the webcam functionality. On one of our devices, the webcam worked fine but on the other it returned an error. This is likely due to different configurations of permissions access which could cause problems for different users.


