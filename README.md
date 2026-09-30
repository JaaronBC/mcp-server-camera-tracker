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
