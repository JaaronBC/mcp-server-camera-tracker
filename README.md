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

Set shared model endpoint configuration in your environment or `.env`:

- `OPENROUTER_API_KEY=<your_api_key>`
- `OPENAI_BASE_URL=https://128.171.10.83:9443/v1`

If `OPENAI_BASE_URL` is unset, the project defaults to `https://128.171.10.83:9443/v1`.

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
- `start_dashboard_demo()`: starts the webcam + radar dashboard demo in a background process.
- `start_dashboard_and_wait_ready(timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: starts dashboard and waits until fresh observations are available or timeout occurs.
- `dashboard_demo_status()`: returns whether the dashboard process is running, PID, and exit code.
- `dashboard_lifecycle_status(max_observation_age_seconds=3.0)`: returns a merged lifecycle snapshot (process + health + calibration) and an overall ready flag.
- `stop_dashboard_demo(force=False)`: stops the running dashboard process.
- `stop_dashboard_and_confirm(force=False, timeout_seconds=5.0, poll_interval_seconds=0.1)`: stops dashboard and polls until process is confirmed down or timeout occurs.
- `restart_dashboard_and_wait_ready(force_stop=False, stop_timeout_seconds=5.0, start_timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: performs stop-confirm then start-and-ready orchestration.
- `dashboard_bootstrap(calibration=None, restart_if_running=False, force_stop=False, stop_timeout_seconds=5.0, start_timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: optional calibration apply + start/restart orchestration + lifecycle snapshot in one call.
- `milestone_progress_status()`: returns current dashboard milestone progress summary against project design goals.
- `milestone_2_progress_status()`: returns milestone 2 progress summary for agent pipeline and observability goals.
- `milestone_3_progress_status()`: returns milestone 3 progress summary for VLM-based tracking quality and evaluation hardening goals.
- `simulate_observation_summary(track_id)`: ingests latest radar state for a track and returns deterministic observation/behavior summary fields.
- `simulate_reasoning_assessment(track_id, request_evidence_on_uncertainty=True)`: ingests latest radar state for a track and returns observation summary, reasoning assessment, and optional bounded evidence request.
- `simulate_evidence_request(track_id, reason)`: issues a bounded evidence request for a track and returns request metadata or a bounded-out indicator.
- `simulate_visual_tracking_association(detections, frame_reference=None, event_reasons=None, timeout_seconds=1.0)`: runs the visual tracking agent on selected frames plus structured perception input and returns coordinator-validated association updates.
- `run_reasoning_cycle_and_log(request_evidence_on_uncertainty=True, max_tracks=10)`: evaluates active tracks, records per-track status/latency/assessment events to `captures/agent_events.jsonl`, and returns cycle summary.
- `get_recent_agent_events(limit=20)`: reads recent structured events from `captures/agent_events.jsonl`.
- `summarize_agent_event_health(window_size=100)`: summarizes recent agent events with success/error rates, bounded-out rate, latency percentiles, and assessment-label counts.
- `prune_agent_event_log(max_events_keep=5000, archive_removed_events=True)`: prunes older event-log lines and optionally writes removed lines to a timestamped archive file.
- `dashboard_latency_continuity_metrics()`: computes update-latency and basic tracking continuity metrics from current dashboard snapshots and latest tracking history file.
- `dashboard_depth_coverage_metrics(sample_limit=50)`: reports latest relative-depth coverage and trend direction from recent radar coverage history samples.
- `dashboard_tracking_quality_report(event_window_size=100, depth_sample_limit=50)`: returns a consolidated tracking-quality status using latency/continuity, depth coverage trend, and agent event health.
- `dashboard_health_status()`: reads latest radar state and returns camera/detector status, mapping mode, track count, and observation age.
- `wait_for_dashboard_ready(timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: polls health until fresh observations are available or timeout.
- `set_ground_plane_calibration(calibration)`: saves and validates the calibration JSON payload.
- `get_ground_plane_calibration()`: returns the current calibration file payload.
- `validate_ground_plane_calibration()`: validates the current calibration file and returns summary metadata.
- `clear_ground_plane_calibration()`: removes the calibration file.

## Object Tracking Dashboard Demo

The dashboard demo is implemented in `tools/components/liveTrackingDisplay.py`.

The Tracking Agent now uses a vision-capable VLM (`qwen-3.8-27b`) asynchronously on selected frames/events while preserving YOLO detections as primary measurements.

It renders:

- live webcam feed with YOLO + ByteTrack overlays,
- a radar-style panel with track markers and short trails,
- status line (`camera`, `detector`, active tracks, last observation, mapping mode),
- track detail panel (ID, timestamp, pixel position, ground estimate, quality flags).

Tracking-agent inputs include YOLO detections, pixel coordinates, bounding boxes, timestamps, optional relative depth field, and motion history snippets. The VLM proposes association continuity and uncertainty; the coordinator validates proposals before use and rejects conflicting overrides.

The live pipeline now attaches a bounded `relative_depth` estimate (`0.0` near to `1.0` far, image-geometry based) and marks quality flags as `relative_depth_estimate:image_geometry` or `relative_depth_unavailable`.

If webcam or detector input is unavailable, the radar automatically switches to clearly labeled simulated tracks so the demo remains usable.

Run from project root:

`python tools/components/liveTrackingDisplay.py`

You can tune radar depth scaling (away-from-camera range) without editing code:

- `DASHBOARD_MAX_GROUND_Z` (default: `20.0`)

Example:

`DASHBOARD_MAX_GROUND_Z=8 python tools/components/liveTrackingDisplay.py`

When launching `main.py`, you can skip startup milestone logging with:

`python main.py --no-milestone`

You can control the periodic agent-ops heartbeat in `main.py`:

- Disable heartbeat:
  `python main.py --disable-agent-ops-heartbeat`
- Faster heartbeat interval:
  `python main.py --agent-ops-interval-seconds 2.5`
- Limit tracks per cycle:
  `python main.py --agent-ops-max-tracks 4`
- Print health summary every 3 heartbeat cycles:
  `python main.py --agent-ops-summary-every-cycles 3`
- Change summary event window size:
  `python main.py --agent-ops-summary-window-size 50`
- Disable periodic summary printing:
  `python main.py --agent-ops-summary-every-cycles 0`
- Disable depth coverage summary:
  `python main.py --disable-depth-coverage-summary`
- Print depth coverage summary every 4 heartbeat cycles:
  `python main.py --depth-coverage-summary-every-cycles 4`
- Change depth coverage trend sample window:
  `python main.py --depth-coverage-sample-limit 80`
- Skip one-time startup depth baseline log:
  `python main.py --no-startup-depth-snapshot`
- Suppress startup banner lines:
  `python main.py --quiet-startup-banner`

Controls:

- `q`: quit dashboard
- `n`: next track selection
- `p`: previous track selection
- `h`: toggle tracking-agent diagnostics (concise/detailed)

Live JSON state snapshots are written to:

- `captures/latest_tracking_frame.json`
- `captures/latest_radar_state.json`

`latest_radar_state.json` now includes `relative_depth_coverage` with:

- `tracks_with_depth`
- `total_tracks`
- `coverage_ratio`
- `coverage_percent`

Depth coverage trend samples are also appended to:

- `captures/radar_depth_coverage_history.jsonl`

For MCP compatibility, the dashboard also emits each frame as UDP JSON on `127.0.0.1:5005` with shape:

- `{ "time": <unix-seconds>, "detections": [...] }`

This keeps `detect_objects(...)` and `main.py`-style consumers working while the dashboard is running.

Coordinate semantics are preserved:

- pixel coordinates (`pixel_bbox`, pixel center) are image-plane measurements,
- relative depth is an estimate field when available,
- calibrated ground-plane coordinates are marked separately,
- uncalibrated estimates are not presented as true 3D world positions.

### Milestone 2 scaffolding

Typed contracts and bounded coordinator logic for the Tracking / Observation / Reasoning flow are now scaffolded in:

- `tools/agent_contracts.py`
- `tools/agent_coordinator.py`

Current coordinator tests:

- `tests/test_agent_coordinator.py`

### Optional ground-plane calibration

If `captures/ground_plane_calibration.json` exists and is valid, the dashboard uses calibrated homography mapping for ground position estimates.

Supported file formats:

1. Point correspondences (minimum 4 points):

```json
{
  "image_points": [
    [120, 460],
    [540, 460],
    [610, 320],
    [60, 320]
  ],
  "ground_points": [
    [-3.0, 1.0],
    [3.0, 1.0],
    [3.5, 9.0],
    [-3.5, 9.0]
  ],
  "ground_coordinate_unit": "meters"
}
```

2. Explicit 3x3 homography matrix:

```json
{
  "matrix": [[...], [...], [...]],
  "ground_coordinate_unit": "meters"
}
```

When calibration is unavailable or invalid, the dashboard falls back to relative image-derived mapping and labels estimates accordingly.

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
- Input: use mcp-server-camera-tracker to analyze the following _json file path_
- Result:
  Dog ID: 1
  Movement: Stationary
  Horizontal displacement: 6.2
  Frames tracked: 214

## MCP Architecture

The MCP server is the local program that calls on our yolo api in a multitude of ways. It can use webcam tracking, instance detection, and movement detections all relating to dogs. The server performs the requested work and returns structured results. The entity that decides how to make these structured calls is the LLM.

The MCP client is the host integration, and in this instance we use Claude Desktop. The client connects the LLM to the MCP server, and the LLM checks the available tools, and sends requests using them. The LLM then returns the corresponding results back to the user. In this case, the user can submit webcam access, video source links, and data file source links.

One limitation we observed when testing is that some devices don’t run the webcam functionality. On one of our devices, the webcam worked fine but on the other it returned an error. This is likely due to different configurations of permissions access which could cause problems for different users.
