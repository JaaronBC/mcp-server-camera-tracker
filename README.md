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

-

## Testing

Default test runs use mocked model responses where appropriate and do not require live endpoint access:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Optional live Observation/Reasoning endpoint checks are provided in `tests/test_orchestration_reasoning_live_llm.py`.
These run only when explicitly enabled and require `OPENROUTER_API_KEY`:

```bash
RUN_LIVE_LLM_TESTS=1 python -m unittest tests.test_orchestration_reasoning_live_llm -v
```

## Migration summary (dashboard → orchestration)

Canonical control-plane imports are now under `agents/orchestration/*`.
The legacy `agents/dashboard/*` package has been removed.

- `tools.dashboard.agent_ops` → `tools.orchestration.agent_ops`
- `tools.dashboard.calibration` → `tools.orchestration.calibration`
- `tools.dashboard.control` → `tools.orchestration.control`
- `tools.dashboard.detect_objects_tool` → `tools.orchestration.detect_objects`
- `tools.dashboard.health` → `tools.orchestration.health`
- `tools.dashboard.live_tracking_tool` → `tools.orchestration.live_tracking`
- `tools.dashboard.metrics` → `tools.orchestration.metrics`
- `tools.dashboard.process` → `tools.orchestration.process`

Dashboard-oriented test modules were also renamed to orchestration naming (for example, `tests/test_dashboard_control.py` → `tests/test_orchestration_control.py`).

### Post-migration cleanup checklist

- Confirm no legacy dashboard imports remain in Python sources:
  `python - <<'PY'\nfrom pathlib import Path\npatterns=['agents.dashboard','from ..dashboard','from .dashboard']\nfor p in Path('.').rglob('*.py'):\n    if any(x in p.parts for x in ['.git','.venv','source','__pycache__']):\n        continue\n    t=p.read_text(encoding='utf-8')\n    if any(s in t for s in patterns):\n        print(p)\nPY`
- Run full test discovery:
  `python -m unittest discover -s tests -p 'test_*.py'`
- If desired, clear local project bytecode caches (keeps `.venv` intact):
  `find . \( -path './.git' -o -path './.venv' -o -path './source' \) -prune -o -type d -name '__pycache__' -exec rm -rf {} +`

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

Canonical orchestration-prefixed tool names are available for control-plane operations.
Existing dashboard-prefixed names are retained as compatibility aliases.

- `start_dashboard_demo()`: starts the webcam + radar dashboard demo in a background process.
- `start_dashboard_and_wait_ready(timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: starts dashboard and waits until fresh observations are available or timeout occurs.
- `dashboard_demo_status()`: returns whether the dashboard process is running, PID, and exit code.
- `dashboard_lifecycle_status(max_observation_age_seconds=3.0)`: returns a merged lifecycle snapshot (process + health + calibration) and an overall ready flag.
- `stop_dashboard_demo(force=False)`: stops the running dashboard process.
- `stop_dashboard_and_confirm(force=False, timeout_seconds=5.0, poll_interval_seconds=0.1)`: stops dashboard and polls until process is confirmed down or timeout occurs.
- `restart_dashboard_and_wait_ready(force_stop=False, stop_timeout_seconds=5.0, start_timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: performs stop-confirm then start-and-ready orchestration.
- `dashboard_bootstrap(calibration=None, restart_if_running=False, force_stop=False, stop_timeout_seconds=5.0, start_timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: optional calibration apply + start/restart orchestration + lifecycle snapshot in one call.
- `simulate_observation_summary(track_id)`: ingests latest radar state for a track and returns an LLM-backed observation/behavior summary with measurements, evidence references, uncertainties, and graceful fallback metadata.
- `simulate_reasoning_assessment(track_id, request_evidence_on_uncertainty=True)`: ingests latest radar state for a track and returns an LLM-backed reasoning assessment plus optional bounded evidence request.
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
- `verify_dashboard_pipeline_contract(max_observation_age_seconds=3.0, require_live_pipeline=False)`: validates freshness, simulation-mode usage, track schema fields, and coordinate-unit consistency for the connected pipeline contract.
- `wait_for_dashboard_ready(timeout_seconds=10.0, max_observation_age_seconds=3.0, poll_interval_seconds=0.25)`: polls health until fresh observations are available or timeout.
- `set_ground_plane_calibration(calibration)`: saves and validates the calibration JSON payload.
- `get_ground_plane_calibration()`: returns the current calibration file payload.
- `validate_ground_plane_calibration()`: validates the current calibration file and returns summary metadata.
- `clear_ground_plane_calibration()`: removes the calibration file.

## Object Tracking Dashboard Demo

The dashboard demo is implemented in `agents/dashboard/live_tracking_display.py`.

The Tracking Agent now uses a vision-capable VLM (`qwen-3.8-27b`) asynchronously on selected frames/events while preserving YOLO detections as primary measurements.

Tracking-agent system guidance is loaded from `TrackingAgentSkill.md`.
You can tune association behavior, uncertainty calibration, and estimation policy there without changing Python code.

Reasoning pipeline agents are also LLM-backed by default:

- Observation/Behavior Agent model: `OBSERVATION_AGENT_MODEL` (default: `gpt-oss-120b`)
- Reasoning/Decision-Support Agent model: `REASONING_AGENT_MODEL` (default: `gpt-oss-120b`)
- Shared call timeout: `DASHBOARD_AGENT_LLM_TIMEOUT_SECONDS` (default: `8.0`)
- Shared retry count: `DASHBOARD_AGENT_LLM_RETRY_LIMIT` (default: `1`, meaning up to 2 attempts)

Example:

`OBSERVATION_AGENT_MODEL=gpt-oss-120b REASONING_AGENT_MODEL=gpt-oss-120b DASHBOARD_AGENT_LLM_TIMEOUT_SECONDS=10 python main.py`

It renders:

- live webcam feed with YOLO + ByteTrack overlays,
- a radar-style panel with track markers and short trails,
- status line (`camera`, `detector`, active tracks, last observation, mapping mode),
- latest validated assessment summary (`status`, `assessment label`, `handoff quality`, freshness) plus limitations/error codes,
- track detail panel (ID, timestamp, pixel position, ground estimate, quality flags).

Tracking-agent inputs include YOLO detections, pixel coordinates, bounding boxes, timestamps, optional relative depth field, and motion history snippets. The VLM proposes association continuity and uncertainty; the coordinator validates proposals before use and rejects conflicting overrides.

The live pipeline now attaches a bounded `relative_depth` estimate (`0.0` near to `1.0` far, image-geometry based) and marks quality flags as `relative_depth_estimate:image_geometry` or `relative_depth_unavailable`.

If webcam or detector input is unavailable, the radar automatically switches to clearly labeled simulated tracks so the demo remains usable.

Run from project root:

`python agents/dashboard/live_tracking_display.py`

You can tune radar depth scaling (away-from-camera range) without editing code:

- `DASHBOARD_MAX_GROUND_Z` (default: `20.0`)

Example:

`DASHBOARD_MAX_GROUND_Z=8 python agents/dashboard/live_tracking_display.py`

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

### End-to-end trace (live frame to assessment)

A stricter call-site matrix and remaining manual live-verification items are documented in `pipeline_trace_matrix.md`.

1. **Frame capture + tracking:** `agents/dashboard/live_tracking_display.py::run_live_tracking_loop`

- Captures webcam frame and runs YOLO/ByteTrack.
- Builds per-track evidence including `pixel_position`, `ground_position`, `ground_coordinate_unit`, `position_kind`, `relative_depth`, `confidence`, and `quality_flags`.

2. **Tracking-agent reasoning:** `agents/visual_tracking_agent.py::LLMVisualTrackingAgent.propose_associations`

- Receives frame reference + structured detections.
- Returns association proposals used to build `tracking_agent_observation`.

3. **Shared state write:** `agents/dashboard/live_tracking_display.py::run_radar_loop`

- Writes `captures/latest_radar_state.json` with `tracks`, `tracking_agent_observation`, `coordinate_system`, and calibration/status fields.

4. **Observation ingestion:** `agents/reasoning/agents/_shared.py::_load_radar_state` and `_ingest_latest_tracks`

- Ingests `tracks` into coordinator history records.

5. **Observation LLM input assembly:** `agents/reasoning/agents/_shared.py::_build_observation_llm_input`

- Packages `history_records` plus `latest_track_snapshot`, `tracking_agent_context`, and `coordinate_context` for Observation agent invocation.

6. **Reasoning cross-check:** `agents/reasoning/agents/_shared.py::_build_reasoning_handoff_payload`

- Validates `supporting_observation_ids` against ingested history before Reasoning invocation.

7. **Ops/event output:** `agents/orchestration/agent_ops.py::run_reasoning_cycle_and_log`

- Logs per-track status, agent metadata, handoff validation, and latency to `captures/agent_events.jsonl` for dashboard/health summaries.

### Milestone 2 scaffolding

Typed contracts and bounded coordinator logic for the Tracking / Observation / Reasoning flow are now scaffolded in:

- `agents/contracts.py`
- `agents/coordinator.py`

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
