# Agentic System File Reference

This document explains the **current agentic runtime** in this repository: what each file does, how function calls flow between files, and why each file matters.

Scope used for this document:

- Included: files that are directly in the runtime path via `main.py`, `server.py`, MCP tool registration, tracking loops, coordinator/contracts, and observation/reasoning agent flows.
- Excluded: test files and non-runtime/experimental scripts not imported into the active runtime pipeline.

---

## 1) System-level architecture

At runtime, the system has two main processes:

1. **Client Orchestrator (`main.py`)**
   - Starts an MCP session against `server.py`
   - Starts live tracking
   - Polls detections
   - Periodically runs agent-ops/reasoning health loops

2. **MCP Server (`server.py` + `agents/*`)**
   - Registers all `@mcp.tool()` endpoints by importing tool modules
   - Hosts tracking, observation/reasoning, lifecycle/control, and health/metrics tools

High-level call chain:

`main.py` → MCP call `live_tracking` → `agents/liveTracking.py` → `agents/dashboard_process.py` → `agents/components/liveTrackingDisplay.py`

Then continuously:

`main.py` → MCP call `detect_objects` → `agents/detectObjects.py` (reads UDP tracking frames)

And periodically:

`main.py` → MCP calls

- `run_reasoning_cycle_and_log` (`agents/dashboardAgentOps.py`)
- `summarize_agent_event_health` (`agents/dashboardAgentOps.py`)
- `dashboard_depth_coverage_metrics` (`agents/dashboardMetrics.py`)
- `verify_dashboard_pipeline_contract` (`agents/dashboardHealth.py`)

---

## 2) Entry points and registration

### `main.py`

**Role**

- Client-side orchestrator (no direct model invocation now).
- Manages startup checks, movement loop, and heartbeat cadence.

**Important functions and calls**

- `parse_cli_args(...)`: validates cadence/threshold options.
- `verify_required_tools(session)`: ensures `live_tracking` and `detect_objects` exist.
- `start_live_tracking(session)`: calls MCP tool `live_tracking`.
- `monitor_movement(session, ...)`:
  - calls MCP tool `detect_objects`
  - schedules `_run_agent_ops_pipeline(...)` tasks.
- `_run_agent_ops_pipeline(...)`:
  - `run_agent_ops_cycle(...)` → MCP `run_reasoning_cycle_and_log`
  - `run_agent_ops_health_summary(...)` → MCP `summarize_agent_event_health`
  - `run_depth_coverage_summary(...)` → MCP `dashboard_depth_coverage_metrics`
- `print_pipeline_contract_startup_snapshot(...)` → MCP `verify_dashboard_pipeline_contract`.
- `run_agent(...)`: creates MCP stdio session and executes the lifecycle.

**Why important**

- It is the **runtime conductor** for agentic behavior across tools.

---

### `server.py`

**Role**

- MCP server bootstrap.
- Imports tool modules so their `@mcp.tool()` functions register.

**Why important**

- Without this import side-effect registration, no tools are available to `main.py` or external MCP clients.

---

### `agents/registry.py`

**Role**

- Defines singleton `mcp = MCPServer("Object Behavior Assistant")`.

**Why important**

- Shared registration anchor used by every MCP tool file.

---

## 3) Tool modules (control, data, health, metrics)

### `agents/liveTracking.py`

**Role**

- MCP wrapper exposing `live_tracking()`.

**Calls**

- `live_tracking()` → `start_dashboard_process()` (`agents/dashboard_process.py`).

**Why important**

- External trigger for starting the tracking/radar process.

---

### `agents/detectObjects.py`

**Role**

- MCP tool `detect_objects(address)`.

**Calls / behavior**

- Binds UDP socket, drains stale packets, returns latest JSON frame.

**Why important**

- Data bridge from live tracker output to client orchestration loop.

---

### `agents/dashboard_process.py`

**Role**

- Process supervisor for dashboard/tracker runtime script.

**Important functions**

- `start_dashboard_process()` launches `agents/components/liveTrackingDisplay.py`.
- `stop_dashboard_process(force=False)` and `dashboard_process_status()`.

**Why important**

- Isolates OS process lifecycle from higher-level MCP orchestration.

---

### `agents/dashboardControl.py`

**Role**

- MCP lifecycle orchestration tools.

**Important MCP tools**

- `start_dashboard_demo`, `stop_dashboard_demo`, `dashboard_demo_status`
- `start_dashboard_and_wait_ready`, `stop_dashboard_and_confirm`
- `restart_dashboard_and_wait_ready`, `dashboard_lifecycle_status`, `dashboard_bootstrap`

**Calls**

- Delegates to `dashboard_process`, `dashboardHealth`, and `dashboardCalibration`.

**Why important**

- Operational control-plane for start/stop/restart/readiness workflows.

---

### `agents/dashboardCalibration.py`

**Role**

- MCP tools for ground-plane calibration artifact management.

**Important MCP tools**

- `set_ground_plane_calibration`, `get_ground_plane_calibration`
- `validate_ground_plane_calibration`, `clear_ground_plane_calibration`

**Calls**

- Uses `tools.components.calibration.load_ground_plane_calibration(...)` for validation.

**Why important**

- Enables calibrated spatial mapping quality and validation.

---

### `agents/dashboardHealth.py`

**Role**

- MCP health + pipeline contract verification over `captures/latest_radar_state.json`.

**Important MCP tools**

- `dashboard_health_status()`
- `verify_dashboard_pipeline_contract(...)`
- `wait_for_dashboard_ready(...)`

**Why important**

- Primary safety/quality gate for runtime readiness and contract sanity.

---

### `agents/dashboardMetrics.py`

**Role**

- MCP telemetry/quality metrics.

**Important MCP tools**

- `dashboard_depth_coverage_metrics(...)`
- `dashboard_latency_continuity_metrics()`
- `dashboard_tracking_quality_report(...)`

**Calls**

- Reads capture artifacts and composes with `dashboardAgentOps.summarize_agent_event_health(...)`.

**Why important**

- Provides measurable quality signals for operators and automation.

---

### `agents/dashboardAgentOps.py`

**Role**

- MCP reasoning-cycle execution + event logging + health summarization.

**Important MCP tools**

- `run_reasoning_cycle_and_log(...)`
- `get_recent_agent_events(...)`
- `summarize_agent_event_health(...)`
- `prune_agent_event_log(...)`

**Calls**

- Per track, invokes `tools.reasoning.tools_reasoning_assessment.simulate_reasoning_assessment(...)`.
- Logs structured events to `captures/agent_events.jsonl`.

**Why important**

- The **agent-ops heartbeat core** that ties reasoning outcomes to observable operations.

---

## 4) Contracts and coordination core

### `agents/contracts.py`

**Role**

- Dataclass contracts shared across tracking/observation/reasoning.

**Key types**

- `TrackingObservationRecord`, `ObservationBehaviorSummary`, `EvidenceRequest`, `ReasoningAssessment`
- `DetectionCandidate`, `VisualAssociationProposal`, `VisualAssociationUpdate`, `TrackingAgentObservation`

**Why important**

- Single schema layer for deterministic handoff and validation.

---

### `agents/coordinator.py`

**Role**

- Deterministic coordinator for history, summaries, bounded evidence requests, and proposal validation.

**Important methods**

- Observation history/summary: `add_tracking_observation`, `get_track_history`, `build_observation_summary`
- Reasoning fallback: `build_reasoning_assessment`
- Evidence budget: `request_evidence`
- Tracking proposal validation: `validate_visual_association_proposal`
- Tracking aggregation: `build_tracking_agent_observation`

**Why important**

- Enforces bounded, deterministic behavior when LLM outputs are missing/uncertain/conflicting.

---

## 5) Tracking subsystem

### `agents/components/liveTrackingDisplay.py`

**Role**

- Live camera + detector + tracker loop, shared-state management, radar publishing, and UI rendering.

**Major loop functions**

- `run_live_tracking_loop(state)`:
  - YOLO + ByteTrack frame processing
  - calls `build_tracking_frame_outputs(...)` (tracking transform)
  - calls `submit_tracking_runtime_candidates(...)` (tracking agent runtime)
  - polls/applies tracking observation updates
  - writes UDP + file frame artifacts via `publish_tracking_frame_message(...)`
- `run_radar_loop(state)`:
  - simulation fallback when live data stale
  - calls `publish_tracking_radar_state(...)`
- `render_dashboard(state)` for UI.

**Why important**

- Produces canonical runtime artifacts consumed by all higher-level tools.

---

### `agents/tracking/agent_tracking.py`

**Role**

- LLM-backed visual tracking association runtime.

**Key classes/functions**

- `LLMVisualTrackingAgent.propose_associations(...)`
- `AsyncVisualTrackingRuntime.submit_if_needed(...)` / `poll_latest_observation(...)`
- `encode_frame_as_data_url(...)`
- internal parsing/normalization: `_extract_json_object`, `_normalize_proposal`

**Calls**

- Uses `AsyncOpenAI` for visual association proposals.
- Uses `AgentCoordinator.build_tracking_agent_observation(...)` to validate/normalize updates.

**Why important**

- Agentic identity continuity for track associations (beyond detector-only IDs).

---

### `agents/tracking/frame_transformer.py`

**Role**

- Converts raw YOLO boxes into:
  - `RadarTrackSnapshot` map,
  - serialized detections,
  - `DetectionCandidate` list for tracking agent.

**Main function**

- `build_tracking_frame_outputs(...)`

**Why important**

- Canonical transformation boundary from perception outputs to agent-ready inputs.

---

### `agents/tracking/runtime_support.py`

**Role**

- Runtime helper/mutation utilities for tracking loops.

**Key functions**

- Depth/metadata: `compute_relative_depth_coverage`, `estimate_relative_depth_from_image_geometry`, `normalize_position_metadata`
- Agent update application: `apply_tracking_agent_updates_to_tracks`, `summarize_tracking_agent_observation_metrics`
- Runtime interaction: `submit_tracking_runtime_candidates`, `collect_tracking_runtime_observation_update`
- State commit: `commit_live_tracking_frame_state`, `apply_tracking_agent_metric_update_to_state`

**Why important**

- Keeps loop logic modular and deterministic while applying agent outputs safely.

---

### `agents/tracking/state_publisher.py`

**Role**

- Publishes frame/radar artifacts and handles simulation fallback state.

**Key functions**

- `publish_tracking_frame_message(...)` (JSON + UDP)
- `publish_tracking_radar_state(...)` (latest state + depth history)
- `apply_simulation_fallback_state(...)`
- `collect_radar_publish_snapshot(...)`

**Why important**

- Persistence/transport layer for downstream tools and monitoring.

---

### `agents/tracking/status_runtime.py`

**Role**

- Status text/health classification utilities for tracking-agent diagnostics.

**Key functions**

- `build_tracking_agent_status_text(...)`
- `classify_tracking_agent_health(...)`
- `tracking_agent_health_color(...)`

**Why important**

- Operator-facing observability and concise health signaling.

---

### `agents/tracking/agents/visual_tracking.py`

**Role**

- MCP endpoint for explicit tracking-association simulation.

**MCP tool**

- `simulate_visual_tracking_association(...)`

**Calls**

- Parses detections into `DetectionCandidate`
- Invokes `LLMVisualTrackingAgent.propose_associations(...)`
- Validates via `AgentCoordinator.build_tracking_agent_observation(...)`

**Why important**

- Isolated, testable entrypoint for tracking-association behavior.

---

## 6) Observation + reasoning subsystem

### `agents/observation/agent_observation.py`

**Role**

- Observation agent wrapper: input building, payload validation, retry/timeout policy, and LLM invocation.

**Key functions**

- Context extraction: `extract_latest_track_snapshot`, `extract_tracking_agent_context`, `extract_coordinate_context`
- Input builder: `build_observation_llm_input(...)`
- Validation: `validate_observation_llm_payload(...)`
- LLM path: `invoke_llm_json_response(...)`, `call_observation_agent_with_retry(...)`

**Why important**

- Converts deterministic measurements into structured observation semantics with controlled fallback behavior.

---

### `agents/reasoning/agent_reasoning.py`

**Role**

- Reasoning agent wrapper parallel to observation module.

**Key functions**

- Handoff validation: `build_reasoning_handoff_payload(...)`
- Output validation: `validate_reasoning_llm_payload(...)`
- LLM path: `invoke_llm_json_response(...)`, `call_reasoning_agent_with_retry(...)`

**Why important**

- Produces operator-facing assessment/recommendation outputs with strict payload safety.

---

### `agents/reasoning/agents/_shared.py`

**Role**

- MCP reasoning workflow tools over live radar state + coordinator history.

**Core behavior**

- Loads and validates radar state freshness
- Ingests tracks into coordinator history
- Builds observation inputs, invokes observation agent (with fallback)
- Builds reasoning handoff, invokes reasoning agent (with fallback)
- Issues bounded evidence requests when uncertainty remains

**MCP tools**

- `simulate_evidence_request(track_id, reason)`
- `simulate_observation_summary(track_id)`
- `simulate_reasoning_assessment(track_id, request_evidence_on_uncertainty=True)`

**Why important**

- Central reasoning pipeline that powers agent-ops cycles.

---

## 7) Shared geometry/data model helpers used by agentic runtime

### `agents/components/dashboard_types.py`

**Role**

- Dataclasses for UI/runtime state artifacts (`Observation`, `RadarTrackSnapshot`, etc.).

**Why important**

- Shared typed structures between tracking loop and publishers.

---

### `agents/components/calibration.py`

**Role**

- Homography-based image-to-ground calibration utilities.

**Key functions**

- `compute_homography(...)`, `map_image_to_ground(...)`, `load_ground_plane_calibration(...)`

**Why important**

- Enables calibrated ground-plane coordinates and higher-quality spatial reasoning.

---

### `agents/components/geometry.py`

**Role**

- Coordinate transform helpers for relative ground and radar rendering.

**Key functions**

- `image_to_relative_ground(...)`, `ground_to_radar_screen(...)`, `planar_distance(...)`

**Why important**

- Baseline spatial conversions when calibration is absent or for visualization.

---

## 8) Package marker files

The following files exist primarily for package structure/import resolution and contain no runtime logic:

- `agents/__init__.py`
- `agents/tracking/__init__.py`
- `agents/reasoning/__init__.py`
- `agents/observation/__init__.py`
- `agents/components/__init__.py`

These are still important for import stability across the agentic modules.

---

## 9) Practical “why this matters” summary

- **`main.py`** is the client orchestrator.
- **`server.py` + `agents/registry.py`** expose the MCP surface.
- **`liveTrackingDisplay.py` + tracking helpers** produce the real-time artifact stream.
- **`contracts.py` + `coordinator.py`** enforce deterministic guardrails.
- **`agent_observation.py` + `agent_reasoning.py`** provide LLM-backed semantic layers with retry/validation.
- **`tools_reasoning_assessment.py` + `dashboardAgentOps.py`** run bounded agentic cycles and log system health.
- **`dashboardHealth.py` + `dashboardMetrics.py`** provide operational correctness and quality signals.

Together, these files form the current end-to-end agentic system.
