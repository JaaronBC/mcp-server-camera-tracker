# Pipeline Trace Matrix (Critical-System Closure)

This matrix traces the **actual repository call sites** for the connected pipeline required by `design.md`.

## 1) Runtime startup and MCP wiring

1. **MCP server startup**
   - File: `main.py`
   - Function: `run_agent(...)`
   - Calls:
     - `session.initialize()`
     - `verify_required_tools(session)`
     - `start_live_tracking(session)`
     - `monitor_movement(...)`

2. **Required tool contract check**
   - File: `main.py`
   - Function: `verify_required_tools(...)`
   - Required names: `live_tracking`, `detect_objects`

3. **Startup pipeline preflight snapshot**
   - File: `main.py`
   - Function: `print_pipeline_contract_startup_snapshot(...)`
   - MCP tool: `verify_dashboard_pipeline_contract`
   - Non-blocking behavior: catches tool/transport errors and prints degraded snapshot output.

## 2) Live perception path (camera -> YOLO -> ByteTrack -> track records)

1. **Live tracking process entry**
   - File: `tools/liveTracking.py`
   - Tool: `live_tracking()`
   - Calls: `start_dashboard_process()` (background process)

2. **Camera + detector loop**
   - File: `tools/components/liveTrackingDisplay.py`
   - Function: `run_live_tracking_loop(state)`
   - Key calls:
     - `camera = cv2.VideoCapture(0)`
     - `ok, frame = camera.read()`
     - `result = model.track(..., tracker="bytetrack.yaml", persist=True)[0]`

3. **Track evidence attached per detection**
   - File: `tools/components/liveTrackingDisplay.py`
   - Within `run_live_tracking_loop(...)`
   - Attached fields include:
     - `track_id`, `timestamp`
     - `bbox_xyxy`, `pixel_position`
     - `detection_confidence`
     - `relative_depth`
     - `ground_position` (`x`, `z`)
     - `position_kind`, `ground_coordinate_unit`
     - `quality_flags`

4. **Coordinate metadata normalization (producer boundary)**
   - File: `tools/components/liveTrackingDisplay.py`
   - Function: `normalize_position_metadata(...)`
   - Ensures consistency between `position_kind` and `ground_coordinate_unit`.

## 3) Shared state and outward publication

1. **Shared in-memory dashboard state**
   - File: `tools/components/liveTrackingDisplay.py`
   - `state.active_tracks` updated with `RadarTrackSnapshot` objects.

2. **Latest tracking frame artifact + UDP event stream**
   - File: `tools/components/liveTrackingDisplay.py`
   - Functions/calls:
     - `build_tracking_message(...)`
     - `update_json(LATEST_TRACKING_FRAME_PATH, tracking_message)`
     - `udp_socket.sendto(json.dumps(tracking_message).encode("utf-8"), ... )`

3. **Radar/shared snapshot artifact**
   - File: `tools/components/liveTrackingDisplay.py`
   - Function: `run_radar_loop(state)`
   - Writes: `captures/latest_radar_state.json`
   - Payload fields include:
     - `timestamp`
     - `coordinate_system`
     - `camera_status`, `detector_status`, `calibration_status`
     - `tracking_agent_status`
     - `active_track_count`
     - `last_observation_timestamp`
     - `relative_depth_coverage`
     - `tracking_agent_observation`
     - `tracks` (list of `RadarTrackSnapshot.to_dict()`)

## 4) Agent toolchain path (state -> observation/reasoning -> events)

1. **Reasoning ingestion/readiness**
   - File: `tools/dashboardReasoning.py`
   - Flow:
     - `_load_radar_state()` reads latest radar payload
     - `_prepare_track_analysis(track_id)` enforces freshness (`radar_state_stale` guard)
     - `_ingest_latest_tracks(...)` maps track payload to coordinator observations

2. **Observation + reasoning tool endpoints**
   - File: `tools/dashboardReasoning.py`
   - MCP tools:
     - `simulate_observation_summary(track_id)`
     - `simulate_reasoning_assessment(track_id, request_evidence_on_uncertainty=True)`
   - Handoff integrity:
     - `_build_reasoning_handoff_payload(...)`
     - `handoff_validation` included in outputs

3. **Cycle orchestration + event log**
   - File: `tools/dashboardAgentOps.py`
   - MCP tool: `run_reasoning_cycle_and_log(...)`
   - Captures:
     - per-track result status
     - observation/reasoning agent metadata
     - `handoff_validation_*`
     - `cycle_error_code`
     - bounded-out and evidence trace fields

4. **Health aggregation over event stream**
   - File: `tools/dashboardAgentOps.py`
   - MCP tool: `summarize_agent_event_health(window_size=...)`
   - Includes cycle-level error trends and top error digests.

## 5) Dashboard contract verification

1. **Read and parse contract source**
   - File: `tools/dashboardHealth.py`
   - Functions:
     - `_read_radar_state_payload()`
     - `_parse_radar_state(...)`

2. **Contract verifier MCP endpoint**
   - File: `tools/dashboardHealth.py`
   - Tool: `verify_dashboard_pipeline_contract(max_observation_age_seconds=3.0, require_live_pipeline=False)`
   - Validates:
     - freshness (`stale_observation`)
     - simulation mode (`simulation_fallback_active`)
     - track schema (`track_id`, timestamps, ground position types)
     - coordinate consistency (`position_kind` ↔ `ground_coordinate_unit`)

## 6) Runtime consumption path for frame summaries

1. **YOLO frame consumer**
   - File: `main.py`
   - Function: `monitor_movement(...)`
   - MCP call: `session.call_tool("detect_objects", arguments={"address": [...]})`

2. **UDP frame source tool**
   - File: `tools/detectObjects.py`
   - Tool: `detect_objects(address)`
   - Reads newest UDP packet and returns latest tracking JSON payload.

## 7) Verified test status for current critical changes

- Full regression command run:
  - `/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'`
- Outcome:
  - `Ran 164 tests ... OK (skipped=1)`

## 8) Remaining unverified live checks (manual/runtime)

These cannot be fully proven by unit tests alone and still require live runtime observation:

1. **Physical camera + model continuity**
   - Confirm camera frames are continuously captured and tracked under real scene dynamics.

2. **Live calibration behavior**
   - Confirm `position_kind=calibrated_ground_plane` and non-relative units are emitted only when calibration is valid and stable.

3. **Display coherence under latency/failure**
   - Confirm camera overlay, radar markers/trails, and agent panels remain responsive while model/LLM calls are slow or intermittently unavailable.

4. **No hidden fallback confusion**
   - Confirm simulation fallback is visibly labeled during live outages and does not silently appear as live data.

## 9) Suggested live validation runbook (critical-first)

1. Start runtime:
   - `python -u main.py`
2. Observe startup snapshots:
   - depth coverage summary
   - pipeline contract summary (`status=ready|degraded`, error/warn counts)
3. Generate movement in camera view and observe:
   - active tracks increasing
   - `latest_radar_state.json` timestamp/age freshness
4. Trigger temporary camera/model interruption and verify:
   - stale/simulation indicators appear
   - contract findings report degradation
5. Restore live feed and verify recovery:
   - freshness and ready status return without restart.
