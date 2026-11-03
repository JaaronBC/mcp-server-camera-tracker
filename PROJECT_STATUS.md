# Project Status

## MCP camera-tracking architecture (current)
- `main.py` runs the bounded/operator loop and MCP tool calls.
- `server.py` exposes orchestration/tracking/reasoning tools.
- Live tracking publishes `latest_tracking_frame.json` and `latest_radar_state.json`.
- Reasoning cycles append structured events to `captures/agent_events.jsonl`.

## Agent responsibilities
- **Tracking Agent:** YOLO/ByteTrack detections and optional VLM association proposals.
- **Observation Agent:** Build observation context from track history; normalize/fallback outputs.
- **Reasoning Agent:** Produce assessment labels/handoffs; normalize/fallback outputs.
- **Coordinator (deterministic):** Validate associations, maintain per-track history, enforce evidence-request budgets.

## Phase 16E pipeline status
### Validated stages
- Bounded run harness executes and writes run-local artifacts.
- Provider-call budget diagnostics and blocked-vs-attempted counters are recorded.
- Real camera → YOLO → radar → production reasoning has been validated in a genuine camera-only run (`camera-only-handoff-1793684861`).
- A genuine `production_monitor` evidence request and accepted verification queue submission have been observed live.
- `TrackingEvidenceStore` → `TrackingEvidencePackager` → Observation evidence consumption/preflight boundary passed a targeted provider-free integration test.
- Shutdown cleanup confirms no residual owned processes.

### Remaining unvalidated boundaries
- Successful visual evidence resolution during a genuine live verification request.
- Observation Visual verification and Coordinator acceptance.
- Feedback publication and Tracking consumer acceptance.

## Current development priorities
- Validate live visual-evidence resolution through verification.
- Validate Observation Visual verification and Coordinator acceptance.
- Validate feedback publication and Tracking consumer acceptance.
- Keep diagnostics concise and avoid unnecessary runtime/test spend.
