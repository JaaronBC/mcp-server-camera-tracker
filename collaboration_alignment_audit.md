# Collaboration Alignment Audit (Counter-UAS Four-Agent Skill)

This audit maps the current implementation to `counter-uas-four-agent-collaboration.SKILL.md` and identifies remaining gaps.

## Scope and intent

- Keep live camera/tracking/radar path real-time and non-blocking.
- Keep LLM specialist work bounded, validated, and orchestrated.
- Ensure dashboard is a state consumer/presenter, not an independent orchestration owner.

## Current data-flow map (producer → schema → transport/storage → consumer)

### 1) Live tracking path

- Producer: `agents/dashboard/tracking_loop.py::run_live_tracking_loop`
- Schema: tracking frame payload (`time`, `detections`) and in-memory `SharedDashboardState` track snapshots
- Transport/storage:
  - UDP datagrams: `127.0.0.1:5005`
  - JSON snapshot: `captures/latest_tracking_frame.json`
  - Shared in-process state (`SharedDashboardState`)
- Consumers:
  - `agents/orchestration/detect_objects_tool.py` (via UDP)
  - Dashboard renderer (`agents/dashboard/rendering.py`)

### 2) Radar/state publication path

- Producer: `agents/dashboard/radar_loop.py::run_radar_loop`
- Schema: radar state with track snapshots, coordinate metadata, camera/detector/calibration status, relative depth coverage
- Transport/storage:
  - JSON snapshot: `captures/latest_radar_state.json`
  - JSONL history: `captures/radar_depth_coverage_history.jsonl`
- Consumers:
  - `agents/orchestration/health.py::dashboard_health_status`
  - `agents/reasoning/agents/_shared.py::_load_radar_state`
  - Dashboard renderer (`agents/dashboard/rendering.py`)

### 3) Orchestration cycle path

- Producer: `main.py::monitor_movement` scheduled loop
- Schema: bounded cycle request (`max_tracks`, uncertainty evidence flag)
- Transport/storage: MCP tool call to `run_reasoning_cycle_and_log`
- Consumers:
  - `agents/orchestration/agent_ops.py::run_reasoning_cycle_and_log`
  - `agents/reasoning/agents/_shared.py::simulate_reasoning_assessment`

### 4) Specialist results and audit path

- Producer: `agents/orchestration/agent_ops.py`
- Schema: per-track event records with cycle status, agent statuses/models/latencies, handoff validation, evidence trace, uncertainty indicators
- Transport/storage:
  - Append-only JSONL: `captures/agent_events.jsonl`
- Consumers:
  - `get_recent_agent_events`, `summarize_agent_event_health`
  - `main.py` status/summary formatting
  - (Dashboard no longer shows agent activity indicators by design)

## Collaboration-skill alignment status

### Aligned

- **Bounded orchestration**: `main.py` now schedules agent-ops cycles on interval and enforces non-overlap.
- **Specialist invocation model**: Observation/Reasoning are invoked by orchestration cycles, not always-on free-running loops.
- **Deterministic coordination/validation**: `AgentCoordinator` and handoff validation logic remain deterministic and separate from LLM prose.
- **Non-blocking real-time path**: camera/tracking/radar loops run independently from LLM cycle latency.
- **Auditability**: cycle outputs and errors are written to `captures/agent_events.jsonl` with structured fields.
- **Correlation traceability**: per-event `cycle_id` and `task_id` fields are emitted by `run_reasoning_cycle_and_log`, and cycle payloads include `cycle_id`.
- **Follow-up lineage traceability**: when evidence is requested, events include deterministic follow-up linkage fields (`evidence_followup_task_id`, `evidence_followup_parent_task_id`, and optional `evidence_followup_previous_task_id`).
- **Disagreement and uncertainty surfacing**: events include explicit `conflicts` and `limitations` arrays derived from deterministic handoff/status/error/uncertainty signals.
- **Summary rollups for collaboration signals**: `summarize_agent_event_health` now reports `collaboration_signals` counts/rates for conflict and limitation prevalence.

### Intentionally changed per user request

- **Dashboard agent activity UI removed**: no Orch/Track/Obs/Reason lights and no tracking-agent activity badges in header.

### Known gaps (non-blocking, future work)

- **Single canonical live bus**: multiple transports (UDP + JSON snapshot + JSONL audit) remain; this is acceptable currently but should be periodically reviewed to avoid divergence.

## Safety and correctness notes

- Observation/reasoning outputs are treated as decision-support artifacts.
- Stale/missing data cases are represented as degraded/unavailable states, not fabricated certainty.
- Failed optional specialist calls do not halt the camera/radar path.

## Recent changes that enforce alignment

- Reverted continuous per-iteration orchestration triggering back to interval-gated scheduling in `main.py`.
- Removed dashboard agent-activity indicators from `agents/dashboard/rendering.py`.
- Removed obsolete dashboard `h` diagnostics control from `README.md`.
- Added dashboard assessment summary lines in `agents/dashboard/rendering.py` sourced from latest validated `agent_events.jsonl` event, including freshness and limitation/error fields.
