# MCP Pipeline Sequence Diagram

This page captures the end-to-end runtime flow for the current architecture.

## End-to-End Sequence

```mermaid
sequenceDiagram
    autonumber

    participant User as Operator
    participant Main as main.py
    participant MCPClient as MCP ClientSession (stdio)
    participant MCPServer as server.py / MCPServer
    participant LiveTool as live_tracking tool
    participant DashProc as dashboard_process
    participant Dashboard as liveTrackingDisplay.py
    participant UDP as UDP 127.0.0.1:5005
    participant DetectTool as detect_objects tool
    participant SummaryLLM as Main Movement LLM
    participant AgentOps as run_reasoning_cycle_and_log
    participant ReasonTool as simulate_reasoning_assessment
    participant ObsAgent as Observation Agent LLM
    participant ReaAgent as Reasoning Agent LLM
    participant Captures as captures/*.json / *.jsonl

    User->>Main: Run `python main.py`
    Main->>MCPClient: start stdio client
    MCPClient->>MCPServer: initialize
    Main->>MCPServer: list_tools()
    MCPServer-->>Main: required tools available

    Main->>MCPServer: call `live_tracking`
    MCPServer->>LiveTool: execute
    LiveTool->>DashProc: start_dashboard_process()
    DashProc->>Dashboard: spawn process
    LiveTool-->>Main: started + PID

    par Dashboard tracking loop
        Dashboard->>Dashboard: camera.read() + YOLO/ByteTrack
        Dashboard->>Dashboard: build DetectionCandidate list
        Dashboard->>Dashboard: AsyncVisualTrackingRuntime.submit_if_needed()
        Dashboard->>Dashboard: LLMVisualTrackingAgent.propose_associations()
        Dashboard->>Dashboard: AgentCoordinator validates proposals
        Dashboard->>Captures: write `latest_tracking_frame.json`
        Dashboard->>UDP: send latest frame packet
    and Dashboard radar/state loop
        Dashboard->>Captures: write `latest_radar_state.json`
        Dashboard->>Captures: append `radar_depth_coverage_history.jsonl`
    end

    loop every ~3s
        Main->>MCPServer: call `detect_objects(address)`
        MCPServer->>DetectTool: execute
        DetectTool->>UDP: bind/recv newest packet
        DetectTool-->>Main: frame JSON
        Main->>SummaryLLM: summarize YOLO frame text
        SummaryLLM-->>Main: operator summary output
    end

    loop heartbeat every N seconds
        Main->>MCPServer: call `run_reasoning_cycle_and_log`
        MCPServer->>AgentOps: execute
        AgentOps->>ReasonTool: per-track `simulate_reasoning_assessment`
        ReasonTool->>Captures: read `latest_radar_state.json`
        ReasonTool->>ReasonTool: ingest tracks into AgentCoordinator history
        ReasonTool->>ObsAgent: observation LLM call (retry/timeout)
        ObsAgent-->>ReasonTool: structured observation payload (or error)
        ReasonTool->>ReasonTool: handoff validation
        alt handoff valid
            ReasonTool->>ReaAgent: reasoning LLM call (retry/timeout)
            ReaAgent-->>ReasonTool: structured assessment payload (or error)
        else handoff invalid
            ReasonTool->>ReasonTool: skip reasoning call, emit invalid_handoff_payload
        end
        ReasonTool->>ReasonTool: bounded evidence request via coordinator
        ReasonTool-->>AgentOps: assessment + metadata + evidence request
        AgentOps->>Captures: append `agent_events.jsonl`
        AgentOps-->>Main: cycle status + evidence trace summary
    end
```

## Data Artifacts

- `captures/latest_tracking_frame.json`: latest detector/tracker frame payload.
- `captures/latest_radar_state.json`: canonical shared runtime state consumed by health/reasoning tools.
- `captures/radar_depth_coverage_history.jsonl`: depth-coverage trend samples.
- `captures/agent_events.jsonl`: reasoning-cycle events + agent health metadata.

## Tool-to-Stage Map

- **Start live process**: `live_tracking`
- **Get newest frame**: `detect_objects`
- **Reasoning cycle + event log**: `run_reasoning_cycle_and_log`
- **Per-track reasoning internals**: `simulate_observation_summary`, `simulate_reasoning_assessment`, `simulate_evidence_request`
- **Health/contract checks**: `dashboard_health_status`, `verify_dashboard_pipeline_contract`, `wait_for_dashboard_ready`
