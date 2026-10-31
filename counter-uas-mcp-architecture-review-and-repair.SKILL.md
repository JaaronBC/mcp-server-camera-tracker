---
name: counter-uas-mcp-architecture-review-and-repair
description: Audit and incrementally improve the Counter-UAS camera-tracking MCP architecture, agent handoffs, runtime ownership, data contracts, and dashboard pipeline without breaking working behavior.
---

# Counter-UAS MCP Architecture Review and Repair

## Purpose

Use this skill to review, debug, and incrementally improve the Counter-UAS / camera-tracking project. The objective is a reliable system with clear ownership, responsive perception, meaningful agent collaboration, stable MCP contracts, validated data, and a dashboard that displays published state.

Do not assume MCP itself makes agents collaborate. Verify the real call paths and structured handoffs in the current repository.

## Project context to verify

The project has previously been described as containing:

- `main.py`: runtime entry point, MCP client setup, service initialization, and operator-loop startup.
- `server.py`: MCP server entry point and tool registration/imports.
- `agents/orchestration/runtime/`: runtime services, startup flow, and monitoring.
- `agents/orchestration/tools/`: MCP-facing orchestration operations.
- `agents/dashboard/`: camera tracking, radar-state, and rendering modules.
- `agents/observation/agent_observation.py`: observation-history interpretation.
- `agents/reasoning/agent_reasoning.py`: reasoning / decision-support interpretation.
- `agents/coordinator.py`: deterministic coordination, validation, history/state management, and bounded evidence requests.
- Tracking-agent code: YOLO/ByteTrack perception and possibly VLM-assisted association.

These are investigation leads, not proof of the current checkout. Inspect actual source before making claims or edits. Treat architecture walkthroughs as documentation, not as a substitute for source inspection.

## Non-negotiable working rules

1. **Explain before editing.** First explain the issue, the source evidence, why it matters, the smallest proposed change, affected files, and regression risks.
2. **Trace before refactoring.** Follow real imports, function calls, MCP registrations, process boundaries, subprocesses, queues, sockets, files, and shared state. Do not infer behavior from filenames.
3. **Preserve working behavior.** Avoid broad rewrites, public tool renames, payload changes, module moves, or file deletion unless justified and approved.
4. **Audit before deleting.** Search all imports, references, tests, scripts, configuration, and runtime registrations before removing code. Ask before deleting anything with uncertain ownership or external use.
5. **Make incremental changes.** Fix one responsibility or data-flow defect at a time. Keep the application runnable after each logical step.
6. **Verify claims.** Report exact tests/checks run. Never claim a fix works if it was not tested. Separate verified results from expected effects.
7. **Never invent evidence.** Agents must not fabricate detections, track continuity, coordinates, depth, measurements, or events. Missing evidence and uncertainty must remain explicit.
8. **Human decision-support only.** Do not infer hostile intent from motion alone, label something a threat without adequate evidence, or implement autonomous countermeasures.

## Phase 1 — Map the current architecture (no edits)

Inspect the current versions of:

- `main.py` and `server.py`
- orchestration runtime app/core/flow modules
- orchestration MCP tool modules
- coordinator implementation
- tracking-agent implementation and tracking runtime
- dashboard entry point, tracking loop, radar loop, and renderer
- Observation and Reasoning agent implementations
- shared schemas/contracts, configuration, tests, and startup scripts

If paths differ, locate current equivalents and report the mapping. Do not create duplicate modules just because a suggested path is absent.

Produce a concise map of:

1. **Processes:** what runs in the main runtime, MCP server, dashboard, and other subprocesses.
2. **Startup:** entry point → service initialization → MCP session/server → camera/dashboard startup → operator loop.
3. **Data path:** camera frame → detector/tracker → normalized observation → transport/storage → MCP read → agent handoffs → assessment → publication → dashboard.
4. **MCP tools:** tool name, implementation, caller, input/output contract, side effects, and whether it can block.
5. **State ownership:** canonical observations, track history, agent results, event logs, and dashboard state.
6. **Scheduling:** who starts each loop and how duplicate cycles are prevented.
7. **Failure and cleanup:** camera failure, MCP disconnect, subprocess exit, timeout, cancellation, and shutdown.

Cite actual file and function names for each important link. Mark anything not verifiable as **unknown**.

## Phase 2 — Diagnose these risks (prove or disprove each)

### A. Runtime-loop ownership

Check whether `OrchestrationAgent.run()`, `AgentCoordinator.run()`, `_run_operator_loop`, `run_operator_loop`, `monitor_movement`, heartbeat cycles, and analysis cycles overlap.

Determine:
- Is there one owner of the top-level runtime loop?
- Does the coordinator execute deterministic workflow steps or start another loop?
- Is a heartbeat confused with an agent-analysis cycle?
- Can scheduled logic and an MCP tool trigger duplicate analysis?
- Are loops cancellable and awaited on shutdown?

Recommend one clear scheduling owner. Keep the Operator Agent's decisions separate from deterministic scheduling, validation, and resource management.

### B. MCP boundary and tool design

For every MCP tool, determine:
- Is it a meaningful bounded capability an external client/agent needs?
- Are input and output schemas clear?
- Are internal implementation details exposed unnecessarily?
- Does it block on long-running camera work?
- What file/process/socket/state side effects does it have?
- Who calls it?
- Is it idempotent, or does it need a task/correlation ID to prevent duplicate work?

Keep real-time frame processing local to the perception pipeline. Do not send every camera frame through an LLM or MCP request. MCP should expose bounded operations such as retrieving observations/history, requesting association review, requesting an assessment, checking status, and reading/publishing validated results.

### C. Perception, tracking, and dashboard coupling

Distinguish these responsibilities:

- **Perception:** camera capture, YOLO detection, ByteTrack IDs.
- **Measurement/normalization:** bbox and pixel coordinates, available depth, calibrated ground-plane projection, timestamps, units, quality, provenance.
- **Tracking interpretation:** ambiguous association proposals and track-history context.
- **Radar state:** deterministic conversion of validated measurements into radar state.
- **Rendering:** drawing webcam/radar/status and displaying published assessments.

A dashboard module may host the camera loop in the current design, but assess whether rendering is too tightly coupled to perception/state production. Find the least disruptive way to separate them.

The renderer should consume published state. It should not run its own reasoning workflow or silently replace missing live data with simulated tracks. Simulation must be clearly labeled.

### D. Real agent collaboration

Verify that the agents exchange structured results, rather than merely existing as separate classes or being invoked independently by scheduled MCP tools.

**Operator / Orchestration Agent**
- Decides which bounded specialist task is needed.
- Delegates with explicit context, scope, evidence references, deadline, and expected schema.
- Reviews completeness, contradictions, and uncertainty.
- Decides whether a bounded follow-up is useful.
- Closes the cycle with a traceable terminal status.

**Tracking Agent**
- Interprets detector/tracker observations, track history, and available measurement context.
- May propose ambiguous associations or explain gaps.
- Returns supporting observation IDs, evidence references, quality flags, and uncertainty.
- Must not invent detections or silently rewrite canonical IDs/measurements.

**Observation Agent**
- Interprets timestamped, validated track history and measured motion.
- Separates measured facts, derived metrics, interpretation, uncertainty, missing intervals, and alternatives.
- Must not convert a gap into a claimed continuous trajectory.

**Reasoning / Decision-Support Agent**
- Synthesizes validated specialist outputs and source evidence.
- Checks evidence support, conflicts, and alternative explanations.
- Produces a bounded, evidence-linked assessment and limitations.
- Must not claim more certainty than the evidence supports.

`AgentCoordinator` should be deterministic application code, not another LLM agent. It should enforce schemas, task permissions, IDs, time windows, timeouts, retries, follow-up budgets, loop prevention, state transitions, and publication rules.

### E. Canonical observation and result contracts

Inspect existing models first. Reuse and evolve them rather than creating duplicate schemas.

A canonical observation should distinguish, where available:

- `observation_id`, `track_id`, `timestamp`, `source_frame_id`
- `bbox_xyxy`
- `pixel_position` and its definition (for example, bbox center or bottom-center)
- `detection_confidence`
- `relative_depth`, `depth_method`, `depth_quality`
- `ground_position`, coordinate frame, units, calibration status/version
- measured motion values and units
- quality flags, provenance/evidence references
- known occlusion, stale-data, and missing-data indicators

Keep image pixel coordinates, relative monocular depth, calibrated 2D ground-plane coordinates, and true 3D position distinct. Do not label monocular relative depth as meters unless metric scale is validated. A floor homography does not recover an airborne object's true 3D position. Label ground-plane projection as an estimate/projection, not a direct 3D measurement.

Agent task/result contracts should include appropriate fields such as task/correlation ID, agent name/version, schema version, source observation IDs, requested time window, status, result, uncertainty, evidence references, validation outcome, and error details.

Avoid introducing a large generic schema framework when a small typed model or existing Pydantic model is sufficient.

### F. Lifecycle and shutdown

Trace who owns:
- camera handle;
- dashboard subprocess;
- MCP client/session;
- background tasks;
- sockets, files, queues, and event logs.

Determine what happens when startup fails partway through, when `operator.stop()` raises, or when a subprocess/camera fails. Ensure one cleanup failure does not prevent other resources from being released. Use nested `try/finally` or an equivalent explicit lifecycle pattern when justified by the actual ownership path. Do not change lifecycle code before tracing it.

## Phase 3 — Target collaboration workflow

Adapt this workflow to existing code; do not rebuild blindly.

1. **Perception:** Camera → YOLO/ByteTrack → available depth/pixel/ground-plane measurements → normalized timestamped observations with provenance and quality flags. This path must not wait for LLM inference.
2. **Operator decision:** Operator receives a bounded state/event summary and decides whether association review, observation analysis, or reasoning is needed.
3. **Tracking review (when needed):** Coordinator dispatches a task with task ID, track ID, relevant observation IDs, time window, permitted evidence, deadline, and expected schema. Tracking Agent returns a proposal with evidence and uncertainty. Deterministic validation checks it before canonical state can change.
4. **Observation analysis:** Observation Agent receives canonical history and validated tracking results. It returns measured facts, derived metrics, interpretation, uncertainty, gaps, and alternatives.
5. **Reasoning assessment:** Reasoning Agent receives source observations and validated specialist results. It checks support and conflicts and returns an evidence-linked assessment with limitations.
6. **Cycle closure:** Operator decides whether a bounded follow-up is warranted. Coordinator enforces a maximum follow-up budget and terminal status. Reuse existing status names if defined; otherwise consider `complete`, `follow_up_required`, `inconclusive`, `failed`, and `expired`.
7. **Publication:** Deterministic code validates and records the result, then publishes it for the dashboard/operator. Dashboard renders it; it does not start a second reasoning loop.
8. **Audit:** Record cycle/task IDs, reasons, track/observation IDs, time window, agent statuses/durations, validation outcomes, evidence references, follow-up count, terminal status, publication status, and errors/conflicts.

Not every cycle needs every agent. Parallelize only independent tasks; keep dependent handoffs sequential. Prevent unbounded agent-to-agent loops.

## Phase 4 — Prioritize repairs

Rank verified issues by:
1. data integrity and correctness;
2. duplicate/conflicting runtime ownership;
3. missing or broken agent handoffs;
4. lifecycle/resource leaks;
5. unclear MCP contracts;
6. dashboard/perception coupling;
7. maintainability and naming.

For each issue report severity, user-visible consequence, source evidence, root cause, minimal repair, tests, dependencies, and regression risks.

Ask before broad refactors or destructive changes. If the user explicitly asks to implement fixes, proceed with the smallest safe change after explaining the plan.

## Phase 5 — Implementation discipline

For each approved repair:

1. State intended behavior and acceptance criteria.
2. Identify exact files/functions to change.
3. Make the smallest coherent edit.
4. Preserve public MCP tool names and payload compatibility unless migration is approved.
5. Add/update contract and behavior tests.
6. Run focused tests, then relevant broader tests, lint, and type checks if configured.
7. Report exact commands and outcomes; say explicitly when a check was not run.
8. Inspect the diff for unrelated edits and accidental deletions.
9. Explain the repaired end-to-end data path.

Use dependency injection or fakes for camera/LLM/network boundaries when hardware or external services are unavailable. Avoid tests that only assert mock calls when actual contract behavior can be tested.

## Acceptance criteria

The architecture is healthier when:

- One clearly documented component owns the top-level runtime schedule.
- Startup and shutdown ownership are explicit and tested.
- Camera/YOLO/ByteTrack processing does not block on LLM calls.
- Canonical observations document units, coordinate frames, timestamps, and provenance.
- MCP tools expose stable, bounded capabilities with explicit contracts and side effects.
- Operator delegates meaningful tasks through structured specialist handoffs.
- Deterministic coordinator validates IDs, evidence, schemas, permissions, timeouts, retries, and follow-up budgets.
- Observation and Reasoning outputs distinguish facts from interpretation and uncertainty.
- Occluded/missing intervals are represented honestly.
- Dashboard consumes published state and clearly distinguishes live data from simulation.
- Failures, timeouts, and incomplete cycles have explicit statuses and audit records.
- Existing startup commands and core dashboard behavior continue to work.

## Expected response format

When using this skill, respond in this order:

1. **What I found** — verified architecture and real call/data paths.
2. **What is actually wrong** — evidence-backed defects, not stylistic preferences.
3. **What should remain as-is** — working boundaries that do not need MCP or LLM involvement.
4. **Recommended repair order** — smallest/highest-value changes first.
5. **Proposed first change** — exact files, behavior, risks, and tests.
6. **Implementation and verification** — after approval or a clear implementation request.

Be candid if the current code is already reasonable. The goal is not to maximize agents, MCP tools, or files. The goal is a reliable, observable, testable system with clear ownership and evidence-grounded collaboration.
