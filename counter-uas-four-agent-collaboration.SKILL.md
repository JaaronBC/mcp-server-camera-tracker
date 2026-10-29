---
name: counter-uas-four-agent-collaboration
description: Defines how the Orchestration, Tracking, Observation, and Reasoning agents collaborate in the Counter-UAS camera tracking MCP project, including task handoffs, evidence rules, lifecycle, failure handling, and integration guardrails.
---

# Counter-UAS Four-Agent Collaboration Skill

## Purpose

Implement and maintain a coordinated four-agent system for Counter-UAS camera tracking and operator decision support. The agents should collaborate through explicit, bounded tasks and structured results. They must not become four independent loops that freely call one another, duplicate state, or block real-time camera processing.

This skill describes **how the agents work together**. Before changing code, inspect the repository and map this design onto the existing implementation rather than assuming files, APIs, or architecture that may not exist.

## The four agents and their responsibilities

### 1. Orchestration / Operator Agent

**Role:** Own the analysis workflow. It decides what question needs answering, which specialist should receive it, whether more evidence is worth requesting, and when the cycle is complete.

It should:
- Receive a trigger, such as an observation gap, uncertain track association, notable measured motion, conflicting outputs, or a scheduled review.
- Create a bounded task with a purpose, time window, track and observation IDs, permitted evidence/tools, expected result schema, and deadline.
- Delegate only the work needed to answer that question.
- Track dependencies, statuses, deadlines, and a small follow-up budget.
- Check whether results are complete, supported, consistent, and useful.
- Request a targeted follow-up when it could materially resolve a stated uncertainty.
- Close the cycle with a terminal status: `complete`, `inconclusive`, `failed`, or `expired` (or the project's established equivalent).
- Return a concise, traceable operator-facing result, including uncertainty and limitations.

It must **not** invent measurements, silently modify authoritative tracking state, bypass deterministic validation, or repeatedly delegate without a bounded stopping condition.

### 2. Tracking Agent

**Role:** Help interpret and validate tracking continuity using existing perception outputs and available history. It supports the tracking pipeline; it is not a replacement for the camera capture loop or deterministic detector/tracker.

It may use:
- YOLO detections and confidence values;
- ByteTrack track IDs and bounding boxes;
- timestamped frame and track history;
- pixel coordinates and an explicitly defined reference point (for example, bbox center or bottom center);
- depth estimates, depth method and quality, if available;
- calibrated ground-plane estimates and calibration metadata, if available;
- occlusion, stale-frame, missing-detection, and quality flags when actually observed or calculated.

It should return:
- the source observation/frame IDs reviewed;
- any proposed association or continuity explanation, clearly marked as a proposal;
- evidence supporting and contradicting the proposal;
- quality flags and uncertainty;
- missing intervals and unavailable evidence;
- a status such as `supported`, `ambiguous`, `insufficient_evidence`, or `failed`.

It must not fabricate observations, claim an object was observed during a gap, silently change a track ID, or present a proposed association as a confirmed measurement. Any proposed state change must pass deterministic validation and the project's defined acceptance policy.

### 3. Observation Agent

**Role:** Explain what the timestamped observations and validated tracking results establish about observed motion and continuity.

It should:
- Use canonical observations and explicit source IDs rather than relying on untraceable prose summaries.
- Separate **measured facts**, **derived metrics**, and **interpretations**.
- State units, coordinate frames, time windows, and the source of each derived metric.
- Describe changes, gaps, stale data, uncertainty, and plausible alternatives.
- Identify what cannot be concluded from the available data.
- Return a structured result with evidence references, limitations, confidence/uncertainty representation, and status.

It must not infer hostile intent from movement alone, treat missing data as evidence of movement, or turn an uncertain track association into a fact.

### 4. Reasoning / Decision-Support Agent

**Role:** Synthesize validated observations and specialist results into a careful, evidence-linked assessment for a human operator.

It should:
- Review the underlying source observation references as well as the Tracking and Observation Agent results.
- Check whether each important statement is supported by evidence.
- Identify contradictions, missing context, alternative explanations, and uncertainty.
- Explain the basis and limits of the assessment in plain language.
- Produce a structured assessment with evidence references, unresolved conflicts, limitations, and a status.

It must not invent facts, hide disagreement between agents, claim certainty unsupported by data, or infer hostile intent from movement alone. Outputs are decision support for a human; they do not authorize autonomous countermeasures or other consequential actions.

## The deterministic coordinator is not a fifth agent

Use ordinary application code (for example, the existing `AgentCoordinator`, if it fits) to enforce workflow and data integrity. It is **not another LLM agent**.

The coordinator should own or enforce:
- task dispatch and allowed agent/tool permissions;
- schema and required-field validation;
- track IDs, observation IDs, timestamps, and source-reference checks;
- acceptance or rejection of proposed tracking associations;
- deadlines, timeouts, retries, and bounded follow-up count;
- duplicate-task and stale-result handling;
- cycle/task correlation IDs and status transitions;
- audit logging and publication of validated results;
- a terminal outcome even when a model call fails.

Do not rely on an LLM to validate its own output or enforce safety-critical invariants. Keep validation deterministic wherever practical.

## Collaboration protocol

Every task request and result should use the project's existing schema where possible. Extend it only when a concrete missing requirement is identified.

A task request should include, as applicable:
- `cycle_id` and unique `task_id`;
- `parent_task_id` or correlation ID;
- requesting and target agent;
- concise question and reason for the task;
- `track_id` and a bounded time window;
- source `observation_ids` / `frame_ids`;
- permitted tools and evidence sources;
- expected output schema;
- deadline and remaining follow-up budget.

A result should include, as applicable:
- matching cycle/task/correlation IDs;
- agent name/version and schema version;
- status (`success`, `partial`, `insufficient_evidence`, `failed`, `expired`, or project equivalent);
- source IDs actually reviewed;
- structured findings, with measured facts separated from interpretations;
- uncertainty, quality flags, conflicts, and limitations;
- errors or missing inputs when applicable.

Never accept a result merely because it is well-written. Check that its IDs, schema, time window, and evidence references match the task. Ignore or quarantine results that are stale, malformed, out of scope, or unsupported.

## Standard collaboration flow

### A. Normal live tracking

1. Camera capture and YOLO/ByteTrack continue on the real-time path.
2. The perception pipeline emits timestamped observations and quality metadata through the project's canonical data path.
3. Deterministic code handles routine updates. Do not invoke every LLM for every frame by default.
4. The dashboard renders current observations and state without waiting for LLM completion.

### B. An observation gap or ambiguous association

1. A deterministic trigger detects a gap/ambiguity or a bounded review request arrives.
2. The Orchestration Agent opens a cycle and sends the Tracking Agent a bounded task containing the track, time window, source IDs, and question.
3. The Tracking Agent returns evidence, proposed continuity/association if appropriate, quality flags, and uncertainty.
4. The coordinator validates the response and any proposed association. Rejected proposals must not silently alter the authoritative track state.
5. The Observation Agent receives the relevant canonical history and validated Tracking result. It describes what is supported, what is missing, and what alternatives remain.
6. The Reasoning Agent synthesizes the source evidence and validated specialist results into a cautious assessment.
7. The Orchestration Agent checks completeness and contradictions. It may request one targeted follow-up if the expected benefit justifies it and budget remains; otherwise it closes as complete or inconclusive.
8. The coordinator validates, logs, and publishes the final cycle result for the dashboard/operator.

**Critical rule:** If the camera did not observe the object during an interval, agents may describe a possible association or uncertainty, but must not claim the missing observations were directly seen.

### C. Other triggers

- For a simple, stable detection, no LLM call may be needed.
- For a motion-summary request with reliable history, the Observation Agent may run first, followed by Reasoning if a synthesis is needed.
- If a tracking ambiguity materially affects the assessment, run Tracking first and pass its validated result downstream.
- If independent specialist analyses can use the same immutable evidence, the coordinator may run them in parallel. Run dependent steps sequentially.
- If the results conflict, preserve the conflict and ask a narrow follow-up only if it could resolve the issue. Otherwise close as inconclusive.

## Agent activation and communication rules

“Available” does not mean “currently executing.” All four agents can be registered and ready while only one or two are active for a given cycle.

- Camera capture, detection, and deterministic tracking run continuously as required by the application.
- LLM agents are invoked on defined events, operator requests, or bounded scheduled cycles.
- Use selective parallelism only for independent tasks; use sequential handoffs when one agent depends on another's validated result.
- Route inter-agent requests through the Orchestration Agent and deterministic coordinator. Do not allow uncontrolled peer-to-peer loops.
- Every cycle must have a maximum number of follow-ups, deadlines, and a terminal state.
- LLM calls must not block camera capture, tracking, rendering, or other time-sensitive work. Use the project's existing asynchronous/queue approach if suitable; do not add infrastructure without demonstrating why it is needed.

## Canonical evidence and coordinate discipline

Use the existing observation schema if one exists. Inspect it before introducing a new one. The canonical observation should represent, where available:
- `observation_id`, `track_id`, timestamp, and source frame ID;
- bounding box and pixel position, including the definition of the selected point;
- detection confidence;
- relative depth, depth method, and depth quality;
- ground-plane position, coordinate frame, units, calibration status/version;
- motion measurements and units;
- quality flags, provenance references, and known occlusion/staleness indicators.

Keep distinct:
- image pixel coordinates;
- relative or monocular depth estimates;
- calibrated 2D ground-plane estimates;
- true 3D position.

Do not label relative monocular depth as meters unless metric scale has been validated. A ground-plane projection does not recover an airborne object's true 3D position. Include calibration and quality status when presenting derived spatial values. Never fabricate missing coordinates, depth, timestamps, or calibration information.

## Integration with the existing project

Before editing, trace one real observation end to end. Inspect the actual repository and identify:
- where camera frames are captured;
- where YOLO/ByteTrack detections and track history are created;
- how `DetectionCandidate`, `AsyncVisualTrackingRuntime`, `LLMVisualTrackingAgent`, and `AgentCoordinator` are currently used, if still present;
- how `liveTrackingDisplay.py` creates or updates `latest_tracking_frame.json`, `latest_radar_state.json`, and `radar_depth_coverage_history.jsonl`, if still present;
- how UDP `127.0.0.1:5005` packets are produced and consumed, if still present;
- how `detect_objects`, `run_reasoning_cycle_and_log`, and `simulate_reasoning_assessment` are called, if still present;
- how Observation and Reasoning Agent outputs reach `agent_events.jsonl` and the dashboard;
- what starts/stops each process and how failures/restarts are handled.

Names above come from the current project description and may have changed. Verify actual code and call sites before relying on them.

Create a short current-state data-flow map showing producer → schema → transport/storage → consumer for each path. Identify duplicate or divergent sources of truth, timestamp/track-ID loss, stale-state risks, and any missing connection between agent output and dashboard. Do not assume a JSONL audit log is a live event bus, or that a latest-value JSON snapshot and UDP packet always describe the same frame.

Prefer one canonical, timestamped observation/state contract and one clearly defined publication path for validated assessments. The dashboard should be a consumer/presenter of shared state, not an independent reasoning engine or owner of authoritative tracking state. It may issue start/stop or selection requests through defined interfaces, but should not duplicate orchestration logic.

Do not immediately remove UDP, JSON snapshots, JSONL logs, processes, or existing tools. First trace real consumers and producers, propose the smallest safe consolidation, explain the tradeoffs, and preserve functionality during migration.

## Dashboard responsibilities

The dashboard should display, as available:
- live camera view and detection boxes/track IDs;
- radar markers/trails based on the same timestamped state;
- selected track details and units/coordinate frame;
- depth and calibration status;
- stale, lost, occluded, simulated, or uncertain state distinctly from live reliable observations;
- agent assessment, evidence references, uncertainty, and limitations;
- camera, tracking, depth, calibration, MCP, and agent status.

Do not let a late LLM result overwrite newer live state. Associate every result with its source time window and track/observation IDs, and visibly distinguish an old assessment from current state.

## Failure, timeout, and disagreement handling

- A missing specialist result is not a successful result.
- On timeout, malformed output, model/provider error, or unavailable evidence, record the failure and continue or terminate according to a bounded policy.
- Retry only when safe and useful; retries must be limited and idempotent where possible.
- A failed optional agent should not stop camera capture or radar rendering.
- If source data are stale, mark the result stale or refuse to assess; do not present it as current.
- If agents disagree, preserve both claims and their evidence, then mark the conflict. Do not silently average, discard, or choose the more confident-sounding prose.
- Always close each cycle with a clear status and error/limitation information.

## Auditability

Log enough to reconstruct what happened without dumping unnecessary sensitive data:
- cycle/task/correlation IDs;
- trigger and task reason;
- agent and version;
- track IDs, observation IDs, and time window;
- status, duration, and timeout/retry details;
- validation result and rejected fields/proposals;
- evidence references, conflicts, uncertainty, and terminal outcome;
- publication outcome and errors.

Use the existing audit mechanism where practical. JSONL may be suitable for an append-only audit trail, but do not treat it as a live event transport unless the application explicitly implements that behavior.

## Safety and interpretation boundaries

This system provides observation and human decision support. Do not infer hostile intent, threat level, or operator action from movement alone. State observable evidence and uncertainty, include plausible alternatives, and leave consequential decisions to a human operator. Do not trigger autonomous countermeasures.

## Required coding-agent workflow

Before making changes, the coding agent must:

1. **Explain why each proposed change matters before coding.** Tie it to a specific defect, missing handoff, data-integrity issue, reliability problem, or requirement.
2. Inspect the relevant files and call sites. Do not assume a named function or file still exists.
3. Trace at least one real observation through capture, tracking, transport, specialist analysis, coordinator validation, logging, and dashboard presentation.
4. Present the current flow and proposed flow, including schemas and handoffs.
5. Identify the smallest coherent implementation step and how it will be tested.
6. Avoid a broad rewrite. Preserve the working camera, tracking, radar, MCP interface, and current behavior unless a change is justified.
7. Before deleting or replacing files, search imports, call sites, entry points, scripts, tests, configuration, and documentation. Explain the replacement and request approval for destructive or uncertain cleanup.
8. Implement and test incrementally. Include failure-path tests for stale/malformed results, timeouts, and agent disagreement where practical.
9. Report files changed, why each change matters, tests run and their actual results, remaining risks, and any unimplemented integration. Never claim live integration based only on mocks or isolated unit tests.

## Definition of done

The collaboration is not complete until:
- all four roles have clear, non-overlapping responsibilities;
- task and result schemas preserve IDs, timestamps, provenance, status, and uncertainty;
- the coordinator enforces bounded handoffs and validates outputs;
- specialist results are consumed by downstream agents through explicit interfaces;
- a final validated assessment is published to the actual dashboard/operator path;
- slow or failed LLM calls cannot block the live camera/tracking path;
- gaps and conflicts remain explicit rather than being turned into fabricated certainty;
- a real end-to-end path is demonstrated and tested, with limitations documented.
