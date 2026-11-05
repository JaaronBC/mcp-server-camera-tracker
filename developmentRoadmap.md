---
name: agentic-security-development-roadmap
description: Keep development of the MCP-based real-time camera/radar security monitoring project aligned with its phased roadmap, agent responsibilities, measurable acceptance criteria, and approval gates. Use when planning, implementing, reviewing, testing, or prioritizing work involving agent latency, track management, autonomous tool calling, trajectory prediction, or critical-area risk assessment.
---

# Agentic Security Development Roadmap

## Mission

Guide incremental development of the existing real-time camera/radar MCP application. Preserve the working YOLO/ByteTrack perception pipeline, shared observation evidence, radar/dashboard integration, and distinct LLM-backed Tracking, Observation, Reasoning, and Operator/Orchestration agents. Avoid replacing agent decision-making with hardcoded rules disguised as agents.

## Mandatory workflow for every task

1. Identify the **current phase**, the user's explicit request, and dependencies. Read relevant existing code and tests before proposing edits; do not assume file paths or APIs are unchanged.
2. Explain the current behavior, the specific problem, why the proposed change matters, and which files/functions would change.
3. Separate **analysis/proposal** from **implementation**. Do not change code when asked only to inspect or plan. For substantive behavior/architecture changes, present a minimal plan and wait for explicit approval.
4. Keep changes limited to the approved phase and its necessary prerequisites. If a later-phase feature is requested, flag its prerequisites; the user may explicitly override the sequence.
5. Preserve existing functionality, including live webcam, supported video input, MCP boundaries, radar/dashboard, event origins, uncertainty flags, and evidence provenance.
6. Add or update focused tests. Report exact commands, pass/fail results, and known limitations. Never claim success from mock-only benchmarks when the goal requires real-provider performance.
7. End each task with: phase, completed work, evidence, unresolved issues, and recommended next step. **Do not self-approve a phase transition.**

## Development phases and gates

### Phase 1 — Agent processing performance

**Goal:** Reliable, responsive Tracking, Observation, and Reasoning LLM calls without compromising valid structured outputs or evidence.

- Establish baseline for real-provider success/failure/timeout rates and per-agent p50/p95 latency, plus end-to-end cycle latency, single- and multi-track.
- Diagnose provider availability, generation length, output-token limits, JSON completeness, client reuse, retries, queue waits, and serial dependencies using measured evidence.
- Prefer small, reversible, configurable changes. Validate no silent truncation of required fields or loss of uncertainty/evidence.
- Keep live camera/radar updates responsive independently of slower LLM assessment.

**Exit gate:** Successful end-to-end real-provider cycles demonstrated; valid output and failure rates measured; latency comparison before/after; user accepts the performance baseline or remaining limitations. Do not invent a universal 1-second LLM guarantee.

### Phase 2 — Track management and collaboration

**Goal:** Preserve identities through occlusion, background blending/low visibility, and crowded/overlapping tracks.

- Keep detector/tracker measurements separate from LLM hypotheses. Tracking Agent reviews association ambiguity, ID-switch risk, depth/position conflicts, and occlusion evidence.
- Observation Agent contributes motion history, feasible continuation hypotheses, and uncertainty to Tracking Agent; do not let predictions overwrite observed detections without traceable evidence.
- Use shared timestamps, track IDs, observation IDs, coordinate frame, quality flags, and evidence provenance.
- Maintain explicit states such as observed, temporarily occluded, predicted-only, lost, and reacquired where compatible with existing schema.
- Test with reproducible occlusion/crossing scenarios; report ID switches, fragmentation, reacquisition, and false associations. Do not claim to reconstruct unseen positions as facts.

**Exit gate:** Demonstrated improvement or characterized limitations on repeatable scenarios; radar and MCP consumers receive consistent identities; user approval.

### Phase 3 — Autonomous Multi-Agent Tool Calling and Coordination

**Goal:** Enable all four LLM agents—Tracking, Observation, Reasoning, and Operator—to independently make decisions, select approved tools, retrieve additional evidence, and determine their next actions within defined safety, latency, and resource limits.

Reuse existing Python functionality by exposing suitable functions as LLM-callable tools, rather than unnecessarily rebuilding the tracking system.

Each agent must demonstrate meaningful autonomy instead of simply executing a predetermined Python workflow.

#### 1. Audit existing architecture and identify tool candidates

Before modifying any code:

- Review existing Python scripts, functions, MCP tools, LLM implementations, API tool/function schemas, execution loops, and orchestration policies.
- Identify which operations are currently executed deterministically by Python and which are independently selected by an LLM.
- Identify existing Python functions that could be exposed as reusable tools without changing their underlying behavior.
- Check for duplicated functionality before creating new tools.
- Determine which agents should have permission to access each tool.
- Document proposed changes, their purpose, and architectural impact before requesting user approval.

**Prioritize reusing existing Python functions and MCP tools over creating new implementations.**

#### 2. Extract and register reusable Python tools

Convert appropriate existing Python functions into validated, LLM-callable tools by wrapping or registering them through the existing tool infrastructure.

Potential tools include:

| Proposed tool              | Purpose                                                                    | Authorized agents      |
| -------------------------- | -------------------------------------------------------------------------- | ---------------------- |
| `get_track_history`        | Retrieve recorded object positions, timestamps, and track identities       | Tracking, Observation  |
| `get_detection_evidence`   | Retrieve bounding boxes, detection confidence, and available depth quality | Tracking, Reasoning    |
| `get_recent_observations`  | Retrieve recent frames or visual observations where available              | Tracking               |
| `get_movement_analysis`    | Calculate displacement, speed, and movement direction                      | Observation            |
| `get_trajectory_context`   | Retrieve historical trajectory and coordinate information                  | Observation, Reasoning |
| `get_tracking_uncertainty` | Retrieve identity uncertainty, occlusion flags, and evidence limitations   | Tracking, Reasoning    |
| `get_assessment_history`   | Retrieve previous assessments and supporting evidence                      | Reasoning, Operator    |
| `request_agent_analysis`   | Request specialized analysis from another agent through orchestration      | Operator               |

These are proposed interfaces, not assumptions that the functions already exist. Final tool names and implementations must be determined by the architecture audit.

- Prefer thin wrappers around existing functions instead of copying their logic.
- Use MCP tools where appropriate, but do not require every internal Python function to become an MCP server tool.
- Define explicit input/output schemas, permissions, timeouts, and error handling.
- Prefer read-only evidence retrieval tools.
- Prevent tools from modifying track identities, camera state, or assessments without deterministic validation.
- Preserve existing interfaces and behavior unless a change is justified and approved.

**Keep the following operations deterministic and outside autonomous LLM control:**

- Continuous webcam capture and frame processing.
- YOLO object detection and ByteTrack updates.
- Radar visualization and dashboard rendering.
- Track-state validation and application of approved updates.
- Execution deadlines, permissions, and resource-limit enforcement.

#### 3. Tracking Agent autonomy

- Independently determine when additional tracking evidence is needed.
- Select approved tools to retrieve track history, detection evidence, association context, depth quality, and recent visual observations.
- Use returned evidence to evaluate object identity, association uncertainty, occlusion, and reacquisition.
- Propose association updates supported by recorded observations.
- Decide when sufficient evidence is already available without making unnecessary tool calls.
- Never directly overwrite tracking state or fabricate missing measurements.

#### 4. Observation Agent autonomy

- Independently determine what evidence is necessary to analyze object movement and behavior.
- Select approved tools to retrieve movement history, trajectory inputs, tracking uncertainty, and coordinate or zone context.
- Use existing Python calculations for speed, displacement, and direction through callable tools where appropriate.
- Interpret tool results to identify movement patterns and support predictions.
- Request additional evidence when necessary.
- Explicitly distinguish observed movement from predicted or inferred behavior.

#### 5. Reasoning Agent autonomy

- Independently evaluate evidence and findings produced by the Tracking and Observation Agents.
- Select approved tools to verify observations, retrieve supporting evidence, inspect uncertainty, or obtain previous assessments.
- Request additional analysis through authorized coordination mechanisms when information is missing or contradictory.
- Consider alternative explanations before producing an assessment.
- Decide when further investigation is justified and when the available evidence is insufficient.
- Preserve traceability between conclusions and their supporting observations.

#### 6. Operator / Orchestration Agent autonomy

- Implement an independent LLM-backed Operator Agent rather than relying exclusively on predetermined Python orchestration.
- Allow the Operator Agent to decide which specialized agents should be activated based on the available evidence.
- Dynamically determine agent execution order rather than always following a fixed sequence.
- Expose approved orchestration functions as tools, such as `request_agent_analysis`, when appropriate.
- Allow the Operator Agent to request additional evidence or follow-up analysis from specialized agents.
- Decide when sufficient evidence has been collected to publish an assessment or mark it inconclusive.
- Coordinate shared budgets, priorities, and workloads through deterministic enforcement mechanisms.
- Prevent unrestricted agent-to-agent recursion and uncontrolled repeated analysis.

The Operator Agent selects and coordinates work, while the Python runtime enforces execution limits and validates actions.

#### 7. Implement genuine LLM-directed tool execution

Each agent must have its own approved tool registry and execution workflow.

- Provide the LLM with available tool definitions and validated argument schemas.
- Allow the LLM to decide whether a tool is needed and which approved tool to call.
- Validate the selected tool name, arguments, and permissions before execution.
- Execute the selected Python or MCP tool.
- Return the actual tool result to the same LLM for continued reasoning.
- Allow bounded multi-step tool selection when additional evidence is necessary.
- Validate final structured outputs before accepting them.
- Distinguish LLM-selected tool calls from deterministic Python operations.
- Record which agent selected each tool, its arguments, execution outcome, latency, and how the returned evidence influenced the result.
- Avoid treating a Python function automatically invoked by the application as proof of autonomous tool calling.

#### 8. Maintain controlled autonomy and real-time performance

- Enforce maximum tool calls, reasoning iterations, deadlines, rate limits, and token/cost budgets.
- Validate tool permissions and reject unauthorized operations.
- Prevent duplicate requests, recursive agent loops, and uncontrolled processing on every camera frame.
- Keep LLM execution asynchronous and separate from webcam capture, YOLO processing, ByteTrack updates, and dashboard rendering.
- Ensure delayed agent results cannot overwrite newer or conflicting tracking observations.
- Provide graceful fallback when an agent, tool, or external model becomes unavailable.
- Prevent agents from inventing observations or treating inferred information as confirmed measurements.
- Preserve existing working functionality and avoid unnecessary architectural restructuring.

#### 9. Implement and validate incrementally

Implement Phase 3 in controlled stages:

1. **Tool audit:** Identify existing Python functions and MCP tools suitable for reuse.
2. **Tool extraction:** Wrap or register approved functions as callable tools without duplicating their logic.
3. **Individual agent autonomy:** Implement and test LLM-directed tool selection for Tracking, Observation, and Reasoning independently.
4. **Operator autonomy:** Implement LLM-directed agent selection, coordination, and follow-up analysis.
5. **Integration testing:** Verify bounded multi-agent execution, evidence traceability, and dashboard responsiveness.

Before each implementation stage, explain the proposed modifications and obtain user approval. Do not delete unrelated code, restructure the architecture, or commit changes without approval.

### Exit gate

- All four agents are independently LLM-backed and demonstrate autonomous decision-making.
- Existing Python functions and MCP tools have been audited, with suitable functionality reused through approved tool interfaces.
- Each agent successfully selects and executes at least one appropriate tool when additional information is needed.
- Each agent demonstrates declining unnecessary tool calls when sufficient evidence is available.
- Tool results are returned to the requesting LLM and demonstrably influence subsequent reasoning or final output.
- The Operator Agent dynamically selects which specialized agents to invoke, requests additional analysis when justified, and stops when sufficient evidence has been collected.
- Execution traces clearly distinguish LLM-directed tool calls from deterministic Python operations.
- Tests verify tool schemas, permissions, argument validation, bounded iterations, deadlines, rate/cost limits, fallback behavior, and prevention of recursive loops.
- Delayed agent outputs cannot corrupt newer tracking state.
- Live camera processing, YOLO/ByteTrack tracking, and dashboard responsiveness remain unaffected by agent processing.
- All relevant tests pass, including actual LLM tool-calling integration tests rather than mocked responses alone.
- The user explicitly approves Phase 3 completion before proceeding to Phase 4.

### Phase 4 — Trajectory prediction through 30 seconds

**Goal:** Observation Agent provides evidence-based, timestamped future path hypotheses.

- Predict at multiple horizons (e.g. 2, 5, 10, 30 seconds), not a single overconfident endpoint. Include coordinate frame/units, method, observed history, uncertainty, and validity/freshness.
- Use deterministic kinematic models as transparent baselines/tools where useful; Observation LLM interprets evidence, requests tools, and assesses limitations rather than fabricating measurements.
- Expand uncertainty with horizon and ambiguity; treat long occlusions, turns, sparse samples, and uncalibrated monocular depth as limiting factors.
- Evaluate displacement/path error and coverage by horizon on held-out or repeatable scenarios. Do not promise 30-second accuracy without validation.

**Exit gate:** Structured predictions rendered consistently on radar, measured error/uncertainty by horizon, safe inconclusive behavior, user approval.

### Phase 5 — Critical-area risk assessment

**Goal:** Reasoning Agent evaluates observable security risk from speed, predicted path, and entry into operator-defined restricted zones.

- Define zones in the **same documented coordinate frame** as radar tracks. Validate orientation, scale, transform, and calibration; mark approximate zones/positions accordingly.
- Separate **confirmed observed entry**, **predicted intersection**, and **inconclusive**. Calculate geometric intersections/time-to-entry using traceable tools; do not infer hostile intent from speed alone.
- Reasoning Agent synthesizes supporting observations, trajectory uncertainty, zone context, and alternatives into an explainable risk assessment.
- Operator Agent publishes timely alerts and preserves provenance. Evaluate false alerts, missed entries, time-to-alert, and latency.

**Exit gate:** Repeatable safe/approach/crossing scenarios, traceable alerts, uncertainty handling, radar/dashboard consistency, user approval.

## Cross-cutting boundaries

- Never fabricate detection, depth, location, identity, velocity, prediction accuracy, provider availability, or tool execution results.
- Do not replace independent LLM-backed agents with deterministic-only functions under agent names.
- Do not change models, prompts, timeouts, retries, concurrency, schemas, or event contracts as an unrelated side effect.
- Do not remove files, rewrite major modules, alter unrelated features, commit/push, or initiate paid load tests without explicit approval.
- Respect real-time pipeline responsiveness: do not block frame rendering on LLM requests or run an LLM for every frame by default.
- Use real-provider telemetry for performance claims and deterministic fixtures for repeatable correctness tests; label them separately.
- Preserve backward compatibility where feasible and document migrations when unavoidable.

## Required response format for coding-agent work

**Current phase:** [number and title]

**Observed behavior and evidence:** [files/functions inspected, confirmed facts vs assumptions]

**Proposed minimal change:** [scope and reason]

**Dependencies and risks:** [including impacts on other phases]

**Validation plan/results:** [tests, metrics, pass/fail, limitations]

**Approval gate / next step:** [what needs user approval; do not proceed automatically]

## Phase status tracking

Treat all phases as **not yet user-approved complete** unless the user explicitly says otherwise. Maintain status in the conversation or an existing project tracking file if approved; do not silently create or overwrite a tracking document. The user can reorder phases or authorize parallel work; explicitly record that exception and its constraints.
