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

### Phase 3 — Bounded autonomous tool calling
**Goal:** Allow LLMs to decide **whether and which approved tool to call**, rather than merely receiving predetermined tool results.

- Audit current API tool/function schemas, execution loops, orchestration policies, and existing MCP tools before modifying.
- Tracking tools may retrieve track history, association evidence, detection/depth quality, and recent frames where available and authorized.
- Observation tools may retrieve motion history, trajectory inputs, tracking uncertainty, and coordinate/zone context.
- Implement actual LLM-directed tool selection and tool-result return to the same LLM, with schema validation, bounded calls/iterations, deadlines, rate/cost limits, and graceful fallback.
- Operator Agent controls budgets and coordination; prevent recursive ping-pong, duplicate requests, and uncontrolled calls on every frame.
- Distinguish tool use selected by the LLM from Python tools invoked deterministically.

**Exit gate:** Tests demonstrate an agent choosing to call a tool when evidence is missing and declining unnecessary calls; tool results influence the final assessment; budget and timeout tests pass; user approval.

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
