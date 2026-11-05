---
name: agentic-security-project-context
description: Restore project history, architecture, constraints, and current plan when resuming work on the MCP camera tracking and agentic predictive security monitoring capstone. Use at the beginning of a new coding-agent conversation or before planning changes.
---

# Agentic Security Monitoring — Project Continuity

## Purpose

Use this skill to resume development without losing architectural decisions, recent integration work, or the current product/demo direction. **This is historical context, not proof of the current repository state.** Inspect the working tree, code, tests, and runtime configuration before making claims or edits.

## Project evolution and goal

- Origin: Counter-UAS / counter-drone capstone using camera/video observations, with possible later RF, radar, or acoustic inputs.
- Current proposed positioning: **agentic predictive security monitoring** for a client with restricted areas (e.g., warehouse, industrial facility, or campus). The immediate demo may track people with a webcam; drones remain a possible specialized application. **This repositioning is a plan, not evidence that code or project branding has been changed.**
- Client problem: conventional camera monitoring is often reactive; operators must review many feeds and alerts, while occlusion, crowded tracks, and poor visibility can degrade tracking. Proposed differentiator: evidence-backed tracking continuity, short-horizon trajectory forecasts, and explainable alerts about possible critical-zone crossings.
- Capstone pitch: demonstrate a concrete customer problem, technical feasibility, measurable benefits, and honest limitations. Do not claim proven commercial effectiveness without measurements.

## Existing architecture: verify against current code

The intended system is one canonical real-time pipeline:

1. Camera or optional video-file source produces frames.
2. YOLO detection and ByteTrack tracking produce observations; depth/ground-plane estimates and quality flags may be available depending on implementation/calibration.
3. Canonical observation/evidence handling records IDs, timestamps, positions, confidence, source frame, track history, and uncertainty.
4. MCP tools and the orchestration runtime coordinate analysis and expose evidence.
5. Independent LLM-backed roles are intended:
   - **Tracking Agent:** gather/assess tracking evidence, identity association, occlusion and ambiguity; never fabricate missing measurements.
   - **Observation Agent:** interpret movement history, estimate gaps, and forecast paths with uncertainty.
   - **Reasoning Agent:** assess evidence and possible zone crossings; distinguish predicted intrusion risk from malicious intent.
   - **Operator/Orchestration Agent:** decide when analysis is warranted, request evidence, coordinate handoffs, validate outputs, and publish traceable assessments or mark them inconclusive.
6. Dashboard displays camera view, tracks, top-down radar/ground-plane representation, and assessments.

Important: YOLO/ByteTrack may provide perception and baseline association, but should not be presented as fulfilling all agent reasoning. Preserve independent LLM roles; also do not make the camera/radar wait on slow LLM reasoning. Use one integrated pipeline rather than a separate dashboard-only tracking path.

### Known implementation areas (historical)

- `main.py`: CLI and runtime startup; optional `--video-path` support was reported present.
- `agents/orchestration/runtime/`: orchestration logic, including `verification_queue.py`.
- `agents/orchestration/ops_orchestration.py`: reported compatibility shim re-exporting runtime core; do not create duplicate orchestration logic.
- `agents/tracking/evidence_store.py`: evidence storage and cleanup/eviction accounting.
- `agents/dashboard/`: tracking loop, rendering, live display; camera/video source should flow through canonical runtime.
- `tests/`: verification, evidence packaging, Stage-D tracking integration, orchestration, CLI, dashboard tests.

Discover the actual module names and call graph in the current repository; historical paths can be renamed.

## Recent integration and validation history

- A prior integration reconciled the modular `Cleanup` work with `main`'s optional video-file input. The intended input path was CLI `--video-path` → runtime configuration → MCP/dashboard tracking loop → `cv2.VideoCapture(source)`.
- An audit reported branch `main`, no active `MERGE_HEAD`, no unresolved conflicts, and six staged code/test files. **This was a snapshot, not current Git status.** Do not assume the merge was committed, that `Cleanup` is an ancestor of HEAD, or that the index is unchanged.
- Six files reported staged at that snapshot:
  - `agents/orchestration/ops_orchestration.py`
  - `agents/orchestration/runtime/verification_queue.py`
  - `agents/tracking/evidence_store.py`
  - `tests/test_stage_d_tracking_integration.py`
  - `tests/test_tracking_evidence_packaging.py`
  - `tests/test_verification_queue_runtime.py`
- Reported fixes: persist `frame_metadata_unavailable` lease-fallback diagnostics on terminal verification records and clean up per-request state; test invalid nonempty `event_origin`; avoid double-counting admission-capacity evictions on repeated cleanup; preserve lease-gated rejection tests while exercising provider failure, timeout, and budget exhaustion using valid evidence fixtures.
- Reported test runs: 33, 4, 25, and 55 successful test executions in overlapping test groups; `git diff --cached --check` passed. These were **agent-reported**, not independently verified here, and must be rerun after changes.
- At that snapshot, runtime `agent_events.jsonl` modifications were unstaged and `merge_conflict_snapshots` were untracked. Preserve them unless the user explicitly authorizes cleanup.
- No confirmed commit/push result is available in this context.

## Current development priorities

### Priority 1 — Profile and reduce agent latency

- Previously observed end-to-end agent processing: roughly **8–12 seconds** in a test; exact cause is unconfirmed.
- Goal: ideally **under 1 second** for agent processing, but first measure feasibility. Separate latency targets for (a) live tracking/radar updates, (b) initial critical-zone alert, and (c) complete multi-agent assessment.
- A historical visual-model test was described as taking about 12.87 seconds for one request; treat as a lead requiring verification, not a confirmed current benchmark.
- **Immediate planned work:** instrumentation and baseline benchmarking before optimization. Measure `time.perf_counter()` durations with correlation IDs across camera inference, evidence acquisition, queue wait, MCP tools, each agent/provider request, orchestration, and dashboard delivery. Record provider/model, token counts, time to first token if available, p50/p95, errors/timeouts, and number of calls. Avoid logging secrets, raw images, or unnecessary sensitive content.
- Benchmark single/multiple tracks, occlusion, critical-zone approach, and multi-agent collaboration; use real providers when available and label mocks distinctly.
- Then propose ranked optimizations: faster models/providers, reduced prompts/tokens and image payloads, event-triggered LLM work, safe concurrency, avoiding repeated calls, nonblocking dashboard, and stale-result handling. **Do not change models or architecture before review/approval.**

### Priority 2 — Occlusion and crowded-track management

- Tracking and Observation Agents should cooperate to handle objects behind obstacles, background blending, overlapping detections, and identity switches.
- Preserve observed vs estimated positions, track confidence, source timestamps, and association evidence. Allow inconclusive outcomes; never silently invent measurements.
- Evaluate recovery accuracy and ID switches under controlled test cases.

### Priority 3 — Path prediction

- Desired long-range feature: predict tracks up to **30 seconds** ahead, but start with reliable 1–5-second horizons and quantify prediction error and widening uncertainty.
- Calibrate the camera/ground-plane mapping before interpreting pixel displacement as physical speed or real-world distances.
- Show forecasts as uncertain estimates, not guaranteed paths.

### Priority 4 — Critical-area risk assessment

- Allow an operator to define a protected area. Reasoning Agent assesses whether an observed/predicted path may cross it, estimated time to entry, evidence quality, and uncertainty.
- Prefer explainable, human-reviewable alerts over asserting that a person is dangerous. Avoid subjective suspicious-person classification.
- Measure false alerts, missed crossings, lead time, and alert latency.

## Suggested live demo

A fixed webcam monitors a designated restricted area. Two people cross paths; one is temporarily occluded. Tracking/Observation maintain or cautiously recover track identity. A path forecast shows possible entry into the restricted zone. Reasoning evaluates the crossing evidence and emits a concise, uncertainty-aware operator alert. Radar and video continue updating during asynchronous agent analysis. If capabilities are incomplete, present them honestly as planned rather than functioning.

## Engineering rules and approval boundaries

1. **Inspect before editing.** Read `git status`, current branch, recent commits, `MERGE_HEAD`, staged/unstaged diff, relevant code and tests. Never infer the latest state from this skill alone.
2. **Explain before changing.** State root cause, ownership boundary, files to touch, risks, and tests; obtain user approval for material changes.
3. Preserve the canonical observation/evidence schema and the single integrated MCP/dashboard pipeline.
4. Preserve distinct LLM agent responsibilities and orchestration; avoid replacing them wholesale with deterministic code merely for speed. Deterministic fast-path checks can support real-time responsiveness, but must not masquerade as agent reasoning.
5. Keep camera/radar processing responsive; handle delayed reasoning and stale observations explicitly.
6. Preserve optional webcam/video-file input and relevant test coverage.
7. Do not weaken evidence validation, lease rules, provider budgets, or uncertainty handling to make tests pass.
8. Do not remove files, generated artifacts, logs, or snapshots without approval.
9. Do not commit, push, merge, reset, abort, or clean the repository unless specifically authorized.
10. Distinguish measured performance from hypothetical estimates, and implemented features from plans.

## Startup checklist for a new agent chat

1. Read this skill and summarize the current product goal and technical constraints.
2. Inspect `git status --short`, `git branch --show-current`, `git log -5 --oneline --decorate`, and whether `MERGE_HEAD` exists; inspect staged and unstaged changes without modifying them.
3. Map the current runtime entry point, video-source selection, MCP tools, orchestration, independent agents, evidence flow, and dashboard update path.
4. Check whether latency instrumentation or benchmark results already exist; avoid duplicating work.
5. Run only appropriate non-destructive tests and report exact commands/results.
6. Present **Implemented / Verified / Planned / Unknown** findings and the next smallest scoped task.
7. Wait for explicit approval before changing behavior or Git state.

## Immediate next action

**Profile the existing agent-processing pipeline and propose a data-driven latency optimization plan.** Instrumentation may be added only after explaining its scope and receiving approval; performance optimizations and model changes require a separate approval. The user wants to understand the source of the 8–12-second delay and move toward a sub-second experience without sacrificing evidence quality or genuine multi-agent behavior.
