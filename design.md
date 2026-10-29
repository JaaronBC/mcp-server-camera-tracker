---
name: multi-agent-architecture-guardrail
description: Keeps changes to the Counter-UAS tracking project aligned with the intended three-agent LLM architecture, evidence-grounded reasoning, asynchronous processing, and validated coordination.
---

# Multi-Agent Architecture Guardrail

## 1. Purpose

You are working on an existing Counter-UAS research prototype that uses real-time visual tracking and three specialized agents.

Your primary responsibility is to improve the existing system without drifting from its intended architecture, duplicating functionality, or introducing unnecessary complexity.

**Do not implement a different architecture unless explicitly asked.**

Before changing code, inspect the existing implementation, identify the relevant files and functions, and explain your planned changes. Preserve working functionality wherever possible.

## 2. Required Architecture

The system must maintain three distinct LLM-backed agents.

### Agent 1: Tracking Agent

Responsibility:

- Interpret selected camera frames alongside available perception data.
- Help associate visual observations with existing tracks.
- Identify possible identity switches, occlusions, ambiguous associations, and conflicting evidence.
- Return structured proposals with uncertainty and references to supporting evidence.

Constraints:

- YOLO and ByteTrack remain responsible for their existing detection and tracking functions.
- Use deterministic measurements when reliable.
- Never allow an LLM to silently overwrite authoritative tracking measurements.
- Validate proposed associations before updating shared state.

### Agent 2: Observation/Behavior Agent

Responsibility:

- Interpret timestamped track histories and measured movement.
- Describe possible trajectory changes, sustained movement, approach or departure patterns, and other evidence-supported movement patterns.
- Identify missing observations, contradictory measurements, and uncertainty.
- Distinguish observed facts from interpretations and hypotheses.

Constraints:

- This must be a genuine LLM-backed agent with its own model invocation and system prompt.
- Do not replace its interpretation task with hardcoded labels, threshold rules, or deterministic summaries.
- Deterministic calculations may supply speed, velocity, distances, elapsed time, and other measurements.
- Do not infer hostile intent from movement alone.

### Agent 3: Reasoning/Decision-Support Agent

Responsibility:

- Evaluate the Observation Agent's findings alongside relevant underlying evidence.
- Identify evidence gaps and alternative explanations.
- Explain whether operator review is warranted based on the available evidence.
- Decide whether additional observation would materially reduce uncertainty.
- Produce an explainable, evidence-linked assessment.

Constraints:

- This must be a genuine LLM-backed agent with its own model invocation and system prompt.
- Do not simply reuse the Observation Agent's output or return predetermined assessment labels.
- Do not infer hostile intent from movement alone.
- Additional observation requests must be bounded and validated by the coordinator.
- Consequential actions remain under operator control.

## 3. Coordinator Responsibilities

AgentCoordinator is the orchestration layer, not a fourth reasoning agent.

It is responsible for:

- Routing messages between agents.
- Maintaining shared track histories and relevant evidence.
- Validating typed inputs and outputs.
- Checking evidence references.
- Enforcing timeouts, retry limits, and evidence-request budgets.
- Scheduling asynchronous work.
- Logging agent invocations and failures.
- Preventing invalid or incomplete model responses from corrupting shared state.

Do not move the agents' core interpretation and reasoning responsibilities back into AgentCoordinator.

## 4. Required Data Flow

Maintain this workflow:

Tracking Agent → Observation/Behavior Agent → Reasoning/Decision-Support Agent → Dashboard/Operator

When additional evidence is requested:

1. The Reasoning Agent returns a structured request with a justification.
2. AgentCoordinator validates the request and checks the configured limits.
3. The tracking pipeline collects additional evidence when permitted and available.
4. The updated evidence is processed by the relevant agents.
5. The new assessment is linked to the evidence that produced it.

Do not create an unbounded autonomous loop. Do not block the live camera loop while waiting for an LLM.

## 5. Implementation Rules

Before editing:

1. Inspect the existing repository structure.
2. Locate the current agent classes, prompts, model clients, schemas, coordinator, MCP tools, and tests.
3. Identify existing functionality that can be reused.
4. Report which files and functions need to change.
5. State how the proposed changes preserve the required architecture.

During implementation:

- Follow existing project conventions and dependencies.
- Reuse the existing OpenAI-compatible model-access pattern where appropriate.
- Keep each agent's model configurable.
- Use distinct prompts, calls, inputs, and structured output schemas for each agent.
- Validate model outputs before downstream use.
- Include evidence identifiers, uncertainty, and limitations where appropriate.
- Add timeouts, bounded retries, exception handling, and graceful failure behavior.
- Keep expensive model calls outside the real-time camera loop.
- Avoid unnecessary refactoring, new abstractions, dependencies, or additional agents.
- Do not remove working functionality without explaining why and obtaining approval when appropriate.

Never claim an agent uses an LLM unless the implementation contains an actual model invocation on the relevant execution path.

## 6. Testing Requirements

Add or update tests covering:

- Each agent's independent LLM invocation.
- Correct prompts and structured output validation.
- Evidence references and uncertainty fields.
- Missing, malformed, and contradictory evidence.
- Model timeouts and failures.
- Bounded retries and evidence requests.
- Asynchronous execution without blocking tracking.
- End-to-end data flow through all three agents.
- Regression of existing MCP tools and tracking behavior.

Mock model responses in unit tests where practical. Clearly distinguish mocked tests from tests that use a real model endpoint.

## 7. Definition of Done

A change is complete only when:

- [ ] The three-agent architecture remains intact.
- [ ] Each of the three agents has a distinct responsibility.
- [ ] All three agents make genuine, independently verifiable LLM calls on their intended execution paths.
- [ ] Observation and Reasoning are not secretly replaced by deterministic shortcuts.
- [ ] Structured outputs are validated before being trusted.
- [ ] Evidence and uncertainty are preserved throughout the workflow.
- [ ] Additional observations are requested only through bounded coordination.
- [ ] The real-time tracking loop remains responsive.
- [ ] Relevant tests pass, or failures are clearly reported.
- [ ] The existing dashboard and MCP interfaces remain compatible unless a change is explicitly required.

## 8. Final Response Format

After completing a task, report:

1. **Files changed:** Exact paths and purpose.
2. **Functions changed:** Exact function or class names.
3. **Agent call sites:** Where each LLM invocation occurs.
4. **Data flow:** How information travels through the three agents.
5. **Validation and failure handling:** How malformed responses, timeouts, and failures are handled.
6. **Tests:** What was run and the actual results.
7. **Remaining work:** Anything incomplete, unverified, or requiring a real model endpoint.

Do not claim tests passed, endpoints work, or LLM calls were verified unless you have evidence that they did.

## 9. Milestones Toward the Goal

Use the following milestones to incrementally harden the current three-agent system without architecture drift.

### Milestone 1: Baseline Stability Lock

Deliverables:

- Confirm full test baseline is green with current architecture.
- Freeze critical contracts for Observation/Reasoning payloads and health summary schemas.
- Document current retry/timeout defaults and fallback behavior.

Acceptance criteria:

- `python -m unittest discover -s tests -p 'test_*.py' -v` passes (except explicitly gated tests).
- No unresolved schema mismatches in `dashboardReasoning` and `dashboardAgentOps` outputs.

### Milestone 2: Live LLM Path Verification

Deliverables:

- Run opt-in live tests for Observation and Reasoning LLM paths.
- Validate bounded evidence-request behavior under real endpoint responses.
- Capture and classify live failures (timeout/auth/rate/format) using existing `error_code` fields.

Acceptance criteria:

- Live suite executes with `RUN_LIVE_LLM_TESTS=1` and produces evidence-backed pass/fail logs.
- Any live failures map to explicit `error_code` values and safe fallback paths.

### Milestone 3: Full Agentic Orchestration

Deliverables:

- Ensure Tracking, Observation, and Reasoning agents operate as distinct components with independent invocation paths.
- Formalize coordinator-mediated communication contracts between agents (inputs, outputs, evidence references, uncertainty, request budgets).
- Add explicit handoff validation so each stage consumes only typed, validated upstream outputs.
- Verify asynchronous orchestration keeps camera/tracking loop responsive while agent work executes out-of-band.

Acceptance criteria:

- End-to-end tests confirm full pipeline flow: Tracking → Observation → Reasoning → Operator summary.
- Tests validate coordinator-enforced evidence request bounds and rejection of malformed inter-agent payloads.
- No direct shortcut paths bypass Observation/Reasoning responsibilities.

### Milestone 4: Agent Health Alerting

Deliverables:

- Add configurable thresholds for warning/critical levels on error rate and stale freshness.
- Emit health status labels in agent-ops summaries while preserving backward compatibility.
- Ensure stale windows do not produce misleading trend or digest claims.

Acceptance criteria:

- Unit tests cover healthy/warn/critical threshold transitions.
- Runtime summary text reflects status labels only when freshness checks pass.

### Milestone 5: Trend Interpretability and Operator Clarity

Deliverables:

- Extend `agent_error_trends` with explicit context: sample size, window age, and recency.
- Improve dashboard/runtime formatting so trend statements are concise and audit-friendly.
- Keep summaries compact, deterministic, and tolerant of sparse data.

Acceptance criteria:

- Tests verify trend metadata presence and formatting under fresh/stale/missing-data cases.
- No regression in existing `main.py` summary behavior.

### Milestone 6: Operational Hardening and Release Readiness

Deliverables:

- Final regression pass across focused suites and full discovery.
- Update `README.md` with new health fields, thresholds, and live-test instructions.
- Verify no changes break MCP tool compatibility or tracking-loop responsiveness.

Acceptance criteria:

- Focused and full suites pass, with any skipped tests explicitly explained.
- Documentation reflects the implemented contracts and environment toggles.

### Execution Rule

Complete milestones in order. Do not start a later milestone until the previous one has passing tests and documented outcomes.
