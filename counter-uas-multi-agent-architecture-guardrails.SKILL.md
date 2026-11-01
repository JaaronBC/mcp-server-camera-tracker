---
name: counter-uas-multi-agent-architecture-guardrails
description: Persistent architecture and implementation guardrails for the Counter-UAS MCP multi-agent camera tracking project.
---

# Counter-UAS Multi-Agent Architecture Guardrails

## Mission
Keep future audits, implementation, and refactoring aligned with an evidence-grounded multi-agent tracking system. Preserve working deterministic perception, explicit agent boundaries, human operator oversight, and incremental testable development. This skill is not a request to implement any particular phase.

## Agent responsibilities

**Tracking Agent:** Consume deterministic YOLO/ByteTrack detections, depth and coordinate estimates, history, and available images; propose associations and report uncertainty. Never claim uncertain continuity as established fact.

**Observation Agent:** Interpret available evidence and, once actual visual inputs are accessible, independently verify tracking claims. Distinguish visible, occluded, not visible, possible ID switch, and insufficient evidence. Do not claim to inspect images when only metadata was supplied.

**Reasoning Agent:** Consume validated observation handoffs, analyze patterns and limitations, and provide evidence-grounded operator decision support. Do not duplicate frame-level verification or tracking.

**Operator/Orchestration Agent:** Schedule and coordinate bounded workflows, manage timeouts and failures, and present qualified outcomes. Do not assume existing deterministic orchestration is already an independent LLM agent.

**Deterministic AgentCoordinator:** Validate schemas, IDs, timestamps, provenance, freshness, conflicting results, budgets, and permitted state transitions. It is not an LLM agent. LLM outputs are proposals, not commands.

**Perception/dashboard:** Keep camera capture and YOLO/ByteTrack local and non-blocking. The dashboard may currently own process and state production; verify the actual code before changing it. Avoid adding reasoning logic to rendering.

## Non-negotiable rules

1. **Deterministic authority.** Observation and Tracking agents cannot directly mutate YOLO/ByteTrack internals. Coordinator validation and explicit deterministic policy govern any accepted feedback.
2. **Evidence before conclusions.** Include source frame/crop/history references, timestamps, supporting and contradictory evidence, uncertainty, and limitations where available. Never fabricate evidence or image access.
3. **Separate confidence concepts.** Detection confidence, association confidence, visual verification, and identity consistency are distinct. Do not automatically overwrite one with another.
4. **Non-blocking frame loop.** Never wait for MCP or an LLM on each frame. Prefer bounded async submission, polling, rate limits, timeouts, and failure fallback.
5. **Preserve agent separation.** Do not merge specialized agents or turn coordinator into a reasoning model for convenience.
6. **Minimal infrastructure.** Prefer existing contracts and async request/response paths. No event bus, broker, database, or duplicate MCP tools without evidence of need and approval.
7. **Human oversight.** Tracking behavior does not establish hostile intent. Do not implement autonomous countermeasures.
8. **Reality over plans.** Label every capability as implemented, partial, proposed, or unknown based on repository inspection; never treat an intended contract as already live.

## Communication design anchors

These are **proposed concepts**, not proof that these classes exist. Inspect existing `agents/contracts.py`, coordinator, tracking runtime, observation input builders, reasoning handoff, and tests before choosing exact fields.

### Tracking → Observation: `TrackingEvidenceEnvelope`

Purpose: describe a tracking claim and evidence available for independent review. Candidate fields: schema version, message ID, track/detection/frame IDs, capture timestamp, pixel bbox and center, depth and ground-position estimates with units, detector and association confidence, visibility/quality flags, association claims and conflicts, track history, uncertainty, and retrievable frame/crop references. Only populate fields grounded in real upstream data.

### Observation → Tracking: `VisualVerificationEnvelope`

Purpose: return an evidence-grounded result for a specific claim. Candidate fields: schema version, observation ID, referenced track and evidence IDs, evaluation and source timestamps, visibility status, verification result, identity consistency, possible ID switch, supporting/contradicting evidence, confidence, uncertainty, limitations, and suggested follow-up.

Prefer separate dimensions: e.g., visibility (`visible`, `partially_occluded`, `occluded`, `not_visible`, `unknown`) and verification (`supported`, `contradicted`, `inconclusive`). Do not force binary judgments. Adopt existing repository naming conventions when possible.

### Observation → Reasoning

Preserve existing structured handoff and supporting-observation-ID validation. If needed, add an optional verification section; do not create a parallel reasoning pipeline. Reasoning must distinguish observed facts, tracker claims, and verification interpretations.

## Visual evidence requirements

Before implementing verification, trace whether Observation actually receives images or only radar metadata. Verify frame/crop reference retrieval, timestamp/track binding, retention, expiration, payload size, and missing-reference behavior. A `frame_reference` string alone is not proof of retrievable evidence. Prefer stable bounded references rather than repeatedly embedding full images in logs or state.

No detection does not prove disappearance. Reappearance near a prior position does not prove identity. Occlusion may justify uncertainty, not an asserted hidden position.

## Feedback lifecycle

1. Tracking produces evidence and an uncertain claim.
2. Orchestration schedules bounded asynchronous verification outside the camera loop.
3. Observation inspects only evidence it can access and returns a typed outcome.
4. AgentCoordinator validates IDs, timestamps, provenance, freshness, consistency, and allowed transitions.
5. Deterministic policy decides whether metadata may change.
6. Tracking applies approved metadata updates at a controlled boundary; Observation never directly changes tracker internals.
7. Validated verification context may flow to Reasoning and operator events.

Explicitly define policy for visibility metadata, verification status, uncertainty flags, association-review status, lifecycle state, and any ID reassignment. Do not assume all are authorized. Handle reused/unknown IDs, out-of-order results, conflicts, missing images, duplicate requests, timeout, track termination, and inconclusive responses. On failure, do not invent a correction.

## Mandatory workflow for coding tasks

**1. Baseline:** Inspect branch, `git status`, and relevant history. Identify pre-existing edits. Never reset, stage, delete, or overwrite unrelated user work.

**2. Trace:** Read implementations, callers, consumers, contracts, tests, MCP registration, serialization, and patch targets. Distinguish verified facts from assumptions.

**3. Explain before editing:** State current data path, problem, why change matters, smallest proposed fix, expected files, compatibility risks, and tests. Seek approval for meaningful architecture changes unless the task explicitly authorizes that scope.

**4. Implement narrowly:** Preserve interfaces, MCP names, import paths, test patch points, fallback semantics, and runtime ownership unless approved otherwise. Avoid opportunistic cleanup and unnecessary dependencies.

**5. Test:** Add focused tests and run relevant regression suites. Cover malformed payloads, schema versions, stale or conflicting evidence, missing images, timeouts, duplicate/out-of-order responses, non-blocking operation, and existing Observation→Reasoning compatibility as applicable. Never claim unrun tests passed.

**6. Review diff:** Inspect `git diff` and `git status`. Remove only generated artifacts created by this task, without discarding user changes. Report exact files changed.

**7. Report:** Explain what changed, why, before/after data flow, test results, compatibility, limitations, and remaining risks.

## Prohibited shortcuts

- Per-frame LLM/MCP blocking.
- Direct LLM mutation of tracker state or IDs.
- Claiming image-based verification from metadata alone.
- Treating missing detections as disappearance or proximity as proven identity.
- Conflating detection, association, identity, and verification confidence.
- Unvalidated agent feedback or invented evidence references.
- Duplicate reasoning pipelines, redundant wrappers, or gratuitous new services.
- Dashboard-owned agent reasoning or another uncontrolled loop.
- Broad rewrites, deletions without reference audit, or weakening tests to pass.
- Inferring hostile intent or introducing autonomous response actions.

## Definition of done

The change supports a stated goal; agent roles remain separate; evidence and uncertainty are traceable; deterministic validation remains authoritative; the camera loop remains responsive; compatibility is preserved or approved changes documented; relevant tests are reported accurately; and the final diff is scoped to the task.

## How to invoke

At the start of each project task, read this skill alongside the phase-specific prompt. Identify the architecture boundaries affected, verify the current repository rather than relying on historical summaries, and flag conflicts before coding. The skill supplies durable guardrails, not permission to expand the requested scope.
