# Agentic Security Monitoring — Project Continuity

## How to use this skill

Read this at the beginning of a new coding-agent chat. It is a **historical handoff based on previous agent reports, not proof of the current repository state**. Inspect code, Git state, tests, configuration, and persisted budget state before claiming anything is implemented, verified, or safe to execute. Do not interpret the task descriptions below as permission to modify code, spend provider requests, reset budgets, or change Git state.

## Product direction and demo

- Origin: Counter-UAS / counter-drone capstone, initially using camera/video, with potential later RF, radar, and acoustic sensing.
- Proposed broader framing: **agentic predictive security monitoring** for restricted areas such as warehouses, facilities, or campuses. A webcam tracking people is a viable immediate demo; drone-specific applications remain possible. This framing is proposed, not proof that branding or production code changed.
- Problem: reactive video monitoring, crowded scenes, occlusion, low visibility, ID switches, and operator overload.
- Goal: evidence-backed identity continuity, short-horizon trajectory prediction, and explainable, uncertainty-aware critical-zone alerts. Do not claim accuracy or commercial impact without measurements.
- Intended demo: two targets cross; one may be occluded; the system maintains or cautiously recovers identity, predicts possible restricted-area entry, and shows camera/radar updates while agents reason asynchronously. Label unimplemented capabilities as planned.

## Architecture and role boundaries — verify in repository

Canonical intended flow: **camera or optional video-file source → YOLO detections / ByteTrack baseline tracks → depth, pixel/ground-plane estimates and quality flags → timestamped canonical evidence/history → MCP/orchestration → independent LLM agents → traceable assessments → camera/radar dashboard**.

- **Tracking Agent (LLM):** evaluate detections and identity associations, investigate switches, gaps, and occlusion, propose or revise identities from evidence. Do not fabricate missing observations.
- **Observation Agent (LLM):** independently verify associations and interpret motion/history; eventually predict trajectories with explicit uncertainty. Its verdict must not simply copy Tracking's conclusion.
- **Reasoning Agent (LLM):** synthesize evidence, possible trajectories and critical-zone crossing risk; do not infer malicious intent from movement alone.
- **Operator/Orchestration Agent (planned role):** decide when additional analysis is warranted, coordinate tools and agent handoffs, enforce budgets and provenance, and publish assessments or mark them inconclusive.
- **Deterministic components:** YOLO/ByteTrack, motion/depth checks, group-assignment constraints, budget enforcement, and diagnostic recovery are useful supporting infrastructure. Do not present them as independent LLM reasoning or let deterministic recovery secretly supply the Tracking Agent's answer.
- Keep the live camera/radar responsive even when LLM calls are slow. Preserve one integrated evidence/MCP/dashboard pipeline, not a second independent dashboard tracker.

Historical repository areas (confirm actual paths): `main.py`; `agents/dashboard/` and `agents/dashboard/tracking_loop.py`; `agents/orchestration/runtime/` including `verification_queue.py`; `agents/orchestration/ops_orchestration.py` (previously reported compatibility shim); `agents/tracking/evidence_store.py`; `agents/tracking/agent_tracking.py`; `agents/observation/agent_observation.py`; `agents/contracts.py`; `agents/coordinator.py`; `agents/provider_call_budget.py`; `agents/llm_call.py`; `tools/tracking_agent_phase1_validation.py`; `tests/`.

Historical integration: optional `--video-path` source routed to `cv2.VideoCapture(source)` through runtime/dashboard. A previous merge reconciliation involved orchestration, verification queue, evidence store and related tests; old reports of staged files, branch status, or merge state are **not current Git facts**. A prior dashboard box-flicker issue was attributed to publishing raw frames before annotated YOLO frames; user reported it fixed. Do not reopen without evidence.

## Roadmap and current phase

1. **Phase 1 — agent processing performance:** measure model/provider latency, errors, token usage, p50/p95, end-to-end responsiveness and dashboard isolation. Historical 8–12-second observations and an aspirational sub-second target are not current benchmarks. Phase 1 has **not been formally approved as complete**.
2. **Phase 2 — track management and Tracking–Observation collaboration: CURRENT FOCUS.** Handle ID swaps, crossings, occlusion, conflicting associations, independent verification, provenance, group uniqueness and cautious recovery. Mocked diagnostic recovery exists by agent report; **live LLM identity recovery has not yet been demonstrated**. Keep Phase 2 open.
3. **Phase 3 — autonomous multi-agent tool calling and Operator coordination:** bounded tools, scheduling, audits, budgets, and state validation. Not claimed complete.
4. **Phase 4 — trajectory prediction:** start with reliable 1–5 seconds, quantify error and uncertainty, eventually explore up to 30 seconds; calibrate ground plane before claiming physical speeds/distances.
5. **Phase 5 — critical-area risk assessment:** operator-defined zone, predicted crossings, lead time, false/missed alerts and human-reviewable explanations. Not claimed complete.

## Phase 2 — verified-by-report development history

All outcomes below are **coding-agent reports**, not independent verification by this skill. Re-run relevant tests and inspect the actual code before relying on them.

### Initial live diagnostic and observed failure

- Tracking model: `qwen-3.8-27b`, with `reasoning_effort=none` and max output 768 tokens in the prior controlled run.
- Synthetic two-target `yolo-id-swap` case: YOLO assigned `det-amb-1 → track-B` and `det-amb-2 → track-A`; independent motion history favored the opposite identities (`track-A`, `track-B`).
- Tracking LLM retained the swapped assignments with confidence 0.45 each. Observation LLM returned `inconclusive` for the first and `contradicted` for the second. Deterministic motion checks flagged inconsistency, yet the **production coordinator accepted both** because it did not gate on that motion mismatch or diagnostic Observation verdict.
- This demonstrated an identity-management gap, **not successful live recovery**. Do not claim the production coordinator has been fixed or that diagnostic corrections affect live tracking state.
- That earlier approved live run attempted 3 provider requests (Tracking 1, Observation visual 2), consuming the then-configured persisted budget.

### Diagnostic adjudication and deterministic recovery

- Diagnostic-only adjudication classifies proposals as `accepted`, `rejected`, or `unresolved`; provisional confidence threshold 0.55. Observation contradictions, insufficient support, deterministic mismatch, and duplicate identity proposals are reported with reasons.
- Added replay tests for confidence boundaries, 2/3/5-target conflicts, occlusion/missing evidence, stale/malformed history, and retention of `alternative_track_candidates` and provenance.
- Diagnostic-only alternative identity recovery evaluates **existing** candidate identities against independent motion endpoints, depth and timestamps, and uses joint assignment to prevent duplicate corrected IDs. It can propose correction or remain unresolved; **it never changes live track state**.
- Agent reported 53 passing focused no-network tests after deterministic recovery; these are historical counts, not the current suite total.

### LLM Tracking–Observation reconsideration

- Diagnostic workflow intended: initial Tracking proposals → Observation round 1 → diagnostic adjudication → conditional **one batched Tracking reconsideration** for eligible ambiguous detections → Observation round 2 per revised detection → deterministic audit and global uniqueness across revised **and non-revised** assignments → final diagnostic adjudication.
- Tracking reconsideration receives original proposals, independent history, alternatives, and Observation feedback, **not** deterministic recovery answers or synthetic ground-truth identity labels. Tracking must originate the revised proposal; Observation independently verifies it.
- Round-specific reports preserve initial and revised identities, verdicts, evidence provenance, status and `identity_origin` such as `revised_tracking`.
- Mocked no-network tests reportedly covered 2-target swap recovery, 3-target permutation, 5-target conflicts, duplicate assignments, insufficient evidence, decline to revise, malformed replies, timeouts and unchanged coordinator behavior.
- A prior audit found reconsideration implemented in `tools/tracking_agent_phase1_validation.py` but real-provider invoker not wired; later work reportedly added `_invoke_tracking_reconsideration_live(...)` and explicit `--tracking-reconsideration-mode {off,live}` with default `off`.
- Agent reported 68 tests in one collaboration/adjacent suite and 18 in a budget/wrapper suite after this wiring. The mocked results **do not prove actual LLM recovery**.

### Provider budgets and observability — latest reported work

- Diagnostic request upper bound for `N` initial detections and `E` reconsidered detections: initial Tracking 1 + conditional reconsideration Tracking 1 if `E>0` + initial Observation `N` + revised Observation `E`, assuming live Observation on all detections and no retries. Worst case `E=N`: **2 + 2N** requests.
- Two-target worst case: **Tracking 2 + Observation visual 4 = total 6**. Three targets: 8 total; five targets: 12 total. These are request ceilings for the specified workflow, not guaranteed call counts or token-cost limits.
- Diagnostic `--max-total-requests` and persisted per-role provider-call budget are distinct safeguards. Verify each attempt is counted exactly once in each applicable mechanism, including failed/time-out attempts, without bypass or accidental double counting.
- Last **read-only preflight** inspected `captures/provider_call_budget_state.json` and reported limits `total=3`, `tracking_vlm=1`, `observation_visual=2`, `observation_text=0`, `reasoning_text=0`; attempts already equaled those limits, so **remaining allowance was zero**. This is a historical snapshot; read current state again. **Do not reset, overwrite, or increase budget without explicit user approval.**
- The preflight gave **NO-GO** for a live six-request experiment because persisted budget was exhausted and per-call token telemetry was incomplete.
- Most recent coding-agent report says it added diagnostic-only per-attempt `provider_call_telemetry` fields: `role`, `round`, `detection_id`, `model`, `status`, `input_tokens`, `output_tokens`, `total_tokens`, `usage_available`, `usage_unavailable_reason`, `latency_ms`, `error_reason`. Unavailable usage should be `None`, not invented zero.
- Reported additions: `_invoke_observation_visual_verification_live(...)` and telemetry in `_invoke_tracking_reconsideration_live(...)`; `_compute_persisted_budget_preflight(...)` reads persisted budget and fail-closes on shortfall; report includes `config.persisted_provider_budget_preflight` and top-level `provider_call_telemetry`.
- Latest reported tests: `python -m unittest tests.test_tracking_agent_phase1_observation_collaboration` → **43 OK**; `python -m unittest tests.test_provider_call_budget tests.test_observation_visual_budget_integration` → **22 OK**. These suites overlap earlier tests; **do not sum all historical counts as a unique total**.
- **Still to audit:** whether initial Tracking is included in the unified telemetry ledger; whether budget preflight occurs **before the first provider call**; whether every live call respects both budgets; whether persisted state stays unchanged in no-network tests; whether failures/timeouts are logged with reasons.
- The agent reported removing a transient `captures/provider_call_budget_state.lock` after tests. Never remove a lockfile without verifying that no process is holding or relying on it; preserve unrelated artifacts.

## Current next task — for the NEW coding-agent chat

**Perform the broader no-network regression and a READ-ONLY final safeguard audit. Do not run a live provider experiment.** This replaces the old skill's immediate instruction to begin latency profiling.

1. Inspect `git status --short`, branch, recent commits, `MERGE_HEAD`, staged/unstaged diffs and any unrelated modified files/logs. Do not change them.
2. Re-read `tools/tracking_agent_phase1_validation.py`, its collaboration tests, the Tracking/Observation provider paths, and `agents/provider_call_budget.py`. Confirm current code matches reported behavior.
3. Run, without network calls:
   ```bash
   .venv/bin/python -m unittest -q \
     tests.test_tracking_agent_phase1_observation_collaboration \
     tests.test_observation_visual_verification \
     tests.test_tracking_runtime_support_metrics \
     tests.test_visual_tracking_runtime
   .venv/bin/python -m unittest -q \
     tests.test_provider_call_budget \
     tests.test_observation_visual_budget_integration
   ```
   Report actual counts/failures; do not assume the earlier 68-test count remains valid after new tests.
4. Trace telemetry for **initial Tracking, initial Observation, Tracking reconsideration, and revised Observation**, including attempted failures and timeouts.
5. Prove persisted-budget preflight executes **before initial Tracking**, computes the two-target worst-case requirement (Tracking 2, Observation 4, total 6), and blocks calls on insufficient allowance. Read current persisted budget without modifying it.
6. Verify diagnostic request-count and persisted role budgets both apply without bypass/double counting; check that tests do not mutate persisted state.
7. Verify no answer keys/deterministic recovery output enter LLM payloads; independent Observation evidence remains independent; coordinator/live state remains unchanged.
8. Check diff for unexpected changes. **Do not fix or delete anything automatically.** Report exact evidence, remaining gaps, and a clear **GO/NO-GO** for a single controlled live diagnostic. Even if GO, wait for explicit approval before any provider request or budget allocation.

## Future controlled live experiment — NOT AUTHORIZED YET

The previously proposed command shape (do **not** execute without explicit authorization) uses `--scenario yolo-id-swap`, `--model qwen-3.8-27b`, `--observation-model qwen-3.8-27b`, `--observation-verification-mode live`, `--tracking-reconsideration-mode live`, `--reasoning-effort none`, `--observation-reasoning-effort none`, tracking timeout 10s, Observation timeout 6s, output caps 768 each, and `--max-total-requests 6`. Exact CLI syntax, provider configuration, output paths, current budget allowance, and state safety must be freshly verified. One run only if separately approved; do not silently reset prior budget or assume budget configuration can be changed safely.

Success requires **actual Tracking LLM** to revise the initially swapped IDs, **actual Observation LLM** to independently support corrected associations, and deterministic validation to confirm evidence consistency **without giving either LLM the answer key**. If inconclusive or failed, report that honestly. Keep Phase 2 open until the user explicitly accepts the evidence.

## Engineering rules and approval boundaries

1. **Inspect before editing.** Historical skill contents and agent reports are not current Git truth.
2. **Explain before changes.** Identify issue, ownership boundary, exact files, risks and tests; obtain approval for material changes.
3. Preserve the canonical evidence schema, integrated MCP/dashboard pipeline, webcam/video-file support, and nonblocking camera/radar rendering.
4. Preserve independent LLM-backed Tracking, Observation, Reasoning and planned Operator responsibilities. Deterministic validation is supporting evidence, not agent reasoning.
5. Preserve measured versus estimated positions, timestamps, quality/confidence, uncertainty, provenance and inconclusive outcomes. Never fabricate measurements or ground-truth labels.
6. Do not weaken tests, evidence validation, lease rules, budgets or uncertainty gates to force success.
7. No live provider calls, model/provider changes, budget resets/increases, production integration, or runtime state mutation without explicit approval.
8. Do not remove files, logs, lockfiles, snapshots or unrelated modifications without approval. Do not commit, push, merge, reset, abort or clean Git unless explicitly authorized.
9. Clearly distinguish **Implemented / Mock-tested / Live-tested / Planned / Unknown**. Give exact commands, results and evidence; do not claim an entire phase complete without user sign-off.
10. At the start of the new chat, summarize the current checkpoint, perform only the approved read-only/no-network verification, and wait for further instructions.
