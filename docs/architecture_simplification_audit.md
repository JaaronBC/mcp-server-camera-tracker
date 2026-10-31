# Architecture Simplification Audit (Skill-Guided)

Date: 2026-10-31  
Scope: `main.py`, `server.py`, `agents/*`, `tests/*`, and current docs.

This audit applies:
- `human-maintainable-code` skill (readability, overloading, duplication, naming, structure, docs/tooling)
- `counter-uas-mcp-architecture-review-and-repair` skill (runtime ownership, MCP boundaries, contracts, lifecycle, collaboration)

## 1) What I verified (source-backed)

### Runtime/process map
- `main.py` is the runtime entrypoint (`cli_main` → `main` → `run_system_with_config`).
- `main.py` launches MCP server via stdio using `server.py` and `ClientSession`.
- `server.py` registers tools by importing:
  - `agents.orchestration.tools`
  - `agents.reasoning.tools`
  - `agents.tracking.tools`
- Dashboard process is started from orchestration control via `agents/orchestration/process.py::start_dashboard_process` using module launch (`python -m agents.dashboard.live_tracking_display`).

### Data path (live)
- Camera + YOLO/ByteTrack run in `agents/dashboard/tracking_loop.py::run_live_tracking_loop`.
- Tracking frame + radar state are written under `captures/` via `agents/tracking/state_publisher.py`.
- Reasoning cycle consumes latest radar state via `agents/reasoning/tools/_shared.py` and logs to `captures/agent_events.jsonl` through `agents/orchestration/tools/ops.py::run_reasoning_cycle_and_log`.
- Renderer consumes shared state and latest agent events in `agents/dashboard/rendering.py::render_dashboard`.

### Baseline behavior check
- Ran focused tests successfully:
  - `python -m unittest tests.test_main_cli tests.test_orchestration_process tests.test_live_tracking_display`
  - Result: `Ran 35 tests ... OK`

## 2) What is actually wrong (evidence-based)

## Critical/High

1. **Dead/duplicate architecture modules create split mental models**
- `agents/orchestration/metrics.py` and `agents/orchestration/tools/metrics.py` both define MCP-decorated metrics tools.
- `agents/reasoning/reasoning_assessment_runtime.py` is a near-duplicate of `agents/reasoning/tools/_shared.py` patterns but appears unreferenced.
- No workspace references to `reasoning_assessment_runtime` were found.
- Impact: onboarding confusion, accidental import/registration risk, duplicate maintenance burden.

2. **MCP boundary is over-expanded and alias-heavy**
- There are 48 MCP tools total in current code.
- `agents/orchestration/tools/control.py` and related modules expose many `dashboard_*` and mirrored `orchestration_*` aliases with same behavior.
- Impact: large surface area, unclear canonical API, higher contract drift risk.

## Medium

3. **Runtime wiring uses too many wrapper layers for simple flows**
- `main.py` → `runtime/core.py` callback factories → `runtime/callbacks.py` wrappers → `tools/control.py` runtime wrappers → `agent_orchestration.py` call helpers.
- Impact: difficult traceability, high indirection, slower debugging, higher cognitive load.

4. **Loop ownership is technically singular but conceptually fragmented**
- Effective owner is `runtime_flow.monitor_movement` loop.
- But ownership path spans `main.py`, `runtime/core.py`, `runtime/callbacks.py`, and `runtime/flow.py` with many pass-through callables.
- Impact: difficult to reason about scheduling/heartbeats and where to patch safely.

5. **Lifecycle cleanup can short-circuit on first exception**
- `main.py::run_system_with_config` calls `await operator.stop()` then `await services.stop()` inside one `finally`.
- If `operator.stop()` raises, `services.stop()` will not execute.
- In `RuntimeServices.stop()`, if session close fails first, stdio close may be skipped.
- Impact: leaked resources during partial-failure shutdown paths.

6. **Naming collisions blur architectural intent**
- Two different `AgentCoordinator` classes exist:
  - `agents/coordinator.py::AgentCoordinator` (domain validation/history)
  - `agents/orchestration/runtime/app.py::AgentCoordinator` (runtime wrapper)
- Impact: high confusion in code review/navigation and unsafe edits.

## Low/Moderate

7. **Large overloaded functions/files reduce readability**
- Examples:
  - `agents/orchestration/tools/ops.py::run_reasoning_cycle_and_log` (~237 lines)
  - `agents/dashboard/rendering.py::render_dashboard` (~201 lines)
  - `agents/dashboard/tracking_loop.py::run_live_tracking_loop` (~141 lines)
- Impact: hard to test in isolation, harder behavior changes, dense branching.

8. **Documentation drift in `README.md`**
- README still references stale paths and contradictory claims (for example says dashboard package removed while `agents/dashboard/*` is active; references non-existent paths like `agents/visual_tracking_agent.py` and `agents/reasoning/agents/_shared.py`).
- Impact: misleading onboarding and operational friction.

9. **Tooling consistency gaps**
- No clear formatter/linter config found in repo root.
- Impact: style drift and review overhead.

## 3) What should remain as-is

- Keep perception local and deterministic in dashboard/tracking loops (do not route every frame through MCP/LLM).
- Keep evidence-aware coordinator validation behavior in `agents/coordinator.py`.
- Keep explicit simulation fallback semantics (`simulation_fallback`, `simulated_data`) in `state_publisher.py`.
- Keep current module-based dashboard process launch fix in `agents/orchestration/process.py`.

## 4) Risk mapping to architecture-review skill (A–F)

- **A Runtime-loop ownership:** single effective loop owner exists, but indirection makes ownership non-obvious.
- **B MCP boundary/tool design:** bounded capabilities exist, but surface is too broad with many aliases.
- **C Perception/dashboard coupling:** mostly acceptable, but render loop remains large and handles event-log interpretation inline.
- **D Real collaboration:** structured handoffs/events exist; major risk is maintainability and readability, not absence of handoff.
- **E Canonical contracts:** contracts are present and typed, but duplicated module paths weaken confidence and discoverability.
- **F Lifecycle/shutdown:** cleanup robustness gap under stop failures.

## 5) Simplification priorities (recommended order)

1. **Canonicalize module ownership**
   - Decide one home for metrics and reasoning shared runtime modules.
   - Remove/retire dead duplicates after reference audit.

2. **Collapse pass-through wrapper layers**
   - Remove purely delegating wrappers where they add no policy/validation.
   - Keep one clear runtime wiring layer.

3. **Trim MCP alias surface**
   - Keep canonical tool names.
   - Keep compatibility aliases only where externally required, with deprecation note.

4. **Harden shutdown lifecycle**
   - Ensure each cleanup step runs even when another fails.
   - Aggregate/report cleanup errors without skipping releases.

5. **Split overloaded functions**
   - Extract event construction/reporting from `run_reasoning_cycle_and_log`.
   - Extract dashboard header/panel composition from `render_dashboard`.

6. **Fix docs + add tooling config**
   - Bring README paths and run commands in sync with current code.
   - Add formatter/linter config and a one-command check.

## 6) Proposed first change (smallest high-value)

**Change:** Remove dead duplicate modules and normalize imports to canonical modules only.

Candidate scope:
- `agents/orchestration/metrics.py` (duplicate/non-canonical)
- `agents/reasoning/reasoning_assessment_runtime.py` (duplicate/non-canonical)
- update any lingering references/docs to canonical paths.

Acceptance criteria:
- No behavior change in runtime startup and dashboard.
- Existing targeted tests still pass.
- Tool registrations remain identical from active canonical packages.

Regression risk:
- Low-to-medium if hidden external imports exist. Mitigate by repo-wide reference search and staged delete behind user approval.

## 7) Human-maintainability scorecard (quick)

- Reader-oriented naming: **mixed** (good domain names + several generic/wrapper names)
- File/function size discipline: **needs work**
- Dead code / duplicates: **needs work**
- One obvious place to edit behavior: **partially**
- Docs accuracy: **needs work**
- Tooling consistency: **needs work**
