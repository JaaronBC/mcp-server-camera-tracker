# Observation Autonomy Provider Harness Runbook

Harness: `tools/observation_autonomy_provider_harness.py`

## Purpose

Run one bounded Observation investigation with:

- real Observation decision flow (`call_observation_agent_with_retry`) with explicit decision contract support,
- real scheduler + IPC (`runtime_flow.monitor_movement` + `TrackingDecisionIpcTransport`),
- mocked MCP evidence source (`_MockEvidenceSession`),
- strict call/time budgets and persisted artifacts.

The harness recognizes optional explicit Observation decisions under `decision`:

- `assess_now`
- `predict_bounded`
- `request_additional_tracking_evidence`
- `inconclusive`

If explicit decision is absent, harness uses legacy deterministic inference and records provenance.

## Commands

Preflight (default, no provider-backed calls):

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python tools/observation_autonomy_provider_harness.py
```

Preflight with explicit fixture/run dir:

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python tools/observation_autonomy_provider_harness.py \
  --fixture tests/fixtures/observation_autonomy/single_investigation_evidence_gap.json \
  --run-dir captures/experiments/observation_autonomy_provider_harness/manual_preflight_run
```

Live provider mode (explicit opt-in only):

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python tools/observation_autonomy_provider_harness.py --live
```

## Default fixture

- `tests/fixtures/observation_autonomy/single_investigation_evidence_gap.json`

## Default budgets and limits

From `HarnessBudgets` defaults:

- provider total attempts: `2`
- provider observation_text attempts: `2`
- provider tracking_vlm attempts: `0`
- provider reasoning_text attempts: `0`
- provider observation_visual attempts: `0`
- max Observation LLM calls per run: `2`
- max MCP evidence requests per run: `1`
- wall clock budget: `45s`
- scheduler analysis interval: `0.01s`
- observation tool timeout env: `OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS = wall_clock/2` (bounded to `5..300s`, default `22.5s`)
- harness-only Observation output cap override: `OBSERVATION_AGENT_MAX_OUTPUT_TOKENS=1024`

## Environment variables set by harness

- `PROVIDER_CALL_BUDGET_ENABLED=1`
- `OBSERVATION_AGENT_MAX_OUTPUT_TOKENS=1024` (harness-only experiment override; production default is unchanged)
- `PROVIDER_CALL_BUDGET_TOTAL_LIMIT`
- `PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT`
- `PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT`
- `PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT`
- `PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT`
- `PROVIDER_CALL_BUDGET_STATE_PATH`
- `PROVIDER_CALL_BUDGET_LOCK_PATH`
- `PROVIDER_CALL_BUDGET_LOG_PATH`
- `PHASE2D_TRACKING_DECISION_IPC_ENABLED=1`
- `OBSERVATION_TOOL_REQUESTS_ENABLED=1`
- `OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS`
- `PHASE2D_TRACKING_DECISION_TOOL_REQUESTS_PATH`
- `PHASE2D_TRACKING_DECISION_TOOL_RESULTS_PATH`
- `PHASE2D_TRACKING_DECISION_REQUEST_INDEX_PATH`
- `PHASE2D_TRACKING_DECISION_RESULT_INDEX_PATH`

## Artifact locations

Default run root:

- `captures/experiments/observation_autonomy_provider_harness/run_<timestamp>_<id>/`

Per-run artifacts:

- `observation_autonomy_harness_report.json`
- `tracking_decision_tool_requests.jsonl`
- `tracking_decision_tool_results.jsonl`
- `latest_tracking_decision_request_index.json`
- `latest_tracking_decision_result_index.json`
- `provider_call_budget_state.json`
- `provider_call_budget_state.lock`
- `provider_call_budget_events.jsonl`

## Expected report fields

Top-level:

- `status`: `ok` or `error`
- `outcome`: `failed_timeout` or `failed_runtime_error` (on top-level errors)
- `mode`: `preflight` or `live_provider`
- `live_mode_enabled`: `false` (default) or `true` (`--live`)
- `fixture_path`, `run_dir`, `budgets`, `safety`

Investigation payload (`investigation`):

- identifiers: `correlation_id`, `investigation_id`, `request_id`
- decisions: `first_pass_decision_source`, `first_decision`, `final_decision`, `final_summary`
- evidence: `request_submitted`, `request_submission`, `correlated_result`, `evidence_references`
- counters: `observation_llm_call_count`, `mcp_evidence_call_count`, `provider_call_attempted_total`, `provider_call_attempted_by_role`
- autonomy flags: `explicit_autonomy_demonstrated`, `tool_request_autonomy_demonstrated`, `successful_explicit_observation_autonomy`
- classification: `classification.explicit_tool_request_validated`, `classification.explicit_tool_request_dispatched`, `classification.explicit_tool_request_followed_by_terminal_second_pass`, `classification.explicit_non_request_decision`, `classification.predict_bounded_downgraded`
- run status: `evidence_retrieval_status`, `outcome`
- diagnostics: `scheduler`, `llm_inputs`, `llm_decisions`, `validation`, `timing`

Decision provenance values:

- `llm_explicit`
- `llm_summary_inference`
- `invalid_explicit_decision`

Outcome semantics:

- `successful_evidence_retrieval`: evidence fetched and investigation completed
- `justified_inconclusive`: final action is bounded/inconclusive and marked allowed by validation
- `failed_timeout`: evidence/tool timeout or timeout-class failure
- `failed_evidence_retrieval`: request made but evidence missing/failed
- `no_evidence_request`: no evidence request required for final decision

Explicit autonomy success rule:

- `successful_explicit_observation_autonomy=true` requires `first_pass_decision_source=llm_explicit`.
- If first pass uses legacy inference (`llm_summary_inference`), outcome can still be valid but `explicit_autonomy_demonstrated=false`.
- If first pass explicit action is non-request (`assess_now`/`predict_bounded`/`inconclusive`), explicit decision is valid but `tool_request_autonomy_demonstrated=false`.
- `predict_bounded` is downgraded to `inconclusive` when deterministic support is insufficient.

Token/latency implication:

- Increasing max Observation output tokens from default to `1024` can increase per-attempt response length and latency.
- Attempt count and budget ceilings remain unchanged (`<=2` provider attempts total; `<=1` MCP request).

## Safety boundaries

- Evidence calls are restricted to mocked `get_track_evidence_history` responses in harness session.
- IPC files are isolated to the harness run directory.
- No camera stream is opened by harness logic.
- Harness does not mutate production tracking identity state; it only emits local run artifacts.
- Live provider mode requires explicit `--live`; default command remains preflight.

## Verification checklist mapping

1. `--live` provider path: in live mode, harness calls `call_observation_agent_with_retry(payload)` without preflight `invoke_fn`.
2. Budget retry counting: each attempted provider call reserves budget in `invoke_llm_json_response`; retries invoke that path per attempt.
3. First-pass decision source: report field `first_pass_decision_source` records explicit/inferred/invalid provenance.
4. Distinct report outcomes: `evidence_retrieval_status` + `investigation.outcome` + top-level `status/outcome`.
5. Production safety: mocked evidence session, no camera I/O, run-local artifacts.
6. Explicit opt-in: `--live` required; default command is preflight.

## Ready-for-live checklist

Before one approved provider-backed run:

- Keep default budgets (`total=2`, `observation_text=2`, MCP requests `<=1`, wall clock `<=45s`).
- Confirm run-local artifact paths are writable.
- Execute live only via explicit command:

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python tools/observation_autonomy_provider_harness.py --live
```
