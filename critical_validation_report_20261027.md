# Critical Validation Report — 2026-10-27

## Scope

Continuation of critical-system closure with evidence-based validation of the connected dashboard/agent pipeline.

## Evidence Collected

### Runtime artifacts inspected

- `captures/latest_radar_state.json`
- `captures/latest_tracking_frame.json`
- `captures/agent_events.jsonl` (tail)
- `captures/radar_depth_coverage_history.jsonl` (tail)

### Contract check executed

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -c "from tools.dashboardHealth import verify_dashboard_pipeline_contract; import json; print(json.dumps(verify_dashboard_pipeline_contract(), indent=2))"
```

Observed findings before fix:

- `stale_observation` (expected for old snapshot age)
- `coordinate_unit_mismatch` on `tracks[0].ground_coordinate_unit`

## Root Cause Found and Fixed

### Issue

`verify_dashboard_pipeline_contract` requires per-track `ground_coordinate_unit` consistency with `position_kind`, but `RadarTrackSnapshot.to_dict()` did not include `ground_coordinate_unit`.

### Fix applied

1. **Schema serialization update**
   - File: `tools/components/dashboard_types.py`
   - Change: add `ground_coordinate_unit` field to `RadarTrackSnapshot` and include it in `to_dict()`.

2. **Producer updates**
   - File: `tools/components/liveTrackingDisplay.py`
   - Change: populate `ground_coordinate_unit` when creating:
     - live `RadarTrackSnapshot` entries
     - simulated fallback `RadarTrackSnapshot` entries (`relative_units`)

3. **Regression updates**
   - File: `tests/test_live_tracking_display.py`
   - Change: update `RadarTrackSnapshot` fixtures and assert serialized `ground_coordinate_unit` presence.

## Verification Results

### Focused tests

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -m unittest tests/test_live_tracking_display.py tests/test_dashboard_health.py
```

- Result: `Ran 34 tests ... OK`

### Full regression

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

- Result: `Ran 164 tests ... OK (skipped=1)`

## `design.md` Critical Checklist Mapping (Current)

- **Live perception path to shared state:** `PASS` (implemented and traced; artifacts present).
- **Radar + camera consistency via same track state:** `PASS` (same track payload path, shared snapshot).
- **Coordinate/unit integrity:** `PASS` for code-path after fix; `PARTIAL` for currently stale artifact file until refreshed runtime snapshot is produced.
- **Agent evidence traceability and disagreement visibility:** `PASS` (event payload fields and handoff/error signals present in code/tests; historical events observed).
- **Non-blocking startup/runtime health visibility:** `PASS` (startup preflight and health summaries implemented and tested).
- **Fresh live-state contract readiness now:** `PARTIAL` (current snapshot age is stale; requires active runtime to confirm fresh `ready=true`).

## Remaining Manual/Runtime Validation (Still Required)

1. Run live runtime and produce a fresh `latest_radar_state.json`.
2. Re-run `verify_dashboard_pipeline_contract` while live updates are active and confirm:
   - no `coordinate_unit_mismatch`
   - no `stale_observation`
   - `ready=true`
3. Validate UI behavior during temporary camera/model interruption and recovery.
4. Confirm simulation fallback labeling remains explicit during outage windows.

## Live Re-Verification (Completed)

After starting the runtime and re-running contract verification with `require_live_pipeline=True`, the latest result is:

- `available=true`
- `ready=true`
- `error_count=0`
- `warn_count=0`
- `findings=[]`
- `camera_status=connected`
- `detector_status=running`
- `last_observation_age_seconds=0.8067`

This confirms the previous `coordinate_unit_mismatch` defect is resolved in live output and that freshness/readiness checks pass under active runtime conditions.

## Outage/Recovery Validation Loop (Completed)

### 1) Outage/stale validation (runtime inactive)

Command:

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -c "from tools.dashboardHealth import verify_dashboard_pipeline_contract; import json; print(json.dumps(verify_dashboard_pipeline_contract(require_live_pipeline=True), indent=2))"
```

Observed result:

- `ready=false`
- `error_count=1`
- finding code: `stale_observation`
- observed age: `last_observation_age_seconds=107.550...`

### 2) Recovery validation (controlled restart)

Command sequence:

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -u main.py > /tmp/mcp_runtime_recovery.log 2>&1 & pid=$!
sleep 6
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -c "from tools.dashboardHealth import verify_dashboard_pipeline_contract; import json; print(json.dumps(verify_dashboard_pipeline_contract(require_live_pipeline=True), indent=2))"
kill -INT "$pid" >/dev/null 2>&1
wait "$pid" >/dev/null 2>&1 || true
```

Observed result:

- `ready=true`
- `error_count=0`
- `warn_count=0`
- `findings=[]`
- `last_observation_age_seconds=0.0501...`

This validates expected degradation under stale/no-live updates and successful recovery to healthy connected-pipeline readiness after runtime restart.

## Recommended Next Command Sequence

```bash
python -u main.py
```

In another shell while runtime is active:

```bash
/Users/jaaroncabsaban/Documents/GitHub/mcp-server-camera-tracker/.venv/bin/python -c "from tools.dashboardHealth import verify_dashboard_pipeline_contract; import json; print(json.dumps(verify_dashboard_pipeline_contract(require_live_pipeline=True), indent=2))"
```
