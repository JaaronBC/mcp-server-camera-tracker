# Dashboard Simplification Map

This is a maintainability-focused map for teammates.

## What changed

- `server.py` now imports a **single dashboard registration module**:
  - `tools/dashboard/tools_dashboard.py`
- Dashboard implementation is now **canonical under `tools/dashboard/`**.
- Root dashboard modules in `tools/` are now compatibility shims so existing imports still work.

## Why this helps

- New contributors no longer need to scan many `server.py` imports to understand dashboard wiring.
- There is now one obvious file to open first for dashboard tool registration.
- Existing tool implementations and tests remain stable.

## Read order for dashboard work

1. `tools/dashboard/tools_dashboard.py`
   - registration entrypoint (what dashboard tool modules are active)
2. `tools/dashboard/control.py`
   - process lifecycle orchestration tools
3. `tools/components/liveTrackingDisplay.py`
   - live tracking/radar runtime loop
4. `tools/dashboard/health.py`
   - readiness and contract checks
5. `tools/dashboard/metrics.py`
   - quality and latency metrics
6. `tools/dashboard/agent_ops.py`
   - periodic reasoning cycle and event health

## Current dashboard tool ownership

- `tools/dashboard/live_tracking_tool.py`: start live tracking process
- `tools/dashboard/detect_objects_tool.py`: pull latest frame payload over UDP
- `tools/dashboard/control.py`: start/stop/restart/status bootstrap tools
- `tools/dashboard/calibration.py`: calibration CRUD + validation
- `tools/dashboard/health.py`: health and contract checks
- `tools/dashboard/metrics.py`: depth/latency/quality metrics
- `tools/dashboard/agent_ops.py`: reasoning-cycle event pipeline

## Compatibility note

Legacy paths like `tools/dashboardHealth.py` remain as thin shims re-exporting canonical modules from `tools/dashboard/`.
