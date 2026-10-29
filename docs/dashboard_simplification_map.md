# Dashboard Simplification Map

This is a maintainability-focused map for teammates.

## What changed

- `server.py` now imports a **single dashboard registration module**:
  - `agents/dashboard/tools_dashboard.py`
- Dashboard implementation is now **canonical under `agents/dashboard/`**.
- Root dashboard modules in `agents/` are now compatibility shims so existing imports still work.

## Why this helps

- New contributors no longer need to scan many `server.py` imports to understand dashboard wiring.
- There is now one obvious file to open first for dashboard tool registration.
- Existing tool implementations and tests remain stable.

## Read order for dashboard work

1. `agents/dashboard/tools_dashboard.py`
   - registration entrypoint (what dashboard tool modules are active)
2. `agents/dashboard/control.py`
   - process lifecycle orchestration tools
3. `agents/components/liveTrackingDisplay.py`
   - live tracking/radar runtime loop
4. `agents/dashboard/health.py`
   - readiness and contract checks
5. `agents/dashboard/metrics.py`
   - quality and latency metrics
6. `agents/dashboard/agent_ops.py`
   - periodic reasoning cycle and event health

## Current dashboard tool ownership

- `agents/dashboard/live_tracking_tool.py`: start live tracking process
- `agents/dashboard/detect_objects_tool.py`: pull latest frame payload over UDP
- `agents/dashboard/control.py`: start/stop/restart/status bootstrap tools
- `agents/dashboard/calibration.py`: calibration CRUD + validation
- `agents/dashboard/health.py`: health and contract checks
- `agents/dashboard/metrics.py`: depth/latency/quality metrics
- `agents/dashboard/agent_ops.py`: reasoning-cycle event pipeline

## Compatibility note

Legacy paths like `agents/dashboardHealth.py` remain as thin shims re-exporting canonical modules from `agents/dashboard/`.
