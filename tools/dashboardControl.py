from __future__ import annotations

import time
from typing import Any

from .dashboardCalibration import (
    set_ground_plane_calibration,
    validate_ground_plane_calibration,
)
from .dashboardHealth import dashboard_health_status, wait_for_dashboard_ready
from .dashboard_process import (
    dashboard_process_status,
    start_dashboard_process,
    stop_dashboard_process,
)
from .registry import mcp


@mcp.tool()
def start_dashboard_demo() -> str:
    """Start the dashboard demo in the background and return immediately."""
    _, message = start_dashboard_process()
    return message


@mcp.tool()
def stop_dashboard_demo(force: bool = False) -> str:
    """Stop the running dashboard demo process."""
    return stop_dashboard_process(force=force)


@mcp.tool()
def dashboard_demo_status() -> dict:
    """Return current dashboard process status including PID and exit code."""
    return dashboard_process_status()


@mcp.tool()
def start_dashboard_and_wait_ready(
    timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    """Start dashboard demo and block until observations are fresh or timeout is reached."""
    started, message = start_dashboard_process()
    readiness = wait_for_dashboard_ready(
        timeout_seconds=timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    return {
        "started": started,
        "start_message": message,
        "readiness": readiness,
    }


@mcp.tool()
def stop_dashboard_and_confirm(
    force: bool = False,
    timeout_seconds: float = 5.0,
    poll_interval_seconds: float = 0.1,
) -> dict:
    """Stop dashboard demo and poll until process is confirmed stopped or timeout occurs."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero")

    stop_message = stop_dashboard_process(force=force)
    deadline = time.time() + timeout_seconds
    attempts = 0
    last_status = dashboard_process_status()

    while True:
        attempts += 1
        last_status = dashboard_process_status()
        if not last_status.get("running", False):
            return {
                "stopped": True,
                "stop_message": stop_message,
                "attempts": attempts,
                "status": last_status,
            }

        now = time.time()
        if now >= deadline:
            return {
                "stopped": False,
                "stop_message": stop_message,
                "attempts": attempts,
                "timeout_seconds": timeout_seconds,
                "status": last_status,
            }

        time.sleep(min(poll_interval_seconds, max(0.0, deadline - now)))


@mcp.tool()
def restart_dashboard_and_wait_ready(
    force_stop: bool = False,
    stop_timeout_seconds: float = 5.0,
    start_timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    """Restart dashboard process and wait until fresh observations are available or timeout."""
    stop_result = stop_dashboard_and_confirm(
        force=force_stop,
        timeout_seconds=stop_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    if not stop_result.get("stopped", False):
        return {
            "restarted": False,
            "stop": stop_result,
            "start": None,
            "readiness": None,
        }

    start_result = start_dashboard_and_wait_ready(
        timeout_seconds=start_timeout_seconds,
        max_observation_age_seconds=max_observation_age_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    readiness = start_result.get("readiness")
    ready = isinstance(readiness, dict) and readiness.get("ready", False)
    return {
        "restarted": bool(ready),
        "stop": stop_result,
        "start": {
            "started": start_result.get("started", False),
            "start_message": start_result.get("start_message"),
        },
        "readiness": readiness,
    }


@mcp.tool()
def dashboard_lifecycle_status(
    max_observation_age_seconds: float = 3.0,
) -> dict:
    """Return merged lifecycle status across process, health, and calibration checks."""
    if max_observation_age_seconds < 0:
        raise ValueError("max_observation_age_seconds must be non-negative")

    process = dashboard_process_status()
    health = dashboard_health_status()
    calibration = validate_ground_plane_calibration()

    running = bool(process.get("running", False))
    available = bool(health.get("available", False))
    age = health.get("last_observation_age_seconds")
    fresh = isinstance(age, (int, float)) and age <= max_observation_age_seconds

    ready = running and available and fresh

    return {
        "ready": ready,
        "running": running,
        "health_available": available,
        "observations_fresh": bool(fresh),
        "max_observation_age_seconds": max_observation_age_seconds,
        "process": process,
        "health": health,
        "calibration": calibration,
    }


@mcp.tool()
def dashboard_bootstrap(
    calibration: dict[str, Any] | None = None,
    restart_if_running: bool = False,
    force_stop: bool = False,
    stop_timeout_seconds: float = 5.0,
    start_timeout_seconds: float = 10.0,
    max_observation_age_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    """Optionally apply calibration, start/restart dashboard, wait readiness, and return lifecycle status."""
    calibration_result: dict[str, Any] | None = None
    if calibration is not None:
        calibration_result = set_ground_plane_calibration(calibration)

    if restart_if_running:
        orchestration = restart_dashboard_and_wait_ready(
            force_stop=force_stop,
            stop_timeout_seconds=stop_timeout_seconds,
            start_timeout_seconds=start_timeout_seconds,
            max_observation_age_seconds=max_observation_age_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        ready = bool(orchestration.get("restarted", False))
    else:
        orchestration = start_dashboard_and_wait_ready(
            timeout_seconds=start_timeout_seconds,
            max_observation_age_seconds=max_observation_age_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        readiness = orchestration.get("readiness")
        ready = isinstance(readiness, dict) and bool(readiness.get("ready", False))

    lifecycle = dashboard_lifecycle_status(
        max_observation_age_seconds=max_observation_age_seconds,
    )

    return {
        "bootstrapped": bool(ready),
        "restart_if_running": restart_if_running,
        "calibration_result": calibration_result,
        "orchestration": orchestration,
        "lifecycle": lifecycle,
    }
