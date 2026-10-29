# Handles dashboard process behavior.
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_SCRIPT_PATH = PROJECT_ROOT / "tools" / "components" / "liveTrackingDisplay.py"

_dashboard_process: subprocess.Popen[Any] | None = None


def _is_running() -> bool:
    # Performs is running logic.
    return _dashboard_process is not None and _dashboard_process.poll() is None


def start_dashboard_process() -> tuple[bool, str]:
    # Performs start dashboard process logic.
    global _dashboard_process

    if _is_running():
        return (
            False,
            "Dashboard demo is already running. "
            f"Process ID: {_dashboard_process.pid}.",
        )

    if not DASHBOARD_SCRIPT_PATH.is_file():
        raise FileNotFoundError(f"Dashboard script not found: {DASHBOARD_SCRIPT_PATH}")

    _dashboard_process = subprocess.Popen(
        [sys.executable, str(DASHBOARD_SCRIPT_PATH)],
        cwd=str(DASHBOARD_SCRIPT_PATH.parent),
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
    )

    return (
        True,
        "Dashboard demo started in the background. "
        f"Process ID: {_dashboard_process.pid}. "
        "Press 'q' in the dashboard window to stop.",
    )


def stop_dashboard_process(force: bool = False) -> str:
    # Performs stop dashboard process logic.
    global _dashboard_process

    if not _is_running():
        _dashboard_process = None
        return "Dashboard demo is not running."

    assert _dashboard_process is not None
    process = _dashboard_process

    if force:
        process.kill()
    else:
        process.terminate()

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)

    _dashboard_process = None
    return "Dashboard demo stopped."


def dashboard_process_status() -> dict[str, Any]:
    # Performs dashboard process status logic.
    if _dashboard_process is None:
        return {
            "running": False,
            "pid": None,
            "return_code": None,
            "script_path": str(DASHBOARD_SCRIPT_PATH),
        }

    return {
        "running": _dashboard_process.poll() is None,
        "pid": _dashboard_process.pid,
        "return_code": _dashboard_process.poll(),
        "script_path": str(DASHBOARD_SCRIPT_PATH),
    }
