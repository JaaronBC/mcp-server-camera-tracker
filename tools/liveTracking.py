
import subprocess
import sys
from pathlib import Path
import threading

from tools.eventReader import read_tracking_events

from .registry import mcp

# Keep a reference to the running subprocess.
tracking_process = None


@mcp.tool()
def run_live_object_tracking() -> str:
    """Start live YOLO tracking in the background and return immediately."""
    global tracking_process

    script_path = (
        Path(__file__).parent
        / "components"
        / "liveTrackingDisplay.py"
    )

    try:
        if (
            tracking_process is not None
            and tracking_process.poll() is None
        ):
            return "Live object tracking is already running."

        if not script_path.exists():
            return f"Tracking script not found: {script_path}"

        # Start the webcam process without waiting for it to finish.
        tracking_process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        threading.Thread(
            target=read_tracking_events,
            args=(tracking_process,),
            daemon=True,
        ).start()

        return (
            "Live object tracking started in the background. "
            f"Process ID: {tracking_process.pid}. "
            "Press 'q' in the webcam window to stop tracking."
        )

    except Exception as error:
        return f"Could not start live tracking: {error}"
