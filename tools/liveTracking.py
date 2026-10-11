
import subprocess
import sys
from pathlib import Path

from .registry import mcp


@mcp.tool()
def run_live_object_tracking() -> str:
    """Open a live webcam window with YOLO detection and ByteTrack IDs. Press q to stop."""

    script_path = (
        Path(__file__).parent
        / "components"
        / "liveTrackingDisplay.py"
    )

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return "Live object tracking stopped."

        return (
            f"Live tracking exited with code {result.returncode}.\n"
            f"Output: {result.stdout}\n"
            f"Errors: {result.stderr}"
        )

    except Exception as error:
        return f"Could not start live tracking: {error}"
