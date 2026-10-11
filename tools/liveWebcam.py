import subprocess
import sys
from pathlib import Path

from tools.registry import mcp


@mcp.tool()
def run_live_webcam() -> str:
    """Launch a live webcam window. Press q in the window to stop."""

    script_path = Path(__file__).with_name("webcamDisplay.py")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return "Webcam closed successfully."

        return (
            f"Webcam process exited with code {result.returncode}.\n"
            f"Output: {result.stdout}\n"
            f"Errors: {result.stderr}"
        )

    except Exception as error:
        return f"Could not launch webcam: {error}"