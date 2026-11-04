# MCP tool that launches and supervises the live dashboard tracking process.
from ..process import start_dashboard_process
from ...registry import mcp


@mcp.tool()
def live_tracking(video_path: str | None = None) -> str:
    # Implements live tracking behavior for this module.
    """Start live YOLO tracking from a webcam or an optional video file."""
    _, message = start_dashboard_process(video_path=video_path)
    return message

if __name__ == "__main__":
    result = live_tracking()
    print(result)
