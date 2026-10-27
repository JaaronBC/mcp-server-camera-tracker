# MCP tool that launches and supervises the live dashboard tracking process.
from .dashboard_process import start_dashboard_process
from .registry import mcp


@mcp.tool()
def live_tracking() -> str:
    # Implements live tracking behavior for this module.
    """Start live YOLO tracking in the background and return immediately."""
    _, message = start_dashboard_process()
    return message

if __name__ == "__main__":
    result = live_tracking()
    print(result)
