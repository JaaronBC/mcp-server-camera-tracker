import json
from pathlib import Path

from .registry import mcp

@mcp.tool()
def analyze_object_movement(tracking_data: str) -> str:
    """Analyze tracked object positions and determine each object's movement direction."""

    print("\n========== ANALYZE MOVEMENT TOOL CALLED ==========", flush=True)
    print(f"Tracking data received: {tracking_data}", flush=True)
    print("==================================================\n", flush=True)

    return "TEST SUCCESS: analyze_object_movement was called."
