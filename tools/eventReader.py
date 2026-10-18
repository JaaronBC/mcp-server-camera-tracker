import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from .registry import mcp

movement_event_queue = queue.Queue()

def read_tracking_events(process):
    if process.stdout is None:
        return

    for line in process.stdout:
        try:
            event = json.loads(line)

            if event.get("type") == "movement_detected":
                print("Received movement event:", event, flush=True)

                if event.get("type") == "movement_detected":
                    movement_event_queue.put(event)

                    print(
                        "Movement event queued:",
                        event,
                        flush=True,
                    )

        except json.JSONDecodeError:
            print(f"Non-JSON subprocess output: {line.rstrip()}", file=sys.stderr)

@mcp.tool()
def get_pending_movement_events() -> list[dict]:
    """Return movement events detected by the live tracking process."""

    events = []

    while True:
        try:
            events.append(movement_event_queue.get_nowait())
        except queue.Empty:
            break

    return events