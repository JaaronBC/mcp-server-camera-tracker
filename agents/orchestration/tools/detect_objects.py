# MCP tool that reads latest UDP tracking payloads and returns normalized detection frames.
import json
import socket
import time

from ..process import dashboard_process_status
from ...registry import mcp

FRAME_RECEIVE_TIMEOUT_SECONDS = 4


@mcp.tool()
def detect_objects(address: list[str | int]) -> str:
    # Implements detect objects behavior for this module.
    """Return the newest available tracking frame from a UDP address."""
    if (
        len(address) != 2
        or not isinstance(address[0], str)
        or not address[0]
        or not isinstance(address[1], int)
        or isinstance(address[1], bool)
        or not 1 <= address[1] <= 65535
    ):
        raise ValueError(
            "address must contain a non-empty host string and a UDP port "
            "integer between 1 and 65535"
        )

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((address[0], address[1]))
        sock.settimeout(FRAME_RECEIVE_TIMEOUT_SECONDS)
        try:
            data, _ = sock.recvfrom(65535)
        except TimeoutError:
            process_status = dashboard_process_status()
            running = bool(process_status.get("running", False))
            status_payload = {
                "time": time.time(),
                "detections": [],
                "status": "frame_timeout" if running else "dashboard_stopped",
                "error": "udp_frame_timeout" if running else "dashboard_not_running",
                "dashboard_running": running,
            }
            return json.dumps(status_payload)

        # Ignore queued older frames and return the most recent one available.
        sock.setblocking(False)
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break

    message = json.loads(data.decode("utf-8"))
    return json.dumps(message)
