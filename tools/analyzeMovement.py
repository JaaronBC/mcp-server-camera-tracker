import json
import socket
from .registry import mcp

@mcp.tool()
def analyze_movement_2(address: list[str | int]) -> str:
    """Receive one tracking frame from the UDP address [host, port]."""
    if (
        len(address) != 2
        or not isinstance(address[0], str)
        or not isinstance(address[1], int)
        or isinstance(address[1], bool)
    ):
        raise ValueError("address must contain a host string and UDP port integer")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((address[0], address[1]))
        data, _ = sock.recvfrom(65535)

    message = json.loads(data.decode("utf-8"))
    return json.dumps(message)
