# listener.py
# Receives detection data from tracker.py and prints it.
#
# Run this in a second terminal:  python listener.py
# (Press Ctrl+C to stop)

import json
import socket

ADDRESS = ("127.0.0.1", 5005)

# Create a UDP socket and listen on the address
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(ADDRESS)

print("Waiting for data from tracker.py...")

while True:
    data, _ = sock.recvfrom(65535)  # wait for one message
    message = json.loads(data.decode())

    detections = message["detections"]
    print(f"Frame at {message['time']:.2f}: {len(detections)} object(s)")

    for d in detections:
        print(f"  ID {d['id']}: {d['label']} ({d['confidence']}) at x={d['x']}, y={d['y']}")