# tracker.py
# Runs YOLO on your webcam and sends the detections to listener.py.
#
# Install:  pip install ultralytics opencv-python
# Run:      python tracker.py   (press 'q' in the video window to quit)

import json
import socket
import time

import cv2
from ultralytics import YOLO

def main(address: list):

    # Where to send the data (listener.py listens on this address)
    ADDRESS = (address[0], address[1])

    # Load a small, fast YOLO model (downloads automatically the first time)
    model = YOLO("yolov8n.pt")

    # Create a UDP socket for sending data
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Open the default webcam (0 = first camera)
    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read from webcam")
            break

        # Run YOLO with tracking so each object keeps the same ID between frames
        result = model.track(frame, persist=True, verbose=False, classes=[0])[0]

        # Turn each detected box into a simple dictionary
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "id": int(box.id[0]) if box.id is not None else None,
                "label": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
                "x": round((x1 + x2) / 2),  # center of the box
                "y": round((y1 + y2) / 2),
            })

        # Send this frame's data as a JSON message
        message = {"time": time.time(), "detections": detections}
        sock.sendto(json.dumps(message).encode(), ADDRESS)

        # Show the webcam feed with boxes drawn on it
        cv2.imshow("YOLO Tracker", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(["127.0.0.1", 5005])