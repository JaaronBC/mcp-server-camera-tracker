
import cv2
import math
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

CAPTURE_SECONDS = 3.0
MOVEMENT_THRESHOLD = 20

# Save captures in the project's root directory.
output_dir = Path(__file__).resolve().parents[2] / "captures"
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    model = YOLO("yolo11n.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        print("Error: Could not open webcam.", file=sys.stderr, flush=True)
        return

    previous_positions = {}
    tracking_data = {}
    frame_number = 0

    capture_start = None
    capture_until = None
    trigger_track_id = None
    trigger_distance = None
    

    # Keep status messages off stdout, which carries MCP events.
    print(
        "Live YOLO + ByteTrack started. Press 'q' to stop.",
        file=sys.stderr,
        flush=True,
    )

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print(
                    "Error: Could not read webcam frame.",
                    file=sys.stderr,
                    flush=True,
                )
                break

            results = model.track(
                source=frame,
                tracker="bytetrack.yaml",
                classes=[0],
                persist=True,
                verbose=False,
            )

            result = results[0]
            boxes = result.boxes

            # 1. Detect meaningful movement and update positions.
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()

                for bbox, track_id in zip(coordinates, track_ids):
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    current_position = (center_x, center_y)

                    previous_position = previous_positions.get(track_id)

                    if previous_position is not None and capture_until is None:
                        previous_x, previous_y = previous_position

                        distance = math.hypot(
                            center_x - previous_x,
                            center_y - previous_y,
                        )

                        if distance >= MOVEMENT_THRESHOLD:
                            capture_start = time.monotonic()
                            capture_until = capture_start + CAPTURE_SECONDS
                            tracking_data = {}

                            trigger_track_id = int(track_id)
                            trigger_distance = round(float(distance), 1)

                            # stderr: diagnostic messages only.
                            print(
                                f"Movement detected: ID {track_id}"
                            )

                    # Update every object's position on every frame.
                    previous_positions[track_id] = current_position

            # 2. Collect data independently of movement detection.
            # This must be outside the per-object loop and detection condition.
            if capture_until is not None:
                now = time.monotonic()

                if now < capture_until:
                    current_boxes = result.boxes

                    if current_boxes is not None and current_boxes.id is not None:
                        track_ids = current_boxes.id.int().cpu().tolist()
                        coordinates = current_boxes.xyxy.cpu().tolist()

                        for track_id, bbox in zip(track_ids, coordinates):
                            x1, y1, x2, y2 = bbox
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            track_id = str(track_id)

                            if track_id not in tracking_data:
                                tracking_data[track_id] = []

                            tracking_data[track_id].append({
                                "frame": frame_number,
                                "elapsed_seconds": round(
                                    now - capture_start, 4
                                ),
                                "center_x": round(center_x, 2),
                                "center_y": round(center_y, 2),
                            })

                else:
                    filename = (
                        f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
                    )
                    output_path = output_dir / filename

                    with output_path.open("w", encoding="utf-8") as file:
                        json.dump(tracking_data, file, indent=2)

                    event = {
                        "type": "movement_detected",
                        "track_id": trigger_track_id,
                        "distance_pixels": trigger_distance,
                        "file_path": str(output_path.resolve()),
                    }
                    # Send the event only after the capture file is complete.
                    print(json.dumps(event), flush=True)

                    print(
                        f"Tracking data saved to {output_path}",
                        file=sys.stderr,
                        flush=True,
                    )

                    capture_until = None
                    capture_start = None
                    tracking_data = {}
                    trigger_track_id = None
                    trigger_distance = None

            # 3. Display and advance the frame counter.
            annotated_frame = result.plot()
            cv2.imshow(
                "Counter-Drone System - Live Tracking",
                annotated_frame,
            )

            frame_number += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Live tracking stopped.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
