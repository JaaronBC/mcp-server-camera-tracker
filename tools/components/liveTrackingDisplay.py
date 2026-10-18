
import cv2
import math
import sys
import json
from ultralytics import YOLO


# Minimum center movement in pixels to trigger detection.
MOVEMENT_THRESHOLD = 20


def main():
    model = YOLO("yolo11n.pt")
    camera = cv2.VideoCapture(0)
    waitForDetection = True

    if not camera.isOpened():
        camera.release()
        print("Error: Could not open webcam.")
        return

    # Store the previous center position for each tracked object.
    previous_positions = {}

    print("Live YOLO + ByteTrack started. Press 'q' to stop.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
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

            if boxes is not None and boxes.id is not None and waitForDetection:
                # Get bounding boxes and their ByteTrack IDs.
                coordinates = boxes.xyxy.cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()

                for bbox, track_id in zip(coordinates, track_ids):
                    x1, y1, x2, y2 = bbox

                    # Calculate the center of the bounding box.
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    current_position = (center_x, center_y)

                    if track_id in previous_positions:
                        previous_x, previous_y = previous_positions[track_id]

                        # Calculate how far the center moved.
                        distance = math.sqrt(
                            (center_x - previous_x) ** 2
                            + (center_y - previous_y) ** 2
                        )

                        if distance >= MOVEMENT_THRESHOLD:
                            waitForDetection = False

                            event = {
                                "type": "movement_detected",
                                "track_id": int(track_id),
                                "distance_pixels": round(float(distance), 1),
                            }
                            #sends the event to the MCP server
                            print(json.dumps(event), flush=True)

                            print(
                                f"Movement detected: ID {track_id}, "
                                f"moved {distance:.1f} pixels",
                                file=sys.stderr,
                                flush=True,
                            )

                    # Save the current position for the next frame.
                    previous_positions[track_id] = current_position

            # Draw bounding boxes, class labels, and track IDs.
            annotated_frame = result.plot()

            cv2.imshow(
                "Counter-Drone System - Live Tracking",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Live tracking stopped.")


if __name__ == "__main__":
    main()
