
import cv2
from ultralytics import YOLO


def main():
    # Load the model once, before starting the camera loop.
    model = YOLO("yolo11n.pt")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        print("Error: Could not open webcam.")
        return

    print("Live YOLO + ByteTrack started. Press 'q' to stop.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            # Detect objects and track them across consecutive frames.
            results = model.track(
                source=frame,
                tracker="bytetrack.yaml",
                classes=[0], 
                persist=True,
                verbose=False,
            )

            # Draw bounding boxes, class labels, and track IDs.
            annotated_frame = results[0].plot()

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
