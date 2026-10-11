
import cv2


def main():
    # Open the default webcam (usually camera 0)
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press 'q' to quit.")

    try:
        while True:
            # Read the next frame from the webcam
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            # Display the live footage
            cv2.imshow("Counter-Drone System - Live Webcam", frame)

            # Press q to close the window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Release the webcam and close all OpenCV windows
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
