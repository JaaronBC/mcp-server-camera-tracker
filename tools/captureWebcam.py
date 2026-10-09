import cv2
import time
from pathlib import Path

from .registry import mcp

@mcp.tool()
def capture_webcam_video(duration_seconds: int) -> str:
    """Record a video from the webcam for the requested number of seconds."""

    if duration_seconds < 1:
        raise ValueError("Duration must be at least 1 second.")

    if duration_seconds > 30:
        raise ValueError("Duration cannot exceed 30 seconds.")

    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"webcam_{int(time.time())}.mp4"

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        30.0,
        (width, height),
    )

    start_time = time.time()

    try:
        while time.time() - start_time < duration_seconds:
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            writer.write(frame)

    finally:
        camera.release()
        writer.release()

    return f"Video recorded successfully: {output_path}"

