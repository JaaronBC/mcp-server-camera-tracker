from mcp.server import MCPServer
import cv2
import time
import json
from pathlib import Path

mcp = MCPServer("Dog Behavior Assistant")


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


@mcp.tool()
def detect_and_track_dogs(video_path: str) -> str:
    """Detect and track dogs in a recorded video."""

    if not video_path.strip():
        raise ValueError("video_path cannot be empty.")

    video_file = Path(video_path)

    if not video_file.exists():
        raise ValueError(f"Video file not found: {video_path}")

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model.track(
        source=str(video_file),
        tracker="bytetrack.yaml",
        classes=[16],
        persist=True,
        stream=True,
        verbose=False,
    )

    tracking_data = {}

    for frame_number, result in enumerate(results):
        boxes = result.boxes

        if boxes is None or boxes.id is None:
            continue

        track_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()

        for track_id, box in zip(track_ids, coordinates):
            x1, y1, x2, y2 = box

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            track_id = str(track_id)

            if track_id not in tracking_data:
                tracking_data[track_id] = []

            tracking_data[track_id].append({
                "frame": frame_number,
                "center_x": center_x,
                "center_y": center_y,
            })

    output_path = video_file.with_suffix(".json")

    with open(output_path, "w") as file:
        json.dump(tracking_data, file, indent=2)

    return f"Tracking complete. Results saved to: {output_path}"

@mcp.tool()
def analyze_dog_movement(tracking_data: str) -> str:
    """Analyze tracked dog positions and determine each dog's movement direction."""

    if not tracking_data.strip():
        raise ValueError("tracking_data cannot be empty.")

    tracking_file = Path(tracking_data)

    if not tracking_file.exists():
        raise ValueError(f"Tracking file not found: {tracking_data}")

    with open(tracking_file, "r") as file:
        data = json.load(file)

    if not data:
        return "No dogs were detected in the tracking data."

    results = {}

    # Movement threshold in pixels
    movement_threshold = 30

    for dog_id, positions in data.items():

        if len(positions) < 2:
            results[dog_id] = {
                "movement": "insufficient data",
                "displacement_x": 0,
            }
            continue

        start_x = positions[0]["center_x"]
        end_x = positions[-1]["center_x"]

        displacement_x = end_x - start_x

        if displacement_x > movement_threshold:
            movement = "right"

        elif displacement_x < -movement_threshold:
            movement = "left"

        else:
            movement = "stationary"

        results[dog_id] = {
            "movement": movement,
            "displacement_x": round(displacement_x, 2),
            "frames_tracked": len(positions),
        }

    return json.dumps(results, indent=2)

if __name__ == "__main__":
    mcp.run()