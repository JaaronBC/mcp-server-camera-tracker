import json
from pathlib import Path

from .registry import mcp

@mcp.tool()
def detect_and_track_objects(video_path: str) -> str:
    """Detect and track object in a recorded video."""

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
        classes=[0], #0 for person, 39 for watter bottle, 16 for dog
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
