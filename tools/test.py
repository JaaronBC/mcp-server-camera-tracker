from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import json 
from ultralytics import YOLO
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline

"""
MODELS = {
    "outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "indoor": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
}
"""

"""
model_id = MODELS["indoor"]
image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForDepthEstimation.from_pretrained(model_id)
"""

# Load a pretrained YOLO26 depth model
model = YOLO("yolo11n.pt")
depth_model = YOLO("yolo26s-depth.pt")


def depth_mapping_webcam() -> str:
    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Initialize the webcam (0 is usually the built-in or default camera)
    cam = cv2.VideoCapture(0)
    # 2. Read a single frame
    ret, frame = cam.read()

    output_path = output_dir / f"webcam_{int(time.time())}.png"
    
    if ret:
        # 3. Save the captured image to a file
        cv2.imwrite(str(output_path), frame)
        sys.stderr.write("Image captured and saved successfully!\n")
    else:
        sys.stderr.write("Error: Could not access the webcam.")

    # 4. Always release the camera resources when finished
    cam.release()
    

    if not ret:
        return "Error: Could not access the webcam."

    # load input image
    image = Image.open(output_path)

    # load input image
    image_file = output_path
    if not image_file.exists():
        raise ValueError(f"Image file not found: {image_file}")

    # Load a pretrained YOLO26 depth model
    model = YOLO("yolo11n.pt")
    depth_model = YOLO("yolo26s-depth.pt")

    #depth mapping
    depth_results = depth_model(str(output_path), save=True, save_txt=True, save_conf=True, conf=0.25, device="cuda" if torch.cuda.is_available() else "cpu")
    depth_map = depth_results[0].depth.data
    output_path = image_file.with_suffix(".json")
    with open(output_path, "w") as file:
        json.dump(depth_map.tolist(), file)

    # Detect and track objects in the image
    results = model.track(
        source=str(image_file),
        tracker="bytetrack.yaml",
        classes=[0], #0 for person, 39 for watter bottle, 16 for dog
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in results:
        for box in result.boxes:
            if box.id is not None:
                track_id = str(box.id.item())
                x1, y1, x2, y2 = box.xyxy.cpu().tolist()[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                depth_at_center = depth_map[int(center_y), int(center_x)].item()
                print(f"Track ID: {track_id}, Center: ({center_x}, {center_y}), Depth: {depth_at_center} meters")


if __name__ == "__main__":
    print(depth_mapping_webcam())