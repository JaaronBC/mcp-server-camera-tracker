from __future__ import annotations

import socket
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from agents.tracking.calibration import GroundPlaneCalibration, load_ground_plane_calibration
from agents.coordinator import AgentCoordinator
from agents.tracking.agent_tracking import (
    AsyncVisualTrackingRuntime,
    LLMVisualTrackingAgent,
    encode_frame_as_data_url,
)
from agents.tracking.frame_transformer import build_tracking_frame_outputs
from agents.tracking.runtime_support import (
    apply_tracking_agent_metric_update_to_state,
    commit_live_tracking_frame_state,
    collect_tracking_runtime_observation_update,
    submit_tracking_runtime_candidates,
)
from agents.tracking.state_publisher import (
    build_tracking_frame_message,
    publish_tracking_frame_message,
)

from .config import (
    GROUND_CALIBRATION_PATH,
    LATEST_TRACKING_FRAME_PATH,
    MAX_GROUND_X,
    MAX_GROUND_Z,
    RADAR_TRAIL_LENGTH,
    UDP_TRACKING_HOST,
    UDP_TRACKING_PORT,
)
from .state import SharedDashboardState

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_path() -> str:
    candidates = [
        Path(__file__).resolve().parent / "yolo11n.pt",
        PROJECT_ROOT / "yolo11n.pt",
        Path("yolov8n.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "yolov8n.pt"


def build_tracking_message(timestamp: float, detections: list[dict]) -> dict:
    return build_tracking_frame_message(timestamp, detections)


def run_live_tracking_loop(
    state: SharedDashboardState,
    video_path: str | None = None,
) -> None:
    with state.lock:
        state.video_path = video_path

    source: str | int = (
        str(Path(video_path).expanduser()) if video_path is not None else 0
    )
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        with state.lock:
            state.camera_status = "unavailable"
            state.detector_status = "idle"
            input_name = (
                f"video file '{source}'"
                if video_path is not None
                else "webcam (camera index 0)"
            )
            state.input_error = f"Could not open {input_name}."
        capture.release()
        return

    model_path = resolve_model_path()

    try:
        model = YOLO(model_path)
    except Exception as error:
        capture.release()
        with state.lock:
            state.detector_status = "error"
            state.camera_status = "error"
            state.input_error = f"{type(error).__name__}: {error}"
        return

    calibration: GroundPlaneCalibration | None = None
    try:
        calibration = load_ground_plane_calibration(GROUND_CALIBRATION_PATH)
        calibration_status = "loaded" if calibration is not None else "missing"
    except Exception:
        calibration = None
        calibration_status = "invalid"

    with state.lock:
        state.calibration_status = calibration_status

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tracking_coordinator = AgentCoordinator()
    tracking_agent = LLMVisualTrackingAgent()
    tracking_runtime = AsyncVisualTrackingRuntime(
        coordinator=tracking_coordinator,
        agent=tracking_agent,
        min_invoke_interval_seconds=0.75,
    )
    with state.lock:
        state.tracking_agent_model = tracking_agent.model

    try:
        while True:
            with state.lock:
                if not state.running:
                    break

            ok, frame = capture.read()
            if not ok:
                is_video_end = False
                if video_path is not None:
                    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    frame_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                    is_video_end = frame_count > 0 and frame_position >= frame_count
                with state.lock:
                    if is_video_end:
                        state.camera_status = "stopped"
                        state.detector_status = "stopped"
                    else:
                        state.camera_status = "frame_error"
                        state.detector_status = "error"
                        state.input_error = (
                            f"Unable to read a frame from {source!r}."
                        )
                if is_video_end:
                    break
                if video_path is not None:
                    break
                time.sleep(0.1)
                continue

            now = time.time()
            with state.lock:
                state.camera_status = "connected"
                state.detector_status = "running"
                state.input_error = None
                state.latest_annotated_frame = frame.copy()
                state.frame_height, state.frame_width = frame.shape[:2]
                state.last_live_frame_time = now

            try:
                result = model.track(
                    source=frame,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    persist=True,
                    verbose=False,
                    conf=0.65,
                )[0]
                now = time.time()
                tracks, detections_for_file, tracking_candidates, selected_event_reasons = build_tracking_frame_outputs(
                    boxes=result.boxes,
                    model_names=model.names,
                    now=now,
                    frame_shape=frame.shape,
                    calibration=calibration,
                    track_trails=state.track_trails,
                    max_ground_x=MAX_GROUND_X,
                    max_ground_z=MAX_GROUND_Z,
                )

                did_submit = submit_tracking_runtime_candidates(
                    runtime=tracking_runtime,
                    now=now,
                    frame=frame,
                    detections=tracking_candidates,
                    event_reasons=selected_event_reasons,
                    encode_frame=encode_frame_as_data_url,
                    request_timeout_seconds=1.0,
                )
                if did_submit:
                    with state.lock:
                        state.tracking_agent_submissions += 1

                with state.lock:
                    track_trails_snapshot = dict(state.track_trails)
                tracking_agent_observation, metric_update = collect_tracking_runtime_observation_update(
                    runtime=tracking_runtime,
                    tracks=tracks,
                    track_trails=track_trails_snapshot,
                    trail_length=RADAR_TRAIL_LENGTH,
                )
                if tracking_agent_observation is not None and metric_update is not None:
                    apply_tracking_agent_metric_update_to_state(
                        state=state,
                        metric_update=metric_update,
                    )

                annotated = result.plot()
                commit_live_tracking_frame_state(
                    state=state,
                    now=now,
                    annotated_frame=annotated,
                    tracks=tracks,
                    tracking_agent_observation=tracking_agent_observation,
                    trail_length=RADAR_TRAIL_LENGTH,
                )

                publish_tracking_frame_message(
                    tracking_frame_path=LATEST_TRACKING_FRAME_PATH,
                    timestamp=now,
                    detections=detections_for_file,
                    udp_socket=udp_socket,
                    udp_host=UDP_TRACKING_HOST,
                    udp_port=UDP_TRACKING_PORT,
                )
            except Exception as error:
                with state.lock:
                    state.camera_status = "frame_error"
                    state.detector_status = "error"
                    state.input_error = f"{type(error).__name__}: {error}"
                break

    except Exception as error:
        with state.lock:
            state.camera_status = "frame_error"
            state.detector_status = "error"
            state.input_error = f"{type(error).__name__}: {error}"
    finally:
        with state.lock:
            if state.detector_status != "error":
                state.camera_status = "stopped"
                state.detector_status = "stopped"
        capture.release()
        udp_socket.close()
        tracking_runtime.shutdown()
