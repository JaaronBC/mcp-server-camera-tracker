from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dashboard._live_tracking_display_impl import *  # noqa: F401,F403


if __name__ == "__main__":
    main()

"""
# Live dashboard pipeline that captures camera frames, tracks detections, applies tracking-agent updates, and renders radar/display state.
from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agents.components.dashboard_types import (
    RadarTrackSnapshot,
)
from agents.components.calibration import (
    GroundPlaneCalibration,
    load_ground_plane_calibration,
)
from agents.components.geometry import ground_to_radar_screen
from agents.contracts import TrackingAgentObservation
from agents.tracking.agent_tracking import (
    AsyncVisualTrackingRuntime,
    LLMVisualTrackingAgent,
    encode_frame_as_data_url,
)
from agents.tracking.state_publisher import (
    apply_simulation_fallback_state,
    build_tracking_frame_message,
    build_tracking_simulated_tracks,
    collect_radar_publish_snapshot,
    publish_tracking_frame_message,
    publish_tracking_radar_state,
)
from agents.tracking.runtime_support import (
    apply_tracking_agent_metric_update_to_state,
    apply_tracking_observation_update,
    apply_tracking_agent_updates_to_tracks,
    build_relative_depth_coverage_text,
    build_tracking_detection_id,
    commit_live_tracking_frame_state,
    collect_tracking_runtime_observation_update,
    compute_relative_depth_coverage,
    estimate_relative_depth_from_image_geometry,
    normalize_position_metadata,
    submit_tracking_runtime_candidates,
    summarize_tracking_agent_observation_metrics,
)
from agents.tracking.status_runtime import (
    build_tracking_agent_health_legend_text,
    build_tracking_agent_status_text,
    build_tracking_debug_toggle_hint,
    classify_tracking_agent_health,
    tracking_agent_health_color,
)
from agents.tracking.frame_transformer import build_tracking_frame_outputs
CAPTURES_DIR = PROJECT_ROOT / "captures"
LATEST_TRACKING_FRAME_PATH = CAPTURES_DIR / "latest_tracking_frame.json"
LATEST_RADAR_STATE_PATH = CAPTURES_DIR / "latest_radar_state.json"
RADAR_DEPTH_COVERAGE_HISTORY_PATH = CAPTURES_DIR / "radar_depth_coverage_history.jsonl"
GROUND_CALIBRATION_PATH = CAPTURES_DIR / "ground_plane_calibration.json"

WINDOW_NAME = "Counter-UAS Dashboard Demo"
RADAR_TRAIL_LENGTH = 20
RADAR_STALE_SECONDS = 1.5
LIVE_DATA_STALE_FOR_SIM_SECONDS = 2.0
SIM_UPDATE_HZ = 10.0
RENDER_HZ = 20.0
MAX_GROUND_X = 10.0
AGENT_GROUND_BLEND_FACTOR = 0.35
AGENT_GROUND_MAX_CORRECTION = 1.5

def _env_positive_float(name: str, default: float) -> float:
    # Implements env positive float behavior for this module.
MAX_GROUND_Z = _env_positive_float("DASHBOARD_MAX_GROUND_Z", 20.0)
UDP_TRACKING_HOST = "127.0.0.1"
UDP_TRACKING_PORT = 5005

class SharedDashboardState:
    def __init__(self) -> None:
        # Implements init behavior for this module.
    self.lock = threading.Lock()
    self.running = True

    self.camera_status = "starting"
    self.detector_status = "starting"
    self.calibration_status = "unknown"
    self.mapping_mode = "relative_image_plane"

    self.latest_annotated_frame = None
    self.frame_width = 1280
    self.frame_height = 720

    self.last_live_frame_time = 0.0
    self.last_observation_time = 0.0

    self.active_tracks: dict[str, RadarTrackSnapshot] = {}
    self.track_trails: dict[str, deque[tuple[float, float]]] = {}
    self.latest_tracking_agent_observation: dict | None = None
    self.tracking_agent_model = "qwen-3.8-27b"
    self.tracking_agent_submissions = 0
    self.tracking_agent_completions = 0
    self.tracking_agent_unavailable_count = 0
    self.tracking_agent_total_updates_accepted = 0
    self.tracking_agent_total_updates_rejected = 0
    self.tracking_agent_total_updates_unmatched = 0
    self.tracking_agent_last_latency_ms: float | None = None
    self.tracking_agent_last_error: str | None = None

    self.selected_track_id: str | None = None
    self.show_tracking_debug_details = False

def resolve_model_path() -> str:
    # Resolves model path from configuration with safe defaults.
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
    # Builds tracking message payloads for pipeline consumers.
    return build_tracking_frame_message(timestamp, detections)

def run_live_tracking_loop(state: SharedDashboardState) -> None:
    # Captures frames, runs detector/tracker, updates live tracks, and ingests tracking-agent observations.
    model_path = resolve_model_path()

    try:
        model = YOLO(model_path)
    except Exception:
        with state.lock:
            state.detector_status = "error"
            state.camera_status = "error"
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        with state.lock:
            state.camera_status = "unavailable"
            state.detector_status = "idle"
        return

    with state.lock:
        state.camera_status = "connected"
        state.detector_status = "running"

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

            ok, frame = camera.read()
            if not ok:
                with state.lock:
                    state.camera_status = "frame_error"
                time.sleep(0.1)
                continue

            try:
                result = model.track(
                    source=frame,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    persist=True,
                    verbose=False,
                    conf=0.65,
                )[0]
            except Exception:
                with state.lock:
                    state.detector_status = "error"
                time.sleep(0.15)
                continue

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
    finally:
        with state.lock:
            state.camera_status = "stopped"
            state.detector_status = "stopped"
        udp_socket.close()
        tracking_runtime.shutdown()
        camera.release()

def run_radar_loop(state: SharedDashboardState) -> None:
    # Maintains radar/shared-state artifacts and simulation fallback when live input is stale.
    sim_step = 0.0
    sim_interval = 1.0 / SIM_UPDATE_HZ

    while True:
        with state.lock:
            if not state.running:
                break
            now = time.time()
            live_is_stale = (now - state.last_live_frame_time) > LIVE_DATA_STALE_FOR_SIM_SECONDS
            camera_status = state.camera_status

        using_simulation = live_is_stale or camera_status in {"unavailable", "error", "frame_error", "stopped"}

        if using_simulation:
            simulated_tracks = build_tracking_simulated_tracks(
                sim_step,
                max_ground_x=MAX_GROUND_X,
                max_ground_z=MAX_GROUND_Z,
            )
            apply_simulation_fallback_state(
                state=state,
                now=now,
                simulated_tracks=simulated_tracks,
                trail_length=RADAR_TRAIL_LENGTH,
            )
            sim_step += sim_interval
        else:
            with state.lock:
                state.mapping_mode = "relative_image_plane"

        radar_publish_snapshot = collect_radar_publish_snapshot(state)

        try:
            publish_tracking_radar_state(
                radar_state_path=LATEST_RADAR_STATE_PATH,
                radar_depth_coverage_history_path=RADAR_DEPTH_COVERAGE_HISTORY_PATH,
                now=now,
                coordinate_system=radar_publish_snapshot["coordinate_system"],
                camera_status=radar_publish_snapshot["camera_status"],
                detector_status=radar_publish_snapshot["detector_status"],
                calibration_status=radar_publish_snapshot["calibration_status"],
                tracking_agent_status=radar_publish_snapshot["tracking_agent_status"],
                last_observation_timestamp=radar_publish_snapshot["last_observation_time"],
                tracking_agent_observation=radar_publish_snapshot["tracking_agent_observation"],
                tracks=radar_publish_snapshot["tracks"],
            )
        except OSError:
            pass

        time.sleep(sim_interval)

def main() -> None:
    # Implements main behavior for this module.
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    state = SharedDashboardState()

    tracking_thread = threading.Thread(target=run_live_tracking_loop, args=(state,), daemon=True)
    radar_thread = threading.Thread(target=run_radar_loop, args=(state,), daemon=True)

    tracking_thread.start()
    radar_thread.start()

    try:
        render_dashboard(state)
    finally:
        with state.lock:
            state.running = False

        tracking_thread.join(timeout=2.0)
        radar_thread.join(timeout=2.0)

if __name__ == "__main__":
    main()
"""
"""
# Live dashboard pipeline that captures camera frames, tracks detections, applies tracking-agent updates, and renders radar/display state.
from __future__ import annotations

import os
import socket
import threading
import time
from collections import deque
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.components.dashboard_types import (
    RadarTrackSnapshot,
)
from agents.components.calibration import (
    GroundPlaneCalibration,
    load_ground_plane_calibration,
)
from agents.components.geometry import ground_to_radar_screen
from agents.contracts import TrackingAgentObservation
from agents.coordinator import AgentCoordinator
from agents.tracking.agent_tracking import (
    AsyncVisualTrackingRuntime,
    LLMVisualTrackingAgent,
    encode_frame_as_data_url,
)
from agents.tracking.state_publisher import (
    apply_simulation_fallback_state,
    build_tracking_frame_message,
    build_tracking_simulated_tracks,
    collect_radar_publish_snapshot,
    publish_tracking_frame_message,
    publish_tracking_radar_state,
)
from agents.tracking.runtime_support import (
    apply_tracking_agent_metric_update_to_state,
    apply_tracking_observation_update,
    apply_tracking_agent_updates_to_tracks,
    build_relative_depth_coverage_text,
    build_tracking_detection_id,
    commit_live_tracking_frame_state,
    collect_tracking_runtime_observation_update,
    compute_relative_depth_coverage,
    estimate_relative_depth_from_image_geometry,
    normalize_position_metadata,
    submit_tracking_runtime_candidates,
    summarize_tracking_agent_observation_metrics,
)
from agents.tracking.status_runtime import (
    build_tracking_agent_health_legend_text,
    build_tracking_agent_status_text,
    build_tracking_debug_toggle_hint,
    classify_tracking_agent_health,
    tracking_agent_health_color,
)
from agents.tracking.frame_transformer import build_tracking_frame_outputs
CAPTURES_DIR = PROJECT_ROOT / "captures"
LATEST_TRACKING_FRAME_PATH = CAPTURES_DIR / "latest_tracking_frame.json"
LATEST_RADAR_STATE_PATH = CAPTURES_DIR / "latest_radar_state.json"
RADAR_DEPTH_COVERAGE_HISTORY_PATH = CAPTURES_DIR / "radar_depth_coverage_history.jsonl"
GROUND_CALIBRATION_PATH = CAPTURES_DIR / "ground_plane_calibration.json"

WINDOW_NAME = "Counter-UAS Dashboard Demo"
RADAR_TRAIL_LENGTH = 20
RADAR_STALE_SECONDS = 1.5
RADAR_LOST_SECONDS = 3.5
LIVE_DATA_STALE_FOR_SIM_SECONDS = 2.0
SIM_UPDATE_HZ = 10.0
RENDER_HZ = 20.0
MAX_GROUND_X = 10.0
AGENT_GROUND_BLEND_FACTOR = 0.35
AGENT_GROUND_MAX_CORRECTION = 1.5


def _env_positive_float(name: str, default: float) -> float:
    # Implements env positive float behavior for this module.
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


MAX_GROUND_Z = _env_positive_float("DASHBOARD_MAX_GROUND_Z", 20.0)
UDP_TRACKING_HOST = "127.0.0.1"
UDP_TRACKING_PORT = 5005


class SharedDashboardState:
    def __init__(self) -> None:
        # Implements init behavior for this module.
        self.lock = threading.Lock()
        self.running = True

        self.camera_status = "starting"
        self.detector_status = "starting"
        self.calibration_status = "unknown"
        self.mapping_mode = "relative_image_plane"

        self.latest_annotated_frame = None
        self.frame_width = 1280
        self.frame_height = 720

        self.last_live_frame_time = 0.0
        self.last_observation_time = 0.0

        self.active_tracks: dict[str, RadarTrackSnapshot] = {}
        self.track_trails: dict[str, deque[tuple[float, float]]] = {}
        self.latest_tracking_agent_observation: dict | None = None
        self.tracking_agent_model = "qwen-3.8-27b"
        self.tracking_agent_submissions = 0
        self.tracking_agent_completions = 0
        self.tracking_agent_unavailable_count = 0
        self.tracking_agent_total_updates_accepted = 0
        self.tracking_agent_total_updates_rejected = 0
        self.tracking_agent_total_updates_unmatched = 0
        self.tracking_agent_last_latency_ms: float | None = None
        self.tracking_agent_last_error: str | None = None

        self.selected_track_id: str | None = None
        self.show_tracking_debug_details = False


def resolve_model_path() -> str:
    # Resolves model path from configuration with safe defaults.
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
    # Builds tracking message payloads for pipeline consumers.
    return build_tracking_frame_message(timestamp, detections)


def run_live_tracking_loop(state: SharedDashboardState) -> None:
    # Captures frames, runs detector/tracker, updates live tracks, and ingests tracking-agent observations.
    model_path = resolve_model_path()

    try:
        model = YOLO(model_path)
    except Exception:
        with state.lock:
            state.detector_status = "error"
            state.camera_status = "error"
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        with state.lock:
            state.camera_status = "unavailable"
            state.detector_status = "idle"
        return

    with state.lock:
        state.camera_status = "connected"
        state.detector_status = "running"

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

            ok, frame = camera.read()
            if not ok:
                with state.lock:
                    state.camera_status = "frame_error"
                time.sleep(0.1)
                continue

            try:
                result = model.track(
                    source=frame,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    persist=True,
                    verbose=False,
                    conf=0.65,
                )[0]
            except Exception:
                with state.lock:
                    state.detector_status = "error"
                time.sleep(0.15)
                continue

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
    finally:
        with state.lock:
            state.camera_status = "stopped"
            state.detector_status = "stopped"
        udp_socket.close()
        tracking_runtime.shutdown()
        camera.release()


def run_radar_loop(state: SharedDashboardState) -> None:
    # Maintains radar/shared-state artifacts and simulation fallback when live input is stale.
    sim_step = 0.0
    sim_interval = 1.0 / SIM_UPDATE_HZ

    while True:
        with state.lock:
            if not state.running:
                break
            now = time.time()
            live_is_stale = (now - state.last_live_frame_time) > LIVE_DATA_STALE_FOR_SIM_SECONDS
            camera_status = state.camera_status

        using_simulation = live_is_stale or camera_status in {"unavailable", "error", "frame_error", "stopped"}

        if using_simulation:
            simulated_tracks = build_tracking_simulated_tracks(
                sim_step,
                max_ground_x=MAX_GROUND_X,
                max_ground_z=MAX_GROUND_Z,
            )
            apply_simulation_fallback_state(
                state=state,
                now=now,
                simulated_tracks=simulated_tracks,
                trail_length=RADAR_TRAIL_LENGTH,
            )
            sim_step += sim_interval
        else:
            with state.lock:
                state.mapping_mode = "relative_image_plane"

        radar_publish_snapshot = collect_radar_publish_snapshot(state)

        try:
            publish_tracking_radar_state(
                radar_state_path=LATEST_RADAR_STATE_PATH,
                radar_depth_coverage_history_path=RADAR_DEPTH_COVERAGE_HISTORY_PATH,
                now=now,
                coordinate_system=radar_publish_snapshot["coordinate_system"],
                camera_status=radar_publish_snapshot["camera_status"],
                detector_status=radar_publish_snapshot["detector_status"],
                calibration_status=radar_publish_snapshot["calibration_status"],
                tracking_agent_status=radar_publish_snapshot["tracking_agent_status"],
                last_observation_timestamp=radar_publish_snapshot["last_observation_time"],
                tracking_agent_observation=radar_publish_snapshot["tracking_agent_observation"],
                tracks=radar_publish_snapshot["tracks"],
            )
        except OSError:
            pass

        time.sleep(sim_interval)


def draw_placeholder_webcam_panel(width: int, height: int, message: str):
    # Implements draw placeholder webcam panel behavior for this module.
    panel = cv2.UMat(height, width, cv2.CV_8UC3).get()
    panel[:] = (22, 22, 22)
    cv2.putText(
        panel,
        "Webcam feed unavailable",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 170, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        message,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def draw_radar_panel(
    width: int,
    height: int,
    tracks: dict[str, RadarTrackSnapshot],
    trails: dict[str, deque[tuple[float, float]]],
    now: float,
):
    # Implements draw radar panel behavior for this module.
    panel = cv2.UMat(height, width, cv2.CV_8UC3).get()
    panel[:] = (15, 18, 22)

    margin = 40
    center_x = margin + (width - 2 * margin) // 2
    camera_y = height - margin

    cv2.line(panel, (center_x, margin), (center_x, camera_y), (70, 70, 70), 1)
    cv2.line(panel, (margin, camera_y), (width - margin, camera_y), (70, 70, 70), 1)
    cv2.putText(panel, "X", (width - margin + 8, camera_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(panel, "Z", (center_x + 8, margin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(panel, "Camera", (center_x - 25, camera_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    for track_id, trail in trails.items():
        if len(trail) < 2:
            continue

        points = [
            ground_to_radar_screen(
                x,
                z,
                width,
                height,
                MAX_GROUND_X,
                MAX_GROUND_Z,
                margin,
            )
            for (x, z) in trail
        ]

        for i in range(1, len(points)):
            cv2.line(panel, points[i - 1], points[i], (80, 130, 240), 1)

        if track_id in tracks:
            snapshot = tracks[track_id]
            marker_point = ground_to_radar_screen(
                snapshot.ground_position.x,
                snapshot.ground_position.z,
                width,
                height,
                MAX_GROUND_X,
                MAX_GROUND_Z,
                margin,
            )
            age_seconds = now - snapshot.timestamp
            if age_seconds > RADAR_LOST_SECONDS:
                marker_color = (0, 0, 255)
                status_label = "lost"
            elif age_seconds > RADAR_STALE_SECONDS:
                marker_color = (0, 180, 255)
                status_label = "stale"
            else:
                marker_color = (70, 230, 70)
                status_label = "fresh"

            cv2.circle(panel, marker_point, 6, marker_color, -1)
            cv2.putText(
                panel,
                f"{track_id} {status_label}",
                (marker_point[0] + 8, marker_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

    return panel


def render_dashboard(state: SharedDashboardState) -> None:
    # Renders the webcam/radar dashboard UI and operator status overlays.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        with state.lock:
            if not state.running:
                break

            camera_status = state.camera_status
            detector_status = state.detector_status
            calibration_status = state.calibration_status
            mapping_mode = state.mapping_mode
            latest_frame = state.latest_annotated_frame
            tracks = dict(state.active_tracks)
            trails = {
                key: deque(value, maxlen=RADAR_TRAIL_LENGTH)
                for key, value in state.track_trails.items()
            }
            last_obs_time = state.last_observation_time
            selected_track_id = state.selected_track_id
            show_tracking_debug_details = state.show_tracking_debug_details
            tracking_agent_status = {
                "model": state.tracking_agent_model,
                "submissions": state.tracking_agent_submissions,
                "completions": state.tracking_agent_completions,
                "unavailable_count": state.tracking_agent_unavailable_count,
                "total_updates_accepted": state.tracking_agent_total_updates_accepted,
                "total_updates_rejected": state.tracking_agent_total_updates_rejected,
                "total_updates_unmatched": state.tracking_agent_total_updates_unmatched,
                "last_latency_ms": state.tracking_agent_last_latency_ms,
                "last_error": state.tracking_agent_last_error,
            }

        now = time.time()

        dashboard_width = 1380
        dashboard_height = 760
        header_height = 112
        panel_height = dashboard_height - header_height
        panel_width = dashboard_width // 2

        if latest_frame is None:
            webcam_panel = draw_placeholder_webcam_panel(
                panel_width,
                panel_height,
                f"camera_status={camera_status}",
            )
        else:
            webcam_panel = cv2.resize(latest_frame, (panel_width, panel_height))
            cv2.putText(
                webcam_panel,
                "Live Webcam (YOLO + ByteTrack)",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        radar_panel = draw_radar_panel(panel_width, panel_height, tracks, trails, now)
        cv2.putText(
            radar_panel,
            "Ground-Plane Radar (relative units)",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        header = cv2.UMat(header_height, dashboard_width, cv2.CV_8UC3).get()
        header[:] = (28, 30, 38)
        cv2.putText(
            header,
            "Counter-UAS Visual Radar Dashboard Demo",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        active_track_count = len(tracks)
        relative_depth_coverage = compute_relative_depth_coverage(list(tracks.values()))
        relative_depth_coverage_text = build_relative_depth_coverage_text(relative_depth_coverage)
        last_obs_ago = "n/a" if last_obs_time == 0 else f"{now - last_obs_time:.2f}s ago"
        status_text = (
            f"camera={camera_status} | detector={detector_status} | "
            f"tracks={active_track_count} | last_obs={last_obs_ago} | "
            f"mode={mapping_mode} | calibration={calibration_status}"
        )
        cv2.putText(
            header,
            status_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        tracking_status_text = build_tracking_agent_status_text(
            tracking_agent_status,
            detailed=show_tracking_debug_details,
        )
        tracking_health = classify_tracking_agent_health(tracking_agent_status)
        cv2.putText(
            header,
            tracking_status_text,
            (20, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            tracking_agent_health_color(tracking_health),
            1,
            cv2.LINE_AA,
        )

        toggle_hint = build_tracking_debug_toggle_hint(show_tracking_debug_details)
        cv2.putText(
            header,
            toggle_hint,
            (980, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            header,
            relative_depth_coverage_text,
            (980, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (170, 220, 255),
            1,
            cv2.LINE_AA,
        )

        legend_text = build_tracking_agent_health_legend_text()
        cv2.putText(
            header,
            legend_text,
            (980, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(header, (1188, 27), 4, tracking_agent_health_color("good"), -1)
        cv2.circle(header, (1248, 27), 4, tracking_agent_health_color("warn"), -1)
        cv2.circle(header, (1308, 27), 4, tracking_agent_health_color("error"), -1)

        if tracks:
            sorted_ids = sorted(tracks.keys())
            if selected_track_id not in tracks:
                selected_track_id = sorted_ids[0]
            with state.lock:
                state.selected_track_id = selected_track_id

            selected = tracks[selected_track_id]
            detail_lines = [
                f"Selected: {selected.track_id}",
                f"timestamp: {selected.timestamp:.3f}",
                (
                    "pixel: n/a"
                    if selected.pixel_position is None
                    else f"pixel: ({selected.pixel_position.x:.1f}, {selected.pixel_position.y:.1f})"
                ),
                f"ground: ({selected.ground_position.x:.2f}, {selected.ground_position.z:.2f})",
                (
                    "relative_depth: n/a"
                    if selected.relative_depth is None
                    else f"relative_depth: {selected.relative_depth:.3f}"
                ),
                f"position_kind: {selected.position_kind}",
                f"quality_flags: {', '.join(selected.quality_flags) if selected.quality_flags else 'none'}",
            ]
            y_cursor = 20
            for line in detail_lines:
                cv2.putText(
                    radar_panel,
                    line,
                    (18, panel_height - 130 + y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                y_cursor += 18

        body = cv2.hconcat([webcam_panel, radar_panel])
        dashboard = cv2.vconcat([header, body])
        cv2.imshow(WINDOW_NAME, dashboard)

        key = cv2.waitKey(int(1000 / RENDER_HZ)) & 0xFF
        if key == ord("q"):
            with state.lock:
                state.running = False
            break

        if key == ord("h"):
            with state.lock:
                state.show_tracking_debug_details = not state.show_tracking_debug_details

        if key in {ord("n"), ord("p")} and tracks:
            sorted_ids = sorted(tracks.keys())
            current_id = state.selected_track_id if state.selected_track_id in sorted_ids else sorted_ids[0]
            current_index = sorted_ids.index(current_id)
            if key == ord("n"):
                next_id = sorted_ids[(current_index + 1) % len(sorted_ids)]
            else:
                next_id = sorted_ids[(current_index - 1) % len(sorted_ids)]
            with state.lock:
                state.selected_track_id = next_id

    cv2.destroyAllWindows()


def main() -> None:
    # Implements main behavior for this module.
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    state = SharedDashboardState()

    tracking_thread = threading.Thread(target=run_live_tracking_loop, args=(state,), daemon=True)
    radar_thread = threading.Thread(target=run_radar_loop, args=(state,), daemon=True)

    tracking_thread.start()
    radar_thread.start()

    try:
        render_dashboard(state)
    finally:
        with state.lock:
            state.running = False

        tracking_thread.join(timeout=2.0)
        radar_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
"""