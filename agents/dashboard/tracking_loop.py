from __future__ import annotations

import socket
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from agents.dashboard.config import CAPTURES_DIR
from agents.tracking.calibration import GroundPlaneCalibration, load_ground_plane_calibration
from agents.coordinator import AgentCoordinator
from agents.tracking.agent_tracking import (
    AsyncVisualTrackingRuntime,
    LLMVisualTrackingAgent,
    encode_frame_as_data_url,
)
from agents.tracking.evidence_packaging import TrackingEvidencePackager
from agents.tracking.evidence_store import TrackingEvidenceStore
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


def _capture_frame_or_mark_error(state: SharedDashboardState, camera) -> tuple[bool, object | None]:
    ok, frame = camera.read()
    if not ok:
        with state.lock:
            state.camera_status = "frame_error"
        time.sleep(0.1)
        return False, None

    with state.lock:
        state.camera_status = "connected"
    return True, frame


def _run_detection_or_mark_error(state: SharedDashboardState, model: YOLO, frame):
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
        return None

    with state.lock:
        state.detector_status = "running"
    return result


def _apply_tracking_agent_runtime_cycle(
    *,
    state: SharedDashboardState,
    tracking_runtime: AsyncVisualTrackingRuntime,
    now: float,
    frame,
    tracks,
    tracking_candidates,
    selected_event_reasons,
) -> object | None:
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

    return tracking_agent_observation


def _publish_tracking_outputs(
    *,
    state: SharedDashboardState,
    result,
    now: float,
    tracks,
    tracking_agent_observation,
    trail_length: int,
    detections_for_file: list[dict],
    udp_socket,
) -> None:
    annotated = result.plot()
    commit_live_tracking_frame_state(
        state=state,
        now=now,
        annotated_frame=annotated,
        tracks=tracks,
        tracking_agent_observation=tracking_agent_observation,
        trail_length=trail_length,
    )

    publish_tracking_frame_message(
        tracking_frame_path=LATEST_TRACKING_FRAME_PATH,
        timestamp=now,
        detections=detections_for_file,
        udp_socket=udp_socket,
        udp_host=UDP_TRACKING_HOST,
        udp_port=UDP_TRACKING_PORT,
    )


def _package_tracking_evidence_if_enabled(
    *,
    state: SharedDashboardState,
    frame,
    now: float,
    tracking_candidates,
    tracking_agent_observation,
) -> None:
    evidence_store = state.tracking_evidence_store
    evidence_packager = state.tracking_evidence_packager
    if evidence_store is None or evidence_packager is None:
        return

    frame_height, frame_width = frame.shape[:2]
    frame_evidence = evidence_store.register_frame(
        frame=frame,
        timestamp=now,
        width=frame_width,
        height=frame_height,
    )

    updates_by_detection: dict[str, dict] = {}
    if tracking_agent_observation is not None:
        updates = tracking_agent_observation.to_dict().get("updates")
        if isinstance(updates, list):
            for item in updates:
                if not isinstance(item, dict):
                    continue
                detection_id = item.get("detection_id")
                if isinstance(detection_id, str):
                    updates_by_detection[detection_id] = item

    for candidate in tracking_candidates:
        update_payload = updates_by_detection.get(candidate.detection_id)
        evidence_packager.package_detection(
            detection=candidate,
            timestamp=now,
            frame_evidence=frame_evidence,
            update_payload=update_payload,
        )


def _initialize_tracking_runtime(
    state: SharedDashboardState,
) -> tuple[YOLO, object, GroundPlaneCalibration | None, socket.socket, AsyncVisualTrackingRuntime] | None:
    model_path = resolve_model_path()

    try:
        model = YOLO(model_path)
    except Exception:
        with state.lock:
            state.detector_status = "error"
            state.camera_status = "error"
        return None

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        with state.lock:
            state.camera_status = "unavailable"
            state.detector_status = "idle"
        return None

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

    udp_socket: socket.socket | None = None
    tracking_runtime: AsyncVisualTrackingRuntime | None = None
    evidence_store: TrackingEvidenceStore | None = None
    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tracking_coordinator = AgentCoordinator()
        tracking_agent = LLMVisualTrackingAgent()
        tracking_runtime = AsyncVisualTrackingRuntime(
            coordinator=tracking_coordinator,
            agent=tracking_agent,
            min_invoke_interval_seconds=0.75,
        )
        evidence_store = TrackingEvidenceStore(
            root_dir=CAPTURES_DIR / "tracking_evidence",
            max_age_seconds=60.0,
            max_frame_records=240,
            max_detection_records=6000,
            max_envelope_records=12000,
        )
        evidence_packager = TrackingEvidencePackager(
            coordinator=tracking_coordinator,
            evidence_store=evidence_store,
            max_history_per_track=20,
        )
        with state.lock:
            state.tracking_agent_model = tracking_agent.model
            state.tracking_evidence_store = evidence_store
            state.tracking_evidence_packager = evidence_packager
    except Exception:
        if evidence_store is not None:
            evidence_store.shutdown()
        if tracking_runtime is not None:
            tracking_runtime.shutdown()
        if udp_socket is not None:
            udp_socket.close()
        camera.release()
        raise

    return model, camera, calibration, udp_socket, tracking_runtime


def run_live_tracking_loop(state: SharedDashboardState) -> None:
    initialized = _initialize_tracking_runtime(state)
    if initialized is None:
        return

    model, camera, calibration, udp_socket, tracking_runtime = initialized

    try:
        while True:
            with state.lock:
                if not state.running:
                    break

            ok, frame = _capture_frame_or_mark_error(state, camera)
            if not ok:
                continue
            assert frame is not None

            result = _run_detection_or_mark_error(state, model, frame)
            if result is None:
                continue

            now = time.time()
            with state.lock:
                track_trails_snapshot = dict(state.track_trails)
            tracks, detections_for_file, tracking_candidates, selected_event_reasons = build_tracking_frame_outputs(
                boxes=result.boxes,
                model_names=model.names,
                now=now,
                frame_shape=frame.shape,
                calibration=calibration,
                track_trails=track_trails_snapshot,
                max_ground_x=MAX_GROUND_X,
                max_ground_z=MAX_GROUND_Z,
            )

            tracking_agent_observation = _apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=tracking_runtime,
                now=now,
                frame=frame,
                tracks=tracks,
                tracking_candidates=tracking_candidates,
                selected_event_reasons=selected_event_reasons,
            )

            _package_tracking_evidence_if_enabled(
                state=state,
                frame=frame,
                now=now,
                tracking_candidates=tracking_candidates,
                tracking_agent_observation=tracking_agent_observation,
            )

            _publish_tracking_outputs(
                state=state,
                result=result,
                now=now,
                tracks=tracks,
                tracking_agent_observation=tracking_agent_observation,
                trail_length=RADAR_TRAIL_LENGTH,
                detections_for_file=detections_for_file,
                udp_socket=udp_socket,
            )
    finally:
        with state.lock:
            state.camera_status = "stopped"
            state.detector_status = "stopped"
            evidence_store = state.tracking_evidence_store
            state.tracking_evidence_store = None
            state.tracking_evidence_packager = None
        if evidence_store is not None:
            evidence_store.shutdown()
        udp_socket.close()
        tracking_runtime.shutdown()
        camera.release()
