# Live dashboard pipeline that captures camera frames, tracks detections, applies tracking-agent updates, and renders radar/display state.
from __future__ import annotations

import json
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

from tools.components.dashboard_types import (
    GroundPosition,
    Observation,
    PixelPosition,
    RadarTrackSnapshot,
)
from tools.components.calibration import (
    GroundPlaneCalibration,
    load_ground_plane_calibration,
    map_image_to_ground,
)
from tools.components.geometry import ground_to_radar_screen, image_to_relative_ground
from tools.agent_contracts import BoundingBox, DetectionCandidate, TrackingAgentObservation
from tools.agent_coordinator import AgentCoordinator
from tools.visual_tracking_agent import (
    AsyncVisualTrackingRuntime,
    LLMVisualTrackingAgent,
    encode_frame_as_data_url,
)
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


def update_json(path: Path, payload: dict) -> None:
    # Updates json persisted state safely.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict) -> None:
    # Appends jsonl records to log/history storage.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, separators=(",", ":")) + "\n")


def build_tracking_message(timestamp: float, detections: list[dict]) -> dict:
    # Builds tracking message payloads for pipeline consumers.
    return {
        "time": timestamp,
        "detections": detections,
    }


def build_tracking_agent_status_text(status: dict[str, object], detailed: bool = True) -> str:
    # Builds tracking agent status text payloads for pipeline consumers.
    model = status.get("model", "unknown")
    submissions = status.get("submissions", 0)
    completions = status.get("completions", 0)
    unavailable_count = status.get("unavailable_count", 0)
    accepted = status.get("total_updates_accepted", 0)
    rejected = status.get("total_updates_rejected", 0)
    unmatched = status.get("total_updates_unmatched", 0)
    latency_ms = status.get("last_latency_ms")
    last_error = status.get("last_error")

    latency_text = "n/a" if not isinstance(latency_ms, (int, float)) else f"{latency_ms:.1f}ms"
    error_text = "none" if not isinstance(last_error, str) or not last_error else last_error

    if not detailed:
        return (
            "tracking_agent_status: "
            f"model={model} | completions={completions} | unavailable={unavailable_count} | "
            f"accepted={accepted} | rejected={rejected} | last_error={error_text}"
        )

    return (
        "tracking_agent_status: "
        f"model={model} | submissions={submissions} | completions={completions} | "
        f"unavailable={unavailable_count} | accepted={accepted} | rejected={rejected} | "
        f"unmatched={unmatched} | last_latency={latency_text} | last_error={error_text}"
    )


def classify_tracking_agent_health(status: dict[str, object]) -> str:
    # Classifies tracking agent health into normalized categories.
    submissions = status.get("submissions", 0)
    completions = status.get("completions", 0)
    unavailable_count = status.get("unavailable_count", 0)
    accepted = status.get("total_updates_accepted", 0)
    rejected = status.get("total_updates_rejected", 0)
    last_error = status.get("last_error")

    if not isinstance(submissions, int):
        submissions = 0
    if not isinstance(completions, int):
        completions = 0
    if not isinstance(unavailable_count, int):
        unavailable_count = 0
    if not isinstance(accepted, int):
        accepted = 0
    if not isinstance(rejected, int):
        rejected = 0

    if isinstance(last_error, str) and last_error in {"vlm_timeout", "vlm_unavailable"}:
        return "error"

    if submissions == 0:
        return "warn"

    unavailable_rate = unavailable_count / submissions
    decisions = accepted + rejected
    rejection_rate = (rejected / decisions) if decisions > 0 else 0.0

    if unavailable_rate >= 0.25 or rejection_rate >= 0.5:
        return "error"
    if unavailable_rate >= 0.1 or rejection_rate >= 0.25 or completions < submissions:
        return "warn"
    return "good"


def tracking_agent_health_color(health: str) -> tuple[int, int, int]:
    # Implements tracking agent health color behavior for this module.
    if health == "good":
        return (90, 220, 120)
    if health == "error":
        return (80, 80, 245)
    return (100, 210, 240)


def build_tracking_agent_health_legend_text() -> str:
    # Builds tracking agent health legend text payloads for pipeline consumers.
    return "tracking health legend: good | warn | error"


def build_tracking_debug_toggle_hint(show_tracking_debug_details: bool) -> str:
    # Builds tracking debug toggle hint payloads for pipeline consumers.
    mode = "detailed" if show_tracking_debug_details else "concise"
    return f"press 'h' to toggle tracking-agent diagnostics ({mode})"


def compute_relative_depth_coverage(
    tracks: list[RadarTrackSnapshot],
) -> dict[str, float | int]:
    # Computes relative depth coverage metrics from available state.
    total_tracks = len(tracks)
    with_depth = 0

    for track in tracks:
        if track.relative_depth is not None:
            with_depth += 1

    coverage_ratio = (with_depth / total_tracks) if total_tracks > 0 else 0.0

    return {
        "tracks_with_depth": with_depth,
        "total_tracks": total_tracks,
        "coverage_ratio": round(coverage_ratio, 3),
        "coverage_percent": round(coverage_ratio * 100.0, 1),
    }


def build_relative_depth_coverage_text(metrics: dict[str, float | int]) -> str:
    # Builds relative depth coverage text payloads for pipeline consumers.
    tracks_with_depth = metrics.get("tracks_with_depth", 0)
    total_tracks = metrics.get("total_tracks", 0)
    coverage_percent = metrics.get("coverage_percent", 0.0)
    return (
        "depth coverage: "
        f"{tracks_with_depth}/{total_tracks} "
        f"({coverage_percent}%)"
    )


def build_tracking_detection_id(track_key: str) -> str:
    # Builds tracking detection id payloads for pipeline consumers.
    return track_key


def normalize_position_metadata(
    position_kind: str,
    ground_coordinate_unit: str,
    quality_flags: list[str],
) -> tuple[str, str, list[str]]:
    # Normalizes position metadata into contract-safe values.
    normalized_flags = list(quality_flags)
    normalized_kind = position_kind
    normalized_unit = ground_coordinate_unit

    valid_kinds = {"relative_ground_estimate", "calibrated_ground_plane"}
    if normalized_kind not in valid_kinds:
        normalized_kind = "relative_ground_estimate"
        normalized_unit = "relative_units"
        normalized_flags.append("position_kind_normalized:unknown_kind")

    if normalized_kind == "relative_ground_estimate":
        if normalized_unit != "relative_units":
            normalized_unit = "relative_units"
            normalized_flags.append("ground_unit_normalized:relative_units")
        return normalized_kind, normalized_unit, normalized_flags

    if not isinstance(normalized_unit, str) or not normalized_unit.strip() or normalized_unit == "relative_units":
        normalized_kind = "relative_ground_estimate"
        normalized_unit = "relative_units"
        normalized_flags.append("position_kind_normalized:invalid_calibrated_unit")

    return normalized_kind, normalized_unit, normalized_flags


def estimate_relative_depth_from_image_geometry(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> float | None:
    # Estimates relative depth from image geometry using deterministic heuristics.
    if frame_width <= 0 or frame_height <= 0:
        return None

    bbox_width = float(x2) - float(x1)
    bbox_height = float(y2) - float(y1)
    if bbox_width <= 0 or bbox_height <= 0:
        return None

    clamped_bottom_y = max(0.0, min(float(y2), float(frame_height)))
    vertical_depth = (float(frame_height) - clamped_bottom_y) / float(frame_height)

    normalized_height = max(0.0, min(bbox_height / float(frame_height), 1.0))
    scale_depth = 1.0 - normalized_height

    relative_depth = 0.7 * vertical_depth + 0.3 * scale_depth
    relative_depth = max(0.0, min(relative_depth, 1.0))
    return round(relative_depth, 3)


def apply_tracking_agent_updates_to_tracks(
    tracks: dict[str, RadarTrackSnapshot],
    tracking_agent_observation: TrackingAgentObservation,
    track_trails: dict[str, deque[tuple[float, float]]] | None = None,
) -> dict[str, int]:
    # Applies validated tracking-agent updates to track state, quality flags, confidence, and bounded continuity corrections.
    accepted = 0
    rejected = 0
    unmatched = 0

    for update in tracking_agent_observation.updates:
        track_key = update.detection_id
        snapshot = tracks.get(track_key)
        if snapshot is None:
            unmatched += 1
            continue

        if update.accepted:
            accepted += 1
            if update.resolved_track_id != track_key:
                snapshot.quality_flags.append(f"vlm_association_candidate:{update.resolved_track_id}")

            if track_trails is not None and update.resolved_track_id:
                resolved_trail = track_trails.get(update.resolved_track_id)
                if isinstance(resolved_trail, deque) and len(resolved_trail) > 0:
                    anchor_x, anchor_z = resolved_trail[-1]
                    current_x = snapshot.ground_position.x
                    current_z = snapshot.ground_position.z
                    delta_x = anchor_x - current_x
                    delta_z = anchor_z - current_z
                    distance = (delta_x * delta_x + delta_z * delta_z) ** 0.5
                    if distance > 0:
                        bounded_scale = min(
                            1.0,
                            AGENT_GROUND_MAX_CORRECTION / distance,
                        )
                        blended_x = current_x + (delta_x * AGENT_GROUND_BLEND_FACTOR * bounded_scale)
                        blended_z = current_z + (delta_z * AGENT_GROUND_BLEND_FACTOR * bounded_scale)
                        snapshot.ground_position = GroundPosition(x=blended_x, z=blended_z)
                        snapshot.quality_flags.append("tracking_agent_ground_anchor_blend")

            if any(
                reason in {"depth_conflict_with_reliable_measurement", "ground_position_conflict_with_calibrated_mapping"}
                for reason in update.rejection_reasons
            ):
                snapshot.quality_flags.append("tracking_agent_conflict_flagged")
                if isinstance(snapshot.confidence, (int, float)):
                    snapshot.confidence = max(0.0, snapshot.confidence * 0.9)
        else:
            rejected += 1
            snapshot.quality_flags.append("vlm_association_rejected")
            snapshot.quality_flags.append("cross_agent_review_required")
            for reason in update.rejection_reasons:
                snapshot.quality_flags.append(f"vlm_rejection:{reason}")
            if isinstance(snapshot.confidence, (int, float)):
                snapshot.confidence = max(0.0, snapshot.confidence * 0.85)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "unmatched": unmatched,
    }


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
            tracks: dict[str, RadarTrackSnapshot] = {}
            detections_for_file: list[dict] = []
            tracking_candidates: list[DetectionCandidate] = []
            selected_event_reasons: list[str] = []

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().tolist()
                track_ids = boxes.id.int().cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                frame_height, frame_width = frame.shape[:2]

                for bbox, track_id, cls_idx, confidence in zip(
                    coordinates,
                    track_ids,
                    classes,
                    confidences,
                ):
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2.0
                    center_y = y2
                    relative_depth = estimate_relative_depth_from_image_geometry(
                        x1,
                        y1,
                        x2,
                        y2,
                        frame_width,
                        frame_height,
                    )
                    quality_flags: list[str] = []
                    position_kind = "relative_ground_estimate"
                    ground_unit = "relative_units"

                    if calibration is not None:
                        try:
                            ground_x, ground_z = map_image_to_ground(
                                calibration.matrix,
                                center_x,
                                center_y,
                            )
                            position_kind = "calibrated_ground_plane"
                            ground_unit = calibration.unit
                            quality_flags.append("ground_plane_homography")
                        except Exception:
                            ground_x, ground_z = image_to_relative_ground(
                                center_x,
                                center_y,
                                frame_width,
                                frame_height,
                                MAX_GROUND_X,
                                MAX_GROUND_Z,
                            )
                            quality_flags.append("calibration_projection_failed")
                            quality_flags.append("estimated_ground_from_image")
                    else:
                        ground_x, _image_based_ground_z = image_to_relative_ground(
                            center_x,
                            center_y,
                            frame_width,
                            frame_height,
                            MAX_GROUND_X,
                            MAX_GROUND_Z,
                        )
                        if relative_depth is not None:
                            ground_z = relative_depth * MAX_GROUND_Z
                            quality_flags.append("estimated_ground_from_relative_depth")
                        else:
                            ground_z = _image_based_ground_z
                        quality_flags.append("estimated_ground_from_image")

                    position_kind, ground_unit, quality_flags = normalize_position_metadata(
                        position_kind=position_kind,
                        ground_coordinate_unit=ground_unit,
                        quality_flags=quality_flags,
                    )

                    track_key = str(track_id)
                    detection_id = build_tracking_detection_id(track_key)
                    if relative_depth is not None:
                        quality_flags.append("relative_depth_estimate:image_geometry")
                    else:
                        quality_flags.append("relative_depth_unavailable")

                    motion_history = []
                    existing_trail = state.track_trails.get(track_key)
                    if existing_trail is not None:
                        for index, (trail_x, trail_z) in enumerate(existing_trail):
                            motion_history.append(
                                {
                                    "history_index": index,
                                    "ground_x": trail_x,
                                    "ground_z": trail_z,
                                }
                            )
                    observation = Observation(
                        track_id=track_key,
                        timestamp=now,
                        source="camera",
                        bbox_xyxy=[x1, y1, x2, y2],
                        pixel_position=PixelPosition(x=center_x, y=center_y),
                        ground_position=GroundPosition(x=ground_x, z=ground_z),
                        ground_coordinate_unit=ground_unit,
                        position_kind=position_kind,
                        detection_confidence=round(float(confidence), 3),
                        quality_flags=quality_flags,
                    )

                    tracks[track_key] = RadarTrackSnapshot(
                        track_id=track_key,
                        timestamp=now,
                        pixel_position=observation.pixel_position,
                        ground_position=observation.ground_position,
                        ground_coordinate_unit=observation.ground_coordinate_unit,
                        position_kind=observation.position_kind,
                        quality_flags=observation.quality_flags,
                        confidence=observation.detection_confidence,
                        relative_depth=relative_depth,
                    )

                    tracking_candidates.append(
                        DetectionCandidate(
                            detection_id=detection_id,
                            timestamp=now,
                            yolo_track_id=track_key,
                            label=model.names[int(cls_idx)],
                            detection_confidence=round(float(confidence), 3),
                            pixel_bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
                            pixel_center_x=float(center_x),
                            pixel_center_y=float(center_y),
                            relative_depth=relative_depth,
                            ground_x=float(ground_x),
                            ground_z=float(ground_z),
                            ground_coordinate_unit=ground_unit,
                            position_kind=position_kind,
                            motion_history=motion_history,
                        )
                    )

                    if float(confidence) < 0.5:
                        selected_event_reasons.append("low_detection_confidence")
                    if "calibration_projection_failed" in quality_flags:
                        selected_event_reasons.append("calibration_projection_failed")

                    detections_for_file.append(
                        {
                            "id": int(track_id),
                            "label": model.names[int(cls_idx)],
                            "confidence": round(float(confidence), 3),
                            "x": round(center_x, 2),
                            "y": round(center_y, 2),
                            "relative_depth": relative_depth,
                            "position_kind": position_kind,
                            "ground_coordinate_unit": ground_unit,
                            "position_frame": position_kind,
                        }
                    )

            frame_reference = None
            if tracking_candidates:
                selected_event_reasons.append("periodic_visual_continuity_check")
                try:
                    frame_reference = encode_frame_as_data_url(frame)
                except Exception:
                    frame_reference = None

                did_submit = tracking_runtime.submit_if_needed(
                    now=now,
                    frame_reference=frame_reference,
                    detections=tracking_candidates,
                    event_reasons=selected_event_reasons,
                    request_timeout_seconds=1.0,
                )
                if did_submit:
                    with state.lock:
                        state.tracking_agent_submissions += 1

            tracking_agent_observation = tracking_runtime.poll_latest_observation()
            if tracking_agent_observation is not None:
                with state.lock:
                    current_trails = {
                        key: deque(value, maxlen=RADAR_TRAIL_LENGTH)
                        for key, value in state.track_trails.items()
                    }
                summary = apply_tracking_agent_updates_to_tracks(
                    tracks,
                    tracking_agent_observation,
                    track_trails=current_trails,
                )
                with state.lock:
                    state.tracking_agent_completions += 1
                    state.tracking_agent_last_latency_ms = tracking_agent_observation.proposal_latency_ms
                    state.tracking_agent_total_updates_accepted += summary["accepted"]
                    state.tracking_agent_total_updates_rejected += summary["rejected"]
                    state.tracking_agent_total_updates_unmatched += summary["unmatched"]

                    uncertainty_values = set(tracking_agent_observation.uncertainties)
                    if any(item.startswith("vlm_") for item in uncertainty_values):
                        state.tracking_agent_unavailable_count += 1

                    last_error = None
                    for item in tracking_agent_observation.uncertainties:
                        if isinstance(item, str) and item.startswith("vlm_"):
                            last_error = item
                            break
                    state.tracking_agent_last_error = last_error

            annotated = result.plot()
            with state.lock:
                state.latest_annotated_frame = annotated
                state.frame_height, state.frame_width = annotated.shape[:2]
                state.last_live_frame_time = now
                state.last_observation_time = now
                state.camera_status = "connected"
                state.detector_status = "running"

                state.active_tracks = tracks
                if tracking_agent_observation is not None:
                    state.latest_tracking_agent_observation = tracking_agent_observation.to_dict()
                for track_key, snapshot in tracks.items():
                    if track_key not in state.track_trails:
                        state.track_trails[track_key] = deque(maxlen=RADAR_TRAIL_LENGTH)
                    state.track_trails[track_key].append(
                        (snapshot.ground_position.x, snapshot.ground_position.z)
                    )

            tracking_message = build_tracking_message(now, detections_for_file)
            update_json(LATEST_TRACKING_FRAME_PATH, tracking_message)

            try:
                udp_socket.sendto(
                    json.dumps(tracking_message).encode("utf-8"),
                    (UDP_TRACKING_HOST, UDP_TRACKING_PORT),
                )
            except OSError:
                pass
    finally:
        with state.lock:
            state.camera_status = "stopped"
            state.detector_status = "stopped"
        udp_socket.close()
        tracking_runtime.shutdown()
        camera.release()


def build_simulated_tracks(sim_time: float) -> dict[str, RadarTrackSnapshot]:
    # Builds simulated tracks payloads for pipeline consumers.
    simulated: dict[str, RadarTrackSnapshot] = {}

    sim_tracks = [
        ("sim-1", -4.0 + 0.8 * sim_time, 8.0 + 1.0 * (sim_time % 4.0)),
        ("sim-2", 3.0 + 1.2 * (sim_time % 2.5), 5.5 + 0.8 * sim_time),
    ]

    for track_id, x_position, z_position in sim_tracks:
        wrapped_x = ((x_position + MAX_GROUND_X) % (2 * MAX_GROUND_X)) - MAX_GROUND_X
        wrapped_z = z_position % MAX_GROUND_Z

        simulated[track_id] = RadarTrackSnapshot(
            track_id=track_id,
            timestamp=time.time(),
            pixel_position=None,
            ground_position=GroundPosition(x=wrapped_x, z=wrapped_z),
            ground_coordinate_unit="relative_units",
            position_kind="simulated_ground_track",
            quality_flags=["simulated_data"],
            confidence=None,
            relative_depth=None,
        )

    return simulated


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
            simulated_tracks = build_simulated_tracks(sim_step)
            with state.lock:
                state.mapping_mode = "simulation_fallback"
                state.active_tracks = simulated_tracks
                state.last_observation_time = now
                for track_key, snapshot in simulated_tracks.items():
                    if track_key not in state.track_trails:
                        state.track_trails[track_key] = deque(maxlen=RADAR_TRAIL_LENGTH)
                    state.track_trails[track_key].append(
                        (snapshot.ground_position.x, snapshot.ground_position.z)
                    )
            sim_step += sim_interval
        else:
            with state.lock:
                state.mapping_mode = "relative_image_plane"

        with state.lock:
            snapshot_tracks = list(state.active_tracks.values())
            coordinate_system = state.mapping_mode
            camera_status = state.camera_status
            detector_status = state.detector_status
            calibration_status = state.calibration_status
            last_observation_time = state.last_observation_time
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

        if calibration_status == "loaded" and coordinate_system != "simulation_fallback":
            coordinate_system = "calibrated_ground_plane"

        relative_depth_coverage = compute_relative_depth_coverage(snapshot_tracks)

        update_json(
            LATEST_RADAR_STATE_PATH,
            {
                "timestamp": now,
                "coordinate_system": coordinate_system,
                "camera_status": camera_status,
                "detector_status": detector_status,
                "calibration_status": calibration_status,
                "tracking_agent_status": tracking_agent_status,
                "active_track_count": len(snapshot_tracks),
                "last_observation_timestamp": last_observation_time,
                "relative_depth_coverage": relative_depth_coverage,
                "tracking_agent_observation": state.latest_tracking_agent_observation,
                "tracks": [track.to_dict() for track in snapshot_tracks],
            },
        )

        try:
            append_jsonl(
                RADAR_DEPTH_COVERAGE_HISTORY_PATH,
                {
                    "timestamp": now,
                    "coverage": relative_depth_coverage,
                },
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