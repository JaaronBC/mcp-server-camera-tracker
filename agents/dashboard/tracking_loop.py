from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

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
from agents.tracking.estimate_proposals import TrackingEstimateProposalEngine
from agents.tracking.evidence_store import TrackingEvidenceStore
from agents.tracking.frame_transformer import build_tracking_frame_outputs
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer
from agents.tracking.runtime_support import (
    apply_tracking_agent_metric_update_to_state,
    commit_live_tracking_frame_state,
    collect_tracking_runtime_observation_update,
    submit_tracking_runtime_candidates,
)
from agents.tracking.state_publisher import (
    build_tracking_frame_message,
    publish_tracking_frame_message,
    write_json_payload,
)

from .config import (
    GROUND_CALIBRATION_PATH,
    LATEST_TRACKING_ESTIMATE_DIAGNOSTICS_PATH,
    LATEST_TRACKING_FRAME_PATH,
    TRACKING_FEEDBACK_CONSUMER_DECISIONS_PATH,
    MAX_GROUND_X,
    MAX_GROUND_Z,
    TRACKING_MAX_EFFECTIVE_HORIZON_SECONDS,
    RADAR_TRAIL_LENGTH,
    TRACKING_MISSING_TRACK_GRACE_SECONDS,
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
        packaged = evidence_packager.package_detection(
            detection=candidate,
            timestamp=now,
            frame_evidence=frame_evidence,
            update_payload=update_payload,
        )
        if bool(packaged.validation.get("valid", False)):
            with state.lock:
                state.latest_tracking_evidence_message_by_track[packaged.envelope.track_id] = packaged.envelope.message_id


def _consume_tracking_observation_feedback(
    *,
    state: SharedDashboardState,
    now: float,
    tracks: dict,
) -> None:
    with state.lock:
        consumer = state.tracking_feedback_consumer
        current_track_ids = set(tracks.keys())
        bounded_grace_seconds = min(
            10.0,
            max(0.1, float(getattr(state, "tracking_missing_track_grace_seconds", 1.5))),
        )
        state.tracking_missing_track_grace_seconds = bounded_grace_seconds

        lost_message_by_track = state.latest_tracking_lost_evidence_message_by_track
        lost_expires_by_track = state.latest_tracking_lost_track_expires_at_by_track

        for track_id in list(lost_expires_by_track.keys()):
            expires_at = lost_expires_by_track.get(track_id)
            if track_id in current_track_ids or not isinstance(expires_at, (int, float)) or float(expires_at) <= float(now):
                lost_expires_by_track.pop(track_id, None)
                lost_message_by_track.pop(track_id, None)

        active_message_by_track = state.latest_tracking_evidence_message_by_track
        for track_id in list(active_message_by_track.keys()):
            if track_id in current_track_ids:
                lost_expires_by_track.pop(track_id, None)
                lost_message_by_track.pop(track_id, None)
                continue
            message_id = active_message_by_track.pop(track_id, None)
            if not isinstance(message_id, str) or not message_id:
                continue
            lost_message_by_track[track_id] = message_id
            lost_expires_by_track[track_id] = float(now) + bounded_grace_seconds

        temporarily_lost_track_ids = {
            track_id
            for track_id, expires_at in lost_expires_by_track.items()
            if isinstance(expires_at, (int, float)) and float(expires_at) > float(now)
        }

        latest_message_id_by_track = dict(active_message_by_track)
        for track_id in temporarily_lost_track_ids:
            message_id = lost_message_by_track.get(track_id)
            if isinstance(message_id, str) and message_id:
                latest_message_id_by_track[track_id] = message_id

    if consumer is None:
        return

    try:
        feedback_summary = consumer.poll(
            current_track_ids=current_track_ids,
            temporarily_lost_track_ids=temporarily_lost_track_ids,
            latest_message_id_by_track=latest_message_id_by_track,
            now_timestamp=now,
        )
    except Exception:
        feedback_summary = {
            "accepted": [],
            "accepted_count": 0,
            "rejected_count": 0,
            "error": "feedback_poll_failed",
        }

    _append_tracking_feedback_consumer_decisions(
        feedback_summary=feedback_summary,
        now=now,
    )

    with state.lock:
        state.latest_tracking_observation_feedback = feedback_summary


def _generate_tracking_estimate_proposals(
    *,
    state: SharedDashboardState,
    now: float,
    tracks: dict,
) -> None:
    with state.lock:
        estimate_engine = state.tracking_estimate_engine
        feedback_summary = state.latest_tracking_observation_feedback
        tracks_snapshot = dict(tracks)

    if estimate_engine is None:
        return

    accepted_feedback = []
    if isinstance(feedback_summary, dict):
        accepted_payload = feedback_summary.get("accepted")
        if isinstance(accepted_payload, list):
            accepted_feedback = accepted_payload

    try:
        proposal_summary = estimate_engine.generate(
            accepted_feedback_records=accepted_feedback,
            current_tracks=tracks_snapshot,
            now_timestamp=now,
        )
    except Exception:
        proposal_summary = {
            "accepted": [],
            "accepted_count": 0,
            "rejected": [],
            "rejected_count": 0,
            "generated_at": now,
            "error": "estimate_generation_failed",
        }

    with state.lock:
        state.latest_tracking_estimate_proposals = proposal_summary


def _build_tracking_estimate_diagnostic_payload(
    *,
    proposal_summary: dict[str, Any] | None,
    now: float,
    max_current_valid_estimates: int = 32,
) -> dict[str, Any]:
    current_valid_estimates: list[dict[str, Any]] = []
    rejected_or_invalid_count = 0
    expired_count = 0

    accepted_payload = None
    if isinstance(proposal_summary, dict):
        accepted_payload = proposal_summary.get("accepted")

    accepted_items = accepted_payload if isinstance(accepted_payload, list) else []
    for item in accepted_items:
        if not isinstance(item, dict):
            rejected_or_invalid_count += 1
            continue

        proposal = item.get("proposal")
        validation = item.get("validation")
        if not isinstance(proposal, dict) or not isinstance(validation, dict):
            rejected_or_invalid_count += 1
            continue

        if not bool(validation.get("valid", False)):
            rejected_or_invalid_count += 1
            continue

        expires_at = proposal.get("expires_at")
        if not isinstance(expires_at, (int, float)) or float(expires_at) <= float(now):
            expired_count += 1
            continue

        estimated_pixel_position = proposal.get("estimated_pixel_center")
        if not isinstance(estimated_pixel_position, dict):
            estimated_pixel_position = None

        estimated_ground_position = proposal.get("estimated_ground_position")
        if (
            proposal.get("coordinate_system") != "calibrated_ground_plane"
            or not isinstance(estimated_ground_position, dict)
        ):
            estimated_ground_position = None

        source_tracking_message_ids = proposal.get("source_tracking_message_ids")
        if not isinstance(source_tracking_message_ids, list):
            source_tracking_message_ids = []

        current_valid_estimates.append(
            {
                "track_id": proposal.get("track_id"),
                "estimate_id": proposal.get("estimate_id"),
                "estimated_pixel_position": estimated_pixel_position,
                "estimated_ground_position": estimated_ground_position,
                "estimated_at": proposal.get("estimated_at"),
                "predicted_for_timestamp": proposal.get("predicted_for_timestamp"),
                "expires_at": proposal.get("expires_at"),
                "source_tracking_evidence_ids": source_tracking_message_ids,
                "source_verification_observation_id": proposal.get("source_verification_observation_id"),
                "source_verification_message_id": proposal.get("source_verification_message_id"),
                "coordinator_validation": {
                    "valid": True,
                    "freshness_status": validation.get("freshness_status"),
                    "plausibility_status": validation.get("plausibility_status"),
                    "errors": validation.get("errors", []),
                    "warnings": validation.get("warnings", []),
                },
                "classification": "predicted_estimate",
            }
        )

    bounded_estimates = current_valid_estimates[-max(1, int(max_current_valid_estimates)) :]
    return {
        "schema_version": "1.0",
        "generated_at": float(now),
        "classification": "predicted_estimate_diagnostics",
        "current_valid_estimates": bounded_estimates,
        "current_valid_count": len(bounded_estimates),
        "expired_count": expired_count,
        "rejected_or_invalid_count": rejected_or_invalid_count,
    }


def _publish_tracking_estimate_diagnostics(
    *,
    state: SharedDashboardState,
    now: float,
    diagnostics_path: Path = LATEST_TRACKING_ESTIMATE_DIAGNOSTICS_PATH,
) -> None:
    with state.lock:
        proposal_summary = state.latest_tracking_estimate_proposals

    payload = _build_tracking_estimate_diagnostic_payload(
        proposal_summary=proposal_summary,
        now=now,
    )
    write_json_payload(diagnostics_path, payload)


def _resolve_tracking_feedback_run_id() -> str | None:
    value = os.getenv("PHASE16E_RUN_ID", "").strip()
    return value if value else None


def _append_tracking_feedback_consumer_decisions(
    *,
    feedback_summary: dict[str, Any],
    now: float,
    diagnostics_path: Path = TRACKING_FEEDBACK_CONSUMER_DECISIONS_PATH,
) -> None:
    evaluated_payload = feedback_summary.get("evaluated_records") if isinstance(feedback_summary, dict) else None
    evaluated_records = [item for item in evaluated_payload if isinstance(item, dict)] if isinstance(evaluated_payload, list) else []

    outcome_payload = feedback_summary.get("consumer_outcome") if isinstance(feedback_summary, dict) else None
    consumer_outcome = outcome_payload if isinstance(outcome_payload, dict) else {}
    poll_error = feedback_summary.get("error") if isinstance(feedback_summary, dict) else None
    run_id = _resolve_tracking_feedback_run_id()

    rows: list[dict[str, Any]] = []
    if evaluated_records:
        for item in evaluated_records:
            rows.append(
                {
                    "schema_version": "1.0",
                    "logged_at": float(now),
                    "run_id": run_id,
                    "feedback_id": item.get("feedback_id"),
                    "verification_request_id": item.get("verification_request_id"),
                    "track_id": item.get("track_id"),
                    "evidence_message_id": item.get("evidence_message_id"),
                    "verification_observation_id": item.get("verification_observation_id"),
                    "decision": item.get("decision"),
                    "reason": item.get("reason"),
                    "consumer_outcome": consumer_outcome,
                    "poll_error": poll_error,
                }
            )
    elif isinstance(poll_error, str) and poll_error:
        rows.append(
            {
                "schema_version": "1.0",
                "logged_at": float(now),
                "run_id": run_id,
                "feedback_id": None,
                "verification_request_id": None,
                "track_id": None,
                "evidence_message_id": None,
                "verification_observation_id": None,
                "decision": "error",
                "reason": poll_error,
                "consumer_outcome": consumer_outcome,
                "poll_error": poll_error,
            }
        )

    if not rows:
        return

    try:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    except OSError:
        return


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
        feedback_consumer = TrackingObservationFeedbackConsumer(
            artifact_path=CAPTURES_DIR / "operator_tracking_feedback.json",
            max_seen_feedback_ids=1024,
            max_artifact_bytes=512_000,
        )
        estimate_engine = TrackingEstimateProposalEngine(
            coordinator=tracking_coordinator,
            evidence_root_dir=CAPTURES_DIR / "tracking_evidence",
            prediction_horizon_seconds=0.5,
            proposal_ttl_seconds=1.0,
            max_feedback_items=32,
            max_history_points=12,
            max_verification_age_seconds=5.0,
            max_evidence_age_seconds=5.0,
            max_effective_horizon_seconds=min(
                10.0,
                max(0.1, TRACKING_MAX_EFFECTIVE_HORIZON_SECONDS),
            ),
        )
        with state.lock:
            state.tracking_agent_model = tracking_agent.model
            state.tracking_evidence_store = evidence_store
            state.tracking_evidence_packager = evidence_packager
            state.tracking_feedback_consumer = feedback_consumer
            state.tracking_estimate_engine = estimate_engine
            state.tracking_missing_track_grace_seconds = min(
                10.0,
                max(0.1, TRACKING_MISSING_TRACK_GRACE_SECONDS),
            )
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

            _consume_tracking_observation_feedback(
                state=state,
                now=now,
                tracks=tracks,
            )

            _generate_tracking_estimate_proposals(
                state=state,
                now=now,
                tracks=tracks,
            )

            _publish_tracking_estimate_diagnostics(
                state=state,
                now=now,
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
            state.tracking_feedback_consumer = None
            state.tracking_estimate_engine = None
            state.latest_tracking_evidence_message_by_track = {}
            state.latest_tracking_lost_evidence_message_by_track = {}
            state.latest_tracking_lost_track_expires_at_by_track = {}
            state.latest_tracking_estimate_proposals = None
        if evidence_store is not None:
            evidence_store.shutdown()
        try:
            write_json_payload(
                LATEST_TRACKING_ESTIMATE_DIAGNOSTICS_PATH,
                _build_tracking_estimate_diagnostic_payload(
                    proposal_summary=None,
                    now=time.time(),
                ),
            )
        except OSError:
            pass
        udp_socket.close()
        tracking_runtime.shutdown()
        camera.release()
