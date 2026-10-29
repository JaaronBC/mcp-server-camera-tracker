from __future__ import annotations

from collections import deque
from typing import Any, Callable

from agents.contracts import DetectionCandidate, TrackingAgentObservation
from agents.tracking.tracking_types import GroundPosition, RadarTrackSnapshot

DEFAULT_AGENT_GROUND_BLEND_FACTOR = 0.35
DEFAULT_AGENT_GROUND_MAX_CORRECTION = 1.5


def compute_relative_depth_coverage(
    tracks: list[RadarTrackSnapshot],
) -> dict[str, float | int]:
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
    tracks_with_depth = metrics.get("tracks_with_depth", 0)
    total_tracks = metrics.get("total_tracks", 0)
    coverage_percent = metrics.get("coverage_percent", 0.0)
    return (
        "depth coverage: "
        f"{tracks_with_depth}/{total_tracks} "
        f"({coverage_percent}%)"
    )


def build_tracking_detection_id(track_key: str) -> str:
    return track_key


def normalize_position_metadata(
    position_kind: str,
    ground_coordinate_unit: str,
    quality_flags: list[str],
) -> tuple[str, str, list[str]]:
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
    *,
    ground_blend_factor: float = DEFAULT_AGENT_GROUND_BLEND_FACTOR,
    ground_max_correction: float = DEFAULT_AGENT_GROUND_MAX_CORRECTION,
) -> dict[str, int]:
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
                            ground_max_correction / distance,
                        )
                        blended_x = current_x + (delta_x * ground_blend_factor * bounded_scale)
                        blended_z = current_z + (delta_z * ground_blend_factor * bounded_scale)
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


def summarize_tracking_agent_observation_metrics(
    tracking_agent_observation: TrackingAgentObservation,
    update_summary: dict[str, int],
) -> dict[str, Any]:
    uncertainty_values = set(tracking_agent_observation.uncertainties)
    unavailable_increment = 1 if any(item.startswith("vlm_") for item in uncertainty_values) else 0

    last_error = None
    for item in tracking_agent_observation.uncertainties:
        if isinstance(item, str) and item.startswith("vlm_"):
            last_error = item
            break

    return {
        "completions_increment": 1,
        "last_latency_ms": tracking_agent_observation.proposal_latency_ms,
        "accepted": update_summary.get("accepted", 0),
        "rejected": update_summary.get("rejected", 0),
        "unmatched": update_summary.get("unmatched", 0),
        "unavailable_increment": unavailable_increment,
        "last_error": last_error,
    }


def submit_tracking_runtime_candidates(
    *,
    runtime: Any,
    now: float,
    frame: Any,
    detections: list[DetectionCandidate],
    event_reasons: list[str],
    encode_frame: Callable[[Any], str],
    request_timeout_seconds: float = 1.0,
) -> bool:
    if not detections:
        return False

    submission_reasons = [*event_reasons, "periodic_visual_continuity_check"]

    frame_reference = None
    try:
        frame_reference = encode_frame(frame)
    except Exception:
        frame_reference = None

    return runtime.submit_if_needed(
        now=now,
        frame_reference=frame_reference,
        detections=detections,
        event_reasons=submission_reasons,
        request_timeout_seconds=request_timeout_seconds,
    )


def apply_tracking_observation_update(
    tracks: dict[str, RadarTrackSnapshot],
    tracking_agent_observation: TrackingAgentObservation,
    track_trails: dict[str, deque[tuple[float, float]]],
) -> dict[str, Any]:
    summary = apply_tracking_agent_updates_to_tracks(
        tracks,
        tracking_agent_observation,
        track_trails=track_trails,
    )
    return summarize_tracking_agent_observation_metrics(
        tracking_agent_observation,
        summary,
    )


def clone_track_trail_snapshot(
    track_trails: dict[str, deque[tuple[float, float]]],
    trail_length: int,
) -> dict[str, deque[tuple[float, float]]]:
    return {
        key: deque(value, maxlen=trail_length)
        for key, value in track_trails.items()
    }


def collect_tracking_runtime_observation_update(
    *,
    runtime: Any,
    tracks: dict[str, RadarTrackSnapshot],
    track_trails: dict[str, deque[tuple[float, float]]],
    trail_length: int,
) -> tuple[TrackingAgentObservation | None, dict[str, Any] | None]:
    tracking_agent_observation = runtime.poll_latest_observation()
    if tracking_agent_observation is None:
        return None, None

    trail_snapshot = clone_track_trail_snapshot(track_trails, trail_length)
    metric_update = apply_tracking_observation_update(
        tracks,
        tracking_agent_observation,
        trail_snapshot,
    )
    return tracking_agent_observation, metric_update


def commit_live_tracking_frame_state(
    *,
    state: Any,
    now: float,
    annotated_frame: Any,
    tracks: dict[str, RadarTrackSnapshot],
    tracking_agent_observation: TrackingAgentObservation | None,
    trail_length: int,
) -> None:
    with state.lock:
        state.latest_annotated_frame = annotated_frame
        state.frame_height, state.frame_width = annotated_frame.shape[:2]
        state.last_live_frame_time = now
        state.last_observation_time = now
        state.camera_status = "connected"
        state.detector_status = "running"

        state.active_tracks = tracks
        if tracking_agent_observation is not None:
            state.latest_tracking_agent_observation = tracking_agent_observation.to_dict()
        for track_key, snapshot in tracks.items():
            if track_key not in state.track_trails:
                state.track_trails[track_key] = deque(maxlen=trail_length)
            state.track_trails[track_key].append(
                (snapshot.ground_position.x, snapshot.ground_position.z)
            )


def apply_tracking_agent_metric_update_to_state(
    *,
    state: Any,
    metric_update: dict[str, Any],
) -> None:
    with state.lock:
        state.tracking_agent_completions += metric_update["completions_increment"]
        state.tracking_agent_last_latency_ms = metric_update["last_latency_ms"]
        state.tracking_agent_total_updates_accepted += metric_update["accepted"]
        state.tracking_agent_total_updates_rejected += metric_update["rejected"]
        state.tracking_agent_total_updates_unmatched += metric_update["unmatched"]
        state.tracking_agent_unavailable_count += metric_update["unavailable_increment"]
        state.tracking_agent_last_error = metric_update["last_error"]
