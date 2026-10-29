from __future__ import annotations

from collections import deque
from typing import Any

from tools.components.calibration import GroundPlaneCalibration, map_image_to_ground
from tools.components.dashboard_types import GroundPosition, Observation, PixelPosition, RadarTrackSnapshot
from tools.components.geometry import image_to_relative_ground
from tools.contracts import BoundingBox, DetectionCandidate
from tools.tracking.runtime_support import (
    build_tracking_detection_id,
    estimate_relative_depth_from_image_geometry,
    normalize_position_metadata,
)


def build_tracking_frame_outputs(
    *,
    boxes: Any,
    model_names: dict[int, str],
    now: float,
    frame_shape: tuple[int, ...],
    calibration: GroundPlaneCalibration | None,
    track_trails: dict[str, deque[tuple[float, float]]],
    max_ground_x: float,
    max_ground_z: float,
) -> tuple[dict[str, RadarTrackSnapshot], list[dict], list[DetectionCandidate], list[str]]:
    tracks: dict[str, RadarTrackSnapshot] = {}
    detections_for_file: list[dict] = []
    tracking_candidates: list[DetectionCandidate] = []
    selected_event_reasons: list[str] = []

    if boxes is None or boxes.id is None:
        return tracks, detections_for_file, tracking_candidates, selected_event_reasons

    coordinates = boxes.xyxy.cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.cpu().tolist()

    frame_height, frame_width = frame_shape[:2]

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
                    max_ground_x,
                    max_ground_z,
                )
                quality_flags.append("calibration_projection_failed")
                quality_flags.append("estimated_ground_from_image")
        else:
            ground_x, image_based_ground_z = image_to_relative_ground(
                center_x,
                center_y,
                frame_width,
                frame_height,
                max_ground_x,
                max_ground_z,
            )
            if relative_depth is not None:
                ground_z = relative_depth * max_ground_z
                quality_flags.append("estimated_ground_from_relative_depth")
            else:
                ground_z = image_based_ground_z
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
        existing_trail = track_trails.get(track_key)
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
                label=model_names[int(cls_idx)],
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
                "label": model_names[int(cls_idx)],
                "confidence": round(float(confidence), 3),
                "x": round(center_x, 2),
                "y": round(center_y, 2),
                "relative_depth": relative_depth,
                "position_kind": position_kind,
                "ground_coordinate_unit": ground_unit,
                "position_frame": position_kind,
            }
        )

    return tracks, detections_for_file, tracking_candidates, selected_event_reasons
