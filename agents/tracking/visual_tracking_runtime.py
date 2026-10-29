# MCP tool endpoints for visual-tracking simulation and diagnostics payloads.
from __future__ import annotations

import asyncio
import time
from typing import Any

from ..contracts import BoundingBox, DetectionCandidate
from ..coordinator import AgentCoordinator
from .agent_tracking import TRACKING_VLM_MODEL, LLMVisualTrackingAgent


def _to_float(value: Any) -> float | None:
    # Implements to float behavior for this module.
    return float(value) if isinstance(value, (int, float)) else None


def _parse_detection_candidates(payload: list[dict[str, Any]]) -> list[DetectionCandidate]:
    # Implements parse detection candidates behavior for this module.
    parsed: list[DetectionCandidate] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        detection_id = item.get("detection_id")
        if not isinstance(detection_id, str) or not detection_id.strip():
            detection_id = f"det-{index}"

        timestamp = _to_float(item.get("timestamp"))
        if timestamp is None:
            timestamp = time.time()

        bbox = item.get("pixel_bbox")
        if not isinstance(bbox, dict):
            continue
        x1 = _to_float(bbox.get("x1"))
        y1 = _to_float(bbox.get("y1"))
        x2 = _to_float(bbox.get("x2"))
        y2 = _to_float(bbox.get("y2"))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue

        parsed.append(
            DetectionCandidate(
                detection_id=detection_id,
                timestamp=timestamp,
                yolo_track_id=(None if item.get("yolo_track_id") is None else str(item.get("yolo_track_id"))),
                label=str(item.get("label", "unknown")),
                detection_confidence=_to_float(item.get("detection_confidence")),
                pixel_bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                pixel_center_x=float(item.get("pixel_center_x", (x1 + x2) / 2.0)),
                pixel_center_y=float(item.get("pixel_center_y", y2)),
                relative_depth=_to_float(item.get("relative_depth")),
                ground_x=_to_float(item.get("ground_x")),
                ground_z=_to_float(item.get("ground_z")),
                ground_coordinate_unit=(None if item.get("ground_coordinate_unit") is None else str(item.get("ground_coordinate_unit"))),
                position_kind=str(item.get("position_kind", "unknown")),
                motion_history=item.get("motion_history") if isinstance(item.get("motion_history"), list) else [],
            )
        )

    return parsed


def simulate_visual_tracking_association(
    detections: list[dict[str, Any]],
    frame_reference: str | None = None,
    event_reasons: list[str] | None = None,
    timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    # Implements simulate visual tracking association behavior for this module.
    """Run visual tracking agent with selected frame + structured detections and return validated updates."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    detection_candidates = _parse_detection_candidates(detections)
    coordinator = AgentCoordinator()
    agent = LLMVisualTrackingAgent(model=TRACKING_VLM_MODEL)

    result = asyncio.run(
        agent.propose_associations(
            frame_reference=frame_reference,
            detections=detection_candidates,
            event_reasons=event_reasons,
            request_timeout_seconds=timeout_seconds,
        )
    )

    observation = coordinator.build_tracking_agent_observation(
        detections=detection_candidates,
        proposals=result.proposals,
        timestamp=time.time(),
        vlm_model=agent.model,
        frame_reference=frame_reference,
        proposal_latency_ms=result.latency_ms,
        additional_uncertainties=result.uncertainty_flags,
    )

    return {
        "available": result.available,
        "error": result.error,
        "vlm_model": agent.model,
        "detection_count": len(detection_candidates),
        "proposal_count": len(result.proposals),
        "observation": observation.to_dict(),
    }
