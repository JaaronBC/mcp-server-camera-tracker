from __future__ import annotations

from typing import Any

from agents.contracts import TrackingEvidenceEnvelope
from agents.coordinator import AgentCoordinator
from agents.tracking.evidence_store import TrackingEvidenceResolver


def _position_kind_from_envelope(envelope: TrackingEvidenceEnvelope) -> str:
    for item in envelope.quality_flags:
        if isinstance(item, str) and item.startswith("position_kind:"):
            return item.split(":", maxsplit=1)[1]
    return "unknown"


def build_measurement_provenance(envelope: TrackingEvidenceEnvelope) -> dict[str, dict[str, Any]]:
    position_kind = _position_kind_from_envelope(envelope)

    ground_source = "unavailable"
    ground_basis = "none"
    if envelope.ground_position is not None:
        if position_kind == "calibrated_ground_plane":
            ground_source = "calculated"
            ground_basis = "homography_calibration"
        elif position_kind in {"relative_ground_estimate", "simulated_ground_track"}:
            ground_source = "inferred"
            ground_basis = "relative_or_fallback_estimate"
        else:
            ground_source = "inferred"
            ground_basis = "unknown_transform"

    depth_source = "unavailable"
    depth_basis = "none"
    if envelope.relative_depth is not None:
        depth_source = "inferred"
        depth_basis = "image_geometry_estimate"

    tracking_confidence_source = "unavailable"
    if envelope.tracking_confidence is not None:
        tracking_confidence_source = "observed"

    return {
        "pixel_bbox": {
            "source": "observed" if envelope.pixel_bbox is not None else "unavailable",
            "basis": "detector_bbox" if envelope.pixel_bbox is not None else "none",
        },
        "pixel_center": {
            "source": "calculated" if envelope.pixel_center is not None else "unavailable",
            "basis": "bbox_center" if envelope.pixel_center is not None else "none",
        },
        "relative_depth": {
            "source": depth_source,
            "basis": depth_basis,
        },
        "ground_position": {
            "source": ground_source,
            "basis": ground_basis,
            "position_kind": position_kind,
            "unit": None if envelope.ground_position is None else envelope.ground_position.unit,
        },
        "tracking_confidence": {
            "source": tracking_confidence_source,
            "basis": "tracking_runtime_metric" if envelope.tracking_confidence is not None else "none",
        },
    }


def preflight_tracking_evidence_for_observation(
    envelope: TrackingEvidenceEnvelope,
    *,
    resolver: TrackingEvidenceResolver,
    coordinator: AgentCoordinator | None = None,
    now_timestamp: float | None = None,
    require_visual_evidence: bool = True,
) -> dict[str, Any]:
    selected_coordinator = coordinator or AgentCoordinator()
    tracking_validation = selected_coordinator.validate_tracking_evidence_envelope(envelope)

    reference_results: list[dict[str, Any]] = []

    if isinstance(envelope.frame_reference, str) and envelope.frame_reference.strip():
        frame_result = resolver.resolve_reference(
            envelope.frame_reference,
            now_timestamp=now_timestamp,
        )
        reference_results.append(
            {
                "reference": envelope.frame_reference,
                "kind": "frame",
                "result": frame_result,
            }
        )

    for ref in envelope.source_evidence_refs:
        if not isinstance(ref, str) or not ref.strip():
            reference_results.append(
                {
                    "reference": ref,
                    "kind": "unknown",
                    "result": {
                        "valid": False,
                        "status": "invalid_reference_format",
                        "reference": ref,
                    },
                }
            )
            continue

        is_detection_ref = ref.startswith("evidence://detection/")
        result = resolver.resolve_reference(
            ref,
            expected_track_id=(envelope.track_id if is_detection_ref else None),
            expected_detection_id=(envelope.detection_id if is_detection_ref else None),
            expected_frame_reference=(envelope.frame_reference if is_detection_ref else None),
            now_timestamp=now_timestamp,
        )
        reference_results.append(
            {
                "reference": ref,
                "kind": "detection" if is_detection_ref else "source",
                "result": result,
            }
        )

    invalid_results = [item for item in reference_results if not bool(item["result"].get("valid"))]
    pending_results = [
        item for item in invalid_results
        if item["result"].get("status") == "pending_reference"
    ]
    expired_results = [
        item for item in invalid_results
        if item["result"].get("status") == "expired_reference"
    ]
    missing_results = [
        item for item in invalid_results
        if item["result"].get("status") == "missing_reference"
    ]

    all_refs_valid = len(invalid_results) == 0
    has_frame_ref = isinstance(envelope.frame_reference, str) and bool(envelope.frame_reference.strip())

    if require_visual_evidence:
        visual_evidence_inspectable = bool(has_frame_ref and all_refs_valid)
    else:
        visual_evidence_inspectable = bool(all_refs_valid)

    gate_errors: list[str] = []
    if not bool(tracking_validation.get("valid", False)):
        gate_errors.append("invalid_tracking_evidence")
    if pending_results:
        gate_errors.append("pending_visual_evidence")
    if expired_results:
        gate_errors.append("expired_visual_evidence")
    if missing_results:
        gate_errors.append("missing_visual_evidence")
    if require_visual_evidence and not has_frame_ref:
        gate_errors.append("missing_frame_reference")
    if not all_refs_valid:
        gate_errors.append("unresolvable_evidence_references")
    if require_visual_evidence and not visual_evidence_inspectable:
        gate_errors.append("visual_evidence_not_inspectable")

    return {
        "preflight_passed": len(gate_errors) == 0,
        "tracking_validation": tracking_validation,
        "evidence_reference_validation": {
            "reference_results": reference_results,
            "all_references_valid": all_refs_valid,
            "pending_count": len(pending_results),
            "expired_count": len(expired_results),
            "missing_count": len(missing_results),
            "no_substitution_policy": "strict_reference_only",
        },
        "visual_evidence_inspectable": visual_evidence_inspectable,
        "measurement_provenance": build_measurement_provenance(envelope),
        "gate_errors": sorted(set(gate_errors)),
    }
