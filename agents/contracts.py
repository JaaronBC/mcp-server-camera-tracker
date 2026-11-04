# Handles agent contracts behavior.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .tracking.tracking_types import PixelPosition


@dataclass(slots=True)
class GroundPoint:
    x: float
    z: float
    unit: str


@dataclass(slots=True)
class TrackingObservationRecord:
    observation_id: str
    track_id: str
    timestamp: float
    source: str
    pixel_x: float | None
    pixel_y: float | None
    ground_point: GroundPoint | None
    confidence: float | None
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "observation_id": self.observation_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "ground_point": (
                None
                if self.ground_point is None
                else {
                    "x": self.ground_point.x,
                    "z": self.ground_point.z,
                    "unit": self.ground_point.unit,
                }
            ),
            "confidence": self.confidence,
            "quality_flags": self.quality_flags,
        }


@dataclass(slots=True)
class ObservationBehaviorSummary:
    track_id: str
    created_at: float
    sample_count: int
    velocity_x: float | None
    velocity_z: float | None
    speed: float | None
    confidence: float | None
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "track_id": self.track_id,
            "created_at": self.created_at,
            "sample_count": self.sample_count,
            "velocity_x": self.velocity_x,
            "velocity_z": self.velocity_z,
            "speed": self.speed,
            "confidence": self.confidence,
            "uncertainties": self.uncertainties,
        }


@dataclass(slots=True)
class EvidenceRequest:
    track_id: str
    reason: str
    requested_at: float
    request_index: int

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "track_id": self.track_id,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "request_index": self.request_index,
        }


@dataclass(slots=True)
class ReasoningAssessment:
    track_id: str
    created_at: float
    assessment_label: str
    summary: str
    supporting_observation_ids: list[str]
    uncertainties: list[str]
    source_agent: str = "reasoning"
    score: float | None = None
    score_method: str | None = None

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "track_id": self.track_id,
            "created_at": self.created_at,
            "assessment_label": self.assessment_label,
            "summary": self.summary,
            "supporting_observation_ids": self.supporting_observation_ids,
            "uncertainties": self.uncertainties,
            "source_agent": self.source_agent,
            "score": self.score,
            "score_method": self.score_method,
        }


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


@dataclass(slots=True)
class DetectionCandidate:
    detection_id: str
    timestamp: float
    yolo_track_id: str | None
    label: str
    detection_confidence: float | None
    pixel_bbox: BoundingBox
    pixel_center_x: float
    pixel_center_y: float
    relative_depth: float | None
    ground_x: float | None
    ground_z: float | None
    ground_coordinate_unit: str | None
    position_kind: str
    motion_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "detection_id": self.detection_id,
            "timestamp": self.timestamp,
            "yolo_track_id": self.yolo_track_id,
            "label": self.label,
            "detection_confidence": self.detection_confidence,
            "pixel_bbox": self.pixel_bbox.to_dict(),
            "pixel_center_x": self.pixel_center_x,
            "pixel_center_y": self.pixel_center_y,
            "relative_depth": self.relative_depth,
            "ground_x": self.ground_x,
            "ground_z": self.ground_z,
            "ground_coordinate_unit": self.ground_coordinate_unit,
            "position_kind": self.position_kind,
            "motion_history": self.motion_history,
        }


@dataclass(slots=True)
class VisualAssociationProposal:
    detection_id: str
    proposed_track_id: str | None
    association_confidence: float | None
    occlusion_likelihood: float | None
    id_switch_risk: float | None
    depth_conflict_flag: bool
    position_conflict_flag: bool
    uncertainties: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "detection_id": self.detection_id,
            "proposed_track_id": self.proposed_track_id,
            "association_confidence": self.association_confidence,
            "occlusion_likelihood": self.occlusion_likelihood,
            "id_switch_risk": self.id_switch_risk,
            "depth_conflict_flag": self.depth_conflict_flag,
            "position_conflict_flag": self.position_conflict_flag,
            "uncertainties": self.uncertainties,
            "supporting_evidence": self.supporting_evidence,
        }


@dataclass(slots=True)
class VisualAssociationUpdate:
    detection_id: str
    resolved_track_id: str
    accepted: bool
    rejection_reasons: list[str]
    uncertainties: list[str]
    supporting_evidence: list[str]

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "detection_id": self.detection_id,
            "resolved_track_id": self.resolved_track_id,
            "accepted": self.accepted,
            "rejection_reasons": self.rejection_reasons,
            "uncertainties": self.uncertainties,
            "supporting_evidence": self.supporting_evidence,
        }


@dataclass(slots=True)
class TrackingAgentObservation:
    timestamp: float
    source_agent: str
    vlm_model: str
    proposal_latency_ms: float | None
    frame_reference: str | None
    updates: list[VisualAssociationUpdate]
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Performs to dict logic.
        return {
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "vlm_model": self.vlm_model,
            "proposal_latency_ms": self.proposal_latency_ms,
            "frame_reference": self.frame_reference,
            "updates": [update.to_dict() for update in self.updates],
            "uncertainties": self.uncertainties,
        }


TrackingAssociationStatus = Literal["accepted", "rejected", "fallback", "unknown"]
VisibilityStatus = Literal["visible", "partially_occluded", "occluded", "not_visible", "unknown"]
VerificationStatus = Literal["supported", "contradicted", "inconclusive"]
IdentityConsistencyStatus = Literal["consistent", "inconsistent", "unknown"]
TrackingEstimateGroundCoordinateMode = Literal[
    "calibrated_metric",
    "relative_uncalibrated",
    "none",
]
TrackingEstimateCoordinateSystem = Literal[
    "calibrated_ground_plane",
    "relative_image_plane",
    "simulation_fallback",
]
TrackingEstimatePlausibilityStatus = Literal[
    "plausible",
    "implausible",
    "not_verifiable",
]
TrackingFeedbackApplicability = Literal["current", "historical"]


@dataclass(slots=True)
class TrackingClaims:
    claimed_resolved_track_id: str | None
    association_status: TrackingAssociationStatus
    possible_id_switch: bool
    visibility_status: VisibilityStatus
    claim_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claimed_resolved_track_id": self.claimed_resolved_track_id,
            "association_status": self.association_status,
            "possible_id_switch": self.possible_id_switch,
            "visibility_status": self.visibility_status,
            "claim_reasons": self.claim_reasons,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrackingClaims":
        return cls(
            claimed_resolved_track_id=(
                payload.get("claimed_resolved_track_id")
                if isinstance(payload.get("claimed_resolved_track_id"), str)
                else None
            ),
            association_status=payload.get("association_status", "unknown"),
            possible_id_switch=bool(payload.get("possible_id_switch", False)),
            visibility_status=payload.get("visibility_status", "unknown"),
            claim_reasons=[
                str(item)
                for item in payload.get("claim_reasons", [])
                if isinstance(item, str)
            ],
        )


@dataclass(slots=True)
class TrackingEvidenceEnvelope:
    schema_version: str
    message_id: str
    track_id: str
    timestamp: float
    tracking_claims: TrackingClaims
    uncertainty: list[str] = field(default_factory=list)
    frame_id: str | None = None
    detection_id: str | None = None
    pixel_bbox: BoundingBox | None = None
    pixel_center: PixelPosition | None = None
    relative_depth: float | None = None
    ground_position: GroundPoint | None = None
    tracking_confidence: float | None = None
    quality_flags: list[str] = field(default_factory=list)
    track_history: list[TrackingObservationRecord] = field(default_factory=list)
    frame_reference: str | None = None
    source_evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "message_id": self.message_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "detection_id": self.detection_id,
            "pixel_bbox": None if self.pixel_bbox is None else self.pixel_bbox.to_dict(),
            "pixel_center": (
                None
                if self.pixel_center is None
                else {
                    "x": self.pixel_center.x,
                    "y": self.pixel_center.y,
                }
            ),
            "relative_depth": self.relative_depth,
            "ground_position": (
                None
                if self.ground_position is None
                else {
                    "x": self.ground_position.x,
                    "z": self.ground_position.z,
                    "unit": self.ground_position.unit,
                }
            ),
            "tracking_confidence": self.tracking_confidence,
            "quality_flags": self.quality_flags,
            "track_history": [record.to_dict() for record in self.track_history],
            "tracking_claims": self.tracking_claims.to_dict(),
            "frame_reference": self.frame_reference,
            "source_evidence_refs": self.source_evidence_refs,
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrackingEvidenceEnvelope":
        pixel_bbox_payload = payload.get("pixel_bbox")
        pixel_bbox = None
        if isinstance(pixel_bbox_payload, dict):
            x1 = pixel_bbox_payload.get("x1")
            y1 = pixel_bbox_payload.get("y1")
            x2 = pixel_bbox_payload.get("x2")
            y2 = pixel_bbox_payload.get("y2")
            if all(isinstance(item, (int, float)) for item in [x1, y1, x2, y2]):
                pixel_bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

        pixel_center_payload = payload.get("pixel_center")
        pixel_center = None
        if isinstance(pixel_center_payload, dict):
            x_value = pixel_center_payload.get("x")
            y_value = pixel_center_payload.get("y")
            if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
                pixel_center = PixelPosition(x=float(x_value), y=float(y_value))

        ground_payload = payload.get("ground_position")
        ground_position = None
        if isinstance(ground_payload, dict):
            x_value = ground_payload.get("x")
            z_value = ground_payload.get("z")
            unit_value = ground_payload.get("unit")
            if isinstance(x_value, (int, float)) and isinstance(z_value, (int, float)) and isinstance(unit_value, str):
                ground_position = GroundPoint(x=float(x_value), z=float(z_value), unit=unit_value)

        parsed_history: list[TrackingObservationRecord] = []
        history_payload = payload.get("track_history")
        if isinstance(history_payload, list):
            for item in history_payload:
                if not isinstance(item, dict):
                    continue

                observation_id = item.get("observation_id")
                track_id = item.get("track_id")
                timestamp = item.get("timestamp")
                source = item.get("source")
                if not isinstance(observation_id, str):
                    continue
                if not isinstance(track_id, str):
                    continue
                if not isinstance(timestamp, (int, float)):
                    continue
                if not isinstance(source, str):
                    continue

                ground_point_payload = item.get("ground_point")
                ground_point = None
                if isinstance(ground_point_payload, dict):
                    gp_x = ground_point_payload.get("x")
                    gp_z = ground_point_payload.get("z")
                    gp_unit = ground_point_payload.get("unit")
                    if isinstance(gp_x, (int, float)) and isinstance(gp_z, (int, float)) and isinstance(gp_unit, str):
                        ground_point = GroundPoint(x=float(gp_x), z=float(gp_z), unit=gp_unit)

                quality_flags = item.get("quality_flags")
                if not isinstance(quality_flags, list):
                    quality_flags = []

                parsed_history.append(
                    TrackingObservationRecord(
                        observation_id=observation_id,
                        track_id=track_id,
                        timestamp=float(timestamp),
                        source=source,
                        pixel_x=(
                            float(item.get("pixel_x"))
                            if isinstance(item.get("pixel_x"), (int, float))
                            else None
                        ),
                        pixel_y=(
                            float(item.get("pixel_y"))
                            if isinstance(item.get("pixel_y"), (int, float))
                            else None
                        ),
                        ground_point=ground_point,
                        confidence=(
                            float(item.get("confidence"))
                            if isinstance(item.get("confidence"), (int, float))
                            else None
                        ),
                        quality_flags=[str(flag) for flag in quality_flags if isinstance(flag, str)],
                    )
                )

        tracking_claims_payload = payload.get("tracking_claims")
        if isinstance(tracking_claims_payload, dict):
            tracking_claims = TrackingClaims.from_dict(tracking_claims_payload)
        else:
            tracking_claims = TrackingClaims(
                claimed_resolved_track_id=None,
                association_status="unknown",
                possible_id_switch=False,
                visibility_status="unknown",
                claim_reasons=[],
            )

        return cls(
            schema_version=str(payload.get("schema_version", "")),
            message_id=str(payload.get("message_id", "")),
            track_id=str(payload.get("track_id", "")),
            timestamp=float(payload.get("timestamp", 0.0)),
            tracking_claims=tracking_claims,
            uncertainty=[
                str(item)
                for item in payload.get("uncertainty", [])
                if isinstance(item, str)
            ],
            frame_id=(payload.get("frame_id") if isinstance(payload.get("frame_id"), str) else None),
            detection_id=(payload.get("detection_id") if isinstance(payload.get("detection_id"), str) else None),
            pixel_bbox=pixel_bbox,
            pixel_center=pixel_center,
            relative_depth=(
                float(payload.get("relative_depth"))
                if isinstance(payload.get("relative_depth"), (int, float))
                else None
            ),
            ground_position=ground_position,
            tracking_confidence=(
                float(payload.get("tracking_confidence"))
                if isinstance(payload.get("tracking_confidence"), (int, float))
                else None
            ),
            quality_flags=[
                str(item)
                for item in payload.get("quality_flags", [])
                if isinstance(item, str)
            ],
            track_history=parsed_history,
            frame_reference=(
                payload.get("frame_reference")
                if isinstance(payload.get("frame_reference"), str)
                else None
            ),
            source_evidence_refs=[
                str(item)
                for item in payload.get("source_evidence_refs", [])
                if isinstance(item, str)
            ],
        )


@dataclass(slots=True)
class VisualVerificationEnvelope:
    schema_version: str
    observation_id: str
    referenced_track_id: str
    referenced_message_id: str
    timestamp: float
    source_evidence_refs: list[str]
    verification_status: VerificationStatus
    visibility_status: VisibilityStatus
    identity_consistency: IdentityConsistencyStatus
    possible_id_switch: bool
    uncertainty: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    supporting_evidence_refs: list[str] = field(default_factory=list)
    contradicting_evidence_refs: list[str] = field(default_factory=list)
    confidence: float | None = None
    recommended_follow_up: str | None = None
    visual_content_inspected: bool | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "observation_id": self.observation_id,
            "referenced_track_id": self.referenced_track_id,
            "referenced_message_id": self.referenced_message_id,
            "timestamp": self.timestamp,
            "source_evidence_refs": self.source_evidence_refs,
            "verification_status": self.verification_status,
            "visibility_status": self.visibility_status,
            "identity_consistency": self.identity_consistency,
            "possible_id_switch": self.possible_id_switch,
            "supporting_evidence_refs": self.supporting_evidence_refs,
            "contradicting_evidence_refs": self.contradicting_evidence_refs,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "limitations": self.limitations,
            "recommended_follow_up": self.recommended_follow_up,
            "visual_content_inspected": self.visual_content_inspected,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "VisualVerificationEnvelope":
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            observation_id=str(payload.get("observation_id", "")),
            referenced_track_id=str(payload.get("referenced_track_id", "")),
            referenced_message_id=str(payload.get("referenced_message_id", "")),
            timestamp=float(payload.get("timestamp", 0.0)),
            source_evidence_refs=[
                str(item)
                for item in payload.get("source_evidence_refs", [])
                if isinstance(item, str)
            ],
            verification_status=payload.get("verification_status", "inconclusive"),
            visibility_status=payload.get("visibility_status", "unknown"),
            identity_consistency=payload.get("identity_consistency", "unknown"),
            possible_id_switch=bool(payload.get("possible_id_switch", False)),
            uncertainty=[
                str(item)
                for item in payload.get("uncertainty", [])
                if isinstance(item, str)
            ],
            limitations=[
                str(item)
                for item in payload.get("limitations", [])
                if isinstance(item, str)
            ],
            supporting_evidence_refs=[
                str(item)
                for item in payload.get("supporting_evidence_refs", [])
                if isinstance(item, str)
            ],
            contradicting_evidence_refs=[
                str(item)
                for item in payload.get("contradicting_evidence_refs", [])
                if isinstance(item, str)
            ],
            confidence=(
                float(payload.get("confidence"))
                if isinstance(payload.get("confidence"), (int, float))
                else None
            ),
            recommended_follow_up=(
                payload.get("recommended_follow_up")
                if isinstance(payload.get("recommended_follow_up"), str)
                else None
            ),
            visual_content_inspected=(
                payload.get("visual_content_inspected")
                if isinstance(payload.get("visual_content_inspected"), bool)
                else None
            ),
        )


@dataclass(slots=True)
class TrackingEstimateProposalEnvelope:
    schema_version: str
    estimate_id: str
    track_id: str
    source_tracking_message_ids: list[str]
    source_verification_observation_id: str
    source_verification_message_id: str
    coordinate_system: TrackingEstimateCoordinateSystem
    ground_coordinate_mode: TrackingEstimateGroundCoordinateMode
    estimated_at: float
    prediction_horizon_seconds: float
    predicted_for_timestamp: float
    expires_at: float
    uncertainty: list[str]
    provenance: dict[str, str]
    estimated_pixel_center: PixelPosition | None = None
    estimated_ground_position: GroundPoint | None = None
    estimation_method: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "estimate_id": self.estimate_id,
            "track_id": self.track_id,
            "source_tracking_message_ids": self.source_tracking_message_ids,
            "source_verification_observation_id": self.source_verification_observation_id,
            "source_verification_message_id": self.source_verification_message_id,
            "coordinate_system": self.coordinate_system,
            "ground_coordinate_mode": self.ground_coordinate_mode,
            "estimated_at": self.estimated_at,
            "prediction_horizon_seconds": self.prediction_horizon_seconds,
            "predicted_for_timestamp": self.predicted_for_timestamp,
            "expires_at": self.expires_at,
            "uncertainty": self.uncertainty,
            "provenance": self.provenance,
            "estimated_pixel_center": (
                None
                if self.estimated_pixel_center is None
                else {
                    "x": self.estimated_pixel_center.x,
                    "y": self.estimated_pixel_center.y,
                }
            ),
            "estimated_ground_position": (
                None
                if self.estimated_ground_position is None
                else {
                    "x": self.estimated_ground_position.x,
                    "z": self.estimated_ground_position.z,
                    "unit": self.estimated_ground_position.unit,
                }
            ),
            "estimation_method": self.estimation_method,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrackingEstimateProposalEnvelope":
        pixel_center_payload = payload.get("estimated_pixel_center")
        estimated_pixel_center = None
        if isinstance(pixel_center_payload, dict):
            x_value = pixel_center_payload.get("x")
            y_value = pixel_center_payload.get("y")
            if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
                estimated_pixel_center = PixelPosition(x=float(x_value), y=float(y_value))

        ground_payload = payload.get("estimated_ground_position")
        estimated_ground_position = None
        if isinstance(ground_payload, dict):
            x_value = ground_payload.get("x")
            z_value = ground_payload.get("z")
            unit_value = ground_payload.get("unit")
            if isinstance(x_value, (int, float)) and isinstance(z_value, (int, float)) and isinstance(unit_value, str):
                estimated_ground_position = GroundPoint(
                    x=float(x_value),
                    z=float(z_value),
                    unit=unit_value,
                )

        return cls(
            schema_version=str(payload.get("schema_version", "")),
            estimate_id=str(payload.get("estimate_id", "")),
            track_id=str(payload.get("track_id", "")),
            source_tracking_message_ids=[
                str(item)
                for item in payload.get("source_tracking_message_ids", [])
                if isinstance(item, str)
            ],
            source_verification_observation_id=str(payload.get("source_verification_observation_id", "")),
            source_verification_message_id=str(payload.get("source_verification_message_id", "")),
            coordinate_system=str(payload.get("coordinate_system", "")),
            ground_coordinate_mode=payload.get("ground_coordinate_mode", "none"),
            estimated_at=float(payload.get("estimated_at", 0.0)),
            prediction_horizon_seconds=float(payload.get("prediction_horizon_seconds", 0.0)),
            predicted_for_timestamp=float(payload.get("predicted_for_timestamp", 0.0)),
            expires_at=float(payload.get("expires_at", 0.0)),
            uncertainty=[
                str(item)
                for item in payload.get("uncertainty", [])
                if isinstance(item, str)
            ],
            provenance={
                str(key): str(value)
                for key, value in payload.get("provenance", {}).items()
                if isinstance(key, str) and isinstance(value, str)
            } if isinstance(payload.get("provenance"), dict) else {},
            estimated_pixel_center=estimated_pixel_center,
            estimated_ground_position=estimated_ground_position,
            estimation_method=(
                payload.get("estimation_method")
                if isinstance(payload.get("estimation_method"), str)
                else None
            ),
        )


@dataclass(slots=True)
class TrackingEstimateValidationResult:
    schema_version: str
    estimate_id: str
    track_id: str
    valid: bool
    errors: list[str]
    warnings: list[str]
    plausibility_status: TrackingEstimatePlausibilityStatus
    freshness_status: str
    linked_tracking_message_ids: list[str]
    linked_verification_observation_id: str | None
    linked_verification_message_id: str | None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "estimate_id": self.estimate_id,
            "track_id": self.track_id,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "plausibility_status": self.plausibility_status,
            "freshness_status": self.freshness_status,
            "linked_tracking_message_ids": self.linked_tracking_message_ids,
            "linked_verification_observation_id": self.linked_verification_observation_id,
            "linked_verification_message_id": self.linked_verification_message_id,
            "details": self.details,
        }


@dataclass(slots=True)
class TrackingObservationFeedbackRecord:
    schema_version: str
    feedback_id: str
    verification_request_id: str
    track_id: str
    evidence_message_id: str
    verification_observation_id: str
    verification_message_id: str
    verification_timestamp: float
    source: str
    current_applicability: TrackingFeedbackApplicability
    historical_observation: bool
    stale_for_current_decisions: bool
    coordinator_valid: bool
    validation_outcome: str
    submitted_at: float | None
    started_at: float | None
    finished_at: float | None
    published_at: float
    expires_at: float
    provenance: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feedback_id": self.feedback_id,
            "verification_request_id": self.verification_request_id,
            "track_id": self.track_id,
            "evidence_message_id": self.evidence_message_id,
            "verification_observation_id": self.verification_observation_id,
            "verification_message_id": self.verification_message_id,
            "verification_timestamp": self.verification_timestamp,
            "source": self.source,
            "current_applicability": self.current_applicability,
            "historical_observation": self.historical_observation,
            "stale_for_current_decisions": self.stale_for_current_decisions,
            "coordinator_valid": self.coordinator_valid,
            "validation_outcome": self.validation_outcome,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "published_at": self.published_at,
            "expires_at": self.expires_at,
            "provenance": self.provenance,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackingObservationFeedbackRecord":
        provenance_payload = payload.get("provenance")
        details_payload = payload.get("details")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            feedback_id=str(payload.get("feedback_id", "")),
            verification_request_id=str(payload.get("verification_request_id", "")),
            track_id=str(payload.get("track_id", "")),
            evidence_message_id=str(payload.get("evidence_message_id", "")),
            verification_observation_id=str(payload.get("verification_observation_id", "")),
            verification_message_id=str(payload.get("verification_message_id", "")),
            verification_timestamp=float(payload.get("verification_timestamp", 0.0)),
            source=str(payload.get("source", "")),
            current_applicability=str(payload.get("current_applicability", "historical")),
            historical_observation=bool(payload.get("historical_observation", False)),
            stale_for_current_decisions=bool(payload.get("stale_for_current_decisions", False)),
            coordinator_valid=bool(payload.get("coordinator_valid", False)),
            validation_outcome=str(payload.get("validation_outcome", "")),
            submitted_at=(
                float(payload.get("submitted_at"))
                if isinstance(payload.get("submitted_at"), (int, float))
                else None
            ),
            started_at=(
                float(payload.get("started_at"))
                if isinstance(payload.get("started_at"), (int, float))
                else None
            ),
            finished_at=(
                float(payload.get("finished_at"))
                if isinstance(payload.get("finished_at"), (int, float))
                else None
            ),
            published_at=float(payload.get("published_at", 0.0)),
            expires_at=float(payload.get("expires_at", 0.0)),
            provenance={
                str(key): str(value)
                for key, value in provenance_payload.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            if isinstance(provenance_payload, dict)
            else {},
            details={
                str(key): value
                for key, value in details_payload.items()
                if isinstance(key, str)
            }
            if isinstance(details_payload, dict)
            else {},
        )


@dataclass(slots=True)
class TrackingObservationFeedbackArtifact:
    schema_version: str
    generated_at: float
    records: list[TrackingObservationFeedbackRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "records": [item.to_dict() for item in self.records],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackingObservationFeedbackArtifact":
        records_payload = payload.get("records")
        records: list[TrackingObservationFeedbackRecord] = []
        if isinstance(records_payload, list):
            for item in records_payload:
                if not isinstance(item, dict):
                    continue
                try:
                    records.append(TrackingObservationFeedbackRecord.from_dict(item))
                except Exception:
                    continue

        return cls(
            schema_version=str(payload.get("schema_version", "")),
            generated_at=float(payload.get("generated_at", 0.0)),
            records=records,
        )
