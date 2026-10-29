# Handles agent contracts behavior.
from __future__ import annotations

from dataclasses import dataclass, field


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
