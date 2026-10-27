from __future__ import annotations

import math
from collections import defaultdict, deque

from .agent_contracts import (
    DetectionCandidate,
    EvidenceRequest,
    ObservationBehaviorSummary,
    ReasoningAssessment,
    TrackingObservationRecord,
    TrackingAgentObservation,
    VisualAssociationProposal,
    VisualAssociationUpdate,
)


class AgentCoordinator:
    def __init__(
        self,
        max_history_per_track: int = 50,
        max_evidence_requests_per_track: int = 3,
    ) -> None:
        self._history: dict[str, deque[TrackingObservationRecord]] = defaultdict(
            lambda: deque(maxlen=max_history_per_track)
        )
        self._evidence_request_count: dict[str, int] = defaultdict(int)
        self._max_evidence_requests_per_track = max_evidence_requests_per_track

    def add_tracking_observation(self, observation: TrackingObservationRecord) -> None:
        self._history[observation.track_id].append(observation)

    def get_track_history(self, track_id: str) -> list[TrackingObservationRecord]:
        return list(self._history.get(track_id, []))

    def build_observation_summary(
        self,
        track_id: str,
        created_at: float,
    ) -> ObservationBehaviorSummary:
        history = self.get_track_history(track_id)

        if len(history) < 2:
            return ObservationBehaviorSummary(
                track_id=track_id,
                created_at=created_at,
                sample_count=len(history),
                velocity_x=None,
                velocity_z=None,
                speed=None,
                confidence=None,
                uncertainties=["insufficient_history"],
            )

        first = history[-2]
        second = history[-1]
        if first.ground_point is None or second.ground_point is None:
            return ObservationBehaviorSummary(
                track_id=track_id,
                created_at=created_at,
                sample_count=len(history),
                velocity_x=None,
                velocity_z=None,
                speed=None,
                confidence=None,
                uncertainties=["missing_ground_coordinates"],
            )

        delta_t = second.timestamp - first.timestamp
        if delta_t <= 0:
            return ObservationBehaviorSummary(
                track_id=track_id,
                created_at=created_at,
                sample_count=len(history),
                velocity_x=None,
                velocity_z=None,
                speed=None,
                confidence=None,
                uncertainties=["invalid_timestamps"],
            )

        velocity_x = (second.ground_point.x - first.ground_point.x) / delta_t
        velocity_z = (second.ground_point.z - first.ground_point.z) / delta_t
        speed = math.hypot(velocity_x, velocity_z)

        confidence_values = [
            obs.confidence
            for obs in history[-5:]
            if isinstance(obs.confidence, (int, float))
        ]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        )

        return ObservationBehaviorSummary(
            track_id=track_id,
            created_at=created_at,
            sample_count=len(history),
            velocity_x=velocity_x,
            velocity_z=velocity_z,
            speed=speed,
            confidence=confidence,
            uncertainties=[],
        )

    def request_evidence(
        self,
        track_id: str,
        reason: str,
        requested_at: float,
    ) -> EvidenceRequest | None:
        count = self._evidence_request_count[track_id]
        if count >= self._max_evidence_requests_per_track:
            return None

        next_index = count + 1
        self._evidence_request_count[track_id] = next_index
        return EvidenceRequest(
            track_id=track_id,
            reason=reason,
            requested_at=requested_at,
            request_index=next_index,
        )

    def build_reasoning_assessment(
        self,
        track_id: str,
        created_at: float,
    ) -> ReasoningAssessment:
        summary = self.build_observation_summary(track_id, created_at)
        history = self.get_track_history(track_id)
        observation_ids = [obs.observation_id for obs in history[-5:]]

        if summary.speed is None:
            return ReasoningAssessment(
                track_id=track_id,
                created_at=created_at,
                assessment_label="requires_more_observation",
                summary="Insufficient evidence for motion assessment.",
                supporting_observation_ids=observation_ids,
                uncertainties=summary.uncertainties or ["insufficient_history"],
            )

        if summary.speed > 2.0:
            label = "requires_review"
            text = "Track shows comparatively high planar speed and should be reviewed."
        else:
            label = "stable_motion"
            text = "Track motion appears stable in the current observation window."

        uncertainties = list(summary.uncertainties)
        return ReasoningAssessment(
            track_id=track_id,
            created_at=created_at,
            assessment_label=label,
            summary=text,
            supporting_observation_ids=observation_ids,
            uncertainties=uncertainties,
            score=None,
            score_method=None,
        )

    def validate_visual_association_proposal(
        self,
        detection: DetectionCandidate,
        proposal: VisualAssociationProposal,
    ) -> VisualAssociationUpdate:
        rejection_reasons: list[str] = []
        uncertainties = list(proposal.uncertainties)
        supporting_evidence = list(proposal.supporting_evidence)

        if proposal.detection_id != detection.detection_id:
            rejection_reasons.append("detection_id_mismatch")

        if not proposal.proposed_track_id:
            rejection_reasons.append("missing_proposed_track_id")

        confidence = proposal.association_confidence
        if confidence is not None and confidence < 0.4:
            rejection_reasons.append("association_confidence_too_low")
            uncertainties.append("low_association_confidence")

        if proposal.depth_conflict_flag and detection.relative_depth is not None:
            rejection_reasons.append("depth_conflict_with_reliable_measurement")

        if proposal.position_conflict_flag and detection.position_kind == "calibrated_ground_plane":
            rejection_reasons.append("ground_position_conflict_with_calibrated_mapping")

        if not supporting_evidence:
            uncertainties.append("no_supporting_evidence")

        accepted = len(rejection_reasons) == 0
        if accepted and proposal.proposed_track_id is not None:
            resolved_track_id = proposal.proposed_track_id
        elif detection.yolo_track_id:
            resolved_track_id = detection.yolo_track_id
        else:
            resolved_track_id = f"candidate:{detection.detection_id}"

        return VisualAssociationUpdate(
            detection_id=detection.detection_id,
            resolved_track_id=resolved_track_id,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            uncertainties=uncertainties,
            supporting_evidence=supporting_evidence,
        )

    def build_tracking_agent_observation(
        self,
        detections: list[DetectionCandidate],
        proposals: list[VisualAssociationProposal],
        timestamp: float,
        vlm_model: str,
        frame_reference: str | None,
        proposal_latency_ms: float | None,
        source_agent: str = "visual_tracking",
        additional_uncertainties: list[str] | None = None,
    ) -> TrackingAgentObservation:
        proposal_by_detection = {proposal.detection_id: proposal for proposal in proposals}
        updates: list[VisualAssociationUpdate] = []

        for detection in detections:
            proposal = proposal_by_detection.get(detection.detection_id)
            if proposal is None:
                fallback_track = detection.yolo_track_id or f"candidate:{detection.detection_id}"
                updates.append(
                    VisualAssociationUpdate(
                        detection_id=detection.detection_id,
                        resolved_track_id=fallback_track,
                        accepted=True,
                        rejection_reasons=[],
                        uncertainties=["no_vlm_proposal_used"],
                        supporting_evidence=["yolo_candidate_track"],
                    )
                )
                continue

            updates.append(self.validate_visual_association_proposal(detection, proposal))

        return TrackingAgentObservation(
            timestamp=timestamp,
            source_agent=source_agent,
            vlm_model=vlm_model,
            proposal_latency_ms=proposal_latency_ms,
            frame_reference=frame_reference,
            updates=updates,
            uncertainties=[] if additional_uncertainties is None else list(additional_uncertainties),
        )
