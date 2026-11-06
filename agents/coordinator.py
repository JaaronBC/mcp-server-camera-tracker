# Coordinator logic for track history, deterministic summaries, evidence requests, and proposal validation.
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Literal

from .contracts import (
    DetectionCandidate,
    EvidenceRequest,
    IdentityConsistencyStatus,
    ObservationBehaviorSummary,
    ReasoningAssessment,
    TrackingAssociationStatus,
    TrackingEvidenceEnvelope,
    TrackingEstimateProposalEnvelope,
    TrackingEstimateValidationResult,
    TrackingClaims,
    TrackingObservationRecord,
    TrackingAgentObservation,
    VerificationStatus,
    VisibilityStatus,
    VisualVerificationEnvelope,
    VisualAssociationProposal,
    VisualAssociationUpdate,
)


SUPPORTED_ENVELOPE_SCHEMA_VERSIONS = frozenset({"1.0"})
TRACKING_ASSOCIATION_STATUS_VALUES: set[TrackingAssociationStatus] = {
    "accepted",
    "rejected",
    "fallback",
    "unknown",
}
VISIBILITY_STATUS_VALUES: set[VisibilityStatus] = {
    "visible",
    "partially_occluded",
    "occluded",
    "not_visible",
    "unknown",
}
VERIFICATION_STATUS_VALUES: set[VerificationStatus] = {
    "supported",
    "contradicted",
    "inconclusive",
}
IDENTITY_CONSISTENCY_VALUES: set[IdentityConsistencyStatus] = {
    "consistent",
    "inconsistent",
    "unknown",
}
TRACKING_ESTIMATE_GROUND_MODES = {
    "calibrated_metric",
    "relative_uncalibrated",
    "none",
}
TRACKING_ESTIMATE_COORDINATE_SYSTEMS = {
    "calibrated_ground_plane",
    "relative_image_plane",
    "simulation_fallback",
}
TRACKING_ESTIMATE_PLAUSIBILITY_VALUES = {
    "plausible",
    "implausible",
    "not_verifiable",
}


class AgentCoordinator:
    def __init__(
        self,
        max_history_per_track: int = 50,
        max_evidence_requests_per_track: int = 3,
    ) -> None:
        # Implements init behavior for this module.
        self._history: dict[str, deque[TrackingObservationRecord]] = defaultdict(
            lambda: deque(maxlen=max_history_per_track)
        )
        self._evidence_request_count: dict[str, int] = defaultdict(int)
        self._max_evidence_requests_per_track = max_evidence_requests_per_track

    def add_tracking_observation(self, observation: TrackingObservationRecord) -> None:
        # Implements add tracking observation behavior for this module.
        self._history[observation.track_id].append(observation)

    def get_track_history(self, track_id: str) -> list[TrackingObservationRecord]:
        # Gets track history values for downstream consumers.
        return list(self._history.get(track_id, []))

    def build_observation_summary(
        self,
        track_id: str,
        created_at: float,
    ) -> ObservationBehaviorSummary:
        # Builds observation summary payloads for pipeline consumers.
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
        # Implements request evidence behavior for this module.
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
        # Builds reasoning assessment payloads for pipeline consumers.
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
        # Implements validate visual association proposal behavior for this module.
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

        if (
            isinstance(proposal.detection_reference, str)
            and isinstance(detection.detection_reference, str)
            and proposal.detection_reference.strip()
            and detection.detection_reference.strip()
            and proposal.detection_reference != detection.detection_reference
        ):
            rejection_reasons.append("identity_detection_reference_mismatch")

        identity_resolution_status = (
            proposal.identity_resolution_status.strip()
            if isinstance(proposal.identity_resolution_status, str)
            else None
        )
        if isinstance(identity_resolution_status, str) and identity_resolution_status.startswith("unresolved_"):
            rejection_reasons.append(identity_resolution_status)

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
            detection_reference=proposal.detection_reference,
            proposed_app_identity_id=proposal.proposed_app_identity_id,
            rejected_app_identity_ids=list(proposal.rejected_app_identity_ids),
            identity_evidence_refs=list(proposal.identity_evidence_refs),
            identity_snapshot_generation=proposal.identity_snapshot_generation,
            identity_snapshot_version=proposal.identity_snapshot_version,
            identity_resolution_status=proposal.identity_resolution_status,
        )

    def build_tracking_agent_observation(
        self,
        detections: list[DetectionCandidate],
        proposals: list[VisualAssociationProposal],
        timestamp: float,
        vlm_model: str,
        frame_reference: str | None,
        proposal_latency_ms: float | None,
        observation_id: str | None = None,
        llm_correlation_id: str | None = None,
        source_agent: str = "visual_tracking",
        additional_uncertainties: list[str] | None = None,
    ) -> TrackingAgentObservation:
        # Builds tracking agent observation payloads for pipeline consumers.
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
            observation_id=observation_id,
            llm_correlation_id=llm_correlation_id,
        )

    @staticmethod
    def _is_nonempty_str(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))

    def _validate_tracking_claims(self, tracking_claims: TrackingClaims, errors: list[str]) -> None:
        if tracking_claims.association_status not in TRACKING_ASSOCIATION_STATUS_VALUES:
            errors.append("invalid_association_status")

        if tracking_claims.visibility_status not in VISIBILITY_STATUS_VALUES:
            errors.append("invalid_visibility_status")

        if tracking_claims.claimed_resolved_track_id is not None and not self._is_nonempty_str(
            tracking_claims.claimed_resolved_track_id
        ):
            errors.append("invalid_claimed_resolved_track_id")

        for reason in tracking_claims.claim_reasons:
            if not self._is_nonempty_str(reason):
                errors.append("invalid_claim_reason")
                break

        identity_claim = tracking_claims.identity_claim
        if identity_claim is not None:
            if not isinstance(identity_claim, dict):
                errors.append("invalid_identity_claim_payload")
                return

            claim_id = identity_claim.get("claim_id")
            detection_reference = identity_claim.get("detection_reference")
            proposed_app_identity_id = identity_claim.get("proposed_app_identity_id")
            claim_status = identity_claim.get("claim_status")
            claim_usable = identity_claim.get("claim_usable")

            if not self._is_nonempty_str(claim_id):
                errors.append("invalid_identity_claim_id")
            if not self._is_nonempty_str(detection_reference):
                errors.append("invalid_identity_claim_detection_reference")
            if proposed_app_identity_id is not None and not self._is_nonempty_str(proposed_app_identity_id):
                errors.append("invalid_identity_claim_proposed_app_identity_id")
            if claim_status is not None and not self._is_nonempty_str(claim_status):
                errors.append("invalid_identity_claim_status")
            if claim_usable is not None and not isinstance(claim_usable, bool):
                errors.append("invalid_identity_claim_usable_flag")

    def validate_tracking_evidence_envelope(
        self,
        envelope: TrackingEvidenceEnvelope,
        *,
        supported_schema_versions: set[str] | None = None,
        known_message_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        allowed_versions = (
            SUPPORTED_ENVELOPE_SCHEMA_VERSIONS
            if supported_schema_versions is None
            else supported_schema_versions
        )
        errors: list[str] = []
        warnings: list[str] = []

        if envelope.schema_version not in allowed_versions:
            errors.append("unsupported_schema_version")

        if not self._is_nonempty_str(envelope.message_id):
            errors.append("invalid_message_id")

        if not self._is_nonempty_str(envelope.track_id):
            errors.append("invalid_track_id")

        if known_message_ids is not None and envelope.message_id in known_message_ids:
            errors.append("duplicate_message_id")

        if not self._is_number(envelope.timestamp):
            errors.append("invalid_timestamp")

        if envelope.tracking_confidence is not None:
            if not self._is_number(envelope.tracking_confidence):
                errors.append("invalid_tracking_confidence")
            elif not 0.0 <= float(envelope.tracking_confidence) <= 1.0:
                errors.append("tracking_confidence_out_of_range")

        if envelope.relative_depth is not None:
            if not self._is_number(envelope.relative_depth):
                errors.append("invalid_relative_depth")
            elif not 0.0 <= float(envelope.relative_depth) <= 1.0:
                warnings.append("relative_depth_outside_expected_range")

        if envelope.pixel_bbox is not None:
            if envelope.pixel_bbox.x2 < envelope.pixel_bbox.x1:
                errors.append("invalid_pixel_bbox_x_order")
            if envelope.pixel_bbox.y2 < envelope.pixel_bbox.y1:
                errors.append("invalid_pixel_bbox_y_order")

        if envelope.ground_position is not None and not self._is_nonempty_str(envelope.ground_position.unit):
            errors.append("invalid_ground_position_unit")

        for ref in envelope.source_evidence_refs:
            if not self._is_nonempty_str(ref):
                errors.append("invalid_source_evidence_ref")
                break

        if len(set(envelope.source_evidence_refs)) != len(envelope.source_evidence_refs):
            warnings.append("duplicate_source_evidence_refs")

        track_timestamps = [item.timestamp for item in envelope.track_history]
        if any(not self._is_number(ts) for ts in track_timestamps):
            errors.append("invalid_track_history_timestamp")
        elif any(later < earlier for earlier, later in zip(track_timestamps, track_timestamps[1:])):
            warnings.append("track_history_out_of_order")

        for record in envelope.track_history:
            if not self._is_nonempty_str(record.observation_id):
                errors.append("invalid_track_history_observation_id")
                break
            if not self._is_nonempty_str(record.track_id):
                errors.append("invalid_track_history_track_id")
                break
            if record.track_id != envelope.track_id:
                warnings.append("track_history_track_id_mismatch")
                break

        self._validate_tracking_claims(envelope.tracking_claims, errors)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "message_id": envelope.message_id,
            "track_id": envelope.track_id,
        }

    def validate_visual_verification_envelope(
        self,
        envelope: VisualVerificationEnvelope,
        *,
        evidence_by_message_id: dict[str, TrackingEvidenceEnvelope],
        supported_schema_versions: set[str] | None = None,
        known_observation_ids: set[str] | None = None,
        now_timestamp: float | None = None,
        max_verification_age_seconds: float | None = None,
        max_negative_skew_seconds: float = 0.0,
        max_positive_skew_seconds: float | None = None,
        newer_message_policy: Literal["ignore", "warn", "reject"] = "warn",
    ) -> dict[str, Any]:
        allowed_versions = (
            SUPPORTED_ENVELOPE_SCHEMA_VERSIONS
            if supported_schema_versions is None
            else supported_schema_versions
        )
        errors: list[str] = []
        warnings: list[str] = []

        if envelope.schema_version not in allowed_versions:
            errors.append("unsupported_schema_version")

        if not self._is_nonempty_str(envelope.observation_id):
            errors.append("invalid_observation_id")

        if known_observation_ids is not None and envelope.observation_id in known_observation_ids:
            errors.append("duplicate_observation_id")

        if not self._is_nonempty_str(envelope.referenced_track_id):
            errors.append("invalid_referenced_track_id")

        if not self._is_nonempty_str(envelope.referenced_message_id):
            errors.append("invalid_referenced_message_id")

        if not self._is_number(envelope.timestamp):
            errors.append("invalid_timestamp")

        if envelope.verification_status not in VERIFICATION_STATUS_VALUES:
            errors.append("invalid_verification_status")

        if envelope.visibility_status not in VISIBILITY_STATUS_VALUES:
            errors.append("invalid_visibility_status")

        if envelope.identity_consistency not in IDENTITY_CONSISTENCY_VALUES:
            errors.append("invalid_identity_consistency")

        if envelope.confidence is not None:
            if not self._is_number(envelope.confidence):
                errors.append("invalid_confidence")
            elif not 0.0 <= float(envelope.confidence) <= 1.0:
                errors.append("confidence_out_of_range")

        if envelope.verification_status in {"supported", "contradicted"} and not envelope.source_evidence_refs:
            errors.append("missing_source_evidence_refs_for_conclusive_verification")

        referenced_evidence = evidence_by_message_id.get(envelope.referenced_message_id)
        if referenced_evidence is None:
            errors.append("missing_referenced_tracking_evidence")
        else:
            if referenced_evidence.track_id != envelope.referenced_track_id:
                errors.append("referenced_track_id_mismatch")

            if self._is_number(envelope.timestamp) and self._is_number(referenced_evidence.timestamp):
                if float(envelope.timestamp) + max_negative_skew_seconds < float(referenced_evidence.timestamp):
                    errors.append("verification_timestamp_before_evidence")

                if max_positive_skew_seconds is not None:
                    if float(envelope.timestamp) - float(referenced_evidence.timestamp) > max_positive_skew_seconds:
                        errors.append("verification_timestamp_too_far_after_evidence")

            if newer_message_policy != "ignore":
                newest_timestamp_for_track = max(
                    (
                        evidence.timestamp
                        for evidence in evidence_by_message_id.values()
                        if evidence.track_id == envelope.referenced_track_id
                    ),
                    default=referenced_evidence.timestamp,
                )
                if self._is_number(newest_timestamp_for_track) and newest_timestamp_for_track > referenced_evidence.timestamp:
                    if newer_message_policy == "reject":
                        errors.append("referenced_message_not_latest_for_track")
                    elif newer_message_policy == "warn":
                        warnings.append("referenced_message_not_latest_for_track")

            available_refs = set(referenced_evidence.source_evidence_refs)
            source_ref_set = set(envelope.source_evidence_refs)

            if available_refs:
                if not source_ref_set.issubset(available_refs):
                    errors.append("source_evidence_refs_not_in_referenced_evidence")
            elif source_ref_set:
                errors.append("source_evidence_refs_without_available_reference_context")

            support_set = set(envelope.supporting_evidence_refs)
            contradict_set = set(envelope.contradicting_evidence_refs)
            if not support_set.issubset(source_ref_set):
                errors.append("supporting_refs_not_subset_of_source_refs")
            if not contradict_set.issubset(source_ref_set):
                errors.append("contradicting_refs_not_subset_of_source_refs")
            if support_set & contradict_set:
                errors.append("overlapping_supporting_and_contradicting_refs")

            claim_payload = referenced_evidence.tracking_claims.identity_claim
            if envelope.identity_claim_id is not None:
                if not isinstance(claim_payload, dict):
                    errors.append("identity_claim_missing_in_referenced_tracking_evidence")
                else:
                    if claim_payload.get("claim_id") != envelope.identity_claim_id:
                        errors.append("identity_claim_id_mismatch")

                    claim_detection_reference = claim_payload.get("detection_reference")
                    if (
                        isinstance(envelope.referenced_detection_reference, str)
                        and envelope.referenced_detection_reference.strip()
                        and claim_detection_reference != envelope.referenced_detection_reference
                    ):
                        errors.append("identity_claim_detection_reference_mismatch")

                    claim_proposed_identity = claim_payload.get("proposed_app_identity_id")
                    if (
                        isinstance(envelope.referenced_proposed_app_identity_id, str)
                        and envelope.referenced_proposed_app_identity_id.strip()
                        and claim_proposed_identity != envelope.referenced_proposed_app_identity_id
                    ):
                        errors.append("identity_claim_proposed_identity_mismatch")

                    claim_generation = claim_payload.get("identity_snapshot_generation")
                    claim_version = claim_payload.get("identity_snapshot_version")
                    if (
                        envelope.referenced_identity_snapshot_generation is not None
                        and claim_generation != envelope.referenced_identity_snapshot_generation
                    ):
                        errors.append("identity_claim_snapshot_generation_mismatch")
                    if (
                        envelope.referenced_identity_snapshot_version is not None
                        and claim_version != envelope.referenced_identity_snapshot_version
                    ):
                        errors.append("identity_claim_snapshot_version_mismatch")

                    claim_usable = claim_payload.get("claim_usable")
                    if claim_usable is False:
                        errors.append("identity_claim_marked_unusable")

        if now_timestamp is not None and max_verification_age_seconds is not None:
            if not self._is_number(now_timestamp):
                errors.append("invalid_now_timestamp")
            elif max_verification_age_seconds < 0:
                errors.append("invalid_max_verification_age_seconds")
            elif self._is_number(envelope.timestamp):
                age_seconds = float(now_timestamp) - float(envelope.timestamp)
                if age_seconds > max_verification_age_seconds:
                    errors.append("stale_verification")
                if age_seconds < 0:
                    warnings.append("verification_timestamp_in_future")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "observation_id": envelope.observation_id,
            "referenced_message_id": envelope.referenced_message_id,
            "evidence_reference_validation": {
                "source_refs_present": len(envelope.source_evidence_refs) > 0,
                "supporting_refs_count": len(envelope.supporting_evidence_refs),
                "contradicting_refs_count": len(envelope.contradicting_evidence_refs),
            },
            "visual_content_access_validation": {
                "claimed_inspected": envelope.visual_content_inspected,
                "coordinator_verified_visual_access": False,
                "note": "Coordinator validates references and linkage only; it does not verify image-content inspection.",
            },
        }

    def validate_tracking_estimate_proposal(
        self,
        proposal: TrackingEstimateProposalEnvelope,
        *,
        evidence_by_message_id: dict[str, TrackingEvidenceEnvelope],
        verification_by_observation_id: dict[str, VisualVerificationEnvelope],
        supported_schema_versions: set[str] | None = None,
        now_timestamp: float | None = None,
        max_prediction_horizon_seconds: float = 3.0,
        max_verification_age_seconds: float = 5.0,
        max_estimate_age_seconds: float | None = None,
        max_plausible_speed_units_per_second: float = 25.0,
        temporal_tolerance_seconds: float = 1e-3,
    ) -> TrackingEstimateValidationResult:
        allowed_versions = (
            SUPPORTED_ENVELOPE_SCHEMA_VERSIONS
            if supported_schema_versions is None
            else supported_schema_versions
        )

        errors: list[str] = []
        warnings: list[str] = []
        details: dict[str, Any] = {}

        linked_tracking_message_ids: list[str] = []
        linked_verification_observation_id: str | None = None
        linked_verification_message_id: str | None = None
        freshness_status = "unknown"
        plausibility_status = "not_verifiable"

        if proposal.schema_version not in allowed_versions:
            errors.append("unsupported_schema_version")

        if not self._is_nonempty_str(proposal.estimate_id):
            errors.append("invalid_estimate_id")
        if not self._is_nonempty_str(proposal.track_id):
            errors.append("invalid_track_id")

        if not proposal.uncertainty:
            errors.append("missing_uncertainty")

        if proposal.ground_coordinate_mode not in TRACKING_ESTIMATE_GROUND_MODES:
            errors.append("invalid_ground_coordinate_mode")

        if proposal.coordinate_system not in TRACKING_ESTIMATE_COORDINATE_SYSTEMS:
            errors.append("unsupported_coordinate_system")

        has_pixel_estimate = proposal.estimated_pixel_center is not None
        has_ground_estimate = proposal.estimated_ground_position is not None
        if not has_pixel_estimate and not has_ground_estimate:
            errors.append("missing_estimated_position")

        if proposal.ground_coordinate_mode == "none" and has_ground_estimate:
            errors.append("ground_estimate_present_when_mode_none")
        if proposal.ground_coordinate_mode != "none" and not has_ground_estimate:
            errors.append("missing_ground_estimate_for_mode")

        if has_ground_estimate:
            assert proposal.estimated_ground_position is not None
            if proposal.ground_coordinate_mode == "calibrated_metric":
                if proposal.coordinate_system != "calibrated_ground_plane":
                    errors.append("metric_position_without_calibrated_coordinate_system")
                if proposal.estimated_ground_position.unit == "relative_units":
                    errors.append("metric_position_uses_relative_units")
            if proposal.ground_coordinate_mode == "relative_uncalibrated":
                if proposal.estimated_ground_position.unit != "relative_units":
                    errors.append("relative_uncalibrated_position_requires_relative_units")

        temporal_fields_valid = True
        if not self._is_number(proposal.estimated_at):
            errors.append("invalid_estimated_at")
            temporal_fields_valid = False
        if not self._is_number(proposal.prediction_horizon_seconds):
            errors.append("invalid_prediction_horizon_seconds")
            temporal_fields_valid = False
        if not self._is_number(proposal.predicted_for_timestamp):
            errors.append("invalid_predicted_for_timestamp")
            temporal_fields_valid = False
        if not self._is_number(proposal.expires_at):
            errors.append("invalid_expires_at")
            temporal_fields_valid = False

        if temporal_fields_valid:
            expected_predicted_for = float(proposal.estimated_at) + float(proposal.prediction_horizon_seconds)
            if float(proposal.prediction_horizon_seconds) <= 0:
                errors.append("non_positive_prediction_horizon")
            if float(proposal.prediction_horizon_seconds) > max_prediction_horizon_seconds:
                errors.append("prediction_horizon_exceeds_limit")

            if abs(float(proposal.predicted_for_timestamp) - expected_predicted_for) > temporal_tolerance_seconds:
                errors.append("predicted_for_timestamp_mismatch")

            if float(proposal.predicted_for_timestamp) < float(proposal.estimated_at):
                errors.append("predicted_time_before_estimated_at")

            if float(proposal.expires_at) < float(proposal.predicted_for_timestamp):
                errors.append("expires_before_prediction_target")

        if now_timestamp is not None:
            if not self._is_number(now_timestamp):
                errors.append("invalid_now_timestamp")
                freshness_status = "unknown"
            else:
                if temporal_fields_valid and max_estimate_age_seconds is not None:
                    if max_estimate_age_seconds < 0:
                        errors.append("invalid_max_estimate_age_seconds")
                    else:
                        estimate_age_seconds = float(now_timestamp) - float(proposal.estimated_at)
                        details["estimate_age_seconds"] = estimate_age_seconds
                        if estimate_age_seconds > max_estimate_age_seconds:
                            errors.append("stale_estimate_proposal")
                            freshness_status = "stale"
                        else:
                            freshness_status = "fresh"

                if temporal_fields_valid and float(now_timestamp) > float(proposal.expires_at):
                    errors.append("estimate_expired")
                    freshness_status = "stale"

                if freshness_status == "unknown" and temporal_fields_valid:
                    freshness_status = "fresh" if float(now_timestamp) <= float(proposal.expires_at) else "stale"

        linked_evidence: list[TrackingEvidenceEnvelope] = []
        source_ids = list(proposal.source_tracking_message_ids)
        if not source_ids:
            errors.append("missing_source_tracking_message_ids")
        else:
            for message_id in source_ids:
                if not self._is_nonempty_str(message_id):
                    errors.append("invalid_source_tracking_message_id")
                    continue

                evidence = evidence_by_message_id.get(message_id)
                if evidence is None:
                    errors.append("missing_referenced_tracking_evidence")
                    continue

                linked_tracking_message_ids.append(message_id)
                linked_evidence.append(evidence)

                if evidence.track_id != proposal.track_id:
                    errors.append("tracking_evidence_track_mismatch")

                if temporal_fields_valid and self._is_number(evidence.timestamp):
                    if float(evidence.timestamp) > float(proposal.estimated_at):
                        errors.append("evidence_timestamp_after_estimated_at")
                    if float(evidence.timestamp) > float(proposal.predicted_for_timestamp):
                        errors.append("evidence_timestamp_after_prediction_target")

        if not self._is_nonempty_str(proposal.source_verification_observation_id):
            errors.append("invalid_source_verification_observation_id")
        if not self._is_nonempty_str(proposal.source_verification_message_id):
            errors.append("invalid_source_verification_message_id")

        verification = verification_by_observation_id.get(proposal.source_verification_observation_id)
        if verification is None:
            errors.append("missing_referenced_visual_verification")
        else:
            linked_verification_observation_id = verification.observation_id
            linked_verification_message_id = verification.referenced_message_id

            if verification.referenced_track_id != proposal.track_id:
                errors.append("verification_track_mismatch")

            if verification.referenced_message_id != proposal.source_verification_message_id:
                errors.append("verification_message_mismatch")

            if source_ids and verification.referenced_message_id not in source_ids:
                errors.append("verification_not_linked_to_tracking_evidence")

            if temporal_fields_valid and self._is_number(verification.timestamp):
                if float(verification.timestamp) > float(proposal.estimated_at):
                    errors.append("verification_timestamp_after_estimated_at")
                if float(verification.timestamp) > float(proposal.predicted_for_timestamp):
                    errors.append("verification_timestamp_after_prediction_target")

            if now_timestamp is not None and self._is_number(now_timestamp) and self._is_number(verification.timestamp):
                verification_age_seconds = float(now_timestamp) - float(verification.timestamp)
                details["verification_age_seconds"] = verification_age_seconds
                if max_verification_age_seconds < 0:
                    errors.append("invalid_max_verification_age_seconds")
                elif verification_age_seconds > max_verification_age_seconds:
                    errors.append("stale_verification")
                    freshness_status = "stale"

            verification_trust_result = self.validate_visual_verification_envelope(
                verification,
                evidence_by_message_id=evidence_by_message_id,
                now_timestamp=now_timestamp,
                max_verification_age_seconds=max_verification_age_seconds,
                newer_message_policy="reject",
            )
            if not bool(verification_trust_result.get("valid", False)):
                errors.append("untrusted_verification_record")
                trust_errors = verification_trust_result.get("errors")
                if isinstance(trust_errors, list):
                    details["verification_trust_errors"] = [
                        str(item)
                        for item in trust_errors
                        if isinstance(item, str)
                    ]

        plausibility_details: dict[str, Any] = {}
        if has_ground_estimate and proposal.ground_coordinate_mode == "calibrated_metric":
            assert proposal.estimated_ground_position is not None
            prior_points: list[tuple[float, float, float, str]] = []
            for evidence in linked_evidence:
                if evidence.ground_position is not None and self._is_number(evidence.timestamp):
                    prior_points.append(
                        (
                            float(evidence.timestamp),
                            float(evidence.ground_position.x),
                            float(evidence.ground_position.z),
                            evidence.ground_position.unit,
                        )
                    )
                for history_item in evidence.track_history:
                    if history_item.ground_point is None:
                        continue
                    if not self._is_number(history_item.timestamp):
                        continue
                    prior_points.append(
                        (
                            float(history_item.timestamp),
                            float(history_item.ground_point.x),
                            float(history_item.ground_point.z),
                            history_item.ground_point.unit,
                        )
                    )

            aligned_points = [
                item
                for item in prior_points
                if item[3] == proposal.estimated_ground_position.unit
                and (not temporal_fields_valid or item[0] <= float(proposal.predicted_for_timestamp))
            ]

            if not aligned_points:
                plausibility_status = "not_verifiable"
                warnings.append("plausibility_not_verifiable_insufficient_measurements")
            else:
                aligned_points.sort(key=lambda item: item[0])
                latest = aligned_points[-1]
                reference_timestamp = latest[0]
                if temporal_fields_valid:
                    delta_t = float(proposal.predicted_for_timestamp) - reference_timestamp
                else:
                    delta_t = float(proposal.prediction_horizon_seconds)

                plausibility_details["reference_timestamp"] = reference_timestamp
                plausibility_details["delta_t_seconds"] = delta_t

                if delta_t <= 0:
                    plausibility_status = "not_verifiable"
                    warnings.append("plausibility_not_verifiable_non_positive_time_delta")
                else:
                    delta_x = float(proposal.estimated_ground_position.x) - latest[1]
                    delta_z = float(proposal.estimated_ground_position.z) - latest[2]
                    distance = math.hypot(delta_x, delta_z)
                    required_speed = distance / delta_t
                    plausibility_details["required_speed_units_per_second"] = required_speed
                    if required_speed > max_plausible_speed_units_per_second:
                        plausibility_status = "implausible"
                        errors.append("implausible_motion_estimate")
                    else:
                        plausibility_status = "plausible"
        else:
            plausibility_status = "not_verifiable"
            warnings.append("plausibility_not_verifiable_coordinate_basis")

        details["plausibility"] = plausibility_details

        if plausibility_status not in TRACKING_ESTIMATE_PLAUSIBILITY_VALUES:
            plausibility_status = "not_verifiable"

        return TrackingEstimateValidationResult(
            schema_version=proposal.schema_version,
            estimate_id=proposal.estimate_id,
            track_id=proposal.track_id,
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            plausibility_status=plausibility_status,
            freshness_status=freshness_status,
            linked_tracking_message_ids=linked_tracking_message_ids,
            linked_verification_observation_id=linked_verification_observation_id,
            linked_verification_message_id=linked_verification_message_id,
            details=details,
        )
