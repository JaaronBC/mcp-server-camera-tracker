from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.contracts import (
    GroundPoint,
    PixelPosition,
    TrackingEstimateProposalEnvelope,
    TrackingObservationFeedbackRecord,
    VisualVerificationEnvelope,
)
from agents.coordinator import AgentCoordinator
from agents.observation.evidence_consumption import load_validated_envelope_by_message_id


OCCLUSION_HINTS = ("occlusion", "occluded", "not_visible", "lost", "intermittent")
IDENTITY_RISK_HINTS = ("identity", "id_switch", "track_swap", "misassociation")


@dataclass(slots=True)
class _ValidatedFindings:
    verification_status: str
    identity_consistency: str
    visibility_status: str
    possible_id_switch: bool
    confidence: float | None
    visual_content_inspected: bool | None
    source_evidence_refs: list[str]
    supporting_evidence_refs: list[str]
    contradicting_evidence_refs: list[str]
    uncertainty: list[str]
    limitations: list[str]


class TrackingEstimateProposalEngine:
    def __init__(
        self,
        *,
        coordinator: AgentCoordinator,
        evidence_root_dir: Path,
        prediction_horizon_seconds: float = 0.5,
        proposal_ttl_seconds: float = 1.0,
        max_feedback_items: int = 32,
        max_history_points: int = 12,
        max_verification_age_seconds: float = 5.0,
        max_evidence_age_seconds: float = 5.0,
        max_effective_horizon_seconds: float = 2.0,
    ) -> None:
        self._coordinator = coordinator
        self._evidence_root_dir = evidence_root_dir
        self._prediction_horizon_seconds = max(0.1, float(prediction_horizon_seconds))
        self._proposal_ttl_seconds = max(0.1, float(proposal_ttl_seconds))
        self._max_feedback_items = max(1, int(max_feedback_items))
        self._max_history_points = max(2, int(max_history_points))
        self._max_verification_age_seconds = max(0.1, float(max_verification_age_seconds))
        self._max_evidence_age_seconds = max(0.1, float(max_evidence_age_seconds))
        self._max_effective_horizon_seconds = max(0.1, float(max_effective_horizon_seconds))

    def generate(
        self,
        *,
        accepted_feedback_records: list[dict[str, Any]],
        current_tracks: dict[str, Any],
        now_timestamp: float,
    ) -> dict[str, Any]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        if not isinstance(accepted_feedback_records, list):
            return {
                "accepted": accepted,
                "accepted_count": 0,
                "rejected": rejected,
                "rejected_count": 0,
                "generated_at": float(now_timestamp),
            }

        for payload in accepted_feedback_records[: self._max_feedback_items]:
            if not isinstance(payload, dict):
                rejected.append({"reason": "invalid_feedback_payload"})
                continue

            try:
                feedback = TrackingObservationFeedbackRecord.from_dict(payload)
            except Exception:
                rejected.append({
                    "feedback_id": payload.get("feedback_id"),
                    "reason": "invalid_feedback_record",
                })
                continue

            findings = self._parse_findings(feedback.details)
            if findings is None:
                rejected.append({"feedback_id": feedback.feedback_id, "reason": "invalid_findings"})
                continue

            if not self._is_estimation_needed(feedback=feedback, findings=findings, current_tracks=current_tracks):
                rejected.append({"feedback_id": feedback.feedback_id, "reason": "estimation_not_warranted"})
                continue

            eligibility_error = self._eligibility_failure(findings)
            if eligibility_error is not None:
                rejected.append({"feedback_id": feedback.feedback_id, "reason": eligibility_error})
                continue

            try:
                evidence_record = load_validated_envelope_by_message_id(
                    feedback.evidence_message_id,
                    evidence_root_dir=self._evidence_root_dir,
                )
            except Exception as error:
                rejected.append(
                    {
                        "feedback_id": feedback.feedback_id,
                        "reason": "trusted_evidence_unavailable",
                        "error": str(error),
                    }
                )
                continue

            if (float(now_timestamp) - float(evidence_record.envelope.timestamp)) > self._max_evidence_age_seconds:
                rejected.append({"feedback_id": feedback.feedback_id, "reason": "stale_evidence"})
                continue

            proposal = self._build_proposal(
                feedback=feedback,
                findings=findings,
                now_timestamp=float(now_timestamp),
                evidence_envelope=evidence_record.envelope,
            )
            if proposal is None:
                rejected.append({"feedback_id": feedback.feedback_id, "reason": "insufficient_or_invalid_history_or_horizon"})
                continue

            verification_envelope = VisualVerificationEnvelope(
                schema_version=feedback.schema_version,
                observation_id=feedback.verification_observation_id,
                referenced_track_id=feedback.track_id,
                referenced_message_id=feedback.evidence_message_id,
                timestamp=feedback.verification_timestamp,
                source_evidence_refs=list(findings.source_evidence_refs),
                verification_status=findings.verification_status,
                visibility_status=findings.visibility_status,
                identity_consistency=findings.identity_consistency,
                possible_id_switch=findings.possible_id_switch,
                uncertainty=list(findings.uncertainty),
                limitations=list(findings.limitations),
                supporting_evidence_refs=list(findings.supporting_evidence_refs),
                contradicting_evidence_refs=list(findings.contradicting_evidence_refs),
                confidence=findings.confidence,
                recommended_follow_up=None,
                visual_content_inspected=findings.visual_content_inspected,
            )

            validation = self._coordinator.validate_tracking_estimate_proposal(
                proposal,
                evidence_by_message_id={evidence_record.message_id: evidence_record.envelope},
                verification_by_observation_id={verification_envelope.observation_id: verification_envelope},
                now_timestamp=float(now_timestamp),
                max_verification_age_seconds=self._max_verification_age_seconds,
            )
            if not validation.valid:
                rejected.append(
                    {
                        "feedback_id": feedback.feedback_id,
                        "reason": "coordinator_rejected",
                        "errors": list(validation.errors),
                    }
                )
                continue

            accepted.append(
                {
                    "proposal": proposal.to_dict(),
                    "validation": validation.to_dict(),
                }
            )

        return {
            "accepted": accepted,
            "accepted_count": len(accepted),
            "rejected": rejected,
            "rejected_count": len(rejected),
            "generated_at": float(now_timestamp),
        }

    def _parse_findings(self, payload: Any) -> _ValidatedFindings | None:
        if not isinstance(payload, dict):
            return None

        verification_status = payload.get("verification_status")
        identity_consistency = payload.get("identity_consistency")
        visibility_status = payload.get("visibility_status")
        possible_id_switch = payload.get("possible_id_switch")
        confidence = payload.get("confidence")
        visual_content_inspected = payload.get("visual_content_inspected")

        if verification_status not in {"supported", "contradicted", "inconclusive"}:
            return None
        if identity_consistency not in {"consistent", "inconsistent", "unknown"}:
            return None
        if visibility_status not in {"visible", "partially_occluded", "occluded", "not_visible", "unknown"}:
            return None
        if not isinstance(possible_id_switch, bool):
            return None
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                return None
            confidence_value = float(confidence)
            if not math.isfinite(confidence_value) or confidence_value < 0.0 or confidence_value > 1.0:
                return None
            confidence = confidence_value
        if visual_content_inspected is not None and not isinstance(visual_content_inspected, bool):
            return None

        source_evidence_refs = self._string_list(payload.get("source_evidence_refs"))
        supporting_evidence_refs = self._string_list(payload.get("supporting_evidence_refs"))
        contradicting_evidence_refs = self._string_list(payload.get("contradicting_evidence_refs"))
        uncertainty = self._string_list(payload.get("uncertainty"))
        limitations = self._string_list(payload.get("limitations"))
        if any(item is None for item in [source_evidence_refs, supporting_evidence_refs, contradicting_evidence_refs, uncertainty, limitations]):
            return None

        return _ValidatedFindings(
            verification_status=verification_status,
            identity_consistency=identity_consistency,
            visibility_status=visibility_status,
            possible_id_switch=possible_id_switch,
            confidence=confidence,
            visual_content_inspected=visual_content_inspected,
            source_evidence_refs=source_evidence_refs or [],
            supporting_evidence_refs=supporting_evidence_refs or [],
            contradicting_evidence_refs=contradicting_evidence_refs or [],
            uncertainty=uncertainty or [],
            limitations=limitations or [],
        )

    def _is_estimation_needed(
        self,
        *,
        feedback: TrackingObservationFeedbackRecord,
        findings: _ValidatedFindings,
        current_tracks: dict[str, Any],
    ) -> bool:
        track_missing = feedback.track_id not in current_tracks
        visibility_trigger = findings.visibility_status in {"partially_occluded", "occluded", "not_visible"}
        uncertainty_trigger = any(
            any(hint in item.lower() for hint in OCCLUSION_HINTS)
            for item in findings.uncertainty
        )
        return track_missing or visibility_trigger or uncertainty_trigger

    def _eligibility_failure(self, findings: _ValidatedFindings) -> str | None:
        if not findings.source_evidence_refs:
            return "missing_source_evidence_refs"
        if findings.verification_status != "supported":
            return "verification_not_supported"
        if findings.identity_consistency != "consistent":
            return "identity_not_consistent"
        if findings.possible_id_switch:
            return "possible_id_switch"
        if findings.visual_content_inspected is not True:
            return "visual_content_not_confirmed"
        if findings.contradicting_evidence_refs:
            return "contradicting_evidence_present"
        if any(
            any(hint in item.lower() for hint in IDENTITY_RISK_HINTS)
            for item in [*findings.uncertainty, *findings.limitations]
        ):
            return "identity_risk_uncertainty"
        return None

    def _build_proposal(
        self,
        *,
        feedback: TrackingObservationFeedbackRecord,
        findings: _ValidatedFindings,
        now_timestamp: float,
        evidence_envelope: Any,
    ) -> TrackingEstimateProposalEnvelope | None:
        pixel_points: list[tuple[float, float, float]] = []
        ground_points: list[tuple[float, float, float, str]] = []

        history = list(getattr(evidence_envelope, "track_history", []))[-self._max_history_points :]
        for item in history:
            timestamp = getattr(item, "timestamp", None)
            pixel_x = getattr(item, "pixel_x", None)
            pixel_y = getattr(item, "pixel_y", None)
            if self._is_finite(timestamp) and self._is_finite(pixel_x) and self._is_finite(pixel_y):
                pixel_points.append((float(timestamp), float(pixel_x), float(pixel_y)))

            ground_point = getattr(item, "ground_point", None)
            unit = getattr(ground_point, "unit", None) if ground_point is not None else None
            quality_flags = getattr(item, "quality_flags", [])
            if (
                ground_point is not None
                and isinstance(unit, str)
                and unit != "relative_units"
                and isinstance(quality_flags, list)
                and any("calibrated_ground_plane" in str(flag) for flag in quality_flags)
                and self._is_finite(timestamp)
                and self._is_finite(getattr(ground_point, "x", None))
                and self._is_finite(getattr(ground_point, "z", None))
            ):
                ground_points.append((float(timestamp), float(ground_point.x), float(ground_point.z), unit))

        envelope_timestamp = getattr(evidence_envelope, "timestamp", None)
        envelope_pixel_center = getattr(evidence_envelope, "pixel_center", None)
        if (
            self._is_finite(envelope_timestamp)
            and envelope_pixel_center is not None
            and self._is_finite(getattr(envelope_pixel_center, "x", None))
            and self._is_finite(getattr(envelope_pixel_center, "y", None))
        ):
            pixel_points.append((float(envelope_timestamp), float(envelope_pixel_center.x), float(envelope_pixel_center.y)))

        envelope_ground = getattr(evidence_envelope, "ground_position", None)
        envelope_unit = getattr(envelope_ground, "unit", None) if envelope_ground is not None else None
        if (
            getattr(evidence_envelope, "position_kind", None) == "calibrated_ground_plane"
            and envelope_ground is not None
            and isinstance(envelope_unit, str)
            and envelope_unit != "relative_units"
            and self._is_finite(envelope_timestamp)
            and self._is_finite(getattr(envelope_ground, "x", None))
            and self._is_finite(getattr(envelope_ground, "z", None))
        ):
            ground_points.append((float(envelope_timestamp), float(envelope_ground.x), float(envelope_ground.z), envelope_unit))

        estimated_ground_position: GroundPoint | None = None
        coordinate_system = "relative_image_plane"
        ground_mode = "none"
        estimated_at = float(now_timestamp)
        horizon = float(self._prediction_horizon_seconds)
        predicted_for = estimated_at + horizon

        pixel_prediction = self._extrapolate_pixel(
            pixel_points,
            predicted_for_timestamp=predicted_for,
        )
        if pixel_prediction is None:
            return None

        estimated_pixel_center = PixelPosition(
            x=pixel_prediction[0],
            y=pixel_prediction[1],
        )

        ground_prediction = self._extrapolate_ground(
            ground_points,
            predicted_for_timestamp=predicted_for,
        )
        if ground_prediction is not None:
            estimated_ground_position = GroundPoint(
                x=ground_prediction[0],
                z=ground_prediction[1],
                unit=ground_prediction[2],
            )
            coordinate_system = "calibrated_ground_plane"
            ground_mode = "calibrated_metric"

        expires_at = predicted_for + float(self._proposal_ttl_seconds)

        proposal_uncertainty = ["predicted_position", *findings.uncertainty]
        deduped_uncertainty = []
        for item in proposal_uncertainty:
            value = str(item).strip()
            if value and value not in deduped_uncertainty:
                deduped_uncertainty.append(value)

        return TrackingEstimateProposalEnvelope(
            schema_version=feedback.schema_version,
            estimate_id=f"estimate-{uuid.uuid4().hex}",
            track_id=feedback.track_id,
            source_tracking_message_ids=[feedback.evidence_message_id],
            source_verification_observation_id=feedback.verification_observation_id,
            source_verification_message_id=feedback.verification_message_id,
            coordinate_system=coordinate_system,
            ground_coordinate_mode=ground_mode,
            estimated_at=estimated_at,
            prediction_horizon_seconds=horizon,
            predicted_for_timestamp=predicted_for,
            expires_at=expires_at,
            uncertainty=deduped_uncertainty,
            provenance={
                "prediction_type": "constant_velocity_extrapolation",
                "source": "tracking_estimate_stage_c",
                "is_prediction": "true",
            },
            estimated_pixel_center=estimated_pixel_center,
            estimated_ground_position=estimated_ground_position,
            estimation_method="constant_velocity_extrapolation",
        )

    def _extrapolate_pixel(
        self,
        points: list[tuple[float, float, float]],
        *,
        predicted_for_timestamp: float,
    ) -> tuple[float, float] | None:
        ordered = self._sorted_unique_points(points)
        if len(ordered) < 2:
            return None

        (t1, x1, y1), (t2, x2, y2) = ordered[-2], ordered[-1]
        dt = t2 - t1
        if dt <= 0.0 or not math.isfinite(dt):
            return None

        effective_horizon = float(predicted_for_timestamp) - t2
        if (
            not math.isfinite(effective_horizon)
            or effective_horizon <= 0.0
            or effective_horizon > float(self._max_effective_horizon_seconds)
        ):
            return None

        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        pred_x = x2 + (vx * effective_horizon)
        pred_y = y2 + (vy * effective_horizon)
        if not math.isfinite(pred_x) or not math.isfinite(pred_y):
            return None
        return pred_x, pred_y

    def _extrapolate_ground(
        self,
        points: list[tuple[float, float, float, str]],
        *,
        predicted_for_timestamp: float,
    ) -> tuple[float, float, str] | None:
        if len(points) < 2:
            return None

        latest_unit = points[-1][3]
        same_unit_points = [item for item in points if item[3] == latest_unit]
        ordered = sorted(same_unit_points, key=lambda item: item[0])
        if len(ordered) < 2:
            return None

        (t1, x1, z1, _), (t2, x2, z2, _) = ordered[-2], ordered[-1]
        dt = t2 - t1
        if dt <= 0.0 or not math.isfinite(dt):
            return None

        effective_horizon = float(predicted_for_timestamp) - t2
        if (
            not math.isfinite(effective_horizon)
            or effective_horizon <= 0.0
            or effective_horizon > float(self._max_effective_horizon_seconds)
        ):
            return None

        vx = (x2 - x1) / dt
        vz = (z2 - z1) / dt
        pred_x = x2 + (vx * effective_horizon)
        pred_z = z2 + (vz * effective_horizon)
        if not math.isfinite(pred_x) or not math.isfinite(pred_z):
            return None
        return pred_x, pred_z, latest_unit

    def _sorted_unique_points(self, points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        seen: set[tuple[float, float, float]] = set()
        unique: list[tuple[float, float, float]] = []
        for item in points:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        unique.sort(key=lambda item: item[0])
        return unique

    def _string_list(self, value: Any) -> list[str] | None:
        if value is None:
            return []
        if not isinstance(value, list):
            return None
        output: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return None
            normalized = item.strip()
            if normalized:
                output.append(normalized)
        return output

    def _is_finite(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(float(value))
