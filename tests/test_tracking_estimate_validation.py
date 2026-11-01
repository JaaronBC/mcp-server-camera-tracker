from __future__ import annotations

import unittest

from agents.contracts import (
    GroundPoint,
    PixelPosition,
    TrackingClaims,
    TrackingEstimateProposalEnvelope,
    TrackingEvidenceEnvelope,
    TrackingObservationRecord,
    VisualVerificationEnvelope,
)
from agents.coordinator import AgentCoordinator


class TestTrackingEstimateValidation(unittest.TestCase):
    def _tracking_evidence(
        self,
        *,
        message_id: str,
        track_id: str,
        timestamp: float,
        ground_point: GroundPoint | None,
        history_records: list[TrackingObservationRecord],
    ) -> TrackingEvidenceEnvelope:
        return TrackingEvidenceEnvelope(
            schema_version="1.0",
            message_id=message_id,
            track_id=track_id,
            timestamp=timestamp,
            tracking_claims=TrackingClaims(
                claimed_resolved_track_id=track_id,
                association_status="accepted",
                possible_id_switch=False,
                visibility_status="visible",
                claim_reasons=["test"],
            ),
            source_evidence_refs=["evidence://frame/frame-1"],
            ground_position=ground_point,
            track_history=history_records,
            uncertainty=[],
        )

    def _verification(
        self,
        *,
        observation_id: str,
        referenced_track_id: str,
        referenced_message_id: str,
        timestamp: float,
    ) -> VisualVerificationEnvelope:
        return VisualVerificationEnvelope(
            schema_version="1.0",
            observation_id=observation_id,
            referenced_track_id=referenced_track_id,
            referenced_message_id=referenced_message_id,
            timestamp=timestamp,
            source_evidence_refs=["evidence://frame/frame-1"],
            verification_status="supported",
            visibility_status="visible",
            identity_consistency="consistent",
            possible_id_switch=False,
            supporting_evidence_refs=["evidence://frame/frame-1"],
            contradicting_evidence_refs=[],
            confidence=0.9,
            uncertainty=[],
            limitations=[],
            visual_content_inspected=True,
        )

    def _proposal(
        self,
        *,
        estimate_id: str,
        track_id: str,
        source_message_ids: list[str],
        verification_observation_id: str,
        verification_message_id: str,
        coordinate_system: str,
        ground_coordinate_mode: str,
        estimated_at: float,
        horizon_seconds: float,
        predicted_for_timestamp: float,
        expires_at: float,
        uncertainty: list[str],
        estimated_pixel_center: PixelPosition | None,
        estimated_ground_position: GroundPoint | None,
    ) -> TrackingEstimateProposalEnvelope:
        return TrackingEstimateProposalEnvelope(
            schema_version="1.0",
            estimate_id=estimate_id,
            track_id=track_id,
            source_tracking_message_ids=source_message_ids,
            source_verification_observation_id=verification_observation_id,
            source_verification_message_id=verification_message_id,
            coordinate_system=coordinate_system,
            ground_coordinate_mode=ground_coordinate_mode,
            estimated_at=estimated_at,
            prediction_horizon_seconds=horizon_seconds,
            predicted_for_timestamp=predicted_for_timestamp,
            expires_at=expires_at,
            uncertainty=uncertainty,
            provenance={
                "pixel_center": "predicted",
                "ground_position": "predicted",
            },
            estimated_pixel_center=estimated_pixel_center,
            estimated_ground_position=estimated_ground_position,
            estimation_method="history_extrapolation_with_verification_gate",
        )

    def test_valid_pixel_only_estimate(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.5,
        )
        proposal = self._proposal(
            estimate_id="est-1",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="relative_image_plane",
            ground_coordinate_mode="none",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["occlusion"],
            estimated_pixel_center=PixelPosition(x=300.0, y=220.0),
            estimated_ground_position=None,
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
            now_timestamp=1001.1,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.plausibility_status, "not_verifiable")

    def test_valid_calibrated_estimate(self) -> None:
        coordinator = AgentCoordinator()
        history = [
            TrackingObservationRecord(
                observation_id="obs-1",
                track_id="track-1",
                timestamp=1000.0,
                source="tracking_frame",
                pixel_x=100.0,
                pixel_y=200.0,
                ground_point=GroundPoint(x=1.0, z=2.0, unit="meters"),
                confidence=0.8,
                quality_flags=["position_kind:calibrated_ground_plane"],
            )
        ]
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.2,
            ground_point=GroundPoint(x=1.2, z=2.1, unit="meters"),
            history_records=history,
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.4,
        )
        proposal = self._proposal(
            estimate_id="est-2",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="calibrated_ground_plane",
            ground_coordinate_mode="calibrated_metric",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["brief_occlusion"],
            estimated_pixel_center=PixelPosition(x=305.0, y=225.0),
            estimated_ground_position=GroundPoint(x=2.0, z=2.8, unit="meters"),
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
            now_timestamp=1001.2,
            max_plausible_speed_units_per_second=10.0,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.plausibility_status, "plausible")

    def test_unsupported_metric_positions_are_rejected(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.1,
        )
        proposal = self._proposal(
            estimate_id="est-3",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="relative_image_plane",
            ground_coordinate_mode="calibrated_metric",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["low_visibility"],
            estimated_pixel_center=None,
            estimated_ground_position=GroundPoint(x=4.0, z=9.0, unit="meters"),
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
            now_timestamp=1001.1,
        )

        self.assertFalse(result.valid)
        self.assertIn("metric_position_without_calibrated_coordinate_system", result.errors)

    def test_mismatched_evidence_is_rejected(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-2",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.1,
        )
        proposal = self._proposal(
            estimate_id="est-4",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="relative_image_plane",
            ground_coordinate_mode="none",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["ambiguous_identity"],
            estimated_pixel_center=PixelPosition(x=300.0, y=200.0),
            estimated_ground_position=None,
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
            now_timestamp=1001.1,
        )

        self.assertFalse(result.valid)
        self.assertIn("tracking_evidence_track_mismatch", result.errors)

    def test_stale_verification_is_rejected(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=900.0,
        )
        proposal = self._proposal(
            estimate_id="est-5",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="relative_image_plane",
            ground_coordinate_mode="none",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["occluded"],
            estimated_pixel_center=PixelPosition(x=301.0, y=201.0),
            estimated_ground_position=None,
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
            now_timestamp=1001.2,
            max_verification_age_seconds=30.0,
        )

        self.assertFalse(result.valid)
        self.assertIn("stale_verification", result.errors)

    def test_invalid_temporal_ordering_is_rejected(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1001.2,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1001.3,
        )
        proposal = self._proposal(
            estimate_id="est-6",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="relative_image_plane",
            ground_coordinate_mode="none",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["occluded"],
            estimated_pixel_center=PixelPosition(x=302.0, y=202.0),
            estimated_ground_position=None,
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
        )

        self.assertFalse(result.valid)
        self.assertIn("evidence_timestamp_after_estimated_at", result.errors)
        self.assertIn("verification_timestamp_after_estimated_at", result.errors)

    def test_missing_uncertainty_is_rejected(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.1,
        )
        proposal = self._proposal(
            estimate_id="est-7",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="relative_image_plane",
            ground_coordinate_mode="none",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=[],
            estimated_pixel_center=PixelPosition(x=303.0, y=203.0),
            estimated_ground_position=None,
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
        )

        self.assertFalse(result.valid)
        self.assertIn("missing_uncertainty", result.errors)

    def test_implausible_motion_is_rejected(self) -> None:
        coordinator = AgentCoordinator()
        history = [
            TrackingObservationRecord(
                observation_id="obs-1",
                track_id="track-1",
                timestamp=1000.0,
                source="tracking_frame",
                pixel_x=100.0,
                pixel_y=200.0,
                ground_point=GroundPoint(x=0.0, z=0.0, unit="meters"),
                confidence=0.9,
                quality_flags=["position_kind:calibrated_ground_plane"],
            )
        ]
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=GroundPoint(x=0.0, z=0.0, unit="meters"),
            history_records=history,
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.2,
        )
        proposal = self._proposal(
            estimate_id="est-8",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="calibrated_ground_plane",
            ground_coordinate_mode="calibrated_metric",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["occluded"],
            estimated_pixel_center=PixelPosition(x=304.0, y=204.0),
            estimated_ground_position=GroundPoint(x=100.0, z=0.0, unit="meters"),
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
            max_plausible_speed_units_per_second=20.0,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.plausibility_status, "implausible")
        self.assertIn("implausible_motion_estimate", result.errors)

    def test_insufficient_data_for_plausibility_is_not_verifiable(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.1,
        )
        proposal = self._proposal(
            estimate_id="est-9",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="calibrated_ground_plane",
            ground_coordinate_mode="calibrated_metric",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["occluded"],
            estimated_pixel_center=PixelPosition(x=305.0, y=205.0),
            estimated_ground_position=GroundPoint(x=2.0, z=4.0, unit="meters"),
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.plausibility_status, "not_verifiable")
        self.assertIn("plausibility_not_verifiable_insufficient_measurements", result.warnings)

    def test_unsupported_coordinate_system_is_rejected(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.1,
        )
        proposal = self._proposal(
            estimate_id="est-10",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="unsupported_custom_system",
            ground_coordinate_mode="none",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["occluded"],
            estimated_pixel_center=PixelPosition(x=200.0, y=300.0),
            estimated_ground_position=None,
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
        )

        self.assertFalse(result.valid)
        self.assertIn("unsupported_coordinate_system", result.errors)

    def test_untrusted_verification_record_is_rejected(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._tracking_evidence(
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            ground_point=None,
            history_records=[],
        )
        verification = self._verification(
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1000.1,
        )
        verification.source_evidence_refs = ["unknown://ref"]

        proposal = self._proposal(
            estimate_id="est-11",
            track_id="track-1",
            source_message_ids=["msg-1"],
            verification_observation_id="verify-1",
            verification_message_id="msg-1",
            coordinate_system="relative_image_plane",
            ground_coordinate_mode="none",
            estimated_at=1001.0,
            horizon_seconds=0.5,
            predicted_for_timestamp=1001.5,
            expires_at=1002.0,
            uncertainty=["occluded"],
            estimated_pixel_center=PixelPosition(x=210.0, y=310.0),
            estimated_ground_position=None,
        )

        result = coordinator.validate_tracking_estimate_proposal(
            proposal,
            evidence_by_message_id={"msg-1": evidence},
            verification_by_observation_id={"verify-1": verification},
        )

        self.assertFalse(result.valid)
        self.assertIn("untrusted_verification_record", result.errors)
        self.assertIn("verification_trust_errors", result.details)


if __name__ == "__main__":
    unittest.main()
