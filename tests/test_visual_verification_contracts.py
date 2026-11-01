from __future__ import annotations

import unittest

from agents.contracts import (
    BoundingBox,
    GroundPoint,
    TrackingClaims,
    TrackingEvidenceEnvelope,
    TrackingObservationRecord,
    VisualVerificationEnvelope,
)
from agents.coordinator import AgentCoordinator
from agents.tracking.tracking_types import PixelPosition


class TestVisualVerificationContracts(unittest.TestCase):
    def _history_record(self) -> TrackingObservationRecord:
        return TrackingObservationRecord(
            observation_id="obs-1",
            track_id="track-1",
            timestamp=1000.0,
            source="radar_state",
            pixel_x=100.0,
            pixel_y=200.0,
            ground_point=GroundPoint(x=1.0, z=2.0, unit="meters"),
            confidence=0.9,
            quality_flags=["ground_plane_homography"],
        )

    def _tracking_claims(self) -> TrackingClaims:
        return TrackingClaims(
            claimed_resolved_track_id="track-1",
            association_status="accepted",
            possible_id_switch=False,
            visibility_status="visible",
            claim_reasons=["visual_continuity"],
        )

    def _valid_tracking_evidence(self) -> TrackingEvidenceEnvelope:
        return TrackingEvidenceEnvelope(
            schema_version="1.0",
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            frame_id="frame-1",
            detection_id="det-1",
            pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
            pixel_center=PixelPosition(x=20.0, y=40.0),
            relative_depth=0.4,
            ground_position=GroundPoint(x=1.0, z=2.0, unit="meters"),
            tracking_confidence=0.75,
            quality_flags=["ground_plane_homography"],
            track_history=[self._history_record()],
            tracking_claims=self._tracking_claims(),
            frame_reference="frame://capture-1",
            source_evidence_refs=["frame://capture-1", "crop://track-1"],
            uncertainty=["brief_occlusion_risk"],
        )

    def _valid_verification(self) -> VisualVerificationEnvelope:
        return VisualVerificationEnvelope(
            schema_version="1.0",
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1001.0,
            source_evidence_refs=["frame://capture-1"],
            verification_status="supported",
            visibility_status="visible",
            identity_consistency="consistent",
            possible_id_switch=False,
            supporting_evidence_refs=["frame://capture-1"],
            contradicting_evidence_refs=[],
            confidence=0.8,
            uncertainty=[],
            limitations=[],
            recommended_follow_up=None,
            visual_content_inspected=True,
        )

    def test_tracking_evidence_round_trip_serialization(self) -> None:
        envelope = self._valid_tracking_evidence()

        payload = envelope.to_dict()
        restored = TrackingEvidenceEnvelope.from_dict(payload)

        self.assertEqual(restored.schema_version, "1.0")
        self.assertEqual(restored.message_id, "msg-1")
        self.assertEqual(restored.track_id, "track-1")
        self.assertEqual(restored.tracking_claims.association_status, "accepted")
        self.assertEqual(len(restored.track_history), 1)

    def test_visual_verification_round_trip_serialization(self) -> None:
        envelope = self._valid_verification()

        payload = envelope.to_dict()
        restored = VisualVerificationEnvelope.from_dict(payload)

        self.assertEqual(restored.referenced_message_id, "msg-1")
        self.assertEqual(restored.verification_status, "supported")
        self.assertEqual(restored.visibility_status, "visible")

    def test_tracking_evidence_valid_minimal_message(self) -> None:
        coordinator = AgentCoordinator()
        envelope = TrackingEvidenceEnvelope(
            schema_version="1.0",
            message_id="msg-min",
            track_id="track-1",
            timestamp=1000.0,
            tracking_claims=TrackingClaims(
                claimed_resolved_track_id=None,
                association_status="unknown",
                possible_id_switch=False,
                visibility_status="unknown",
                claim_reasons=[],
            ),
            uncertainty=[],
        )

        result = coordinator.validate_tracking_evidence_envelope(envelope)

        self.assertTrue(result["valid"])

    def test_tracking_evidence_invalid_required_like_fields(self) -> None:
        coordinator = AgentCoordinator()
        envelope = self._valid_tracking_evidence()
        envelope.message_id = ""

        result = coordinator.validate_tracking_evidence_envelope(envelope)

        self.assertFalse(result["valid"])
        self.assertIn("invalid_message_id", result["errors"])

    def test_tracking_evidence_invalid_coordinates(self) -> None:
        coordinator = AgentCoordinator()
        envelope = self._valid_tracking_evidence()
        envelope.pixel_bbox = BoundingBox(x1=5.0, y1=10.0, x2=4.0, y2=8.0)

        result = coordinator.validate_tracking_evidence_envelope(envelope)

        self.assertFalse(result["valid"])
        self.assertIn("invalid_pixel_bbox_x_order", result["errors"])
        self.assertIn("invalid_pixel_bbox_y_order", result["errors"])

    def test_tracking_evidence_invalid_confidence_and_schema(self) -> None:
        coordinator = AgentCoordinator()
        envelope = self._valid_tracking_evidence()
        envelope.schema_version = "2.0"
        envelope.tracking_confidence = 1.5

        result = coordinator.validate_tracking_evidence_envelope(envelope)

        self.assertFalse(result["valid"])
        self.assertIn("unsupported_schema_version", result["errors"])
        self.assertIn("tracking_confidence_out_of_range", result["errors"])

    def test_tracking_evidence_missing_optional_frame_reference_is_valid(self) -> None:
        coordinator = AgentCoordinator()
        envelope = self._valid_tracking_evidence()
        envelope.frame_reference = None

        result = coordinator.validate_tracking_evidence_envelope(envelope)

        self.assertTrue(result["valid"])

    def test_visual_verification_supported_contradicted_and_inconclusive(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._valid_tracking_evidence()
        context = {evidence.message_id: evidence}

        supported = self._valid_verification()
        contradicted = self._valid_verification()
        contradicted.observation_id = "verify-2"
        contradicted.verification_status = "contradicted"
        contradicted.identity_consistency = "inconsistent"
        contradicted.possible_id_switch = True
        contradicted.supporting_evidence_refs = []
        contradicted.contradicting_evidence_refs = ["frame://capture-1"]

        inconclusive = self._valid_verification()
        inconclusive.observation_id = "verify-3"
        inconclusive.verification_status = "inconclusive"
        inconclusive.visibility_status = "occluded"
        inconclusive.identity_consistency = "unknown"
        inconclusive.source_evidence_refs = []
        inconclusive.supporting_evidence_refs = []
        inconclusive.contradicting_evidence_refs = []
        inconclusive.visual_content_inspected = False

        self.assertTrue(coordinator.validate_visual_verification_envelope(supported, evidence_by_message_id=context)["valid"])
        self.assertTrue(coordinator.validate_visual_verification_envelope(contradicted, evidence_by_message_id=context)["valid"])
        self.assertTrue(coordinator.validate_visual_verification_envelope(inconclusive, evidence_by_message_id=context)["valid"])

    def test_visual_verification_missing_or_unknown_references(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._valid_tracking_evidence()
        context = {evidence.message_id: evidence}

        verification = self._valid_verification()
        verification.source_evidence_refs = ["unknown://ref"]

        result = coordinator.validate_visual_verification_envelope(
            verification,
            evidence_by_message_id=context,
        )

        self.assertFalse(result["valid"])
        self.assertIn("source_evidence_refs_not_in_referenced_evidence", result["errors"])

    def test_visual_verification_unknown_message_and_track_mismatch(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._valid_tracking_evidence()
        context = {evidence.message_id: evidence}

        verification_unknown = self._valid_verification()
        verification_unknown.referenced_message_id = "missing-msg"

        result_unknown = coordinator.validate_visual_verification_envelope(
            verification_unknown,
            evidence_by_message_id=context,
        )
        self.assertFalse(result_unknown["valid"])
        self.assertIn("missing_referenced_tracking_evidence", result_unknown["errors"])

        verification_mismatch = self._valid_verification()
        verification_mismatch.referenced_track_id = "track-2"

        result_mismatch = coordinator.validate_visual_verification_envelope(
            verification_mismatch,
            evidence_by_message_id=context,
        )
        self.assertFalse(result_mismatch["valid"])
        self.assertIn("referenced_track_id_mismatch", result_mismatch["errors"])

    def test_visual_verification_invalid_status_confidence_and_ref_conflicts(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._valid_tracking_evidence()
        context = {evidence.message_id: evidence}

        verification = self._valid_verification()
        verification.verification_status = "bad-status"  # type: ignore[assignment]
        verification.visibility_status = "bad-visibility"  # type: ignore[assignment]
        verification.identity_consistency = "bad-identity"  # type: ignore[assignment]
        verification.confidence = 1.2
        verification.supporting_evidence_refs = ["frame://capture-1"]
        verification.contradicting_evidence_refs = ["frame://capture-1"]

        result = coordinator.validate_visual_verification_envelope(
            verification,
            evidence_by_message_id=context,
        )

        self.assertFalse(result["valid"])
        self.assertIn("invalid_verification_status", result["errors"])
        self.assertIn("invalid_visibility_status", result["errors"])
        self.assertIn("invalid_identity_consistency", result["errors"])
        self.assertIn("confidence_out_of_range", result["errors"])
        self.assertIn("overlapping_supporting_and_contradicting_refs", result["errors"])

    def test_visual_verification_stale_out_of_order_and_duplicate(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._valid_tracking_evidence()
        context = {evidence.message_id: evidence}

        verification = self._valid_verification()
        verification.timestamp = 995.0

        result = coordinator.validate_visual_verification_envelope(
            verification,
            evidence_by_message_id=context,
            known_observation_ids={"verify-1"},
            now_timestamp=1010.0,
            max_verification_age_seconds=5.0,
            max_negative_skew_seconds=0.0,
        )

        self.assertFalse(result["valid"])
        self.assertIn("duplicate_observation_id", result["errors"])
        self.assertIn("verification_timestamp_before_evidence", result["errors"])
        self.assertIn("stale_verification", result["errors"])

    def test_newer_message_policy_configurable(self) -> None:
        coordinator = AgentCoordinator()

        older = self._valid_tracking_evidence()
        older.message_id = "msg-older"
        older.timestamp = 1000.0

        newer = self._valid_tracking_evidence()
        newer.message_id = "msg-newer"
        newer.timestamp = 1005.0

        verification = self._valid_verification()
        verification.referenced_message_id = "msg-older"

        context = {"msg-older": older, "msg-newer": newer}

        warn_result = coordinator.validate_visual_verification_envelope(
            verification,
            evidence_by_message_id=context,
            newer_message_policy="warn",
        )
        self.assertTrue(warn_result["valid"])
        self.assertIn("referenced_message_not_latest_for_track", warn_result["warnings"])

        reject_result = coordinator.validate_visual_verification_envelope(
            verification,
            evidence_by_message_id=context,
            newer_message_policy="reject",
        )
        self.assertFalse(reject_result["valid"])
        self.assertIn("referenced_message_not_latest_for_track", reject_result["errors"])

    def test_validation_is_pure_and_does_not_mutate_tracking_state(self) -> None:
        coordinator = AgentCoordinator()
        coordinator.add_tracking_observation(self._history_record())
        before_history = coordinator.get_track_history("track-1")

        evidence = self._valid_tracking_evidence()
        verification = self._valid_verification()

        coordinator.validate_tracking_evidence_envelope(evidence)
        coordinator.validate_visual_verification_envelope(
            verification,
            evidence_by_message_id={evidence.message_id: evidence},
        )

        after_history = coordinator.get_track_history("track-1")
        self.assertEqual(len(before_history), len(after_history))
        self.assertEqual(before_history[0].observation_id, after_history[0].observation_id)

    def test_visual_access_validation_distinct_from_reference_validation(self) -> None:
        coordinator = AgentCoordinator()
        evidence = self._valid_tracking_evidence()
        verification = self._valid_verification()

        result = coordinator.validate_visual_verification_envelope(
            verification,
            evidence_by_message_id={evidence.message_id: evidence},
        )

        self.assertTrue(result["valid"])
        self.assertTrue(result["evidence_reference_validation"]["source_refs_present"])
        self.assertFalse(result["visual_content_access_validation"]["coordinator_verified_visual_access"])


if __name__ == "__main__":
    unittest.main()
