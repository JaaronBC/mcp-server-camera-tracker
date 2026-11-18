from __future__ import annotations

import unittest

from agents.contracts import BoundingBox, DetectionCandidate
from agents.dashboard.state import SharedDashboardState
from agents.dashboard.tracking_loop import _promote_and_prune_tracking_prediction_evidence
from agents.tracking.identity_registry import PersistentIdentityRegistry
from agents.tracking.tracking_types import GroundPosition, PixelPosition, RadarTrackSnapshot


class TestTrackingPredictionEvidenceRuntime(unittest.TestCase):
    def _candidate(self, *, detection_id: str, yolo_track_id: str, timestamp: float) -> DetectionCandidate:
        return DetectionCandidate(
            detection_id=detection_id,
            timestamp=timestamp,
            yolo_track_id=yolo_track_id,
            label="person",
            detection_confidence=0.9,
            pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=40.0, y2=80.0),
            pixel_center_x=25.0,
            pixel_center_y=50.0,
            relative_depth=0.3,
            ground_x=1.0,
            ground_z=2.0,
            ground_coordinate_unit="meters",
            position_kind="calibrated_ground_plane",
            motion_history=[],
        )

    def _proposal_item(
        self,
        *,
        track_id: str,
        estimate_id: str,
        predicted_for: float,
        expires_at: float,
        estimated_at: float,
        observation_id: str,
        message_id: str,
    ) -> dict:
        return {
            "proposal": {
                "schema_version": "1.0",
                "estimate_id": estimate_id,
                "track_id": track_id,
                "source_tracking_message_ids": [message_id],
                "source_verification_observation_id": observation_id,
                "source_verification_message_id": message_id,
                "coordinate_system": "calibrated_ground_plane",
                "ground_coordinate_mode": "calibrated_metric",
                "estimated_at": estimated_at,
                "prediction_horizon_seconds": 0.5,
                "predicted_for_timestamp": predicted_for,
                "expires_at": expires_at,
                "uncertainty": ["brief_occlusion"],
                "provenance": {"pixel_center": "predicted", "ground_position": "predicted"},
                "estimated_pixel_center": {"x": 120.0, "y": 80.0},
                "estimated_ground_position": {"x": 2.0, "z": 3.0, "unit": "meters"},
                "estimation_method": "constant_velocity_extrapolation",
            },
            "validation": {
                "valid": True,
                "freshness_status": "fresh",
                "plausibility_status": "plausible",
                "warnings": [],
            },
        }

    def _feedback(self, *, track_id: str, observation_id: str, message_id: str, expires_at: float, visibility: str = "occluded") -> dict:
        return {
            "schema_version": "1.0",
            "feedback_id": f"fb-{message_id}",
            "verification_request_id": f"req-{message_id}",
            "track_id": track_id,
            "evidence_message_id": message_id,
            "verification_observation_id": observation_id,
            "verification_message_id": message_id,
            "verification_timestamp": 1000.0,
            "source": "reasoning_cycle",
            "current_applicability": "current",
            "historical_observation": False,
            "stale_for_current_decisions": False,
            "coordinator_valid": True,
            "validation_outcome": "accepted",
            "submitted_at": 999.8,
            "started_at": 999.9,
            "finished_at": 1000.0,
            "published_at": 1000.1,
            "expires_at": expires_at,
            "provenance": {"publisher": "verification_queue", "source": "reasoning_cycle"},
            "details": {
                "verification_status": "supported",
                "identity_consistency": "consistent",
                "visibility_status": visibility,
                "possible_id_switch": False,
                "confidence": 0.9,
                "visual_content_inspected": True,
                "source_evidence_refs": ["evidence://frame/frame-1"],
                "supporting_evidence_refs": ["evidence://frame/frame-1"],
                "contradicting_evidence_refs": [],
                "uncertainty": ["brief_occlusion"],
                "limitations": ["single_view"],
            },
        }

    def test_valid_promotion_during_temporary_loss_and_no_observed_track_mutation(self) -> None:
        state = SharedDashboardState()
        state.calibration_status = "loaded"
        registry = PersistentIdentityRegistry()
        registry.process_detections(now=1000.0, detections=[self._candidate(detection_id="d1", yolo_track_id="1", timestamp=1000.0)])
        registry.process_detections(now=1000.2, detections=[])
        app_identity_id = registry.get_identity_for_yolo_track("1")
        self.assertIsInstance(app_identity_id, str)

        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.latest_tracking_observation_feedback = {
            "accepted": [self._feedback(track_id="1", observation_id="obs-1", message_id="msg-1", expires_at=1004.0)]
        }
        state.latest_tracking_estimate_proposals = {
            "generated_at": 1000.3,
            "accepted": [
                self._proposal_item(
                    track_id="1",
                    estimate_id="est-1",
                    predicted_for=1000.8,
                    expires_at=1001.2,
                    estimated_at=1000.3,
                    observation_id="obs-1",
                    message_id="msg-1",
                )
            ],
        }

        measured_track = RadarTrackSnapshot(
            track_id="other-track",
            timestamp=1000.3,
            pixel_position=PixelPosition(x=10.0, y=20.0),
            ground_position=GroundPosition(x=0.5, z=1.0),
            ground_coordinate_unit="meters",
            position_kind="calibrated_ground_plane",
            quality_flags=["position_kind:calibrated_ground_plane"],
            confidence=0.7,
            relative_depth=0.2,
        )
        state.active_tracks = {"other-track": measured_track}
        before_track = measured_track.to_dict()

        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.4, tracks={})

        assert isinstance(app_identity_id, str)
        promoted = state.latest_tracking_prediction_evidence_by_identity.get(app_identity_id)
        self.assertIsInstance(promoted, dict)
        assert isinstance(promoted, dict)
        self.assertEqual(promoted.get("classification"), "occlusion_candidate")
        self.assertEqual(promoted.get("source_yolo_track_id"), "1")
        self.assertEqual(state.active_tracks["other-track"].to_dict(), before_track)

    def test_invalid_or_expired_proposals_are_rejected(self) -> None:
        state = SharedDashboardState()
        registry = PersistentIdentityRegistry()
        registry.process_detections(now=1000.0, detections=[self._candidate(detection_id="d1", yolo_track_id="1", timestamp=1000.0)])
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.latest_tracking_observation_feedback = {
            "accepted": [self._feedback(track_id="1", observation_id="obs-1", message_id="msg-1", expires_at=1003.0)]
        }

        expired_item = self._proposal_item(
            track_id="1",
            estimate_id="est-expired",
            predicted_for=1000.8,
            expires_at=999.9,
            estimated_at=1000.3,
            observation_id="obs-1",
            message_id="msg-1",
        )
        expired_item["validation"]["valid"] = True

        malformed_item = self._proposal_item(
            track_id="1",
            estimate_id="est-malformed",
            predicted_for=1000.9,
            expires_at=1001.5,
            estimated_at=1000.4,
            observation_id="obs-1",
            message_id="msg-1",
        )
        malformed_item["proposal"]["estimated_pixel_center"] = None
        malformed_item["proposal"]["estimated_ground_position"] = None

        state.latest_tracking_estimate_proposals = {
            "generated_at": 1000.5,
            "accepted": [expired_item, malformed_item],
        }

        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.6, tracks={})
        self.assertEqual(state.latest_tracking_prediction_evidence_by_identity, {})

    def test_out_of_order_and_reappearance_invalidation(self) -> None:
        state = SharedDashboardState()
        registry = PersistentIdentityRegistry()
        registry.process_detections(now=1000.0, detections=[self._candidate(detection_id="d1", yolo_track_id="1", timestamp=1000.0)])
        registry.process_detections(now=1000.2, detections=[])
        app_identity_id = registry.get_identity_for_yolo_track("1")
        assert isinstance(app_identity_id, str)

        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.latest_tracking_observation_feedback = {
            "accepted": [self._feedback(track_id="1", observation_id="obs-1", message_id="msg-1", expires_at=1004.0)]
        }

        state.latest_tracking_estimate_proposals = {
            "generated_at": 1000.3,
            "accepted": [
                self._proposal_item(
                    track_id="1",
                    estimate_id="est-newer",
                    predicted_for=1001.0,
                    expires_at=1002.0,
                    estimated_at=1000.3,
                    observation_id="obs-1",
                    message_id="msg-1",
                )
            ],
        }
        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.4, tracks={})
        self.assertEqual(
            state.latest_tracking_prediction_evidence_by_identity[app_identity_id]["estimate_id"],
            "est-newer",
        )

        state.latest_tracking_estimate_proposals = {
            "generated_at": 1000.35,
            "accepted": [
                self._proposal_item(
                    track_id="1",
                    estimate_id="est-older",
                    predicted_for=1000.9,
                    expires_at=1002.0,
                    estimated_at=1000.2,
                    observation_id="obs-1",
                    message_id="msg-1",
                )
            ],
        }
        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.45, tracks={})
        self.assertEqual(
            state.latest_tracking_prediction_evidence_by_identity[app_identity_id]["estimate_id"],
            "est-newer",
        )

        state.latest_tracking_estimate_proposals = {"generated_at": 1000.5, "accepted": []}
        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.5, tracks={"1": object()})
        self.assertEqual(state.latest_tracking_prediction_evidence_by_identity, {})

    def test_identity_retirement_transfer_calibration_and_resolution_handling(self) -> None:
        state = SharedDashboardState()
        state.tracking_prediction_evidence_max_records = 2
        state.tracking_prediction_unresolved_max_records = 2

        registry = PersistentIdentityRegistry(temporary_loss_seconds=0.1, retirement_seconds=0.1)
        registry.process_detections(now=1000.0, detections=[self._candidate(detection_id="d1", yolo_track_id="1", timestamp=1000.0)])
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry

        state.latest_tracking_observation_feedback = {
            "accepted": [
                self._feedback(track_id="1", observation_id="obs-1", message_id="msg-1", expires_at=1004.0),
                self._feedback(track_id="2", observation_id="obs-2", message_id="msg-2", expires_at=1004.0),
                self._feedback(track_id="3", observation_id="obs-3", message_id="msg-3", expires_at=1004.0),
            ]
        }
        state.latest_tracking_estimate_proposals = {
            "generated_at": 1000.3,
            "accepted": [
                self._proposal_item(
                    track_id="1",
                    estimate_id="est-1",
                    predicted_for=1001.0,
                    expires_at=1001.5,
                    estimated_at=1000.3,
                    observation_id="obs-1",
                    message_id="msg-1",
                ),
                self._proposal_item(
                    track_id="2",
                    estimate_id="est-2",
                    predicted_for=1001.1,
                    expires_at=1001.6,
                    estimated_at=1000.31,
                    observation_id="obs-2",
                    message_id="msg-2",
                ),
                self._proposal_item(
                    track_id="3",
                    estimate_id="est-3",
                    predicted_for=1001.2,
                    expires_at=1001.7,
                    estimated_at=1000.32,
                    observation_id="obs-3",
                    message_id="msg-3",
                ),
            ],
        }

        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.4, tracks={})
        self.assertEqual(len(state.latest_tracking_prediction_evidence_by_identity), 1)
        self.assertEqual(len(state.latest_tracking_prediction_evidence_unresolved), 2)
        unresolved_reasons = {
            item.get("diagnostic_reason")
            for item in state.latest_tracking_prediction_evidence_unresolved
            if isinstance(item, dict)
        }
        self.assertIn("identity_resolution_unavailable", unresolved_reasons)

        app_identity_id = registry.get_identity_for_yolo_track("1")
        assert isinstance(app_identity_id, str)
        snapshot = registry.snapshot()
        mutation = registry.apply_identity_mutation(
            app_identity_id=app_identity_id,
            new_yolo_track_id="9",
            now=1000.5,
            expected_generation_at_least=int(snapshot["generation"]),
            expected_version=int(snapshot["version"]),
        )
        self.assertTrue(mutation.accepted)

        state.latest_tracking_estimate_proposals = {"generated_at": 1000.6, "accepted": []}
        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.6, tracks={})
        self.assertEqual(state.latest_tracking_prediction_evidence_by_identity, {})

        state.tracking_identity_registry_enabled = False
        state.latest_tracking_estimate_proposals = {
            "generated_at": 1000.7,
            "accepted": [
                self._proposal_item(
                    track_id="1",
                    estimate_id="est-no-registry",
                    predicted_for=1001.3,
                    expires_at=1001.8,
                    estimated_at=1000.7,
                    observation_id="obs-1",
                    message_id="msg-1",
                )
            ],
        }
        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.7, tracks={})
        self.assertEqual(len(state.latest_tracking_prediction_evidence_unresolved), 2)
        self.assertEqual(
            state.latest_tracking_prediction_evidence_unresolved[-1].get("diagnostic_reason"),
            "identity_registry_unavailable",
        )

        state.calibration_status = "invalid"
        state.latest_tracking_estimate_proposals = {"generated_at": 1000.8, "accepted": []}
        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.8, tracks={})
        self.assertEqual(state.latest_tracking_prediction_evidence_by_identity, {})
        self.assertEqual(state.latest_tracking_prediction_evidence_unresolved, [])

        retirement_state = SharedDashboardState()
        retirement_registry = PersistentIdentityRegistry(temporary_loss_seconds=0.1, retirement_seconds=0.1)
        retirement_registry.process_detections(
            now=2000.0,
            detections=[self._candidate(detection_id="d-retire", yolo_track_id="r1", timestamp=2000.0)],
        )
        retirement_registry.process_detections(now=2000.2, detections=[])
        retirement_state.tracking_identity_registry_enabled = True
        retirement_state.tracking_identity_registry = retirement_registry
        retirement_state.latest_tracking_observation_feedback = {
            "accepted": [
                self._feedback(track_id="r1", observation_id="obs-r1", message_id="msg-r1", expires_at=2004.0)
            ]
        }
        retirement_state.latest_tracking_estimate_proposals = {
            "generated_at": 2000.3,
            "accepted": [
                self._proposal_item(
                    track_id="r1",
                    estimate_id="est-r1",
                    predicted_for=2000.8,
                    expires_at=2001.2,
                    estimated_at=2000.3,
                    observation_id="obs-r1",
                    message_id="msg-r1",
                )
            ],
        }
        _promote_and_prune_tracking_prediction_evidence(state=retirement_state, now=2000.35, tracks={})
        self.assertEqual(len(retirement_state.latest_tracking_prediction_evidence_by_identity), 1)

        retirement_registry.process_detections(now=2000.35, detections=[])
        retirement_registry.process_detections(now=2000.55, detections=[])
        retirement_state.latest_tracking_estimate_proposals = {"generated_at": 2000.56, "accepted": []}
        _promote_and_prune_tracking_prediction_evidence(state=retirement_state, now=2000.56, tracks={})
        self.assertEqual(retirement_state.latest_tracking_prediction_evidence_by_identity, {})

    def test_bounded_memory_growth_for_resolved_predictions(self) -> None:
        state = SharedDashboardState()
        state.tracking_prediction_evidence_max_records = 2

        registry = PersistentIdentityRegistry()
        registry.process_detections(
            now=3000.0,
            detections=[
                self._candidate(detection_id="d1", yolo_track_id="1", timestamp=3000.0),
                self._candidate(detection_id="d2", yolo_track_id="2", timestamp=3000.0),
                self._candidate(detection_id="d3", yolo_track_id="3", timestamp=3000.0),
            ],
        )
        registry.process_detections(now=3000.2, detections=[])

        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.latest_tracking_observation_feedback = {
            "accepted": [
                self._feedback(track_id="1", observation_id="obs-1", message_id="msg-1", expires_at=3004.0),
                self._feedback(track_id="2", observation_id="obs-2", message_id="msg-2", expires_at=3004.0),
                self._feedback(track_id="3", observation_id="obs-3", message_id="msg-3", expires_at=3004.0),
            ]
        }
        state.latest_tracking_estimate_proposals = {
            "generated_at": 3000.3,
            "accepted": [
                self._proposal_item(
                    track_id="1",
                    estimate_id="est-1",
                    predicted_for=3000.7,
                    expires_at=3001.3,
                    estimated_at=3000.3,
                    observation_id="obs-1",
                    message_id="msg-1",
                ),
                self._proposal_item(
                    track_id="2",
                    estimate_id="est-2",
                    predicted_for=3000.8,
                    expires_at=3001.4,
                    estimated_at=3000.31,
                    observation_id="obs-2",
                    message_id="msg-2",
                ),
                self._proposal_item(
                    track_id="3",
                    estimate_id="est-3",
                    predicted_for=3000.9,
                    expires_at=3001.5,
                    estimated_at=3000.32,
                    observation_id="obs-3",
                    message_id="msg-3",
                ),
            ],
        }

        _promote_and_prune_tracking_prediction_evidence(state=state, now=3000.35, tracks={})
        self.assertEqual(len(state.latest_tracking_prediction_evidence_by_identity), 2)
        estimate_ids = {
            item.get("estimate_id")
            for item in state.latest_tracking_prediction_evidence_by_identity.values()
            if isinstance(item, dict)
        }
        self.assertEqual(estimate_ids, {"est-2", "est-3"})

    def test_ttl_expiry_and_feedback_expiry_pruning(self) -> None:
        state = SharedDashboardState()
        registry = PersistentIdentityRegistry()
        registry.process_detections(now=1000.0, detections=[self._candidate(detection_id="d1", yolo_track_id="1", timestamp=1000.0)])
        registry.process_detections(now=1000.2, detections=[])

        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.latest_tracking_observation_feedback = {
            "accepted": [self._feedback(track_id="1", observation_id="obs-1", message_id="msg-1", expires_at=1000.75)]
        }
        state.latest_tracking_estimate_proposals = {
            "generated_at": 1000.3,
            "accepted": [
                self._proposal_item(
                    track_id="1",
                    estimate_id="est-ttl",
                    predicted_for=1000.6,
                    expires_at=1000.9,
                    estimated_at=1000.3,
                    observation_id="obs-1",
                    message_id="msg-1",
                )
            ],
        }

        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.4, tracks={})
        self.assertEqual(len(state.latest_tracking_prediction_evidence_by_identity), 1)

        state.latest_tracking_estimate_proposals = {"generated_at": 1000.8, "accepted": []}
        _promote_and_prune_tracking_prediction_evidence(state=state, now=1000.8, tracks={})
        self.assertEqual(state.latest_tracking_prediction_evidence_by_identity, {})


if __name__ == "__main__":
    unittest.main()
