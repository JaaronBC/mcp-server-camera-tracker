from __future__ import annotations

import unittest

from tools.agent_contracts import (
    BoundingBox,
    DetectionCandidate,
    GroundPoint,
    TrackingObservationRecord,
    VisualAssociationProposal,
)
from tools.agent_coordinator import AgentCoordinator


class TestAgentCoordinator(unittest.TestCase):
    def _obs(
        self,
        observation_id: str,
        track_id: str,
        timestamp: float,
        x: float,
        z: float,
        confidence: float = 0.9,
    ) -> TrackingObservationRecord:
        return TrackingObservationRecord(
            observation_id=observation_id,
            track_id=track_id,
            timestamp=timestamp,
            source="camera",
            pixel_x=None,
            pixel_y=None,
            ground_point=GroundPoint(x=x, z=z, unit="meters"),
            confidence=confidence,
            quality_flags=[],
        )

    def test_summary_insufficient_history(self) -> None:
        coordinator = AgentCoordinator()
        coordinator.add_tracking_observation(self._obs("o1", "t1", 1.0, 0.0, 0.0))

        summary = coordinator.build_observation_summary("t1", created_at=2.0)

        self.assertIsNone(summary.speed)
        self.assertIn("insufficient_history", summary.uncertainties)

    def test_summary_velocity_computation(self) -> None:
        coordinator = AgentCoordinator()
        coordinator.add_tracking_observation(self._obs("o1", "t1", 1.0, 0.0, 0.0))
        coordinator.add_tracking_observation(self._obs("o2", "t1", 2.0, 2.0, 1.0))

        summary = coordinator.build_observation_summary("t1", created_at=2.5)

        self.assertAlmostEqual(summary.velocity_x or 0.0, 2.0, places=6)
        self.assertAlmostEqual(summary.velocity_z or 0.0, 1.0, places=6)
        self.assertAlmostEqual(summary.speed or 0.0, (5 ** 0.5), places=6)

    def test_evidence_request_is_bounded(self) -> None:
        coordinator = AgentCoordinator(max_evidence_requests_per_track=2)

        req1 = coordinator.request_evidence("t1", "need_more_frames", requested_at=1.0)
        req2 = coordinator.request_evidence("t1", "still_uncertain", requested_at=2.0)
        req3 = coordinator.request_evidence("t1", "extra", requested_at=3.0)

        self.assertIsNotNone(req1)
        self.assertIsNotNone(req2)
        self.assertIsNone(req3)

    def test_reasoning_assessment_requires_more_observation(self) -> None:
        coordinator = AgentCoordinator()
        coordinator.add_tracking_observation(self._obs("o1", "t1", 1.0, 0.0, 0.0))

        assessment = coordinator.build_reasoning_assessment("t1", created_at=2.0)

        self.assertEqual(assessment.assessment_label, "requires_more_observation")
        self.assertIn("o1", assessment.supporting_observation_ids)

    def test_reasoning_assessment_requires_review_for_fast_motion(self) -> None:
        coordinator = AgentCoordinator()
        coordinator.add_tracking_observation(self._obs("o1", "t1", 1.0, 0.0, 0.0))
        coordinator.add_tracking_observation(self._obs("o2", "t1", 2.0, 3.0, 0.0))

        assessment = coordinator.build_reasoning_assessment("t1", created_at=2.1)

        self.assertEqual(assessment.assessment_label, "requires_review")
        self.assertIn("o2", assessment.supporting_observation_ids)

    def test_visual_proposal_conflict_uses_yolo_fallback(self) -> None:
        coordinator = AgentCoordinator()
        detection = DetectionCandidate(
            detection_id="det-1",
            timestamp=10.0,
            yolo_track_id="yolo-3",
            label="person",
            detection_confidence=0.8,
            pixel_bbox=BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0),
            pixel_center_x=2.0,
            pixel_center_y=4.0,
            relative_depth=3.0,
            ground_x=0.5,
            ground_z=1.2,
            ground_coordinate_unit="meters",
            position_kind="calibrated_ground_plane",
            motion_history=[],
        )
        proposal = VisualAssociationProposal(
            detection_id="det-1",
            proposed_track_id="vlm-9",
            association_confidence=0.92,
            occlusion_likelihood=0.4,
            id_switch_risk=0.6,
            depth_conflict_flag=True,
            position_conflict_flag=True,
            uncertainties=[],
            supporting_evidence=["appearance_shift"],
        )

        update = coordinator.validate_visual_association_proposal(detection, proposal)

        self.assertFalse(update.accepted)
        self.assertEqual(update.resolved_track_id, "yolo-3")
        self.assertIn("depth_conflict_with_reliable_measurement", update.rejection_reasons)
        self.assertIn("ground_position_conflict_with_calibrated_mapping", update.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
