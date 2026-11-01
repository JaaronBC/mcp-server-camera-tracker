from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.dashboard.state import SharedDashboardState
from agents.dashboard.tracking_loop import _consume_tracking_observation_feedback
from agents.orchestration.runtime.verification_feedback import VerificationFeedbackPublisher
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer


def _accepted_result(*, request_id: str, track_id: str, message_id: str) -> dict:
    return {
        "request_id": request_id,
        "source": "reasoning_cycle",
        "track_id": track_id,
        "evidence_message_id": message_id,
        "status": "completed",
        "validation_outcome": "accepted",
        "submitted_at": 1000.0,
        "started_at": 1000.1,
        "finished_at": 1000.2,
        "current_applicability": "current",
        "stale_for_current_decisions": False,
        "coordinator_valid": True,
        "error_code": None,
        "error": None,
        "details": {
            "verification_observation_id": f"verify-{message_id}",
            "verification_referenced_message_id": message_id,
            "verification_timestamp": 999.8,
            "verification_status": "supported",
            "identity_consistency": "consistent",
            "visibility_status": "occluded",
            "possible_id_switch": False,
            "confidence": 0.91,
            "visual_content_inspected": True,
            "source_evidence_refs": ["evidence://frame/frame-1"],
            "supporting_evidence_refs": ["evidence://frame/frame-1"],
            "contradicting_evidence_refs": [],
            "uncertainty": ["partial_occlusion"],
            "limitations": ["single_view"],
        },
    }


class TestTrackingLoopMissingTrackGrace(unittest.TestCase):
    def test_track_disappears_briefly_and_remains_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            publisher.publish_from_results(
                [_accepted_result(request_id="req-1", track_id="track-1", message_id="msg-1")],
                now_timestamp=1001.0,
            )

            state = SharedDashboardState()
            state.tracking_feedback_consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)
            state.tracking_missing_track_grace_seconds = 1.5
            state.latest_tracking_evidence_message_by_track = {"track-1": "msg-1"}

            _consume_tracking_observation_feedback(state=state, now=1001.1, tracks={})

            self.assertEqual(state.latest_tracking_lost_evidence_message_by_track.get("track-1"), "msg-1")
            feedback = state.latest_tracking_observation_feedback
            assert isinstance(feedback, dict)
            self.assertEqual(feedback.get("accepted_count"), 1)
            self.assertEqual(feedback["accepted"][0]["evidence_message_id"], "msg-1")

    def test_track_exceeds_grace_period_and_becomes_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            publisher.publish_from_results(
                [_accepted_result(request_id="req-1", track_id="track-1", message_id="msg-1")],
                now_timestamp=1001.0,
            )

            state = SharedDashboardState()
            state.tracking_feedback_consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)
            state.tracking_missing_track_grace_seconds = 1.0
            state.latest_tracking_evidence_message_by_track = {"track-1": "msg-1"}

            _consume_tracking_observation_feedback(state=state, now=1001.1, tracks={})

            publisher.publish_from_results(
                [_accepted_result(request_id="req-2", track_id="track-1", message_id="msg-1")],
                now_timestamp=1002.3,
            )
            _consume_tracking_observation_feedback(state=state, now=1002.3, tracks={})

            self.assertNotIn("track-1", state.latest_tracking_lost_evidence_message_by_track)
            self.assertNotIn("track-1", state.latest_tracking_lost_track_expires_at_by_track)
            feedback = state.latest_tracking_observation_feedback
            assert isinstance(feedback, dict)
            self.assertEqual(feedback.get("accepted_count"), 0)

    def test_track_reappears_and_lost_state_clears(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            publisher.publish_from_results(
                [_accepted_result(request_id="req-1", track_id="track-1", message_id="msg-1")],
                now_timestamp=1001.0,
            )

            state = SharedDashboardState()
            state.tracking_feedback_consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)
            state.tracking_missing_track_grace_seconds = 2.0
            state.latest_tracking_evidence_message_by_track = {"track-1": "msg-1"}

            _consume_tracking_observation_feedback(state=state, now=1001.1, tracks={})
            self.assertIn("track-1", state.latest_tracking_lost_evidence_message_by_track)

            state.latest_tracking_evidence_message_by_track = {"track-1": "msg-1"}
            _consume_tracking_observation_feedback(state=state, now=1001.3, tracks={"track-1": object()})

            self.assertNotIn("track-1", state.latest_tracking_lost_evidence_message_by_track)
            self.assertNotIn("track-1", state.latest_tracking_lost_track_expires_at_by_track)


if __name__ == "__main__":
    unittest.main()
