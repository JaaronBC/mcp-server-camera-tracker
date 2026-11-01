from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.orchestration.runtime.verification_feedback import VerificationFeedbackPublisher
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer


def _accepted_result(
    *,
    request_id: str,
    track_id: str,
    message_id: str,
    applicability: str = "current",
    stale_for_current: bool = False,
    details_overrides: dict | None = None,
) -> dict:
    details = {
        "verification_observation_id": f"verify-{message_id}",
        "verification_referenced_message_id": message_id,
        "verification_timestamp": 999.8,
        "verification_status": "supported",
        "identity_consistency": "consistent",
        "visibility_status": "visible",
        "possible_id_switch": False,
        "confidence": 0.91,
        "visual_content_inspected": True,
        "source_evidence_refs": ["evidence://frame/frame-1"],
        "supporting_evidence_refs": ["evidence://frame/frame-1"],
        "contradicting_evidence_refs": [],
        "uncertainty": ["partial_occlusion"],
        "limitations": ["single_view"],
    }
    if isinstance(details_overrides, dict):
        details.update(details_overrides)

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
        "current_applicability": applicability,
        "stale_for_current_decisions": stale_for_current,
        "coordinator_valid": True,
        "error_code": None,
        "error": None,
        "details": details,
    }


class TestTrackingObservationFeedback(unittest.TestCase):
    def test_publish_and_consume_successful_current_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            publish_result = publisher.publish_from_results(
                [_accepted_result(request_id="req-1", track_id="track-1", message_id="msg-1")],
                now_timestamp=1001.0,
            )
            self.assertEqual(publish_result["published"], 1)

            consumed = consumer.poll(
                current_track_ids={"track-1"},
                latest_message_id_by_track={"track-1": "msg-1"},
                now_timestamp=1001.2,
            )

            self.assertEqual(consumed["accepted_count"], 1)
            feedback = consumed["accepted"][0]
            self.assertEqual(feedback["track_id"], "track-1")
            self.assertEqual(feedback["evidence_message_id"], "msg-1")
            self.assertEqual(feedback["current_applicability"], "current")
            self.assertEqual(
                feedback["details"]["supporting_evidence_refs"],
                ["evidence://frame/frame-1"],
            )
            self.assertEqual(feedback["details"]["contradicting_evidence_refs"], [])
            self.assertEqual(feedback["details"]["uncertainty"], ["partial_occlusion"])
            self.assertEqual(feedback["details"]["limitations"], ["single_view"])
            self.assertFalse(feedback["details"]["possible_id_switch"])
            self.assertTrue(feedback["details"]["visual_content_inspected"])
            self.assertEqual(feedback["details"]["confidence"], 0.91)

    def test_supporting_and_contradicting_refs_preserved_without_alteration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            supporting_refs = [
                "evidence://frame/frame-1",
                "evidence://frame/frame-2",
            ]
            contradicting_refs = ["evidence://frame/frame-3"]
            publisher.publish_from_results(
                [
                    _accepted_result(
                        request_id="req-refs",
                        track_id="track-refs",
                        message_id="msg-refs",
                        details_overrides={
                            "source_evidence_refs": [
                                "evidence://frame/frame-1",
                                "evidence://frame/frame-2",
                                "evidence://frame/frame-3",
                            ],
                            "supporting_evidence_refs": supporting_refs,
                            "contradicting_evidence_refs": contradicting_refs,
                            "uncertainty": ["brief_occlusion"],
                            "limitations": ["low_resolution_segment"],
                        },
                    )
                ],
                now_timestamp=2001.0,
            )

            consumed = consumer.poll(
                current_track_ids={"track-refs"},
                latest_message_id_by_track={"track-refs": "msg-refs"},
                now_timestamp=2001.1,
            )

            self.assertEqual(consumed["accepted_count"], 1)
            details = consumed["accepted"][0]["details"]
            self.assertEqual(details["supporting_evidence_refs"], supporting_refs)
            self.assertEqual(details["contradicting_evidence_refs"], contradicting_refs)
            self.assertEqual(details["uncertainty"], ["brief_occlusion"])
            self.assertEqual(details["limitations"], ["low_resolution_segment"])

    def test_missing_optional_findings_defaults_to_empty_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            publish_result = publisher.publish_from_results(
                [
                    _accepted_result(
                        request_id="req-optional",
                        track_id="track-optional",
                        message_id="msg-optional",
                        details_overrides={
                            "source_evidence_refs": None,
                            "supporting_evidence_refs": None,
                            "contradicting_evidence_refs": None,
                            "uncertainty": None,
                            "limitations": None,
                        },
                    )
                ],
                now_timestamp=3001.0,
            )
            self.assertEqual(publish_result["published"], 1)

            consumed = consumer.poll(
                current_track_ids={"track-optional"},
                latest_message_id_by_track={"track-optional": "msg-optional"},
                now_timestamp=3001.1,
            )
            self.assertEqual(consumed["accepted_count"], 1)
            details = consumed["accepted"][0]["details"]
            self.assertEqual(details["source_evidence_refs"], [])
            self.assertEqual(details["supporting_evidence_refs"], [])
            self.assertEqual(details["contradicting_evidence_refs"], [])
            self.assertEqual(details["uncertainty"], [])
            self.assertEqual(details["limitations"], [])

    def test_malformed_findings_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)

            malformed_required_status = _accepted_result(
                request_id="req-bad-status",
                track_id="track-bad",
                message_id="msg-bad-status",
                details_overrides={"verification_status": "maybe"},
            )
            malformed_refs = _accepted_result(
                request_id="req-bad-refs",
                track_id="track-bad",
                message_id="msg-bad-refs",
                details_overrides={"supporting_evidence_refs": "not-a-list"},
            )
            malformed_possible_switch = _accepted_result(
                request_id="req-bad-switch",
                track_id="track-bad",
                message_id="msg-bad-switch",
                details_overrides={"possible_id_switch": "false"},
            )
            malformed_confidence = _accepted_result(
                request_id="req-bad-confidence",
                track_id="track-bad",
                message_id="msg-bad-confidence",
                details_overrides={"confidence": 1.2},
            )
            malformed_inspected = _accepted_result(
                request_id="req-bad-inspected",
                track_id="track-bad",
                message_id="msg-bad-inspected",
                details_overrides={"visual_content_inspected": "yes"},
            )

            publish_result = publisher.publish_from_results(
                [
                    malformed_required_status,
                    malformed_refs,
                    malformed_possible_switch,
                    malformed_confidence,
                    malformed_inspected,
                ],
                now_timestamp=4001.0,
            )
            self.assertEqual(publish_result["published"], 0)
            self.assertEqual(publish_result["malformed"], 5)

    def test_oversized_payload_is_ignored_by_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            artifact_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "generated_at": 5000.0,
                        "records": [],
                        "padding": "x" * 4096,
                    }
                ),
                encoding="utf-8",
            )

            consumer = TrackingObservationFeedbackConsumer(
                artifact_path=artifact_path,
                max_artifact_bytes=128,
            )
            consumed = consumer.poll(
                current_track_ids={"track-1"},
                latest_message_id_by_track={"track-1": "msg-1"},
                now_timestamp=5001.0,
            )

            self.assertEqual(consumed["accepted_count"], 0)
            self.assertEqual(consumed["rejected_count"], 0)

    def test_invalid_verification_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)

            invalid = _accepted_result(request_id="req-2", track_id="track-2", message_id="msg-2")
            invalid["coordinator_valid"] = False

            publish_result = publisher.publish_from_results([invalid], now_timestamp=1001.0)
            self.assertEqual(publish_result["published"], 0)
            self.assertEqual(publish_result["malformed"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"], [])

    def test_stale_feedback_rejected_on_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(
                artifact_path=artifact_path,
                record_ttl_seconds=1.0,
            )
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            publisher.publish_from_results(
                [_accepted_result(request_id="req-3", track_id="track-3", message_id="msg-3")],
                now_timestamp=1001.0,
            )

            consumed = consumer.poll(
                current_track_ids={"track-3"},
                latest_message_id_by_track={"track-3": "msg-3"},
                now_timestamp=1002.5,
            )

            self.assertEqual(consumed["accepted_count"], 0)
            self.assertGreaterEqual(consumed["rejected_count"], 1)

    def test_mismatched_track_or_message_rejected_for_current_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            publisher.publish_from_results(
                [_accepted_result(request_id="req-4", track_id="track-4", message_id="msg-4")],
                now_timestamp=1001.0,
            )

            missing_track = consumer.poll(
                current_track_ids=set(),
                latest_message_id_by_track={},
                now_timestamp=1001.1,
            )
            self.assertEqual(missing_track["accepted_count"], 0)
            self.assertGreaterEqual(missing_track["rejected_count"], 1)

            wrong_message = consumer.poll(
                current_track_ids={"track-4"},
                latest_message_id_by_track={"track-4": "msg-other"},
                now_timestamp=1001.2,
            )
            self.assertEqual(wrong_message["accepted_count"], 0)
            self.assertGreaterEqual(wrong_message["rejected_count"], 1)

    def test_current_feedback_can_be_accepted_for_temporarily_lost_track(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            publisher.publish_from_results(
                [_accepted_result(request_id="req-lost", track_id="track-lost", message_id="msg-lost")],
                now_timestamp=1101.0,
            )

            consumed = consumer.poll(
                current_track_ids=set(),
                temporarily_lost_track_ids={"track-lost"},
                latest_message_id_by_track={"track-lost": "msg-lost"},
                now_timestamp=1101.1,
            )

            self.assertEqual(consumed["accepted_count"], 1)
            self.assertEqual(consumed["accepted"][0]["track_id"], "track-lost")
            self.assertEqual(consumed["accepted"][0]["evidence_message_id"], "msg-lost")

    def test_historical_feedback_remains_historical(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            publisher.publish_from_results(
                [
                    _accepted_result(
                        request_id="req-historical",
                        track_id="track-historical",
                        message_id="msg-historical",
                        applicability="historical",
                        stale_for_current=True,
                    )
                ],
                now_timestamp=1201.0,
            )

            consumed = consumer.poll(
                current_track_ids=set(),
                temporarily_lost_track_ids={"track-historical"},
                latest_message_id_by_track={"track-historical": "msg-historical"},
                now_timestamp=1201.1,
            )

            self.assertEqual(consumed["accepted_count"], 1)
            self.assertEqual(consumed["accepted"][0]["current_applicability"], "historical")
            self.assertTrue(consumed["accepted"][0]["historical_observation"])

    def test_atomic_file_handling_and_recovery_from_partial_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            artifact_path.write_text('{"schema_version":"1.0","records":[', encoding="utf-8")

            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)
            consumed = consumer.poll(
                current_track_ids={"track-1"},
                latest_message_id_by_track={"track-1": "msg-1"},
                now_timestamp=1001.0,
            )
            self.assertEqual(consumed["accepted_count"], 0)

            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            publish_result = publisher.publish_from_results(
                [_accepted_result(request_id="req-5", track_id="track-1", message_id="msg-1")],
                now_timestamp=1001.1,
            )
            self.assertEqual(publish_result["published"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(len(payload["records"]), 1)

    def test_deduplication_and_failure_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path, max_records=8)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            first = _accepted_result(request_id="req-6", track_id="track-6", message_id="msg-6")
            publish_one = publisher.publish_from_results([first], now_timestamp=1001.0)
            publish_two = publisher.publish_from_results([first], now_timestamp=1001.2)

            self.assertEqual(publish_one["published"], 1)
            self.assertEqual(publish_two["published"], 0)
            self.assertEqual(publish_two["skipped"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            payload["records"].append({"feedback_id": "bad-record"})
            artifact_path.write_text(json.dumps(payload), encoding="utf-8")

            first_poll = consumer.poll(
                current_track_ids={"track-6"},
                latest_message_id_by_track={"track-6": "msg-6"},
                now_timestamp=1001.3,
            )
            second_poll = consumer.poll(
                current_track_ids={"track-6"},
                latest_message_id_by_track={"track-6": "msg-6"},
                now_timestamp=1001.4,
            )

            self.assertEqual(first_poll["accepted_count"], 1)
            self.assertGreaterEqual(first_poll["rejected_count"], 1)
            self.assertEqual(second_poll["accepted_count"], 0)


if __name__ == "__main__":
    unittest.main()
