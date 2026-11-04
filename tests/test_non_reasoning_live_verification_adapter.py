from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tests.adapters.non_reasoning_live_verification_adapter import (
    EvidenceCandidate,
    SingleVerificationRequestSource,
    evaluate_boundaries,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestNonReasoningLiveVerificationAdapter(unittest.TestCase):
    def test_request_source_submits_exactly_once_for_current_valid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            frames_dir = evidence_root / "frames"
            refs_dir = evidence_root / "refs"
            envelopes_dir = evidence_root / "envelopes"
            frames_dir.mkdir(parents=True, exist_ok=True)
            refs_dir.mkdir(parents=True, exist_ok=True)
            envelopes_dir.mkdir(parents=True, exist_ok=True)

            frame_id = "frame-live"
            detection_ref_id = "det-live"
            frame_ref = f"evidence://frame/{frame_id}"
            detection_ref = f"evidence://detection/{detection_ref_id}"
            frame_path = frames_dir / f"{frame_id}.jpg"
            frame_path.write_bytes(b"jpeg")

            _write_json(
                refs_dir / f"{frame_id}.json",
                {
                    "kind": "frame",
                    "id": frame_id,
                    "reference": frame_ref,
                    "status": "available",
                    "timestamp": 1000.1,
                    "expires_at": 9999999999.0,
                    "frame_path": str(frame_path),
                    "width": 640,
                    "height": 480,
                },
            )
            _write_json(
                refs_dir / f"{detection_ref_id}.json",
                {
                    "kind": "detection",
                    "id": detection_ref_id,
                    "reference": detection_ref,
                    "timestamp": 1000.1,
                    "expires_at": 9999999999.0,
                    "track_id": "track-live",
                    "detection_id": "det-1",
                    "frame_reference": frame_ref,
                },
            )
            _write_json(
                envelopes_dir / "msg-live.json",
                {
                    "message_id": "msg-live",
                    "stored_at": 1000.2,
                    "validation": {"valid": True, "errors": [], "warnings": []},
                    "envelope": {
                        "schema_version": "1.0",
                        "message_id": "msg-live",
                        "track_id": "track-live",
                        "timestamp": 1000.1,
                        "tracking_claims": {
                            "claimed_resolved_track_id": "track-live",
                            "association_status": "accepted",
                            "possible_id_switch": False,
                            "visibility_status": "visible",
                            "claim_reasons": [],
                        },
                        "frame_id": frame_id,
                        "detection_id": "det-1",
                        "frame_reference": frame_ref,
                        "source_evidence_refs": [frame_ref, detection_ref],
                        "uncertainty": [],
                        "quality_flags": [],
                        "track_history": [],
                    },
                },
            )

            radar_path = root / "latest_radar_state.json"
            _write_json(radar_path, {"tracks": [{"track_id": "track-live"}]})

            source = SingleVerificationRequestSource(
                evidence_root_dir=evidence_root,
                radar_state_path=radar_path,
                run_started_at_epoch=0.0,
            )

            first_batch = source()
            second_batch = source()

            self.assertEqual(len(first_batch), 1)
            self.assertEqual(len(second_batch), 0)
            self.assertIsNotNone(source.selected_candidate)
            self.assertEqual(source.selected_candidate.track_id, "track-live")
            self.assertEqual(source.selected_candidate.message_id, "msg-live")
            self.assertEqual(first_batch[0].track_id, "track-live")
            self.assertEqual(first_batch[0].message_id, "msg-live")
            self.assertEqual(first_batch[0].source, "external_source")

    def test_evaluate_boundaries_reports_separate_publication_consumer_estimate_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-123"
            (run_dir / "run_id.txt").write_text("run-1\n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-1",
                message_id="msg-1",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-1",
                detection_reference="evidence://detection/det-1",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-1", "message_id": "msg-1", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-1", "verification_referenced_message_id": "msg-1"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-1", "evidence_message_id": "msg-1"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "verification_observation_id": "verify-msg-1",
                        "evidence_message_id": "msg-1",
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertTrue(evaluation["visual_verification"]["success"])
            self.assertTrue(evaluation["feedback_publication"]["success"])
            self.assertTrue(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertFalse(evaluation["estimate_generation"]["success"])
            self.assertEqual(
                evaluation["estimate_generation"]["reason"],
                "likely_stale_or_mismatched_latest_message",
            )

    def test_evaluate_boundaries_requires_both_observation_and_message_for_consumer_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-both-ids"
            (run_dir / "run_id.txt").write_text("run-ids\n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-ids",
                message_id="msg-ids",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-ids",
                detection_reference="evidence://detection/det-ids",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-ids", "message_id": "msg-ids", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-ids", "verification_referenced_message_id": "msg-ids"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-ids", "evidence_message_id": "msg-ids"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "run-ids",
                                "verification_observation_id": "verify-msg-ids",
                                "evidence_message_id": "msg-other",
                                "decision": "accepted",
                            }
                        ),
                        json.dumps(
                            {
                                "run_id": "run-ids",
                                "verification_observation_id": "verify-msg-ids",
                                "evidence_message_id": "msg-ids",
                                "decision": "rejected",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertFalse(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(
                evaluation["feedback_consumer_acceptance"]["reason"],
                "consumer_rejected",
            )

    def test_evaluate_boundaries_ignores_stale_run_consumer_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-run-filter"
            (run_dir / "run_id.txt").write_text("run-current\n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-current",
                message_id="msg-current",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-current",
                detection_reference="evidence://detection/det-current",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-current", "message_id": "msg-current", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-current", "verification_referenced_message_id": "msg-current"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-current", "evidence_message_id": "msg-current"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-stale",
                        "verification_observation_id": "verify-msg-current",
                        "evidence_message_id": "msg-current",
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertFalse(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(
                evaluation["feedback_consumer_acceptance"]["reason"],
                "consumer_decision_log_missing",
            )


    def test_evaluate_boundaries_reports_contradiction_for_matching_accepted_and_rejected_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-contradiction"
            (run_dir / "run_id.txt").write_text("run-contradiction\n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-contradiction",
                message_id="msg-contradiction",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-contradiction",
                detection_reference="evidence://detection/det-contradiction",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-contradiction", "message_id": "msg-contradiction", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-contradiction", "verification_referenced_message_id": "msg-contradiction"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-contradiction", "evidence_message_id": "msg-contradiction"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "run-contradiction",
                                "verification_observation_id": "verify-msg-contradiction",
                                "evidence_message_id": "msg-contradiction",
                                "decision": "accepted",
                            }
                        ),
                        json.dumps(
                            {
                                "run_id": "run-contradiction",
                                "verification_observation_id": "verify-msg-contradiction",
                                "evidence_message_id": "msg-contradiction",
                                "decision": "rejected",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertFalse(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(
                evaluation["feedback_consumer_acceptance"]["reason"],
                "contradictory_consumer_decisions",
            )
            self.assertFalse(evaluation["feedback_consumption"]["success"])


    def test_run_id_missing_marks_consumer_acceptance_as_run_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-missing-run-id"

            candidate = EvidenceCandidate(
                track_id="track-missing-run-id",
                message_id="msg-missing-run-id",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-missing-run-id",
                detection_reference="evidence://detection/det-missing-run-id",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-missing-run-id", "message_id": "msg-missing-run-id", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-missing-run-id", "verification_referenced_message_id": "msg-missing-run-id"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-missing-run-id", "evidence_message_id": "msg-missing-run-id"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "other-run",
                        "verification_observation_id": "verify-msg-missing-run-id",
                        "evidence_message_id": "msg-missing-run-id",
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertFalse(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(evaluation["feedback_consumer_acceptance"]["reason"], "run_id_missing")

    def test_empty_run_id_marks_consumer_acceptance_as_run_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-empty-run-id"
            (run_dir / "run_id.txt").write_text("   \n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-empty-run-id",
                message_id="msg-empty-run-id",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-empty-run-id",
                detection_reference="evidence://detection/det-empty-run-id",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-empty-run-id", "message_id": "msg-empty-run-id", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-empty-run-id", "verification_referenced_message_id": "msg-empty-run-id"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-empty-run-id", "evidence_message_id": "msg-empty-run-id"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-empty-other",
                        "verification_observation_id": "verify-msg-empty-run-id",
                        "evidence_message_id": "msg-empty-run-id",
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertFalse(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(evaluation["feedback_consumer_acceptance"]["reason"], "run_id_missing")

    def test_unreadable_run_id_marks_consumer_acceptance_as_run_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-unreadable-run-id"
            run_id_path = run_dir / "run_id.txt"
            run_id_path.write_text("run-unreadable\n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-unreadable-run-id",
                message_id="msg-unreadable-run-id",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-unreadable-run-id",
                detection_reference="evidence://detection/det-unreadable-run-id",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-unreadable-run-id", "message_id": "msg-unreadable-run-id", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-unreadable-run-id", "verification_referenced_message_id": "msg-unreadable-run-id"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-unreadable-run-id", "evidence_message_id": "msg-unreadable-run-id"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-unreadable",
                        "verification_observation_id": "verify-msg-unreadable-run-id",
                        "evidence_message_id": "msg-unreadable-run-id",
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            original_read_text = Path.read_text

            def _patched_read_text(self: Path, *args, **kwargs):
                if self == run_id_path:
                    raise OSError("simulated unreadable run_id")
                return original_read_text(self, *args, **kwargs)

            with patch.object(Path, "read_text", _patched_read_text):
                evaluation = evaluate_boundaries(
                    run_dir=run_dir,
                    selected_candidate=candidate,
                    submitted_request=request,
                )

            self.assertFalse(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(evaluation["feedback_consumer_acceptance"]["reason"], "run_id_missing")

    def test_valid_run_id_ignores_stale_accepted_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-valid-stale"
            (run_dir / "run_id.txt").write_text("run-current-valid\n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-valid-stale",
                message_id="msg-valid-stale",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-valid-stale",
                detection_reference="evidence://detection/det-valid-stale",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-valid-stale", "message_id": "msg-valid-stale", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-valid-stale", "verification_referenced_message_id": "msg-valid-stale"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-valid-stale", "evidence_message_id": "msg-valid-stale"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-stale-other",
                        "verification_observation_id": "verify-msg-valid-stale",
                        "evidence_message_id": "msg-valid-stale",
                        "decision": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertFalse(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(
                evaluation["feedback_consumer_acceptance"]["reason"],
                "consumer_decision_log_missing",
            )

    def test_valid_run_id_accepts_matching_current_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-valid-match"
            (run_dir / "run_id.txt").write_text("run-current-match\n", encoding="utf-8")

            candidate = EvidenceCandidate(
                track_id="track-valid-match",
                message_id="msg-valid-match",
                stored_at=1000.0,
                frame_reference="evidence://frame/frame-valid-match",
                detection_reference="evidence://detection/det-valid-match",
            )

            class _Request:
                def __init__(self, request_id: str):
                    self.request_id = request_id

            request = _Request(request_id=request_id)

            _write_json(
                run_dir / "latest_verification_queue_submissions.json",
                [{"request_id": request_id, "source": "external_source", "track_id": "track-valid-match", "message_id": "msg-valid-match", "accepted": True}],
            )
            _write_json(
                run_dir / "latest_verification_queue_results.json",
                [{"request_id": request_id, "status": "completed", "validation_outcome": "accepted", "coordinator_valid": True, "details": {"verification_observation_id": "verify-msg-valid-match", "verification_referenced_message_id": "msg-valid-match"}}],
            )
            _write_json(
                run_dir / "operator_tracking_feedback.json",
                {"schema_version": "1.0", "generated_at": 1001.0, "records": [{"feedback_id": f"obsfeedback-{request_id}", "verification_request_id": request_id, "track_id": "track-valid-match", "evidence_message_id": "msg-valid-match"}]},
            )
            (run_dir / "tracking_feedback_consumer_decisions.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "run_id": "run-stale-other",
                                "verification_observation_id": "verify-msg-valid-match",
                                "evidence_message_id": "msg-valid-match",
                                "decision": "accepted",
                            }
                        ),
                        json.dumps(
                            {
                                "run_id": "run-current-match",
                                "verification_observation_id": "verify-msg-valid-match",
                                "evidence_message_id": "msg-valid-match",
                                "decision": "accepted",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertTrue(evaluation["feedback_consumer_acceptance"]["success"])
            self.assertEqual(evaluation["feedback_consumer_acceptance"]["reason"], "ok")


if __name__ == "__main__":
    unittest.main()
