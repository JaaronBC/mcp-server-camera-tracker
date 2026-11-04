from __future__ import annotations

import json
import tempfile
import unittest
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

    def test_evaluate_boundaries_distinguishes_visual_and_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            request_id = "verifyreq-123"
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
            _write_json(run_dir / "latest_tracking_estimate_diagnostics.json", {"current_valid_estimates": []})

            evaluation = evaluate_boundaries(
                run_dir=run_dir,
                selected_candidate=candidate,
                submitted_request=request,
            )

            self.assertTrue(evaluation["visual_verification"]["success"])
            self.assertTrue(evaluation["feedback_publication"]["success"])
            self.assertFalse(evaluation["feedback_consumption"]["success"])
            self.assertEqual(
                evaluation["feedback_consumption"]["reason"],
                "likely_stale_or_mismatched_latest_message",
            )


if __name__ == "__main__":
    unittest.main()
