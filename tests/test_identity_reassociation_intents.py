from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.orchestration.runtime.identity_reassociation_intents import (
    IdentityReassociationIntentPublisher,
)


def _base_result(*, request_id: str = "req-1", track_id: str = "track-1") -> dict:
    return {
        "request_id": request_id,
        "source": "reasoning_cycle",
        "track_id": track_id,
        "evidence_message_id": "trkmsg-1",
        "status": "completed",
        "validation_outcome": "accepted",
        "current_applicability": "current",
        "stale_for_current_decisions": False,
        "coordinator_valid": True,
        "details": {
            "verification_observation_id": "verify-1",
            "verification_referenced_message_id": "trkmsg-1",
            "verification_timestamp": 1000.5,
            "verification_status": "supported",
            "identity_consistency": "consistent",
            "visual_content_inspected": True,
            "source_evidence_refs": ["evidence://frame/f1"],
            "identity_claim_id": "idclaim-1",
            "identity_claim_status": "usable",
            "referenced_detection_reference": "evidence://detection/d1",
            "referenced_proposed_app_identity_id": "app-1",
            "referenced_identity_snapshot_generation": 12,
            "referenced_identity_snapshot_version": 21,
            "identity_claim_evidence_refs": ["identity_snapshot:gen:12:ver:21"],
            "identity_claim_provenance": {
                "source_agent": "visual_tracking",
                "source_contract": "VisualAssociationUpdate",
                "mutation_authorized": False,
            },
        },
    }


def _seed_tracking_evidence(
    root_dir: Path,
    *,
    detection_ref_id: str = "d1",
    track_id: str = "track-1",
    detection_id: str = "det-1",
    frame_id: str = "f1",
    canonical_detection_reference: str = "fdet-1000000-0-det-1-track-1",
    process_nonce: str = "proc-test-1",
) -> tuple[str, str]:
    refs_dir = root_dir / "refs"
    frames_dir = root_dir / "frames"
    refs_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_path = frames_dir / f"{frame_id}.jpg"
    frame_path.write_bytes(b"jpeg")
    frame_reference = f"evidence://frame/{frame_id}"

    (refs_dir / f"{frame_id}.json").write_text(
        json.dumps(
            {
                "kind": "frame",
                "id": frame_id,
                "reference": frame_reference,
                "timestamp": 1000.0,
                "created_at": 1000.0,
                "expires_at": 9999999999.0,
                "status": "available",
                "frame_path": str(frame_path),
                "width": 16,
                "height": 16,
            }
        ),
        encoding="utf-8",
    )

    detection_reference = f"evidence://detection/{detection_ref_id}"
    (refs_dir / f"{detection_ref_id}.json").write_text(
        json.dumps(
            {
                "kind": "detection",
                "id": detection_ref_id,
                "reference": detection_reference,
                "timestamp": 1000.0,
                "created_at": 1000.0,
                "expires_at": 9999999999.0,
                "track_id": track_id,
                "detection_id": detection_id,
                "frame_reference": frame_reference,
                "pixel_bbox": None,
                "process_nonce": process_nonce,
                "tracking_detection_reference": canonical_detection_reference,
            }
        ),
        encoding="utf-8",
    )

    return detection_reference, frame_reference


class TestIdentityReassociationIntentPublisher(unittest.TestCase):
    def test_supported_consistent_usable_claim_creates_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )

            result = _base_result()
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]

            summary = publisher.publish_from_results([result], now_timestamp=1010.0)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected"], 0)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 1)
            intent = payload["intents"][0]
            self.assertEqual(intent["claim_id"], "idclaim-1")
            self.assertEqual(intent["app_identity_id"], "app-1")
            self.assertEqual(intent["detection_reference"], "fdet-1000000-0-det-1-track-1")
            self.assertEqual(intent["evidence_detection_reference"], detection_reference)
            self.assertEqual(intent["frame_reference"], frame_reference)
            self.assertIn("process_nonce", intent["capture_context"])
            self.assertEqual(intent["target_yolo_track_id"], "track-1")
            self.assertEqual(intent["identity_snapshot_generation"], 12)
            self.assertEqual(intent["identity_snapshot_version"], 21)
            self.assertEqual(intent["source_evidence_refs"], [frame_reference])
            self.assertEqual(intent["identity_evidence_refs"], ["identity_snapshot:gen:12:ver:21"])
            self.assertEqual(intent["generated_at"], 1010.0)
            self.assertEqual(intent["expires_at"], 1055.0)

    def test_contradicted_and_inconclusive_verdicts_do_not_create_intents(self) -> None:
        contradicted = _base_result(request_id="req-contradicted")
        contradicted["details"]["verification_status"] = "contradicted"

        inconclusive = _base_result(request_id="req-inconclusive")
        inconclusive["details"]["verification_status"] = "inconclusive"

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            for item in (contradicted, inconclusive):
                item["details"]["referenced_detection_reference"] = detection_reference
                item["details"]["source_evidence_refs"] = [frame_reference]
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            summary = publisher.publish_from_results(
                [contradicted, inconclusive],
                now_timestamp=1011.0,
            )

            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 2)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 0)
            reasons = {item.get("reason") for item in payload["decisions"]}
            self.assertIn("verification_not_supported", reasons)

    def test_stale_or_noncurrent_records_are_rejected(self) -> None:
        stale = _base_result(request_id="req-stale")
        stale["stale_for_current_decisions"] = True

        historical = _base_result(request_id="req-historical")
        historical["current_applicability"] = "historical"

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            for item in (stale, historical):
                item["details"]["referenced_detection_reference"] = detection_reference
                item["details"]["source_evidence_refs"] = [frame_reference]
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )
            summary = publisher.publish_from_results([stale, historical], now_timestamp=1012.0)

            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 2)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 0)
            reasons = {item.get("reason") for item in payload["decisions"]}
            self.assertIn("verification_stale_for_current", reasons)
            self.assertIn("verification_not_current", reasons)

    def test_malformed_claim_binding_is_rejected(self) -> None:
        malformed = _base_result(request_id="req-malformed")
        malformed["details"]["identity_claim_id"] = None

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            malformed["details"]["referenced_detection_reference"] = detection_reference
            malformed["details"]["source_evidence_refs"] = [frame_reference]
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )
            summary = publisher.publish_from_results([malformed], now_timestamp=1013.0)

            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 0)
            self.assertEqual(payload["decisions"][-1]["reason"], "missing_claim_id")

    def test_duplicate_intent_is_deduplicated(self) -> None:
        first = _base_result(request_id="req-dup")
        second = _base_result(request_id="req-dup")

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            for item in (first, second):
                item["details"]["referenced_detection_reference"] = detection_reference
                item["details"]["source_evidence_refs"] = [frame_reference]
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )
            summary = publisher.publish_from_results([first, second], now_timestamp=1014.0)

            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["duplicates"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 1)
            self.assertEqual(payload["decisions"][-1]["reason"], "duplicate_intent")

    def test_conflicting_claim_binding_is_rejected(self) -> None:
        base = _base_result(request_id="req-1", track_id="track-1")
        conflicting = _base_result(request_id="req-2", track_id="track-9")
        conflicting["details"]["identity_claim_id"] = "idclaim-1"

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            base_detection_reference, base_frame_reference = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d1",
                track_id="track-1",
                canonical_detection_reference="fdet-1000000-0-det-1-track-1",
            )
            conflicting_detection_reference, conflicting_frame_reference = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-9",
                canonical_detection_reference="fdet-1001000-0-det-1-track-9",
                process_nonce="proc-test-2",
            )
            base["details"]["referenced_detection_reference"] = base_detection_reference
            base["details"]["source_evidence_refs"] = [base_frame_reference]
            conflicting["details"]["referenced_detection_reference"] = conflicting_detection_reference
            conflicting["details"]["source_evidence_refs"] = [conflicting_frame_reference]
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )
            summary = publisher.publish_from_results([base, conflicting], now_timestamp=1015.0)

            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["conflicts"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 1)
            self.assertEqual(payload["decisions"][-1]["reason"], "conflicting_claim_binding")

    def test_intent_id_stable_across_publisher_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"

            result = _base_result(request_id="req-stable")
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]

            publisher_a = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )
            publisher_b = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )

            publisher_a.publish_from_results([result], now_timestamp=2000.0)
            payload_a = json.loads(artifact_path.read_text(encoding="utf-8"))
            intent_id_a = payload_a["intents"][0]["intent_id"]

            publisher_b.publish_from_results([result], now_timestamp=2001.0)
            payload_b = json.loads(artifact_path.read_text(encoding="utf-8"))
            intent_id_b = payload_b["intents"][0]["intent_id"]

            self.assertEqual(intent_id_a, intent_id_b)
            self.assertEqual(payload_b["intent_count"], 1)

    def test_multiple_valid_intents_published_between_polls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            d1, f1 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d1",
                track_id="track-1",
                canonical_detection_reference="fdet-1000000-0-det-1-track-1",
            )
            d2, f2 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-2",
                detection_id="det-2",
                frame_id="f2",
                canonical_detection_reference="fdet-1000001-0-det-2-track-2",
                process_nonce="proc-test-2",
            )
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            r1 = _base_result(request_id="req-1", track_id="track-1")
            r1["details"]["identity_claim_id"] = "idclaim-1"
            r1["details"]["referenced_detection_reference"] = d1
            r1["details"]["source_evidence_refs"] = [f1]

            r2 = _base_result(request_id="req-2", track_id="track-2")
            r2["details"]["identity_claim_id"] = "idclaim-2"
            r2["details"]["referenced_detection_reference"] = d2
            r2["details"]["source_evidence_refs"] = [f2]

            summary = publisher.publish_from_results([r1, r2], now_timestamp=3000.0)
            self.assertEqual(summary["accepted"], 2)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 2)

    def test_overflow_rejects_new_without_losing_existing_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            d1, f1 = _seed_tracking_evidence(evidence_root, detection_ref_id="d1", track_id="track-1")
            d2, f2 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-2",
                detection_id="det-2",
                frame_id="f2",
                canonical_detection_reference="fdet-1000001-0-det-2-track-2",
                process_nonce="proc-test-2",
            )
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
                max_intents=1,
            )

            r1 = _base_result(request_id="req-a", track_id="track-1")
            r1["details"]["identity_claim_id"] = "idclaim-a"
            r1["details"]["referenced_detection_reference"] = d1
            r1["details"]["source_evidence_refs"] = [f1]

            r2 = _base_result(request_id="req-b", track_id="track-2")
            r2["details"]["identity_claim_id"] = "idclaim-b"
            r2["details"]["referenced_detection_reference"] = d2
            r2["details"]["source_evidence_refs"] = [f2]

            publisher.publish_from_results([r1], now_timestamp=4000.0)
            summary = publisher.publish_from_results([r2], now_timestamp=4001.0)
            self.assertEqual(summary["overflow"], 1)
            self.assertEqual(summary["accepted"], 0)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 1)
            self.assertEqual(payload["intents"][0]["claim_id"], "idclaim-a")

    def test_expired_intent_pruned_and_not_reactivated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
                intent_ttl_seconds=2.0,
            )

            result = _base_result(request_id="req-exp", track_id="track-1")
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]
            result["details"]["identity_claim_id"] = "idclaim-exp"

            publisher.publish_from_results([result], now_timestamp=5000.0)
            summary = publisher.publish_from_results([result], now_timestamp=5003.0)
            self.assertEqual(summary["accepted"], 0)
            self.assertGreaterEqual(summary["expired_pruned"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            reasons = {item.get("reason") for item in payload.get("decisions", [])}
            self.assertIn("intent_expired", reasons)
            self.assertIn("intent_previously_expired", reasons)
            self.assertEqual(payload["intent_count"], 0)

    def test_missing_frame_or_capture_context_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            refs_dir = evidence_root / "refs"
            detection_payload_path = refs_dir / "d1.json"
            detection_payload = json.loads(detection_payload_path.read_text(encoding="utf-8"))
            detection_payload["process_nonce"] = None
            detection_payload_path.write_text(json.dumps(detection_payload), encoding="utf-8")

            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            result = _base_result(request_id="req-missing-context")
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]

            summary = publisher.publish_from_results([result], now_timestamp=6000.0)
            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 1)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intents"], [])
            self.assertEqual(payload["decisions"][-1]["reason"], "detection_binding_invalid")

    def test_recover_from_corrupt_primary_using_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            result = _base_result(request_id="req-recovery")
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]
            result["details"]["identity_claim_id"] = "idclaim-recovery"

            publisher.publish_from_results([result], now_timestamp=7000.0)
            backup_path = Path(f"{artifact_path}.bak")
            self.assertTrue(backup_path.exists())

            artifact_path.write_text("{bad-json", encoding="utf-8")
            summary = publisher.publish_from_results([], now_timestamp=7001.0)
            self.assertFalse(summary["load_failed"])
            self.assertTrue(summary["recovered_from_backup"])

    def test_unrecoverable_corruption_fails_closed_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            backup_path = Path(f"{artifact_path}.bak")
            artifact_path.write_text("{bad-json", encoding="utf-8")
            backup_path.write_text("{bad-json", encoding="utf-8")

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )
            summary = publisher.publish_from_results([], now_timestamp=8000.0)
            self.assertTrue(summary["load_failed"])
            self.assertEqual(summary["load_failure_reason"], "intent_artifact_unrecoverable")

    def test_ack_ledger_schema_created_and_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )

            result = _base_result(request_id="req-ack")
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]

            publisher.publish_from_results([result], now_timestamp=8100.0)
            self.assertTrue(ack_path.exists())
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertEqual(ack_payload.get("schema_version"), "1.0")
            self.assertEqual(ack_payload.get("acknowledgments"), [])


if __name__ == "__main__":
    unittest.main()
