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


def _write_minimal_intent_artifact(
    artifact_path: Path,
    *,
    generated_at: float,
    decisions: list[dict] | None = None,
    terminal_investigations: list[dict] | None = None,
) -> None:
    payload = {
        "schema_version": "1.1",
        "generated_at": float(generated_at),
        "intent_count": 0,
        "decision_count": len(decisions or []),
        "intent_ttl_seconds": 45.0,
        "intents": [],
        "decisions": list(decisions or []),
        "terminal_investigations": list(terminal_investigations or []),
    }
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    backup_path = Path(f"{artifact_path}.bak")
    backup_path.write_text(json.dumps(payload), encoding="utf-8")


class TestIdentityReassociationIntentPublisher(unittest.TestCase):
    def test_investigation_metadata_propagates_to_intent_and_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            result = _base_result(request_id="req-investigation-1")
            result["investigation_id"] = "inv-xyz-1"
            result["investigation_state"] = "awaiting_reassociation"
            result["verification_outcome"] = "supported"
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]

            summary = publisher.publish_from_results([result], now_timestamp=1010.0)
            self.assertEqual(summary["accepted"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 1)
            intent = payload["intents"][0]
            self.assertEqual(intent.get("investigation_id"), "inv-xyz-1")
            self.assertEqual(intent.get("investigation_state"), "awaiting_reassociation")
            self.assertEqual(intent.get("verification_outcome"), "supported")

            decision = payload["decisions"][-1]
            self.assertEqual(decision.get("investigation_id"), "inv-xyz-1")
            self.assertEqual(decision.get("investigation_state"), "awaiting_reassociation")
            self.assertEqual(decision.get("verification_outcome"), "supported")
            self.assertIsNone(decision.get("investigation_termination_reason"))

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

    def test_competing_candidates_same_track_keep_investigation_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference_1, frame_reference_1 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d1",
                track_id="track-1",
                canonical_detection_reference="fdet-1000-0-det-1-track-1",
            )
            detection_reference_2, frame_reference_2 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-1",
                canonical_detection_reference="fdet-1001-0-det-2-track-1",
                frame_id="f2",
                detection_id="det-2",
                process_nonce="proc-test-2",
            )
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            first = _base_result(request_id="req-candidate-1", track_id="track-1")
            first["investigation_id"] = "inv-shared-track-1"
            first["details"]["identity_claim_id"] = "idclaim-a"
            first["details"]["referenced_proposed_app_identity_id"] = "app-a"
            first["details"]["referenced_detection_reference"] = detection_reference_1
            first["details"]["source_evidence_refs"] = [frame_reference_1]

            second = _base_result(request_id="req-candidate-2", track_id="track-1")
            second["investigation_id"] = "inv-shared-track-1"
            second["details"]["identity_claim_id"] = "idclaim-b"
            second["details"]["referenced_proposed_app_identity_id"] = "app-b"
            second["details"]["referenced_detection_reference"] = detection_reference_2
            second["details"]["source_evidence_refs"] = [frame_reference_2]

            summary = publisher.publish_from_results([first, second], now_timestamp=1010.0)
            self.assertEqual(summary["accepted"], 2)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            intents = payload["intents"]
            self.assertEqual(len(intents), 2)
            self.assertEqual(intents[0].get("investigation_id"), "inv-shared-track-1")
            self.assertEqual(intents[1].get("investigation_id"), "inv-shared-track-1")
            self.assertNotEqual(intents[0].get("intent_id"), intents[1].get("intent_id"))

    def test_late_result_for_terminal_investigation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference_1, frame_reference_1 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d1",
                track_id="track-1",
                canonical_detection_reference="fdet-2000-0-det-1-track-1",
            )
            detection_reference_2, frame_reference_2 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-1",
                canonical_detection_reference="fdet-2001-0-det-2-track-1",
                frame_id="f2",
                detection_id="det-2",
                process_nonce="proc-test-2",
            )
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            terminal_result = _base_result(request_id="req-terminal", track_id="track-1")
            terminal_result["investigation_id"] = "inv-terminal-1"
            terminal_result["details"]["verification_status"] = "contradicted"
            terminal_result["details"]["identity_consistency"] = "inconsistent"
            terminal_result["details"]["referenced_detection_reference"] = detection_reference_1
            terminal_result["details"]["source_evidence_refs"] = [frame_reference_1]

            late_result = _base_result(request_id="req-late", track_id="track-1")
            late_result["investigation_id"] = "inv-terminal-1"
            late_result["details"]["identity_claim_id"] = "idclaim-late"
            late_result["details"]["referenced_detection_reference"] = detection_reference_2
            late_result["details"]["source_evidence_refs"] = [frame_reference_2]

            first_summary = publisher.publish_from_results([terminal_result], now_timestamp=1500.0)
            self.assertEqual(first_summary["accepted"], 0)

            second_summary = publisher.publish_from_results([late_result], now_timestamp=1501.0)
            self.assertEqual(second_summary["accepted"], 0)
            self.assertEqual(second_summary["rejected"], 1)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 0)
            self.assertEqual(payload["decisions"][-1]["reason"], "investigation_already_terminal")
            self.assertEqual(payload["decisions"][-1]["investigation_state"], "resolved")

    def test_terminal_decision_truncation_restart_still_rejects_late_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference_1, frame_reference_1 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d1",
                track_id="track-1",
                canonical_detection_reference="fdet-r1",
            )
            detection_reference_2, frame_reference_2 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-1",
                canonical_detection_reference="fdet-r2",
                frame_id="f2",
                detection_id="det-2",
                process_nonce="proc-test-2",
            )

            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                max_decisions=1,
            )

            terminal_result = _base_result(request_id="req-terminal-a", track_id="track-1")
            terminal_result["investigation_id"] = "inv-truncated-restart"
            terminal_result["details"]["verification_status"] = "contradicted"
            terminal_result["details"]["identity_consistency"] = "inconsistent"
            terminal_result["details"]["referenced_detection_reference"] = detection_reference_1
            terminal_result["details"]["source_evidence_refs"] = [frame_reference_1]
            publisher.publish_from_results([terminal_result], now_timestamp=2000.0)

            filler = _base_result(request_id="req-filler-b", track_id="track-1")
            filler["investigation_id"] = "inv-filler"
            filler["details"]["verification_status"] = "contradicted"
            filler["details"]["identity_consistency"] = "inconsistent"
            filler["details"]["referenced_detection_reference"] = detection_reference_1
            filler["details"]["source_evidence_refs"] = [frame_reference_1]
            publisher.publish_from_results([filler], now_timestamp=2001.0)

            restarted = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                max_decisions=1,
            )

            late = _base_result(request_id="req-late-c", track_id="track-1")
            late["investigation_id"] = "inv-truncated-restart"
            late["details"]["identity_claim_id"] = "idclaim-late"
            late["details"]["referenced_detection_reference"] = detection_reference_2
            late["details"]["source_evidence_refs"] = [frame_reference_2]
            summary = restarted.publish_from_results([late], now_timestamp=2002.0)

            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 1)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["decisions"][-1]["reason"], "investigation_already_terminal")
            self.assertEqual(payload["intent_count"], 0)

    def test_terminal_without_reassociation_ack_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference_1, frame_reference_1 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d1",
                track_id="track-1",
                canonical_detection_reference="fdet-a1",
            )
            detection_reference_2, frame_reference_2 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-1",
                canonical_detection_reference="fdet-a2",
                frame_id="f2",
                detection_id="det-2",
                process_nonce="proc-test-2",
            )

            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                max_decisions=1,
            )

            contradicted = _base_result(request_id="req-noack", track_id="track-1")
            contradicted["investigation_id"] = "inv-no-ack-terminal"
            contradicted["details"]["verification_status"] = "contradicted"
            contradicted["details"]["identity_consistency"] = "inconsistent"
            contradicted["details"]["referenced_detection_reference"] = detection_reference_1
            contradicted["details"]["source_evidence_refs"] = [frame_reference_1]
            publisher.publish_from_results([contradicted], now_timestamp=2100.0)

            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertEqual(ack_payload.get("acknowledgments"), [])

            restarted = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                max_decisions=1,
            )
            late = _base_result(request_id="req-noack-late", track_id="track-1")
            late["investigation_id"] = "inv-no-ack-terminal"
            late["details"]["identity_claim_id"] = "idclaim-noack"
            late["details"]["referenced_detection_reference"] = detection_reference_2
            late["details"]["source_evidence_refs"] = [frame_reference_2]

            summary = restarted.publish_from_results([late], now_timestamp=2101.0)
            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 1)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["decisions"][-1]["reason"], "investigation_already_terminal")

    def test_reassociation_ack_recovery_for_applied_and_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference_1, frame_reference_1 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d1",
                track_id="track-1",
                canonical_detection_reference="fdet-b1",
            )
            detection_reference_2, frame_reference_2 = _seed_tracking_evidence(
                evidence_root,
                detection_ref_id="d2",
                track_id="track-2",
                canonical_detection_reference="fdet-b2",
                frame_id="f2",
                detection_id="det-2",
                process_nonce="proc-test-2",
            )

            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"

            _write_minimal_intent_artifact(
                artifact_path,
                generated_at=2200.0,
                decisions=[],
                terminal_investigations=[],
            )
            ack_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "generated_at": 2200.0,
                        "acknowledgments": [
                            {
                                "intent_id": "intent-applied",
                                "outcome": "applied",
                                "timestamp": 2199.0,
                                "reason": "mutation_applied",
                                "investigation_id": "inv-ack-applied",
                                "investigation_state": "closed",
                                "investigation_termination_reason": "mutation_applied",
                            },
                            {
                                "intent_id": "intent-rejected",
                                "outcome": "rejected",
                                "timestamp": 2199.1,
                                "reason": "unknown_identity",
                                "investigation_id": "inv-ack-rejected",
                                "investigation_state": "closed",
                                "investigation_termination_reason": "unknown_identity",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )

            late_applied = _base_result(request_id="req-ack-applied", track_id="track-1")
            late_applied["investigation_id"] = "inv-ack-applied"
            late_applied["details"]["identity_claim_id"] = "idclaim-ack-a"
            late_applied["details"]["referenced_detection_reference"] = detection_reference_1
            late_applied["details"]["source_evidence_refs"] = [frame_reference_1]

            late_rejected = _base_result(request_id="req-ack-rejected", track_id="track-2")
            late_rejected["investigation_id"] = "inv-ack-rejected"
            late_rejected["details"]["identity_claim_id"] = "idclaim-ack-b"
            late_rejected["details"]["referenced_detection_reference"] = detection_reference_2
            late_rejected["details"]["source_evidence_refs"] = [frame_reference_2]

            summary = publisher.publish_from_results(
                [late_applied, late_rejected],
                now_timestamp=2201.0,
            )

            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 2)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            reasons = [item.get("reason") for item in payload.get("decisions", [])[-2:]]
            self.assertEqual(reasons, ["investigation_already_terminal", "investigation_already_terminal"])

    def test_legacy_records_missing_investigation_metadata_remain_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"

            _write_minimal_intent_artifact(
                artifact_path,
                generated_at=2300.0,
                decisions=[
                    {
                        "schema_version": "1.1",
                        "generated_at": 2299.0,
                        "verification_request_id": "req-old",
                        "reason": "verification_not_supported",
                        "accepted": False,
                    }
                ],
                terminal_investigations=[
                    {
                        "generated_at": 2298.0,
                        "investigation_state": "closed",
                    }
                ],
            )
            ack_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "generated_at": 2300.0,
                        "acknowledgments": [
                            {
                                "intent_id": "legacy-intent-1",
                                "outcome": "rejected",
                                "timestamp": 2298.0,
                                "reason": "legacy",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )
            result = _base_result(request_id="req-legacy-compatible", track_id="track-1")
            result.pop("investigation_id", None)
            result.pop("investigation_state", None)
            result["details"].pop("investigation_id", None)
            result["details"].pop("investigation_state", None)
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]

            summary = publisher.publish_from_results([result], now_timestamp=2301.0)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected"], 0)

    def test_duplicate_delayed_results_for_terminal_investigation_are_all_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            ack_path = Path(temp_dir) / "tracking_identity_reassociation_ack_ledger.json"

            _write_minimal_intent_artifact(
                artifact_path,
                generated_at=2400.0,
                terminal_investigations=[
                    {
                        "investigation_id": "inv-duplicate-late",
                        "investigation_state": "resolved",
                        "investigation_termination_reason": "verification_contradicted",
                        "generated_at": 2399.0,
                    }
                ],
            )
            ack_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "generated_at": 2400.0,
                        "acknowledgments": [],
                    }
                ),
                encoding="utf-8",
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )

            delayed_a = _base_result(request_id="req-delayed-a", track_id="track-1")
            delayed_a["investigation_id"] = "inv-duplicate-late"
            delayed_a["details"]["identity_claim_id"] = "idclaim-da"
            delayed_a["details"]["referenced_detection_reference"] = detection_reference
            delayed_a["details"]["source_evidence_refs"] = [frame_reference]

            delayed_b = _base_result(request_id="req-delayed-b", track_id="track-1")
            delayed_b["investigation_id"] = "inv-duplicate-late"
            delayed_b["details"]["identity_claim_id"] = "idclaim-db"
            delayed_b["details"]["referenced_detection_reference"] = detection_reference
            delayed_b["details"]["source_evidence_refs"] = [frame_reference]

            summary = publisher.publish_from_results([delayed_a, delayed_b], now_timestamp=2401.0)
            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 2)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            terminal_reasons = [
                item.get("reason")
                for item in payload.get("decisions", [])
                if item.get("verification_request_id") in {"req-delayed-a", "req-delayed-b"}
            ]
            self.assertEqual(terminal_reasons, ["investigation_already_terminal", "investigation_already_terminal"])

    def test_result_without_investigation_id_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            detection_reference, frame_reference = _seed_tracking_evidence(evidence_root)
            artifact_path = Path(temp_dir) / "tracking_identity_reassociation_intents.json"
            publisher = IdentityReassociationIntentPublisher(
                artifact_path=artifact_path,
                evidence_root_dir=evidence_root,
            )

            result = _base_result(request_id="req-legacy")
            result.pop("investigation_id", None)
            result.pop("investigation_state", None)
            result["details"].pop("investigation_id", None)
            result["details"].pop("investigation_state", None)
            result["details"]["referenced_detection_reference"] = detection_reference
            result["details"]["source_evidence_refs"] = [frame_reference]

            summary = publisher.publish_from_results([result], now_timestamp=1600.0)
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected"], 0)

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["intent_count"], 1)
            self.assertTrue(str(payload["intents"][0]["investigation_id"]).startswith("inv-"))

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
