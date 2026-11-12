from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.contracts import BoundingBox, DetectionCandidate, VisualAssociationProposal
from agents.dashboard.identity_reassociation_consumer import IdentityReassociationIntentConsumer
from agents.orchestration.runtime.identity_reassociation_intents import IdentityReassociationIntentPublisher
from agents.tracking.identity_registry import PersistentIdentityRegistry


def _candidate(
    *,
    detection_id: str,
    yolo_track_id: str,
    timestamp: float,
    detection_reference: str,
) -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        timestamp=timestamp,
        yolo_track_id=yolo_track_id,
        label="person",
        detection_confidence=0.9,
        pixel_bbox=BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0),
        pixel_center_x=2.0,
        pixel_center_y=4.0,
        relative_depth=0.4,
        ground_x=0.5,
        ground_z=3.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
        detection_reference=detection_reference,
    )


def _seed_evidence(
    *,
    evidence_root: Path,
    frame_id: str,
    detection_ref_id: str,
    track_id: str,
    detection_id: str,
    canonical_detection_reference: str,
    timestamp: float,
    process_nonce: str = "proc-e2e-1",
) -> tuple[str, str]:
    refs_dir = evidence_root / "refs"
    frames_dir = evidence_root / "frames"
    refs_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_reference = f"evidence://frame/{frame_id}"
    frame_path = frames_dir / f"{frame_id}.jpg"
    frame_path.write_bytes(b"jpeg")
    (refs_dir / f"{frame_id}.json").write_text(
        json.dumps(
            {
                "kind": "frame",
                "id": frame_id,
                "reference": frame_reference,
                "timestamp": float(timestamp),
                "created_at": float(timestamp),
                "expires_at": float(timestamp + 600.0),
                "status": "available",
                "frame_path": str(frame_path),
                "width": 16,
                "height": 16,
                "process_nonce": process_nonce,
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
                "timestamp": float(timestamp),
                "created_at": float(timestamp),
                "expires_at": float(timestamp + 600.0),
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


def _verification_result_from_proposal(
    *,
    request_id: str,
    track_id: str,
    evidence_message_id: str,
    observation_id: str,
    verification_timestamp: float,
    evidence_detection_reference: str,
    frame_reference: str,
    proposal: VisualAssociationProposal,
) -> dict:
    return {
        "request_id": request_id,
        "source": "reasoning_cycle",
        "track_id": track_id,
        "evidence_message_id": evidence_message_id,
        "status": "completed",
        "validation_outcome": "accepted",
        "current_applicability": "current",
        "stale_for_current_decisions": False,
        "coordinator_valid": True,
        "details": {
            "verification_observation_id": observation_id,
            "verification_referenced_message_id": evidence_message_id,
            "verification_timestamp": verification_timestamp,
            "verification_status": "supported",
            "identity_consistency": "consistent",
            "visual_content_inspected": True,
            "source_evidence_refs": [frame_reference],
            "identity_claim_id": f"idclaim-{request_id}",
            "identity_claim_status": "usable",
            "referenced_detection_reference": evidence_detection_reference,
            "referenced_proposed_app_identity_id": proposal.proposed_app_identity_id,
            "referenced_identity_snapshot_generation": proposal.identity_snapshot_generation,
            "referenced_identity_snapshot_version": proposal.identity_snapshot_version,
            "identity_claim_evidence_refs": list(proposal.identity_evidence_refs),
            "identity_claim_provenance": {
                "source_agent": "visual_tracking",
                "source_contract": "VisualAssociationUpdate",
                "mutation_authorized": False,
            },
        },
    }


def _build_registry_with_app_140_on_track_191() -> tuple[PersistentIdentityRegistry, list[str]]:
    registry = PersistentIdentityRegistry(retirement_seconds=20.0)
    base_track_ids = [str(value) for value in range(1, 140)] + ["191"] + [str(value) for value in range(140, 150)]
    initial_detections = [
        _candidate(
            detection_id=f"det-{track_id}-g1",
            yolo_track_id=track_id,
            timestamp=1000.0,
            detection_reference=f"fdet-1000000-0-det-{track_id}-track-{track_id}",
        )
        for track_id in base_track_ids
    ]
    registry.process_detections(now=1000.0, detections=initial_detections)
    return registry, base_track_ids


class TestIdentityReassociationOfflineE2E(unittest.TestCase):
    def test_e2e_valid_verified_claim_transfers_track_from_provisional_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry, base_track_ids = _build_registry_with_app_140_on_track_191()
            self.assertEqual(registry.get_identity_for_yolo_track("191"), "app-140")

            frame_two_track_ids = [track_id for track_id in base_track_ids if track_id != "191"] + ["210"]
            frame_two_detections = [
                _candidate(
                    detection_id=f"det-{track_id}-g2",
                    yolo_track_id=track_id,
                    timestamp=1001.0,
                    detection_reference=f"fdet-1001000-0-det-{track_id}-track-{track_id}",
                )
                for track_id in frame_two_track_ids
            ]
            registry.process_detections(now=1001.0, detections=frame_two_detections)

            self.assertEqual(registry.get_identity_for_yolo_track("210"), "app-151")
            lost_record = registry.get_record("app-140")
            self.assertIsNotNone(lost_record)
            assert lost_record is not None
            self.assertEqual(lost_record.status, "temporarily_lost")

            snapshot = registry.snapshot()
            canonical_reference = "fdet-1001000-0-det-210-track-210"
            evidence_detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="frame-210",
                detection_ref_id="detref-210",
                track_id="210",
                detection_id="det-210-g2",
                canonical_detection_reference=canonical_reference,
                timestamp=1001.1,
            )

            tracking_agent_proposal = VisualAssociationProposal(
                detection_id="det-210-g2",
                proposed_track_id="210",
                association_confidence=0.92,
                occlusion_likelihood=0.15,
                id_switch_risk=0.05,
                depth_conflict_flag=False,
                position_conflict_flag=False,
                uncertainties=[],
                supporting_evidence=["reappearance_pattern"],
                detection_reference=canonical_reference,
                proposed_app_identity_id="app-140",
                rejected_app_identity_ids=["app-151"],
                identity_evidence_refs=[
                    f"identity_snapshot:gen:{int(snapshot['generation'])}:ver:{int(snapshot['version'])}",
                    "identity_candidate:app-140",
                ],
                identity_snapshot_generation=int(snapshot["generation"]),
                identity_snapshot_version=int(snapshot["version"]),
                identity_resolution_status="identity_proposed",
            )
            verification_result = _verification_result_from_proposal(
                request_id="req-e2e-1",
                track_id="210",
                evidence_message_id="trkmsg-e2e-1",
                observation_id="verify-e2e-1",
                verification_timestamp=1001.15,
                evidence_detection_reference=evidence_detection_reference,
                frame_reference=frame_reference,
                proposal=tracking_agent_proposal,
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )
            publish_summary = publisher.publish_from_results([verification_result], now_timestamp=1001.2)
            self.assertEqual(publish_summary["accepted"], 1)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            consume_summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(
                        detection_id="det-210-g2",
                        yolo_track_id="210",
                        timestamp=1001.2,
                        detection_reference=canonical_reference,
                    )
                ],
                now_timestamp=1001.25,
            )

            self.assertEqual(consume_summary["processed"], 1)
            self.assertEqual(consume_summary["applied"], 1)
            self.assertEqual(consume_summary["rejected"], 0)

            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertEqual(ack_payload["acknowledgments"][-1]["reason"], "provisional_transfer_applied")
            self.assertIsNone(registry.get_identity_for_yolo_track("191"))
            self.assertEqual(registry.get_identity_for_yolo_track("210"), "app-140")

            recovered_record = registry.get_record("app-140")
            provisional_record = registry.get_record("app-151")
            self.assertIsNotNone(recovered_record)
            self.assertIsNotNone(provisional_record)
            assert recovered_record is not None
            assert provisional_record is not None

            self.assertEqual(recovered_record.status, "visible")
            self.assertEqual(recovered_record.yolo_track_id, "210")
            self.assertEqual(provisional_record.status, "temporarily_lost")
            self.assertNotEqual(provisional_record.yolo_track_id, "210")

            provisional_history_events = [item.event for item in provisional_record.history]
            self.assertIn("provisional_transfer_released_track", provisional_history_events)

    def test_negative_distinct_visible_objects_remain_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry, base_track_ids = _build_registry_with_app_140_on_track_191()
            snapshot = registry.snapshot()
            self.assertEqual(registry.get_identity_for_yolo_track("191"), "app-140")
            self.assertEqual(registry.get_identity_for_yolo_track("149"), "app-150")

            canonical_reference = "fdet-1000000-0-det-149-track-149"
            evidence_detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="frame-149",
                detection_ref_id="detref-149",
                track_id="149",
                detection_id="det-149-g1",
                canonical_detection_reference=canonical_reference,
                timestamp=1000.5,
            )

            tracking_agent_proposal = VisualAssociationProposal(
                detection_id="det-149-g1",
                proposed_track_id="149",
                association_confidence=0.55,
                occlusion_likelihood=0.2,
                id_switch_risk=0.3,
                depth_conflict_flag=False,
                position_conflict_flag=False,
                uncertainties=["ambiguous_similarity"],
                supporting_evidence=["appearance_overlap"],
                detection_reference=canonical_reference,
                proposed_app_identity_id="app-140",
                rejected_app_identity_ids=["app-150"],
                identity_evidence_refs=[
                    f"identity_snapshot:gen:{int(snapshot['generation'])}:ver:{int(snapshot['version'])}"
                ],
                identity_snapshot_generation=int(snapshot["generation"]),
                identity_snapshot_version=int(snapshot["version"]),
                identity_resolution_status="identity_proposed",
            )
            verification_result = _verification_result_from_proposal(
                request_id="req-e2e-2",
                track_id="149",
                evidence_message_id="trkmsg-e2e-2",
                observation_id="verify-e2e-2",
                verification_timestamp=1000.6,
                evidence_detection_reference=evidence_detection_reference,
                frame_reference=frame_reference,
                proposal=tracking_agent_proposal,
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )
            publish_summary = publisher.publish_from_results([verification_result], now_timestamp=1000.7)
            self.assertEqual(publish_summary["accepted"], 1)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            consume_summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(
                        detection_id=f"det-{track_id}-g1",
                        yolo_track_id=track_id,
                        timestamp=1000.8,
                        detection_reference=f"fdet-1000000-0-det-{track_id}-track-{track_id}",
                    )
                    for track_id in base_track_ids
                ],
                now_timestamp=1000.8,
            )

            self.assertEqual(consume_summary["applied"], 0)
            self.assertEqual(consume_summary["rejected"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("191"), "app-140")
            self.assertEqual(registry.get_identity_for_yolo_track("149"), "app-150")

    def test_stale_snapshot_binding_is_rejected_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry, base_track_ids = _build_registry_with_app_140_on_track_191()
            registry.process_detections(
                now=1001.0,
                detections=[
                    _candidate(
                        detection_id=f"det-{track_id}-g2",
                        yolo_track_id=track_id,
                        timestamp=1001.0,
                        detection_reference=f"fdet-1001000-0-det-{track_id}-track-{track_id}",
                    )
                    for track_id in base_track_ids
                ],
            )

            current_snapshot = registry.snapshot()
            canonical_reference = "fdet-1001000-0-det-191-track-191"
            evidence_detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="frame-191",
                detection_ref_id="detref-191",
                track_id="191",
                detection_id="det-191-g2",
                canonical_detection_reference=canonical_reference,
                timestamp=1001.1,
            )

            tracking_agent_proposal = VisualAssociationProposal(
                detection_id="det-191-g2",
                proposed_track_id="191",
                association_confidence=0.8,
                occlusion_likelihood=0.1,
                id_switch_risk=0.1,
                depth_conflict_flag=False,
                position_conflict_flag=False,
                uncertainties=[],
                supporting_evidence=["trajectory_match"],
                detection_reference=canonical_reference,
                proposed_app_identity_id="app-140",
                rejected_app_identity_ids=[],
                identity_evidence_refs=["identity_snapshot:gen:0:ver:0"],
                identity_snapshot_generation=max(0, int(current_snapshot["generation"]) - 1),
                identity_snapshot_version=max(0, int(current_snapshot["version"]) - 1),
                identity_resolution_status="identity_proposed",
            )
            verification_result = _verification_result_from_proposal(
                request_id="req-e2e-3",
                track_id="191",
                evidence_message_id="trkmsg-e2e-3",
                observation_id="verify-e2e-3",
                verification_timestamp=1001.2,
                evidence_detection_reference=evidence_detection_reference,
                frame_reference=frame_reference,
                proposal=tracking_agent_proposal,
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )
            publish_summary = publisher.publish_from_results([verification_result], now_timestamp=1001.25)
            self.assertEqual(publish_summary["accepted"], 1)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            consume_summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(
                        detection_id="det-191-g2",
                        yolo_track_id="191",
                        timestamp=1001.3,
                        detection_reference=canonical_reference,
                    )
                ],
                now_timestamp=1001.3,
            )

            self.assertEqual(consume_summary["applied"], 0)
            self.assertEqual(consume_summary["rejected"], 1)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertIn(
                ack_payload["acknowledgments"][-1]["reason"],
                {"stale_snapshot_generation", "stale_snapshot_version"},
            )


if __name__ == "__main__":
    unittest.main()
