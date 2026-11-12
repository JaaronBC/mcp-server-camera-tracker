from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.contracts import BoundingBox, DetectionCandidate
from agents.dashboard.identity_reassociation_consumer import IdentityReassociationIntentConsumer
from agents.dashboard.identity_snapshot_bridge import build_identity_snapshot_payload
from agents.orchestration.runtime.identity_reassociation_intents import (
    INTENT_ACK_LEDGER_SCHEMA_VERSION,
    INTENT_ARTIFACT_SCHEMA_VERSION,
    IdentityReassociationMutationIntent,
)
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
    process_nonce: str = "proc-1",
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


def _build_intent(
    *,
    intent_id: str,
    target_yolo_track_id: str,
    app_identity_id: str,
    canonical_detection_reference: str,
    evidence_detection_reference: str,
    frame_reference: str,
    process_nonce: str,
    detection_ref_id: str,
    snapshot_generation: int,
    snapshot_version: int,
    generated_at: float,
    expires_at: float,
) -> IdentityReassociationMutationIntent:
    return IdentityReassociationMutationIntent(
        schema_version=INTENT_ARTIFACT_SCHEMA_VERSION,
        intent_id=intent_id,
        generated_at=generated_at,
        expires_at=expires_at,
        verification_request_id=f"req-{intent_id}",
        source="reasoning_cycle",
        evidence_message_id=f"trkmsg-{intent_id}",
        target_yolo_track_id=target_yolo_track_id,
        claim_id=f"claim-{intent_id}",
        app_identity_id=app_identity_id,
        detection_reference=canonical_detection_reference,
        evidence_detection_reference=evidence_detection_reference,
        frame_reference=frame_reference,
        capture_context={
            "process_nonce": process_nonce,
            "evidence_detection_ref_id": detection_ref_id,
        },
        source_evidence_refs=[frame_reference],
        identity_evidence_refs=[f"identity_snapshot:gen:{snapshot_generation}:ver:{snapshot_version}"],
        identity_snapshot_generation=snapshot_generation,
        identity_snapshot_version=snapshot_version,
        verification_provenance={
            "verification_observation_id": f"verify-{intent_id}",
            "verification_referenced_message_id": f"trkmsg-{intent_id}",
            "verification_timestamp": generated_at,
            "verification_status": "supported",
            "identity_consistency": "consistent",
            "visual_content_inspected": True,
        },
        identity_claim_provenance={
            "source_agent": "visual_tracking",
            "source_contract": "VisualAssociationUpdate",
            "mutation_authorized": False,
        },
    )


def _write_intents(path: Path, intents: list[IdentityReassociationMutationIntent], now: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": INTENT_ARTIFACT_SCHEMA_VERSION,
        "generated_at": float(now),
        "intent_count": len(intents),
        "decision_count": 0,
        "intent_ttl_seconds": 45.0,
        "intents": [item.to_dict() for item in intents],
        "decisions": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    backup_path = path.with_name(f"{path.name}.bak")
    backup_path.write_text(json.dumps(payload), encoding="utf-8")


class TestIdentityReassociationIntentConsumer(unittest.TestCase):
    def test_valid_reassociation_applies_temporarily_lost_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            old_detection = _candidate(
                detection_id="det-old",
                yolo_track_id="track-old",
                timestamp=1000.0,
                detection_reference="fdet-old",
            )
            registry.process_detections(now=1000.0, detections=[old_detection])
            app_identity_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(app_identity_id)
            assert isinstance(app_identity_id, str)

            registry.process_detections(now=1001.0, detections=[])

            snapshot = registry.snapshot()
            detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-new",
                detection_ref_id="d-new",
                track_id="track-new",
                detection_id="det-new",
                canonical_detection_reference="fdet-new",
                timestamp=1002.0,
            )
            intent = _build_intent(
                intent_id="intent-1",
                target_yolo_track_id="track-new",
                app_identity_id=app_identity_id,
                canonical_detection_reference="fdet-new",
                evidence_detection_reference=detection_reference,
                frame_reference=frame_reference,
                process_nonce="proc-1",
                detection_ref_id="d-new",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=1002.0,
                expires_at=1100.0,
            )
            _write_intents(intents_path, [intent], now=1002.0)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(
                        detection_id="det-new",
                        yolo_track_id="track-new",
                        timestamp=1002.0,
                        detection_reference="fdet-new",
                    )
                ],
                now_timestamp=1002.5,
            )

            self.assertEqual(summary["applied"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("track-new"), app_identity_id)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertEqual(ack_payload["schema_version"], INTENT_ACK_LEDGER_SCHEMA_VERSION)
            self.assertEqual(ack_payload["acknowledgments"][-1]["outcome"], "applied")

    def test_rejects_stale_snapshot_generation_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"
            registry = PersistentIdentityRegistry()
            registry.process_detections(
                now=10.0,
                detections=[_candidate(detection_id="det-1", yolo_track_id="track-1", timestamp=10.0, detection_reference="fdet-1")],
            )
            app_identity_id = registry.get_identity_for_yolo_track("track-1")
            self.assertIsNotNone(app_identity_id)
            assert isinstance(app_identity_id, str)

            detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-1",
                detection_ref_id="d-1",
                track_id="track-1",
                detection_id="det-1",
                canonical_detection_reference="fdet-1",
                timestamp=11.0,
            )

            stale_generation = _build_intent(
                intent_id="intent-g",
                target_yolo_track_id="track-1",
                app_identity_id=app_identity_id,
                canonical_detection_reference="fdet-1",
                evidence_detection_reference=detection_reference,
                frame_reference=frame_reference,
                process_nonce="proc-1",
                detection_ref_id="d-1",
                snapshot_generation=0,
                snapshot_version=999,
                generated_at=11.0,
                expires_at=30.0,
            )
            current_snapshot = registry.snapshot()
            stale_version = _build_intent(
                intent_id="intent-v",
                target_yolo_track_id="track-1",
                app_identity_id=app_identity_id,
                canonical_detection_reference="fdet-1",
                evidence_detection_reference=detection_reference,
                frame_reference=frame_reference,
                process_nonce="proc-1",
                detection_ref_id="d-1",
                snapshot_generation=int(current_snapshot["generation"]),
                snapshot_version=int(current_snapshot["version"]) - 1,
                generated_at=11.0,
                expires_at=30.0,
            )
            _write_intents(intents_path, [stale_generation, stale_version], now=11.0)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-1", yolo_track_id="track-1", timestamp=11.0, detection_reference="fdet-1")
                ],
                now_timestamp=11.5,
            )
            self.assertEqual(summary["rejected"], 2)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            reasons = {item.get("reason") for item in ack_payload.get("acknowledgments", [])}
            self.assertIn("stale_snapshot_generation", reasons)
            self.assertIn("stale_snapshot_version", reasons)

    def test_rejects_target_track_id_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"
            registry = PersistentIdentityRegistry(retirement_seconds=20.0)

            registry.process_detections(
                now=20.0,
                detections=[_candidate(detection_id="det-old", yolo_track_id="track-old", timestamp=20.0, detection_reference="fdet-old")],
            )
            app_identity_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(app_identity_id)
            assert isinstance(app_identity_id, str)
            registry.process_detections(now=21.0, detections=[])

            detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-reuse",
                detection_ref_id="d-reuse",
                track_id="track-new",
                detection_id="det-new",
                canonical_detection_reference="fdet-intended",
                timestamp=22.0,
            )
            snapshot = registry.snapshot()
            intent = _build_intent(
                intent_id="intent-reuse",
                target_yolo_track_id="track-new",
                app_identity_id=app_identity_id,
                canonical_detection_reference="fdet-intended",
                evidence_detection_reference=detection_reference,
                frame_reference=frame_reference,
                process_nonce="proc-1",
                detection_ref_id="d-reuse",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=22.0,
                expires_at=60.0,
            )
            _write_intents(intents_path, [intent], now=22.0)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(
                        detection_id="det-reused",
                        yolo_track_id="track-new",
                        timestamp=22.0,
                        detection_reference="fdet-different-now",
                    )
                ],
                now_timestamp=22.2,
            )
            self.assertEqual(summary["rejected"], 1)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ack_payload["acknowledgments"][-1]["reason"],
                "target_track_detection_reference_mismatch",
            )

    def test_rejects_unknown_retired_and_conflicting_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(
                temporary_loss_seconds=0.1,
                retirement_seconds=0.5,
            )
            registry.process_detections(
                now=30.0,
                detections=[
                    _candidate(detection_id="det-a", yolo_track_id="track-a", timestamp=30.0, detection_reference="fdet-a"),
                    _candidate(detection_id="det-b", yolo_track_id="track-b", timestamp=30.0, detection_reference="fdet-b"),
                    _candidate(detection_id="det-c", yolo_track_id="track-c", timestamp=30.0, detection_reference="fdet-c"),
                ],
            )
            app_a = registry.get_identity_for_yolo_track("track-a")
            app_b = registry.get_identity_for_yolo_track("track-b")
            app_c = registry.get_identity_for_yolo_track("track-c")
            self.assertIsNotNone(app_a)
            self.assertIsNotNone(app_b)
            self.assertIsNotNone(app_c)
            assert isinstance(app_a, str)
            assert isinstance(app_b, str)
            assert isinstance(app_c, str)

            registry.process_detections(
                now=31.0,
                detections=[
                    _candidate(detection_id="det-b", yolo_track_id="track-b", timestamp=31.0, detection_reference="fdet-b"),
                    _candidate(detection_id="det-c", yolo_track_id="track-c", timestamp=31.0, detection_reference="fdet-c"),
                ],
            )
            registry.process_detections(
                now=32.0,
                detections=[
                    _candidate(detection_id="det-b", yolo_track_id="track-b", timestamp=32.0, detection_reference="fdet-b"),
                    _candidate(detection_id="det-c", yolo_track_id="track-c", timestamp=32.0, detection_reference="fdet-c"),
                ],
            )
            retired_record = registry.get_record(app_a)
            self.assertIsNotNone(retired_record)
            self.assertEqual(retired_record.status, "retired")

            d_unknown, f_unknown = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-u",
                detection_ref_id="d-u",
                track_id="track-u",
                detection_id="det-u",
                canonical_detection_reference="fdet-u",
                timestamp=31.5,
            )
            d_retired, f_retired = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-r",
                detection_ref_id="d-r",
                track_id="track-r",
                detection_id="det-r",
                canonical_detection_reference="fdet-r",
                timestamp=31.5,
            )
            d_conflict, f_conflict = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-c",
                detection_ref_id="d-c",
                track_id="track-b",
                detection_id="det-c",
                canonical_detection_reference="fdet-b",
                timestamp=31.5,
            )
            snapshot = registry.snapshot()
            intents = [
                _build_intent(
                    intent_id="intent-unknown",
                    target_yolo_track_id="track-u",
                    app_identity_id="app-missing",
                    canonical_detection_reference="fdet-u",
                    evidence_detection_reference=d_unknown,
                    frame_reference=f_unknown,
                    process_nonce="proc-1",
                    detection_ref_id="d-u",
                    snapshot_generation=int(snapshot["generation"]),
                    snapshot_version=int(snapshot["version"]),
                    generated_at=31.5,
                    expires_at=80.0,
                ),
                _build_intent(
                    intent_id="intent-retired",
                    target_yolo_track_id="track-r",
                    app_identity_id=app_a,
                    canonical_detection_reference="fdet-r",
                    evidence_detection_reference=d_retired,
                    frame_reference=f_retired,
                    process_nonce="proc-1",
                    detection_ref_id="d-r",
                    snapshot_generation=int(snapshot["generation"]),
                    snapshot_version=int(snapshot["version"]),
                    generated_at=31.5,
                    expires_at=80.0,
                ),
                _build_intent(
                    intent_id="intent-conflict",
                    target_yolo_track_id="track-b",
                    app_identity_id=app_c,
                    canonical_detection_reference="fdet-b",
                    evidence_detection_reference=d_conflict,
                    frame_reference=f_conflict,
                    process_nonce="proc-1",
                    detection_ref_id="d-c",
                    snapshot_generation=int(snapshot["generation"]),
                    snapshot_version=int(snapshot["version"]),
                    generated_at=31.5,
                    expires_at=80.0,
                ),
            ]
            _write_intents(intents_path, intents, now=31.5)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-u", yolo_track_id="track-u", timestamp=31.6, detection_reference="fdet-u"),
                    _candidate(detection_id="det-r", yolo_track_id="track-r", timestamp=31.6, detection_reference="fdet-r"),
                    _candidate(detection_id="det-c", yolo_track_id="track-b", timestamp=31.6, detection_reference="fdet-b"),
                    _candidate(detection_id="det-live-c", yolo_track_id="track-c", timestamp=31.6, detection_reference="fdet-c"),
                ],
                now_timestamp=31.6,
            )
            self.assertEqual(summary["rejected"], 3)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            reasons = {item["reason"] for item in ack_payload["acknowledgments"]}
            self.assertIn("unknown_identity", reasons)
            self.assertIn("identity_retired", reasons)
            self.assertIn("target_track_owned_by_different_identity", reasons)
            self.assertEqual(registry.get_identity_for_yolo_track("track-b"), app_b)

    def test_marks_expired_and_preserves_unrelated_acknowledgments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry()
            registry.process_detections(
                now=40.0,
                detections=[_candidate(detection_id="det-1", yolo_track_id="track-1", timestamp=40.0, detection_reference="fdet-1")],
            )
            app_id = registry.get_identity_for_yolo_track("track-1")
            self.assertIsNotNone(app_id)
            assert isinstance(app_id, str)
            d1, f1 = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-1",
                detection_ref_id="d-1",
                track_id="track-1",
                detection_id="det-1",
                canonical_detection_reference="fdet-1",
                timestamp=40.0,
            )
            snapshot = registry.snapshot()
            intent = _build_intent(
                intent_id="intent-expired",
                target_yolo_track_id="track-1",
                app_identity_id=app_id,
                canonical_detection_reference="fdet-1",
                evidence_detection_reference=d1,
                frame_reference=f1,
                process_nonce="proc-1",
                detection_ref_id="d-1",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=40.0,
                expires_at=40.1,
            )
            _write_intents(intents_path, [intent], now=40.0)

            ack_path.write_text(
                json.dumps(
                    {
                        "schema_version": INTENT_ACK_LEDGER_SCHEMA_VERSION,
                        "generated_at": 40.0,
                        "acknowledgments": [
                            {
                                "intent_id": "other-intent",
                                "outcome": "rejected",
                                "timestamp": 40.0,
                                "reason": "other_reason",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-1", yolo_track_id="track-1", timestamp=41.0, detection_reference="fdet-1")
                ],
                now_timestamp=41.0,
            )

            self.assertEqual(summary["expired"], 1)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            intent_ids = [item["intent_id"] for item in ack_payload["acknowledgments"]]
            self.assertIn("other-intent", intent_ids)
            self.assertIn("intent-expired", intent_ids)

    def test_duplicate_delivery_process_restart_and_feature_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"
            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            registry.process_detections(
                now=50.0,
                detections=[_candidate(detection_id="det-old", yolo_track_id="track-old", timestamp=50.0, detection_reference="fdet-old")],
            )
            app_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(app_id)
            assert isinstance(app_id, str)
            registry.process_detections(now=51.0, detections=[])

            d_new, f_new = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-new",
                detection_ref_id="d-new",
                track_id="track-new",
                detection_id="det-new",
                canonical_detection_reference="fdet-new",
                timestamp=52.0,
            )
            snapshot = registry.snapshot()
            intent = _build_intent(
                intent_id="intent-dup",
                target_yolo_track_id="track-new",
                app_identity_id=app_id,
                canonical_detection_reference="fdet-new",
                evidence_detection_reference=d_new,
                frame_reference=f_new,
                process_nonce="proc-1",
                detection_ref_id="d-new",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=52.0,
                expires_at=100.0,
            )
            _write_intents(intents_path, [intent], now=52.0)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            first = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=52.1, detection_reference="fdet-new")
                ],
                now_timestamp=52.1,
            )
            self.assertEqual(first["applied"], 1)

            restarted = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            second = restarted.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=52.2, detection_reference="fdet-new")
                ],
                now_timestamp=52.2,
            )
            self.assertEqual(second["processed"], 0)

            disabled = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=False,
            )
            disabled_summary = disabled.poll_and_apply(
                registry=registry,
                tracking_candidates=[],
                now_timestamp=52.3,
            )
            self.assertTrue(disabled_summary["skipped"])
            self.assertEqual(disabled_summary["reason"], "feature_disabled")

    def test_multiple_pending_intents_have_independent_ack_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"
            registry = PersistentIdentityRegistry(retirement_seconds=20.0)

            registry.process_detections(
                now=60.0,
                detections=[_candidate(detection_id="det-old", yolo_track_id="track-old", timestamp=60.0, detection_reference="fdet-old")],
            )
            app_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(app_id)
            assert isinstance(app_id, str)
            registry.process_detections(now=61.0, detections=[])
            snapshot = registry.snapshot()

            d_good, f_good = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-good",
                detection_ref_id="d-good",
                track_id="track-good",
                detection_id="det-good",
                canonical_detection_reference="fdet-good",
                timestamp=62.0,
            )
            d_bad, f_bad = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-bad",
                detection_ref_id="d-bad",
                track_id="track-bad",
                detection_id="det-bad",
                canonical_detection_reference="fdet-bad",
                timestamp=62.0,
            )

            intents = [
                _build_intent(
                    intent_id="intent-good",
                    target_yolo_track_id="track-good",
                    app_identity_id=app_id,
                    canonical_detection_reference="fdet-good",
                    evidence_detection_reference=d_good,
                    frame_reference=f_good,
                    process_nonce="proc-1",
                    detection_ref_id="d-good",
                    snapshot_generation=int(snapshot["generation"]),
                    snapshot_version=int(snapshot["version"]),
                    generated_at=62.0,
                    expires_at=120.0,
                ),
                _build_intent(
                    intent_id="intent-bad",
                    target_yolo_track_id="track-bad",
                    app_identity_id="app-missing",
                    canonical_detection_reference="fdet-bad",
                    evidence_detection_reference=d_bad,
                    frame_reference=f_bad,
                    process_nonce="proc-1",
                    detection_ref_id="d-bad",
                    snapshot_generation=int(snapshot["generation"]),
                    snapshot_version=int(snapshot["version"]),
                    generated_at=62.0,
                    expires_at=120.0,
                ),
            ]
            _write_intents(intents_path, intents, now=62.0)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-good", yolo_track_id="track-good", timestamp=62.1, detection_reference="fdet-good"),
                    _candidate(detection_id="det-bad", yolo_track_id="track-bad", timestamp=62.1, detection_reference="fdet-bad"),
                ],
                now_timestamp=62.1,
            )

            self.assertEqual(summary["applied"], 1)
            self.assertEqual(summary["rejected"], 1)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            outcomes = {
                item["intent_id"]: item["outcome"]
                for item in ack_payload["acknowledgments"]
            }
            self.assertEqual(outcomes.get("intent-good"), "applied")
            self.assertEqual(outcomes.get("intent-bad"), "rejected")

    def test_reassociation_continuity_persists_across_future_frames_and_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            initial_detection = _candidate(
                detection_id="det-initial",
                yolo_track_id="4",
                timestamp=100.0,
                detection_reference="fdet-4",
            )
            registry.process_detections(now=100.0, detections=[initial_detection])
            app_identity_id = registry.get_identity_for_yolo_track("4")
            self.assertIsNotNone(app_identity_id)
            assert isinstance(app_identity_id, str)

            registry.process_detections(now=101.0, detections=[])
            record_lost = registry.get_record(app_identity_id)
            self.assertIsNotNone(record_lost)
            assert record_lost is not None
            self.assertEqual(record_lost.status, "temporarily_lost")

            detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="frame-12",
                detection_ref_id="detref-12",
                track_id="12",
                detection_id="det-reappear",
                canonical_detection_reference="fdet-12",
                timestamp=102.0,
            )
            snapshot_before = registry.snapshot()
            intent = _build_intent(
                intent_id="intent-continuity",
                target_yolo_track_id="12",
                app_identity_id=app_identity_id,
                canonical_detection_reference="fdet-12",
                evidence_detection_reference=detection_reference,
                frame_reference=frame_reference,
                process_nonce="proc-1",
                detection_ref_id="detref-12",
                snapshot_generation=int(snapshot_before["generation"]),
                snapshot_version=int(snapshot_before["version"]),
                generated_at=102.0,
                expires_at=160.0,
            )
            _write_intents(intents_path, [intent], now=102.0)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            apply_summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(
                        detection_id="det-reappear",
                        yolo_track_id="12",
                        timestamp=102.1,
                        detection_reference="fdet-12",
                    )
                ],
                now_timestamp=102.1,
            )

            self.assertEqual(apply_summary["applied"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("12"), app_identity_id)
            self.assertIsNone(registry.get_identity_for_yolo_track("4"))

            registry.process_detections(
                now=103.0,
                detections=[
                    _candidate(
                        detection_id="det-next-1",
                        yolo_track_id="12",
                        timestamp=103.0,
                        detection_reference="fdet-12-next-1",
                    )
                ],
            )
            registry.process_detections(
                now=104.0,
                detections=[
                    _candidate(
                        detection_id="det-next-2",
                        yolo_track_id="12",
                        timestamp=104.0,
                        detection_reference="fdet-12-next-2",
                    )
                ],
            )

            self.assertEqual(registry.get_identity_for_yolo_track("12"), app_identity_id)
            record_after = registry.get_record(app_identity_id)
            self.assertIsNotNone(record_after)
            assert record_after is not None
            self.assertEqual(record_after.status, "visible")
            self.assertEqual(record_after.yolo_track_id, "12")

            snapshot_payload = build_identity_snapshot_payload(
                registry=registry,
                now=104.1,
                freshness_window_seconds=3.0,
                max_identities=64,
                max_history_per_identity=64,
                max_historical_track_ids=64,
                max_payload_bytes=262_144,
            )
            identities_payload = snapshot_payload.get("identities")
            self.assertIsInstance(identities_payload, list)
            assert isinstance(identities_payload, list)

            matched_identity_payload = None
            for item in identities_payload:
                if isinstance(item, dict) and item.get("app_identity_id") == app_identity_id:
                    matched_identity_payload = item
                    break

            self.assertIsNotNone(matched_identity_payload)
            assert isinstance(matched_identity_payload, dict)
            self.assertEqual(matched_identity_payload.get("yolo_track_id"), "12")

            observation_history = matched_identity_payload.get("observation_history")
            self.assertIsInstance(observation_history, list)
            assert isinstance(observation_history, list)
            history_events = [
                item.get("event")
                for item in observation_history
                if isinstance(item, dict)
            ]
            self.assertIn("mutation_applied", history_events)
            self.assertIn("visible", history_events)

            active_on_track_12 = [
                item
                for item in identities_payload
                if isinstance(item, dict)
                and item.get("yolo_track_id") == "12"
                and item.get("status") != "retired"
            ]
            self.assertEqual(len(active_on_track_12), 1)
            self.assertEqual(active_on_track_12[0].get("app_identity_id"), app_identity_id)

            restarted_consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            restart_summary = restarted_consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(
                        detection_id="det-next-2",
                        yolo_track_id="12",
                        timestamp=104.2,
                        detection_reference="fdet-12",
                    )
                ],
                now_timestamp=104.2,
            )
            self.assertEqual(restart_summary["processed"], 0)
            self.assertEqual(registry.get_identity_for_yolo_track("12"), app_identity_id)

    def test_verified_provisional_conflict_transfers_and_repeat_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            registry.process_detections(
                now=200.0,
                detections=[_candidate(detection_id="det-old", yolo_track_id="track-old", timestamp=200.0, detection_reference="fdet-old")],
            )
            source_identity_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(source_identity_id)
            assert isinstance(source_identity_id, str)

            registry.process_detections(
                now=201.0,
                detections=[_candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=201.0, detection_reference="fdet-new")],
            )
            provisional_owner_id = registry.get_identity_for_yolo_track("track-new")
            self.assertIsNotNone(provisional_owner_id)
            assert isinstance(provisional_owner_id, str)
            self.assertNotEqual(source_identity_id, provisional_owner_id)

            snapshot = registry.snapshot()
            d_new, f_new = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-new",
                detection_ref_id="d-new",
                track_id="track-new",
                detection_id="det-new",
                canonical_detection_reference="fdet-new",
                timestamp=201.0,
            )
            intent = _build_intent(
                intent_id="intent-transfer",
                target_yolo_track_id="track-new",
                app_identity_id=source_identity_id,
                canonical_detection_reference="fdet-new",
                evidence_detection_reference=d_new,
                frame_reference=f_new,
                process_nonce="proc-1",
                detection_ref_id="d-new",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=201.1,
                expires_at=260.0,
            )
            _write_intents(intents_path, [intent], now=201.1)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            first = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=201.2, detection_reference="fdet-new")
                ],
                now_timestamp=201.2,
            )

            self.assertEqual(first["applied"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("track-new"), source_identity_id)
            self.assertIsNone(registry.get_identity_for_yolo_track("track-old"))

            source_record = registry.get_record(source_identity_id)
            owner_record = registry.get_record(provisional_owner_id)
            self.assertIsNotNone(source_record)
            self.assertIsNotNone(owner_record)
            assert source_record is not None
            assert owner_record is not None
            self.assertEqual(source_record.status, "visible")
            self.assertEqual(owner_record.status, "temporarily_lost")
            self.assertIn("provisional_transfer_released_track", [item.event for item in owner_record.history])

            restarted_consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            second = restarted_consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=201.3, detection_reference="fdet-new")
                ],
                now_timestamp=201.3,
            )
            self.assertEqual(second["processed"], 0)

    def test_transfer_conflict_rejected_when_owner_is_not_provisional(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            registry.process_detections(
                now=300.0,
                detections=[_candidate(detection_id="det-old", yolo_track_id="track-old", timestamp=300.0, detection_reference="fdet-old")],
            )
            source_identity_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(source_identity_id)
            assert isinstance(source_identity_id, str)
            registry.process_detections(
                now=301.0,
                detections=[_candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=301.0, detection_reference="fdet-new")],
            )
            provisional_owner_id = registry.get_identity_for_yolo_track("track-new")
            self.assertIsNotNone(provisional_owner_id)
            assert isinstance(provisional_owner_id, str)

            with registry._lock:
                registry._records[provisional_owner_id].provisional = False

            snapshot = registry.snapshot()
            d_new, f_new = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-new",
                detection_ref_id="d-new",
                track_id="track-new",
                detection_id="det-new",
                canonical_detection_reference="fdet-new",
                timestamp=301.0,
            )
            intent = _build_intent(
                intent_id="intent-non-provisional-owner",
                target_yolo_track_id="track-new",
                app_identity_id=source_identity_id,
                canonical_detection_reference="fdet-new",
                evidence_detection_reference=d_new,
                frame_reference=f_new,
                process_nonce="proc-1",
                detection_ref_id="d-new",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=301.1,
                expires_at=360.0,
            )
            _write_intents(intents_path, [intent], now=301.1)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=301.2, detection_reference="fdet-new")
                ],
                now_timestamp=301.2,
            )

            self.assertEqual(summary["applied"], 0)
            self.assertEqual(summary["rejected"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("track-new"), provisional_owner_id)
            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertEqual(
                ack_payload["acknowledgments"][-1]["reason"],
                "target_track_owned_by_different_identity",
            )

    def test_transfer_conflict_rejected_when_source_identity_not_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            registry.process_detections(
                now=400.0,
                detections=[
                    _candidate(detection_id="det-a", yolo_track_id="track-a", timestamp=400.0, detection_reference="fdet-a"),
                    _candidate(detection_id="det-b", yolo_track_id="track-b", timestamp=400.0, detection_reference="fdet-b"),
                ],
            )
            source_identity_id = registry.get_identity_for_yolo_track("track-a")
            owner_identity_id = registry.get_identity_for_yolo_track("track-b")
            self.assertIsNotNone(source_identity_id)
            self.assertIsNotNone(owner_identity_id)
            assert isinstance(source_identity_id, str)
            assert isinstance(owner_identity_id, str)

            snapshot = registry.snapshot()
            d_b, f_b = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-b",
                detection_ref_id="d-b",
                track_id="track-b",
                detection_id="det-b",
                canonical_detection_reference="fdet-b",
                timestamp=400.1,
            )
            intent = _build_intent(
                intent_id="intent-source-visible",
                target_yolo_track_id="track-b",
                app_identity_id=source_identity_id,
                canonical_detection_reference="fdet-b",
                evidence_detection_reference=d_b,
                frame_reference=f_b,
                process_nonce="proc-1",
                detection_ref_id="d-b",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=400.2,
                expires_at=460.0,
            )
            _write_intents(intents_path, [intent], now=400.2)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-b", yolo_track_id="track-b", timestamp=400.3, detection_reference="fdet-b")
                ],
                now_timestamp=400.3,
            )

            self.assertEqual(summary["applied"], 0)
            self.assertEqual(summary["rejected"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("track-b"), owner_identity_id)

    def test_transfer_conflict_rejected_when_verification_provenance_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            registry.process_detections(
                now=500.0,
                detections=[_candidate(detection_id="det-old", yolo_track_id="track-old", timestamp=500.0, detection_reference="fdet-old")],
            )
            source_identity_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(source_identity_id)
            assert isinstance(source_identity_id, str)

            registry.process_detections(
                now=501.0,
                detections=[_candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=501.0, detection_reference="fdet-new")],
            )
            owner_identity_id = registry.get_identity_for_yolo_track("track-new")
            self.assertIsNotNone(owner_identity_id)
            assert isinstance(owner_identity_id, str)

            snapshot = registry.snapshot()
            d_new, f_new = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-new",
                detection_ref_id="d-new",
                track_id="track-new",
                detection_id="det-new",
                canonical_detection_reference="fdet-new",
                timestamp=501.1,
            )
            intent = _build_intent(
                intent_id="intent-unverified",
                target_yolo_track_id="track-new",
                app_identity_id=source_identity_id,
                canonical_detection_reference="fdet-new",
                evidence_detection_reference=d_new,
                frame_reference=f_new,
                process_nonce="proc-1",
                detection_ref_id="d-new",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=501.2,
                expires_at=560.0,
            )
            intent.verification_provenance["verification_status"] = "contradicted"
            _write_intents(intents_path, [intent], now=501.2)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=501.3, detection_reference="fdet-new")
                ],
                now_timestamp=501.3,
            )

            self.assertEqual(summary["applied"], 0)
            self.assertEqual(summary["rejected"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("track-new"), owner_identity_id)

    def test_malformed_or_corrupt_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            backup_path = intents_path.with_name(f"{intents_path.name}.bak")
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            intents_path.write_text("{bad-json", encoding="utf-8")
            backup_path.write_text("{bad-json", encoding="utf-8")

            registry = PersistentIdentityRegistry()
            before = registry.snapshot()

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[],
                now_timestamp=70.0,
            )
            self.assertTrue(summary["load_failed"])
            after = registry.snapshot()
            self.assertEqual(before, after)

    def test_interruption_between_mutation_and_ack_is_safe_on_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=20.0)
            registry.process_detections(
                now=80.0,
                detections=[_candidate(detection_id="det-old", yolo_track_id="track-old", timestamp=80.0, detection_reference="fdet-old")],
            )
            app_id = registry.get_identity_for_yolo_track("track-old")
            self.assertIsNotNone(app_id)
            assert isinstance(app_id, str)
            registry.process_detections(now=81.0, detections=[])

            pre_mutation_snapshot = registry.snapshot()
            manual_mutation = registry.apply_identity_mutation(
                app_identity_id=app_id,
                new_yolo_track_id="track-new",
                now=81.5,
                expected_generation_at_least=int(pre_mutation_snapshot["generation"]),
                expected_version=int(pre_mutation_snapshot["version"]),
            )
            self.assertTrue(manual_mutation.accepted)

            d_new, f_new = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="f-new",
                detection_ref_id="d-new",
                track_id="track-new",
                detection_id="det-new",
                canonical_detection_reference="fdet-new",
                timestamp=82.0,
            )
            snapshot = registry.snapshot()
            intent = _build_intent(
                intent_id="intent-crash",
                target_yolo_track_id="track-new",
                app_identity_id=app_id,
                canonical_detection_reference="fdet-new",
                evidence_detection_reference=d_new,
                frame_reference=f_new,
                process_nonce="proc-1",
                detection_ref_id="d-new",
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                generated_at=82.0,
                expires_at=140.0,
            )
            _write_intents(intents_path, [intent], now=82.0)

            class _FailAfterMutationConsumer(IdentityReassociationIntentConsumer):
                def _write_ack_ledger(self, payload: dict[str, object]) -> None:
                    raise OSError("simulated_ack_write_failure")

            failing_consumer = _FailAfterMutationConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )

            with self.assertRaises(OSError):
                failing_consumer.poll_and_apply(
                    registry=registry,
                    tracking_candidates=[
                        _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=82.1, detection_reference="fdet-new")
                    ],
                    now_timestamp=82.1,
                )

            self.assertEqual(registry.get_identity_for_yolo_track("track-new"), app_id)

            retry_consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            retry_summary = retry_consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=[
                    _candidate(detection_id="det-new", yolo_track_id="track-new", timestamp=82.2, detection_reference="fdet-new")
                ],
                now_timestamp=82.2,
            )
            self.assertEqual(retry_summary["applied"], 1)

            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            self.assertEqual(ack_payload["acknowledgments"][-1]["reason"], "already_applied")


if __name__ == "__main__":
    unittest.main()
