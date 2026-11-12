from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.contracts import BoundingBox, DetectionCandidate, VisualAssociationProposal
from agents.dashboard.identity_reassociation_consumer import IdentityReassociationIntentConsumer
from agents.dashboard import tracking_loop
from agents.orchestration.runtime.identity_reassociation_intents import IdentityReassociationIntentPublisher
from agents.tracking.identity_registry import PersistentIdentityRegistry
from agents.tracking.state_publisher import publish_tracking_radar_state
from agents.tracking.tracking_types import GroundPosition, PixelPosition, RadarTrackSnapshot


"""
Phase 1 deterministic continuity validation using video-derived evidence.

Ground truth scope:
- User-verified: ByteTrack `21 -> 35` is the same physical person for this video.

Evidence boundary:
- Real evidence: frame/track bbox-center-confidence values come from the analyzed video.
- Simulated decisions: reassociation verification outcomes are deterministic test inputs,
  not live provider outputs.
"""


FRAME_309_EVIDENCE = {
    "21": {
        "bbox": [313.45, 893.48, 469.16, 1078.96],
        "center": [391.31, 986.22],
        "conf": 0.658,
    },
    "23": {
        "bbox": [757.34, 953.41, 851.50, 1080.00],
        "center": [804.42, 1016.70],
        "conf": 0.736,
    },
    "24": {
        "bbox": [1143.66, 929.10, 1281.03, 1077.66],
        "center": [1212.34, 1003.38],
        "conf": 0.839,
    },
    "25": {
        "bbox": [1203.82, 472.19, 1333.09, 660.69],
        "center": [1268.45, 566.44],
        "conf": 0.771,
    },
}

FRAME_345_EVIDENCE = {
    "23": {
        "bbox": [757.44, 953.50, 851.04, 1080.00],
        "center": [804.24, 1016.75],
        "conf": 0.746,
    },
    "24": {
        "bbox": [1140.75, 929.01, 1282.93, 1077.72],
        "center": [1211.84, 1003.36],
        "conf": 0.734,
    },
    "25": {
        "bbox": [1224.54, 507.98, 1363.54, 668.59],
        "center": [1294.04, 588.29],
        "conf": 0.764,
    },
    "35": {
        "bbox": [306.22, 909.15, 447.30, 1079.88],
        "center": [376.76, 994.51],
        "conf": 0.743,
    },
}


def _candidate(
    *,
    track_id: str,
    timestamp: float,
    detection_reference: str,
    bbox: list[float],
    center: list[float],
    confidence: float,
) -> DetectionCandidate:
    x1, y1, x2, y2 = bbox
    center_x, center_y = center
    return DetectionCandidate(
        detection_id=f"det-{track_id}-{int(timestamp)}",
        timestamp=timestamp,
        yolo_track_id=track_id,
        label="person",
        detection_confidence=confidence,
        pixel_bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        pixel_center_x=center_x,
        pixel_center_y=center_y,
        relative_depth=0.35,
        ground_x=round(center_x / 1920.0, 6),
        ground_z=round(center_y / 1080.0, 6),
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
        detection_reference=detection_reference,
    )


def _build_candidates(frame_data: dict[str, dict], *, frame: int) -> list[DetectionCandidate]:
    timestamp = float(frame)
    return [
        _candidate(
            track_id=track_id,
            timestamp=timestamp,
            detection_reference=f"fdet-{frame}-0-det-{track_id}-track-{track_id}",
            bbox=payload["bbox"],
            center=payload["center"],
            confidence=payload["conf"],
        )
        for track_id, payload in frame_data.items()
    ]


def _seed_evidence(
    *,
    evidence_root: Path,
    frame_id: str,
    detection_ref_id: str,
    track_id: str,
    detection_id: str,
    canonical_detection_reference: str,
    timestamp: float,
    process_nonce: str = "phase1-video-1",
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
                "width": 1920,
                "height": 1080,
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


def _proposal(
    *,
    detection_id: str,
    track_id: str,
    detection_reference: str,
    app_identity_id: str,
    snapshot_generation: int,
    snapshot_version: int,
    rejected_ids: list[str],
) -> VisualAssociationProposal:
    return VisualAssociationProposal(
        detection_id=detection_id,
        proposed_track_id=track_id,
        association_confidence=0.9,
        occlusion_likelihood=0.2,
        id_switch_risk=0.1,
        depth_conflict_flag=False,
        position_conflict_flag=False,
        uncertainties=[],
        supporting_evidence=["video_reappearance_pattern"],
        detection_reference=detection_reference,
        proposed_app_identity_id=app_identity_id,
        rejected_app_identity_ids=rejected_ids,
        identity_evidence_refs=[
            f"identity_snapshot:gen:{snapshot_generation}:ver:{snapshot_version}",
            f"identity_candidate:{app_identity_id}",
        ],
        identity_snapshot_generation=snapshot_generation,
        identity_snapshot_version=snapshot_version,
        identity_resolution_status="identity_proposed",
    )


class TestPhase1VideoIdentityContinuity(unittest.TestCase):
    def test_video_derived_reassociation_and_radar_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"
            radar_state_before = root / "radar_before.json"
            radar_state_after = root / "radar_after.json"
            radar_history_before = root / "radar_before.jsonl"
            radar_history_after = root / "radar_after.jsonl"

            registry = PersistentIdentityRegistry(retirement_seconds=120.0)

            frame_309_candidates = _build_candidates(FRAME_309_EVIDENCE, frame=309)
            registry.process_detections(now=309.0, detections=frame_309_candidates)

            app_identity_21 = registry.get_identity_for_yolo_track("21")
            app_identity_23 = registry.get_identity_for_yolo_track("23")
            self.assertIsNotNone(app_identity_21)
            self.assertIsNotNone(app_identity_23)
            assert isinstance(app_identity_21, str)
            assert isinstance(app_identity_23, str)

            frame_330_candidates = _build_candidates(
                {"23": FRAME_345_EVIDENCE["23"], "24": FRAME_345_EVIDENCE["24"], "25": FRAME_345_EVIDENCE["25"]},
                frame=330,
            )
            registry.process_detections(now=330.0, detections=frame_330_candidates)

            lost_record = registry.get_record(app_identity_21)
            self.assertIsNotNone(lost_record)
            assert lost_record is not None
            self.assertEqual(lost_record.status, "temporarily_lost")

            frame_345_candidates = _build_candidates(FRAME_345_EVIDENCE, frame=345)
            registry.process_detections(now=345.0, detections=frame_345_candidates)

            provisional_owner_35 = registry.get_identity_for_yolo_track("35")
            self.assertIsNotNone(provisional_owner_35)
            assert isinstance(provisional_owner_35, str)
            self.assertNotEqual(provisional_owner_35, app_identity_21)

            snapshot = registry.snapshot()
            canonical_reference = "fdet-345-0-det-35-track-35"
            evidence_detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="frame-345",
                detection_ref_id="detref-35",
                track_id="35",
                detection_id="det-35-345",
                canonical_detection_reference=canonical_reference,
                timestamp=345.1,
            )

            accepted_proposal = _proposal(
                detection_id="det-35-345",
                track_id="35",
                detection_reference=canonical_reference,
                app_identity_id=app_identity_21,
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                rejected_ids=[provisional_owner_35],
            )
            verification_result = _verification_result_from_proposal(
                request_id="req-phase1-accept",
                track_id="35",
                evidence_message_id="trkmsg-phase1-accept",
                observation_id="verify-phase1-accept",
                verification_timestamp=345.15,
                evidence_detection_reference=evidence_detection_reference,
                frame_reference=frame_reference,
                proposal=accepted_proposal,
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )
            publish_summary = publisher.publish_from_results([verification_result], now_timestamp=345.2)
            self.assertEqual(publish_summary["accepted"], 1)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            consume_summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=frame_345_candidates,
                now_timestamp=345.25,
            )
            self.assertEqual(consume_summary["applied"], 1)
            self.assertEqual(consume_summary["rejected"], 0)

            self.assertEqual(registry.get_identity_for_yolo_track("35"), app_identity_21)
            self.assertEqual(registry.get_identity_for_yolo_track("23"), app_identity_23)

            display_map, summary = tracking_loop._build_stable_identity_display_map(
                registry=registry,
                tracking_candidates=frame_345_candidates,
            )
            self.assertGreaterEqual(summary["mapped_count"], 2)
            self.assertEqual(display_map["35"]["app_identity_id"], app_identity_21)
            self.assertEqual(display_map["23"]["app_identity_id"], app_identity_23)
            mapped_ids = [metadata["app_identity_id"] for metadata in display_map.values()]
            self.assertEqual(len(mapped_ids), len(set(mapped_ids)))

            conflicting_snapshot = registry.snapshot()
            conflicting_proposal = _proposal(
                detection_id="det-35-345",
                track_id="35",
                detection_reference=canonical_reference,
                app_identity_id=app_identity_23,
                snapshot_generation=int(conflicting_snapshot["generation"]),
                snapshot_version=int(conflicting_snapshot["version"]),
                rejected_ids=[app_identity_21],
            )
            conflicting_verification_result = _verification_result_from_proposal(
                request_id="req-phase1-conflict",
                track_id="35",
                evidence_message_id="trkmsg-phase1-conflict",
                observation_id="verify-phase1-conflict",
                verification_timestamp=345.35,
                evidence_detection_reference=evidence_detection_reference,
                frame_reference=frame_reference,
                proposal=conflicting_proposal,
            )
            second_publish_summary = publisher.publish_from_results(
                [conflicting_verification_result],
                now_timestamp=345.4,
            )
            self.assertEqual(second_publish_summary["accepted"], 1)

            second_consume_summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=frame_345_candidates,
                now_timestamp=345.7,
            )
            self.assertEqual(second_consume_summary["rejected"], 1)
            self.assertEqual(registry.get_identity_for_yolo_track("35"), app_identity_21)

            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            reasons = {
                entry.get("reason")
                for entry in ack_payload.get("acknowledgments", [])
                if isinstance(entry, dict)
            }
            self.assertIn("target_track_owned_by_different_identity", reasons)

            radar_tracks = [
                RadarTrackSnapshot(
                    track_id="23",
                    timestamp=345.0,
                    pixel_position=PixelPosition(
                        x=FRAME_345_EVIDENCE["23"]["center"][0],
                        y=FRAME_345_EVIDENCE["23"]["center"][1],
                    ),
                    ground_position=GroundPosition(
                        x=round(FRAME_345_EVIDENCE["23"]["center"][0] / 1920.0, 6),
                        z=round(FRAME_345_EVIDENCE["23"]["center"][1] / 1080.0, 6),
                    ),
                    ground_coordinate_unit="relative_units",
                    position_kind="relative_ground_estimate",
                    quality_flags=[],
                    confidence=FRAME_345_EVIDENCE["23"]["conf"],
                    relative_depth=0.35,
                ),
                RadarTrackSnapshot(
                    track_id="35",
                    timestamp=345.0,
                    pixel_position=PixelPosition(
                        x=FRAME_345_EVIDENCE["35"]["center"][0],
                        y=FRAME_345_EVIDENCE["35"]["center"][1],
                    ),
                    ground_position=GroundPosition(
                        x=round(FRAME_345_EVIDENCE["35"]["center"][0] / 1920.0, 6),
                        z=round(FRAME_345_EVIDENCE["35"]["center"][1] / 1080.0, 6),
                    ),
                    ground_coordinate_unit="relative_units",
                    position_kind="relative_ground_estimate",
                    quality_flags=[],
                    confidence=FRAME_345_EVIDENCE["35"]["conf"],
                    relative_depth=0.35,
                ),
            ]

            publish_tracking_radar_state(
                radar_state_path=radar_state_before,
                radar_depth_coverage_history_path=radar_history_before,
                now=345.5,
                coordinate_system="relative",
                camera_status="connected",
                detector_status="running",
                calibration_status="loaded",
                tracking_agent_status={"status": "idle"},
                last_observation_timestamp=345.5,
                tracking_agent_observation=None,
                tracks=radar_tracks,
            )
            publish_tracking_radar_state(
                radar_state_path=radar_state_after,
                radar_depth_coverage_history_path=radar_history_after,
                now=345.6,
                coordinate_system="relative",
                camera_status="connected",
                detector_status="running",
                calibration_status="loaded",
                tracking_agent_status={"status": "idle"},
                last_observation_timestamp=345.6,
                tracking_agent_observation=None,
                tracks=radar_tracks,
            )

            before_payload = json.loads(radar_state_before.read_text(encoding="utf-8"))
            after_payload = json.loads(radar_state_after.read_text(encoding="utf-8"))
            self.assertEqual(before_payload["active_track_count"], 2)
            self.assertEqual(after_payload["active_track_count"], 2)

            before_coords = {
                track["track_id"]: (
                    track["ground_position"]["x"],
                    track["ground_position"]["z"],
                    track["pixel_position"]["x"],
                    track["pixel_position"]["y"],
                )
                for track in before_payload["tracks"]
            }
            after_coords = {
                track["track_id"]: (
                    track["ground_position"]["x"],
                    track["ground_position"]["z"],
                    track["pixel_position"]["x"],
                    track["pixel_position"]["y"],
                )
                for track in after_payload["tracks"]
            }
            self.assertEqual(before_coords, after_coords)

    def test_video_derived_negative_distinct_people_do_not_merge_by_proximity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            intents_path = root / "tracking_identity_reassociation_intents.json"
            ack_path = root / "tracking_identity_reassociation_ack_ledger.json"

            registry = PersistentIdentityRegistry(retirement_seconds=120.0)
            frame_345_candidates = _build_candidates(FRAME_345_EVIDENCE, frame=345)
            registry.process_detections(now=345.0, detections=frame_345_candidates)

            app_identity_23 = registry.get_identity_for_yolo_track("23")
            app_identity_24 = registry.get_identity_for_yolo_track("24")
            self.assertIsNotNone(app_identity_23)
            self.assertIsNotNone(app_identity_24)
            assert isinstance(app_identity_23, str)
            assert isinstance(app_identity_24, str)
            self.assertNotEqual(app_identity_23, app_identity_24)

            snapshot = registry.snapshot()
            canonical_reference = "fdet-345-0-det-24-track-24"
            evidence_detection_reference, frame_reference = _seed_evidence(
                evidence_root=evidence_root,
                frame_id="frame-345-neg",
                detection_ref_id="detref-24-neg",
                track_id="24",
                detection_id="det-24-345",
                canonical_detection_reference=canonical_reference,
                timestamp=345.1,
            )

            unsupported_merge_proposal = _proposal(
                detection_id="det-24-345",
                track_id="24",
                detection_reference=canonical_reference,
                app_identity_id=app_identity_23,
                snapshot_generation=int(snapshot["generation"]),
                snapshot_version=int(snapshot["version"]),
                rejected_ids=[app_identity_24],
            )
            unsupported_merge_verification_result = _verification_result_from_proposal(
                request_id="req-phase1-negative-merge",
                track_id="24",
                evidence_message_id="trkmsg-phase1-negative-merge",
                observation_id="verify-phase1-negative-merge",
                verification_timestamp=345.15,
                evidence_detection_reference=evidence_detection_reference,
                frame_reference=frame_reference,
                proposal=unsupported_merge_proposal,
            )

            publisher = IdentityReassociationIntentPublisher(
                artifact_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
            )
            publish_summary = publisher.publish_from_results(
                [unsupported_merge_verification_result],
                now_timestamp=345.2,
            )
            self.assertEqual(publish_summary["accepted"], 1)

            consumer = IdentityReassociationIntentConsumer(
                intents_path=intents_path,
                ack_ledger_path=ack_path,
                evidence_root_dir=evidence_root,
                enabled=True,
            )
            consume_summary = consumer.poll_and_apply(
                registry=registry,
                tracking_candidates=frame_345_candidates,
                now_timestamp=345.25,
            )
            self.assertEqual(consume_summary["applied"], 0)
            self.assertEqual(consume_summary["rejected"], 1)

            self.assertEqual(registry.get_identity_for_yolo_track("23"), app_identity_23)
            self.assertEqual(registry.get_identity_for_yolo_track("24"), app_identity_24)

            ack_payload = json.loads(ack_path.read_text(encoding="utf-8"))
            reasons = {
                entry.get("reason")
                for entry in ack_payload.get("acknowledgments", [])
                if isinstance(entry, dict)
            }
            self.assertIn("target_track_owned_by_different_identity", reasons)

            display_map, summary = tracking_loop._build_stable_identity_display_map(
                registry=registry,
                tracking_candidates=frame_345_candidates,
            )
            self.assertGreaterEqual(summary["mapped_count"], 2)
            self.assertEqual(display_map["23"]["app_identity_id"], app_identity_23)
            self.assertEqual(display_map["24"]["app_identity_id"], app_identity_24)
            mapped_ids = [metadata["app_identity_id"] for metadata in display_map.values()]
            self.assertEqual(len(mapped_ids), len(set(mapped_ids)))


if __name__ == "__main__":
    unittest.main()
