from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from agents.contracts import BoundingBox, DetectionCandidate
from agents.coordinator import AgentCoordinator
from agents.observation.evidence_consumption import ObservationEvidenceRequest, consume_tracking_evidence_by_message_id
from agents.tracking.evidence_packaging import TrackingEvidencePackager
from agents.tracking.evidence_store import TrackingEvidenceResolver, TrackingEvidenceStore
from agents.tracking.state_publisher import publish_tracking_radar_state
from agents.tracking.tracking_types import GroundPosition, RadarTrackSnapshot


class _InvalidCoordinator:
    def validate_tracking_evidence_envelope(self, envelope, *, supported_schema_versions=None, known_message_ids=None):
        return {
            "valid": False,
            "errors": ["forced_invalid"],
            "warnings": [],
            "message_id": envelope.message_id,
            "track_id": envelope.track_id,
        }


class TestTrackingEvidencePackaging(unittest.TestCase):
    def _detection(self, track_id: str, detection_id: str, timestamp: float) -> DetectionCandidate:
        return DetectionCandidate(
            detection_id=detection_id,
            timestamp=timestamp,
            yolo_track_id=track_id,
            label="person",
            detection_confidence=0.92,
            pixel_bbox=BoundingBox(x1=100.0, y1=120.0, x2=180.0, y2=260.0),
            pixel_center_x=140.0,
            pixel_center_y=260.0,
            relative_depth=0.4,
            ground_x=1.5,
            ground_z=5.0,
            ground_coordinate_unit="meters",
            position_kind="calibrated_ground_plane",
            motion_history=[],
        )

    def _wait_for_frame_available(self, resolver: TrackingEvidenceResolver, frame_reference: str, timeout_seconds: float = 1.0) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            result = resolver.resolve_reference(frame_reference)
            if result.get("valid"):
                return True
            time.sleep(0.01)
        return False

    def test_valid_envelope_from_tracking_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = TrackingEvidenceStore(root / "tracking_evidence")
            packager = TrackingEvidencePackager(coordinator=AgentCoordinator(), evidence_store=store)

            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame_evidence = store.register_frame(frame=frame, timestamp=1000.0, width=32, height=32)

            result = packager.package_detection(
                detection=self._detection("track-1", "det-1", 1000.0),
                timestamp=1000.0,
                frame_evidence=frame_evidence,
                update_payload={
                    "detection_id": "det-1",
                    "resolved_track_id": "track-1",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )

            self.assertTrue(result.validation["valid"])
            self.assertEqual(result.envelope.track_id, "track-1")
            self.assertEqual(result.envelope.timestamp, 1000.0)
            self.assertEqual(result.envelope.pixel_bbox.x1, 100.0)
            self.assertEqual(result.envelope.pixel_center.x, 140.0)
            self.assertEqual(result.envelope.ground_position.unit, "meters")
            self.assertIsNone(result.envelope.tracking_confidence)
            self.assertEqual(result.envelope.tracking_claims.association_status, "accepted")
            self.assertEqual(result.envelope.tracking_claims.claimed_resolved_track_id, "track-1")
            self.assertEqual(result.envelope.frame_reference, frame_evidence.frame_reference)
            self.assertGreaterEqual(len(result.envelope.source_evidence_refs), 1)

            store.shutdown()

    def test_missing_optional_fields_remain_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = TrackingEvidenceStore(root / "tracking_evidence")
            packager = TrackingEvidencePackager(coordinator=AgentCoordinator(), evidence_store=store)

            candidate = self._detection("track-2", "det-2", 1002.0)
            candidate.relative_depth = None
            candidate.ground_x = None
            candidate.ground_z = None
            candidate.ground_coordinate_unit = None
            candidate.position_kind = "relative_ground_estimate"

            result = packager.package_detection(
                detection=candidate,
                timestamp=1002.0,
                frame_evidence=None,
                update_payload=None,
            )

            self.assertTrue(result.validation["valid"])
            self.assertIsNone(result.envelope.relative_depth)
            self.assertIsNone(result.envelope.ground_position)
            self.assertIsNone(result.envelope.frame_reference)
            self.assertIn("relative_depth_unavailable", result.envelope.uncertainty)
            self.assertIn("ground_position_relative_or_estimated", result.envelope.uncertainty)

            store.shutdown()

    def test_message_ids_are_unique_across_packaging_and_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_store = TrackingEvidenceStore(root / "tracking_evidence")
            first_packager = TrackingEvidencePackager(coordinator=AgentCoordinator(), evidence_store=first_store)

            result_a = first_packager.package_detection(
                detection=self._detection("track-a", "det-a", 1003.0),
                timestamp=1003.0,
                frame_evidence=None,
                update_payload=None,
            )
            result_b = first_packager.package_detection(
                detection=self._detection("track-a", "det-b", 1003.1),
                timestamp=1003.1,
                frame_evidence=None,
                update_payload=None,
            )
            self.assertNotEqual(result_a.envelope.message_id, result_b.envelope.message_id)

            first_store.shutdown()

            second_store = TrackingEvidenceStore(root / "tracking_evidence")
            second_packager = TrackingEvidencePackager(coordinator=AgentCoordinator(), evidence_store=second_store)
            result_c = second_packager.package_detection(
                detection=self._detection("track-a", "det-c", 1003.2),
                timestamp=1003.2,
                frame_evidence=None,
                update_payload=None,
            )
            self.assertNotEqual(result_a.envelope.message_id, result_c.envelope.message_id)
            second_store.shutdown()

    def test_shared_frame_reference_and_cross_process_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = TrackingEvidenceStore(root / "tracking_evidence")
            packager = TrackingEvidencePackager(coordinator=AgentCoordinator(), evidence_store=store)
            resolver = TrackingEvidenceResolver(root / "tracking_evidence")

            frame = np.zeros((24, 24, 3), dtype=np.uint8)
            frame_evidence = store.register_frame(frame=frame, timestamp=1004.0, width=24, height=24)

            result_one = packager.package_detection(
                detection=self._detection("track-1", "det-1", 1004.0),
                timestamp=1004.0,
                frame_evidence=frame_evidence,
                update_payload=None,
            )
            result_two = packager.package_detection(
                detection=self._detection("track-2", "det-2", 1004.0),
                timestamp=1004.0,
                frame_evidence=frame_evidence,
                update_payload=None,
            )

            self.assertEqual(result_one.envelope.frame_reference, result_two.envelope.frame_reference)
            self.assertTrue(self._wait_for_frame_available(resolver, frame_evidence.frame_reference))

            detection_ref = None
            for item in result_one.envelope.source_evidence_refs:
                if item.startswith("evidence://detection/"):
                    detection_ref = item
                    break
            self.assertIsNotNone(detection_ref)
            assert detection_ref is not None

            resolved = resolver.resolve_reference(
                detection_ref,
                expected_track_id="track-1",
                expected_detection_id="det-1",
                expected_frame_reference=frame_evidence.frame_reference,
            )
            self.assertTrue(resolved["valid"])

            wrong_linkage = resolver.resolve_reference(
                detection_ref,
                expected_track_id="track-other",
            )
            self.assertFalse(wrong_linkage["valid"])
            self.assertEqual(wrong_linkage["status"], "track_linkage_mismatch")

            resolver.shutdown()
            store.shutdown()

    def test_real_visual_evidence_lifecycle_store_to_observation_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            store = TrackingEvidenceStore(evidence_root)
            packager = TrackingEvidencePackager(coordinator=AgentCoordinator(), evidence_store=store)
            resolver = TrackingEvidenceResolver(evidence_root)

            now = time.time()
            frame = np.full((40, 60, 3), 127, dtype=np.uint8)
            frame_evidence = store.register_frame(
                frame=frame,
                timestamp=now,
                width=60,
                height=40,
            )

            packaged = packager.package_detection(
                detection=self._detection("track-1", "det-1", now),
                timestamp=now,
                frame_evidence=frame_evidence,
                update_payload={
                    "detection_id": "det-1",
                    "resolved_track_id": "track-1",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )

            self.assertTrue(packaged.validation["valid"])
            self.assertEqual(packaged.envelope.track_id, "track-1")
            self.assertEqual(packaged.envelope.frame_reference, frame_evidence.frame_reference)
            self.assertTrue(self._wait_for_frame_available(resolver, frame_evidence.frame_reference, timeout_seconds=1.5))

            frame_resolution = resolver.resolve_reference(
                frame_evidence.frame_reference,
                now_timestamp=now + 0.1,
            )
            self.assertTrue(frame_resolution["valid"])
            self.assertEqual(frame_resolution["status"], "ok")

            detection_reference = next(
                ref for ref in packaged.envelope.source_evidence_refs if ref.startswith("evidence://detection/")
            )
            detection_resolution = resolver.resolve_reference(
                detection_reference,
                expected_track_id="track-1",
                expected_detection_id="det-1",
                expected_frame_reference=frame_evidence.frame_reference,
                now_timestamp=now + 0.1,
            )
            self.assertTrue(detection_resolution["valid"])
            self.assertEqual(detection_resolution["status"], "ok")

            package = consume_tracking_evidence_by_message_id(
                ObservationEvidenceRequest(
                    message_id=packaged.envelope.message_id,
                    now_timestamp=now + 0.2,
                    require_visual_evidence=True,
                ),
                evidence_root_dir=evidence_root,
            )

            self.assertTrue(package.preflight["preflight_passed"])
            self.assertTrue(package.preflight["visual_evidence_inspectable"])
            self.assertIsNotNone(package.image_evidence)
            assert package.image_evidence is not None
            self.assertEqual(package.image_evidence.reference, frame_evidence.frame_reference)
            self.assertEqual(package.envelope.track_id, "track-1")
            self.assertEqual(package.envelope.frame_reference, frame_evidence.frame_reference)
            self.assertGreater(len(package.image_evidence.image_bytes), 0)

            resolver.shutdown()
            store.shutdown()

    def test_expiration_and_capacity_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = TrackingEvidenceStore(
                root / "tracking_evidence",
                max_age_seconds=0.01,
                max_frame_records=1,
                max_detection_records=1,
                max_envelope_records=1,
            )
            resolver = TrackingEvidenceResolver(root / "tracking_evidence")

            frame_a = store.register_frame(frame=np.zeros((8, 8, 3), dtype=np.uint8), timestamp=1005.0, width=8, height=8)
            frame_b = store.register_frame(frame=np.zeros((8, 8, 3), dtype=np.uint8), timestamp=1005.1, width=8, height=8)

            time.sleep(0.03)
            store.cleanup(now_timestamp=time.time())

            expired = resolver.resolve_reference(frame_a.frame_reference, now_timestamp=time.time())
            self.assertFalse(expired["valid"])
            self.assertIn(expired["status"], {"expired_reference", "missing_reference"})

            recent = resolver.resolve_reference(frame_b.frame_reference, now_timestamp=time.time())
            if not recent["valid"]:
                self.assertIn(recent["status"], {"expired_reference", "pending_reference", "missing_reference", "frame_unavailable"})
            else:
                self.assertEqual(recent["status"], "ok")

            resolver.shutdown()
            store.shutdown()

    def test_invalid_evidence_not_stored_as_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = TrackingEvidenceStore(root / "tracking_evidence")
            packager = TrackingEvidencePackager(coordinator=_InvalidCoordinator(), evidence_store=store)

            result = packager.package_detection(
                detection=self._detection("track-x", "det-x", 1006.0),
                timestamp=1006.0,
                frame_evidence=None,
                update_payload=None,
            )

            self.assertFalse(result.validation["valid"])
            self.assertIsNone(result.envelope_reference)

            envelope_files = list((root / "tracking_evidence" / "envelopes").glob("*.json"))
            self.assertEqual(envelope_files, [])
            store.shutdown()

    def test_radar_payload_unchanged_no_envelope_or_base64(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            radar_state_path = root / "latest_radar_state.json"
            depth_history_path = root / "radar_depth_coverage_history.jsonl"

            tracks = [
                RadarTrackSnapshot(
                    track_id="1",
                    timestamp=1007.0,
                    pixel_position=None,
                    ground_position=GroundPosition(x=1.0, z=2.0),
                    ground_coordinate_unit="relative_units",
                    position_kind="relative_ground_estimate",
                    quality_flags=["estimated_ground_from_image"],
                    confidence=0.88,
                    relative_depth=0.4,
                )
            ]

            publish_tracking_radar_state(
                radar_state_path=radar_state_path,
                radar_depth_coverage_history_path=depth_history_path,
                now=1007.0,
                coordinate_system="relative_image_plane",
                camera_status="connected",
                detector_status="running",
                calibration_status="missing",
                tracking_agent_status={"model": "qwen"},
                last_observation_timestamp=1007.0,
                tracking_agent_observation={"updates": []},
                tracks=tracks,
            )

            payload = json.loads(radar_state_path.read_text(encoding="utf-8"))
            self.assertNotIn("tracking_evidence", payload)
            self.assertNotIn("tracking_evidence_envelopes", payload)
            serialized = json.dumps(payload)
            self.assertNotIn("data:image/jpeg;base64", serialized)


    def test_frame_capacity_cleanup_invalidates_unexpired_linked_detection_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            store = TrackingEvidenceStore(
                evidence_root,
                max_age_seconds=120.0,
                max_frame_records=1,
                max_detection_records=100,
                max_envelope_records=100,
            )
            resolver = TrackingEvidenceResolver(evidence_root)
            packager = TrackingEvidencePackager(coordinator=AgentCoordinator(), evidence_store=store)

            now = time.time()
            frame_one = np.full((24, 24, 3), 64, dtype=np.uint8)
            frame_one_record = store.register_frame(
                frame=frame_one,
                timestamp=now,
                width=24,
                height=24,
            )

            self.assertTrue(self._wait_for_frame_available(resolver, frame_one_record.frame_reference, timeout_seconds=1.5))

            packaged = packager.package_detection(
                detection=self._detection("track-cap", "det-cap", now + 0.01),
                timestamp=now + 0.01,
                frame_evidence=frame_one_record,
                update_payload={
                    "detection_id": "det-cap",
                    "resolved_track_id": "track-cap",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )

            self.assertTrue(packaged.validation["valid"])
            detection_reference = next(
                ref for ref in packaged.envelope.source_evidence_refs if ref.startswith("evidence://detection/")
            )
            envelope_path = evidence_root / "envelopes" / f"{packaged.envelope.message_id}.json"
            self.assertTrue(envelope_path.is_file())

            frame_two = np.full((24, 24, 3), 128, dtype=np.uint8)
            frame_two_record = store.register_frame(
                frame=frame_two,
                timestamp=now + 0.2,
                width=24,
                height=24,
            )
            self.assertTrue(self._wait_for_frame_available(resolver, frame_two_record.frame_reference, timeout_seconds=1.5))

            cleanup_summary = store.cleanup(now_timestamp=now + 1.0)
            self.assertGreaterEqual(cleanup_summary["removed_capacity"], 1)

            repeated_cleanup_summary = store.cleanup(now_timestamp=now + 1.1)
            self.assertEqual(repeated_cleanup_summary["removed_capacity"], 0)

            frame_resolution = resolver.resolve_reference(
                frame_one_record.frame_reference,
                now_timestamp=now + 1.0,
            )
            detection_resolution = resolver.resolve_reference(
                detection_reference,
                expected_track_id="track-cap",
                expected_detection_id="det-cap",
                expected_frame_reference=frame_one_record.frame_reference,
                now_timestamp=now + 1.0,
            )

            self.assertFalse(frame_resolution["valid"])
            self.assertEqual(frame_resolution["status"], "missing_reference")
            self.assertFalse(detection_resolution["valid"])
            self.assertEqual(detection_resolution["status"], "linked_frame_unavailable")
            self.assertEqual(detection_resolution["linked_frame"]["status"], "missing_reference")

            resolver.shutdown()
            store.shutdown()


if __name__ == "__main__":
    unittest.main()
