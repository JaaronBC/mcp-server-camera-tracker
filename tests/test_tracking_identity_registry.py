from __future__ import annotations

import unittest

from agents.contracts import BoundingBox, DetectionCandidate
from agents.dashboard.state import SharedDashboardState
from agents.dashboard.tracking_loop import _update_identity_registry_if_enabled
from agents.tracking.identity_registry import PersistentIdentityRegistry


class TestPersistentIdentityRegistry(unittest.TestCase):
    def _candidate(
        self,
        *,
        detection_id: str,
        yolo_track_id: str,
        timestamp: float = 100.0,
        detection_reference: str | None = None,
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

    def test_registry_prefers_candidate_detection_reference_with_legacy_fallback(self) -> None:
        registry = PersistentIdentityRegistry(max_history_per_identity=4)

        preferred_reference = "fdet-1000000-0-det-a-track-a"
        preferred_detection = self._candidate(
            detection_id="det-a",
            yolo_track_id="track-a",
            timestamp=1.0,
            detection_reference=preferred_reference,
        )
        registry.process_detections(now=1.0, detections=[preferred_detection])

        preferred_app_id = registry.get_identity_for_yolo_track("track-a")
        self.assertIsNotNone(preferred_app_id)
        preferred_record = registry.get_record(preferred_app_id)
        self.assertIsNotNone(preferred_record)
        self.assertEqual(preferred_record.detection_reference, preferred_reference)

        fallback_detection = self._candidate(
            detection_id="det-b",
            yolo_track_id="track-b",
            timestamp=2.0,
            detection_reference=None,
        )
        registry.process_detections(now=2.0, detections=[fallback_detection])

        fallback_app_id = registry.get_identity_for_yolo_track("track-b")
        self.assertIsNotNone(fallback_app_id)
        fallback_record = registry.get_record(fallback_app_id)
        self.assertIsNotNone(fallback_record)
        self.assertIsInstance(fallback_record.detection_reference, str)
        assert isinstance(fallback_record.detection_reference, str)
        self.assertTrue(fallback_record.detection_reference.startswith("frame:"))

    def test_registry_creates_identities_and_updates_records(self) -> None:
        registry = PersistentIdentityRegistry(max_history_per_identity=4)
        detections = [
            self._candidate(detection_id="11", yolo_track_id="11"),
            self._candidate(detection_id="12", yolo_track_id="12"),
        ]

        summary = registry.process_detections(now=101.0, detections=detections)

        self.assertEqual(summary["created_count"], 2)
        self.assertEqual(summary["visible_count"], 2)
        self.assertEqual(summary["temporarily_lost_count"], 0)
        self.assertEqual(summary["retired_count"], 0)

        app_id_11 = registry.get_identity_for_yolo_track("11")
        app_id_12 = registry.get_identity_for_yolo_track("12")
        self.assertIsNotNone(app_id_11)
        self.assertIsNotNone(app_id_12)
        self.assertNotEqual(app_id_11, app_id_12)

        record_11 = registry.get_record(app_id_11)
        self.assertIsNotNone(record_11)
        self.assertEqual(record_11.status, "visible")
        self.assertEqual(record_11.detection_id, "11")
        self.assertTrue(record_11.provisional)
        self.assertEqual(detections[0].app_identity_id, app_id_11)

    def test_temporary_loss_and_retention_then_retirement(self) -> None:
        registry = PersistentIdentityRegistry(
            temporary_loss_seconds=1.0,
            retirement_seconds=2.0,
            max_history_per_identity=8,
        )
        detection = self._candidate(detection_id="21", yolo_track_id="21")
        registry.process_detections(now=200.0, detections=[detection])

        app_id = registry.get_identity_for_yolo_track("21")
        self.assertIsNotNone(app_id)

        summary_lost = registry.process_detections(now=201.0, detections=[])
        self.assertEqual(summary_lost["temporarily_lost_count"], 1)
        self.assertEqual(summary_lost["retired_count"], 0)

        record_lost = registry.get_record(app_id)
        self.assertIsNotNone(record_lost)
        self.assertEqual(record_lost.status, "temporarily_lost")
        self.assertIsNotNone(record_lost.detection_reference)

        summary_retired = registry.process_detections(now=203.5, detections=[])
        self.assertEqual(summary_retired["retired_count"], 1)

        record_retired = registry.get_record(app_id)
        self.assertIsNotNone(record_retired)
        self.assertEqual(record_retired.status, "retired")

    def test_reappearance_with_new_yolo_track_remains_provisional(self) -> None:
        registry = PersistentIdentityRegistry(retirement_seconds=10.0)
        registry.process_detections(
            now=300.0,
            detections=[self._candidate(detection_id="31", yolo_track_id="31")],
        )
        old_app_id = registry.get_identity_for_yolo_track("31")

        registry.process_detections(now=301.0, detections=[])

        new_detection = self._candidate(detection_id="99", yolo_track_id="99")
        registry.process_detections(now=302.0, detections=[new_detection])
        new_app_id = registry.get_identity_for_yolo_track("99")

        self.assertIsNotNone(old_app_id)
        self.assertIsNotNone(new_app_id)
        self.assertNotEqual(old_app_id, new_app_id)

        old_record = registry.get_record(old_app_id)
        new_record = registry.get_record(new_app_id)
        self.assertEqual(old_record.status, "temporarily_lost")
        self.assertTrue(new_record.provisional)
        self.assertEqual(new_record.status, "visible")

    def test_identity_version_and_generation_conflict_rejection(self) -> None:
        registry = PersistentIdentityRegistry()
        registry.process_detections(
            now=400.0,
            detections=[self._candidate(detection_id="41", yolo_track_id="41")],
        )
        app_id = registry.get_identity_for_yolo_track("41")
        self.assertIsNotNone(app_id)

        stale_generation_result = registry.apply_identity_mutation(
            app_identity_id=app_id,
            new_yolo_track_id="42",
            now=401.0,
            expected_generation_at_least=0,
        )
        self.assertFalse(stale_generation_result.accepted)
        self.assertEqual(stale_generation_result.reason, "stale_generation")

        wrong_version_result = registry.apply_identity_mutation(
            app_identity_id=app_id,
            new_yolo_track_id="42",
            now=401.1,
            expected_generation_at_least=registry.generation,
            expected_version=registry.version + 5,
        )
        self.assertFalse(wrong_version_result.accepted)
        self.assertEqual(wrong_version_result.reason, "version_conflict")

    def test_multiple_tracks_prevent_identity_collision(self) -> None:
        registry = PersistentIdentityRegistry()
        registry.process_detections(
            now=500.0,
            detections=[
                self._candidate(detection_id="51", yolo_track_id="51"),
                self._candidate(detection_id="52", yolo_track_id="52"),
            ],
        )
        app_id_51 = registry.get_identity_for_yolo_track("51")
        app_id_52 = registry.get_identity_for_yolo_track("52")
        self.assertIsNotNone(app_id_51)
        self.assertIsNotNone(app_id_52)
        self.assertNotEqual(app_id_51, app_id_52)

        conflict = registry.apply_identity_mutation(
            app_identity_id=app_id_51,
            new_yolo_track_id="52",
            now=501.0,
            expected_generation_at_least=registry.generation,
            expected_version=registry.version,
        )
        self.assertFalse(conflict.accepted)
        self.assertEqual(conflict.reason, "track_id_already_assigned")

    def test_verified_provisional_owner_transfer_moves_track_atomically(self) -> None:
        registry = PersistentIdentityRegistry(retirement_seconds=20.0)
        registry.process_detections(
            now=700.0,
            detections=[self._candidate(detection_id="det-old", yolo_track_id="191")],
        )
        source_app_identity_id = registry.get_identity_for_yolo_track("191")
        self.assertIsNotNone(source_app_identity_id)
        assert isinstance(source_app_identity_id, str)

        registry.process_detections(
            now=701.0,
            detections=[self._candidate(detection_id="det-new", yolo_track_id="210")],
        )

        provisional_owner_identity_id = registry.get_identity_for_yolo_track("210")
        self.assertIsNotNone(provisional_owner_identity_id)
        assert isinstance(provisional_owner_identity_id, str)
        self.assertNotEqual(source_app_identity_id, provisional_owner_identity_id)

        source_before = registry.get_record(source_app_identity_id)
        owner_before = registry.get_record(provisional_owner_identity_id)
        snapshot_before = registry.snapshot()
        self.assertIsNotNone(source_before)
        self.assertIsNotNone(owner_before)
        assert source_before is not None
        assert owner_before is not None
        self.assertEqual(source_before.status, "temporarily_lost")
        self.assertTrue(owner_before.provisional)

        transfer_result = registry.apply_verified_provisional_owner_transfer(
            app_identity_id=source_app_identity_id,
            provisional_owner_identity_id=provisional_owner_identity_id,
            target_yolo_track_id="210",
            now=701.5,
            expected_generation_at_least=int(snapshot_before["generation"]),
            expected_version=int(snapshot_before["version"]),
        )
        self.assertTrue(transfer_result.accepted)

        source_after = registry.get_record(source_app_identity_id)
        owner_after = registry.get_record(provisional_owner_identity_id)
        self.assertIsNotNone(source_after)
        self.assertIsNotNone(owner_after)
        assert source_after is not None
        assert owner_after is not None

        self.assertEqual(registry.get_identity_for_yolo_track("210"), source_app_identity_id)
        self.assertIsNone(registry.get_identity_for_yolo_track("191"))
        self.assertEqual(source_after.status, "visible")
        self.assertEqual(source_after.yolo_track_id, "210")
        self.assertEqual(owner_after.status, "temporarily_lost")
        self.assertNotEqual(owner_after.yolo_track_id, "210")

        owner_history_events = [item.event for item in owner_after.history]
        self.assertIn("provisional_transfer_released_track", owner_history_events)

    def test_verified_provisional_transfer_failure_is_noop(self) -> None:
        registry = PersistentIdentityRegistry(retirement_seconds=20.0)
        registry.process_detections(
            now=800.0,
            detections=[self._candidate(detection_id="det-old", yolo_track_id="191")],
        )
        source_app_identity_id = registry.get_identity_for_yolo_track("191")
        self.assertIsNotNone(source_app_identity_id)
        assert isinstance(source_app_identity_id, str)

        registry.process_detections(
            now=801.0,
            detections=[self._candidate(detection_id="det-new", yolo_track_id="210")],
        )
        provisional_owner_identity_id = registry.get_identity_for_yolo_track("210")
        self.assertIsNotNone(provisional_owner_identity_id)
        assert isinstance(provisional_owner_identity_id, str)

        with registry._lock:
            owner_record = registry._records[provisional_owner_identity_id]
            owner_record.provisional = False

        snapshot_before = registry.snapshot()
        source_owner_before = registry.get_identity_for_yolo_track("191")
        target_owner_before = registry.get_identity_for_yolo_track("210")

        transfer_result = registry.apply_verified_provisional_owner_transfer(
            app_identity_id=source_app_identity_id,
            provisional_owner_identity_id=provisional_owner_identity_id,
            target_yolo_track_id="210",
            now=801.5,
            expected_generation_at_least=int(snapshot_before["generation"]),
            expected_version=int(snapshot_before["version"]),
        )
        self.assertFalse(transfer_result.accepted)
        self.assertEqual(transfer_result.reason, "target_owner_not_provisional")

        snapshot_after = registry.snapshot()
        self.assertEqual(snapshot_after["generation"], snapshot_before["generation"])
        self.assertEqual(snapshot_after["version"], snapshot_before["version"])
        self.assertEqual(registry.get_identity_for_yolo_track("191"), source_owner_before)
        self.assertEqual(registry.get_identity_for_yolo_track("210"), target_owner_before)

    def test_disabled_mode_is_backward_compatible(self) -> None:
        state = SharedDashboardState(video_path=None)
        state.tracking_identity_registry_enabled = False
        state.tracking_identity_registry = None

        _update_identity_registry_if_enabled(
            state=state,
            now=600.0,
            tracking_candidates=[self._candidate(detection_id="61", yolo_track_id="61")],
        )

        self.assertEqual(state.tracking_identity_registry_generation, 0)
        self.assertEqual(state.tracking_identity_registry_version, 0)
        self.assertIsNone(state.latest_tracking_identity_registry_summary)


if __name__ == "__main__":
    unittest.main()
