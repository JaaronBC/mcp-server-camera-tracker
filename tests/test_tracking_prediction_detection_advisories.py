from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.contracts import BoundingBox, DetectionCandidate
from agents.dashboard.state import SharedDashboardState
from agents.dashboard.tracking_loop import _build_prediction_detection_advisories
from agents.tracking.identity_registry import PersistentIdentityRegistry


class TestTrackingPredictionDetectionAdvisories(unittest.TestCase):
    def _candidate(
        self,
        *,
        detection_id: str,
        yolo_track_id: str,
        timestamp: float,
        ground_x: float,
        ground_z: float,
        unit: str = "meters",
        position_kind: str = "calibrated_ground_plane",
        pixel_center_x: float = 100.0,
        pixel_center_y: float = 80.0,
    ) -> DetectionCandidate:
        return DetectionCandidate(
            detection_id=detection_id,
            timestamp=timestamp,
            yolo_track_id=yolo_track_id,
            label="person",
            detection_confidence=0.92,
            pixel_bbox=BoundingBox(x1=10.0, y1=15.0, x2=60.0, y2=110.0),
            pixel_center_x=pixel_center_x,
            pixel_center_y=pixel_center_y,
            relative_depth=0.3,
            ground_x=ground_x,
            ground_z=ground_z,
            ground_coordinate_unit=unit,
            position_kind=position_kind,
            motion_history=[],
        )

    def _seed_lost_identity(self, *, track_id: str = "1") -> tuple[PersistentIdentityRegistry, str]:
        registry = PersistentIdentityRegistry()
        registry.process_detections(
            now=1000.0,
            detections=[
                self._candidate(
                    detection_id="seed",
                    yolo_track_id=track_id,
                    timestamp=1000.0,
                    ground_x=2.0,
                    ground_z=3.0,
                )
            ],
        )
        app_identity_id = registry.get_identity_for_yolo_track(track_id)
        assert isinstance(app_identity_id, str)
        registry.process_detections(now=1000.1, detections=[])
        return registry, app_identity_id

    def _prediction_record(
        self,
        *,
        source_track_id: str,
        estimate_id: str,
        predicted_for: float,
        expires_at: float,
        source_estimated_at: float,
        registry_generation: int,
        registry_version: int,
        ground_x: float,
        ground_z: float,
        unit: str = "meters",
    ) -> dict:
        return {
            "app_identity_id": None,
            "source_yolo_track_id": source_track_id,
            "estimate_id": estimate_id,
            "source_generated_at": source_estimated_at,
            "source_identity_registry_generation": registry_generation,
            "source_identity_registry_version": registry_version,
            "source_estimated_at": source_estimated_at,
            "source_verification_timestamp": source_estimated_at,
            "predicted_for_timestamp": predicted_for,
            "expires_at": expires_at,
            "coordinate_system": "calibrated_ground_plane",
            "ground_coordinate_mode": "calibrated_metric",
            "coordinate_unit": unit,
            "estimated_pixel_position": {"x": 100.0, "y": 80.0},
            "estimated_ground_position": {"x": ground_x, "z": ground_z, "unit": unit},
            "uncertainty": ["brief_occlusion"],
            "classification": "occlusion_candidate",
            "source_tracking_message_ids": ["msg-1"],
            "source_verification_observation_id": "obs-1",
            "source_verification_message_id": "msg-1",
            "validation": {"valid": True},
            "source_feedback_expires_at": expires_at,
            "promoted_at": source_estimated_at,
        }

    def test_valid_match_for_temporarily_lost_identity_with_new_track(self) -> None:
        registry, app_identity_id = self._seed_lost_identity(track_id="1")
        snapshot = registry.snapshot()

        state = SharedDashboardState()
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.calibration_status = "loaded"
        state.latest_tracking_prediction_evidence_by_identity = {
            app_identity_id: self._prediction_record(
                source_track_id="1",
                estimate_id="est-1",
                predicted_for=1000.45,
                expires_at=1001.2,
                source_estimated_at=1000.2,
                registry_generation=int(snapshot["generation"]),
                registry_version=int(snapshot["version"]),
                ground_x=2.0,
                ground_z=3.0,
            )
        }

        candidates = [
            self._candidate(
                detection_id="new",
                yolo_track_id="9",
                timestamp=1000.4,
                ground_x=2.2,
                ground_z=3.1,
            )
        ]

        _build_prediction_detection_advisories(state=state, now=1000.4, tracking_candidates=candidates)

        advisories = state.latest_tracking_prediction_detection_advisories
        self.assertEqual(len(advisories), 1)
        self.assertEqual(advisories[0]["decision"], "matched")
        self.assertEqual(advisories[0]["app_identity_id"], app_identity_id)
        self.assertEqual(advisories[0]["candidate_yolo_track_id"], "9")

    def test_stale_or_invalid_prediction_is_not_matched(self) -> None:
        registry, app_identity_id = self._seed_lost_identity(track_id="2")
        snapshot = registry.snapshot()

        state = SharedDashboardState()
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.calibration_status = "loaded"
        state.latest_tracking_prediction_evidence_by_identity = {
            app_identity_id: self._prediction_record(
                source_track_id="2",
                estimate_id="est-stale",
                predicted_for=995.0,
                expires_at=995.2,
                source_estimated_at=994.9,
                registry_generation=int(snapshot["generation"]),
                registry_version=int(snapshot["version"]),
                ground_x=2.0,
                ground_z=3.0,
            )
        }

        candidates = [
            self._candidate(
                detection_id="new",
                yolo_track_id="20",
                timestamp=1000.4,
                ground_x=2.0,
                ground_z=3.0,
            )
        ]

        _build_prediction_detection_advisories(state=state, now=1000.4, tracking_candidates=candidates)

        self.assertEqual(state.latest_tracking_prediction_detection_advisories, [])
        summary = state.latest_tracking_prediction_detection_advisory_summary
        assert isinstance(summary, dict)
        self.assertGreaterEqual(int(summary.get("stale_or_invalid_count", 0)), 1)

    def test_competing_candidates_are_marked_ambiguous(self) -> None:
        registry, app_identity_id = self._seed_lost_identity(track_id="3")
        snapshot = registry.snapshot()

        state = SharedDashboardState()
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.calibration_status = "loaded"
        state.latest_tracking_prediction_evidence_by_identity = {
            app_identity_id: self._prediction_record(
                source_track_id="3",
                estimate_id="est-ambig",
                predicted_for=1000.45,
                expires_at=1001.2,
                source_estimated_at=1000.2,
                registry_generation=int(snapshot["generation"]),
                registry_version=int(snapshot["version"]),
                ground_x=4.0,
                ground_z=6.0,
            )
        }

        candidates = [
            self._candidate(
                detection_id="a",
                yolo_track_id="30",
                timestamp=1000.4,
                ground_x=4.2,
                ground_z=6.1,
            ),
            self._candidate(
                detection_id="b",
                yolo_track_id="31",
                timestamp=1000.4,
                ground_x=4.3,
                ground_z=6.1,
            ),
        ]

        _build_prediction_detection_advisories(state=state, now=1000.4, tracking_candidates=candidates)

        advisories = state.latest_tracking_prediction_detection_advisories
        self.assertTrue(advisories)
        self.assertTrue(all(item.get("decision") == "ambiguous" for item in advisories))

    def test_competing_identities_for_single_track_are_ambiguous(self) -> None:
        registry = PersistentIdentityRegistry()
        first = self._candidate(detection_id="f", yolo_track_id="40", timestamp=1000.0, ground_x=1.0, ground_z=1.0)
        second = self._candidate(detection_id="s", yolo_track_id="41", timestamp=1000.0, ground_x=2.0, ground_z=2.0)
        registry.process_detections(now=1000.0, detections=[first, second])
        first_id = registry.get_identity_for_yolo_track("40")
        second_id = registry.get_identity_for_yolo_track("41")
        assert isinstance(first_id, str)
        assert isinstance(second_id, str)
        registry.process_detections(now=1000.1, detections=[])
        snapshot = registry.snapshot()

        state = SharedDashboardState()
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.calibration_status = "loaded"
        state.latest_tracking_prediction_evidence_by_identity = {
            first_id: self._prediction_record(
                source_track_id="40",
                estimate_id="est-first",
                predicted_for=1000.45,
                expires_at=1001.2,
                source_estimated_at=1000.2,
                registry_generation=int(snapshot["generation"]),
                registry_version=int(snapshot["version"]),
                ground_x=3.0,
                ground_z=3.0,
            ),
            second_id: self._prediction_record(
                source_track_id="41",
                estimate_id="est-second",
                predicted_for=1000.45,
                expires_at=1001.2,
                source_estimated_at=1000.2,
                registry_generation=int(snapshot["generation"]),
                registry_version=int(snapshot["version"]),
                ground_x=3.1,
                ground_z=3.0,
            ),
        }

        candidates = [
            self._candidate(
                detection_id="new",
                yolo_track_id="49",
                timestamp=1000.4,
                ground_x=3.05,
                ground_z=3.0,
            )
        ]

        _build_prediction_detection_advisories(state=state, now=1000.4, tracking_candidates=candidates)

        advisories = state.latest_tracking_prediction_detection_advisories
        self.assertTrue(advisories)
        self.assertTrue(all(item.get("decision") == "ambiguous" for item in advisories))

    def test_no_identity_mutation_side_effects(self) -> None:
        registry, app_identity_id = self._seed_lost_identity(track_id="5")
        snapshot_before = registry.snapshot()

        state = SharedDashboardState()
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.calibration_status = "loaded"
        state.latest_tracking_prediction_evidence_by_identity = {
            app_identity_id: self._prediction_record(
                source_track_id="5",
                estimate_id="est-5",
                predicted_for=1000.45,
                expires_at=1001.2,
                source_estimated_at=1000.2,
                registry_generation=int(snapshot_before["generation"]),
                registry_version=int(snapshot_before["version"]),
                ground_x=2.0,
                ground_z=3.0,
            )
        }

        candidates = [
            self._candidate(
                detection_id="new",
                yolo_track_id="55",
                timestamp=1000.4,
                ground_x=2.1,
                ground_z=3.1,
            )
        ]

        _build_prediction_detection_advisories(state=state, now=1000.4, tracking_candidates=candidates)

        snapshot_after = registry.snapshot()
        self.assertEqual(snapshot_before["generation"], snapshot_after["generation"])
        self.assertEqual(snapshot_before["version"], snapshot_after["version"])
        self.assertEqual(registry.get_identity_for_yolo_track("5"), app_identity_id)
        self.assertIsNone(registry.get_identity_for_yolo_track("55"))

    def test_unavailable_registry_publishes_unavailable_artifact_payload(self) -> None:
        state = SharedDashboardState()
        state.tracking_identity_registry_enabled = False
        state.tracking_identity_registry = None

        with patch("agents.dashboard.tracking_loop.write_json_payload") as write_mock:
            _build_prediction_detection_advisories(
                state=state,
                now=1000.0,
                tracking_candidates=[],
            )

        write_mock.assert_called_once()
        payload = write_mock.call_args.args[1]
        self.assertFalse(payload["available"])
        self.assertEqual(payload["summary"]["reason"], "identity_registry_unavailable")
        self.assertEqual(payload["advisories"], [])

    def test_publish_replaces_previous_artifact_payload(self) -> None:
        registry, app_identity_id = self._seed_lost_identity(track_id="6")
        snapshot = registry.snapshot()

        state = SharedDashboardState()
        state.tracking_identity_registry_enabled = True
        state.tracking_identity_registry = registry
        state.calibration_status = "loaded"
        state.latest_tracking_prediction_evidence_by_identity = {
            app_identity_id: self._prediction_record(
                source_track_id="6",
                estimate_id="est-6",
                predicted_for=1000.45,
                expires_at=1001.2,
                source_estimated_at=1000.2,
                registry_generation=int(snapshot["generation"]),
                registry_version=int(snapshot["version"]),
                ground_x=2.0,
                ground_z=3.0,
            )
        }

        with patch("agents.dashboard.tracking_loop.write_json_payload") as write_mock:
            _build_prediction_detection_advisories(
                state=state,
                now=1000.4,
                tracking_candidates=[
                    self._candidate(
                        detection_id="first",
                        yolo_track_id="66",
                        timestamp=1000.4,
                        ground_x=2.1,
                        ground_z=3.1,
                    )
                ],
            )
            _build_prediction_detection_advisories(
                state=state,
                now=1000.8,
                tracking_candidates=[
                    self._candidate(
                        detection_id="second",
                        yolo_track_id="67",
                        timestamp=1000.8,
                        ground_x=2.1,
                        ground_z=3.1,
                    )
                ],
            )

        self.assertEqual(write_mock.call_count, 2)
        first_payload = write_mock.call_args_list[0].args[1]
        second_payload = write_mock.call_args_list[1].args[1]
        self.assertTrue(first_payload["available"])
        self.assertTrue(second_payload["available"])
        self.assertNotEqual(first_payload["updated_at"], second_payload["updated_at"])
        self.assertEqual(first_payload["advisories"][0]["candidate_yolo_track_id"], "66")
        self.assertEqual(second_payload["advisories"][0]["candidate_yolo_track_id"], "67")


if __name__ == "__main__":
    unittest.main()
