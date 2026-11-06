from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agents.contracts import BoundingBox, DetectionCandidate
from agents.dashboard.identity_snapshot_bridge import (
    build_identity_snapshot_payload,
    build_identity_snapshot_unavailable_payload,
    publish_identity_snapshot,
)
from agents.orchestration.runtime.identity_snapshot_service import load_identity_snapshot
from agents.orchestration.runtime.identity_snapshot_service import (
    find_identity_candidates,
    get_identity_history,
    get_identity_status,
)
from agents.tracking.identity_registry import PersistentIdentityRegistry


class TestIdentitySnapshotBridge(unittest.TestCase):
    def _candidate(self, *, detection_id: str, yolo_track_id: str, timestamp: float = 100.0) -> DetectionCandidate:
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
        )

    def test_payload_is_bounded_contains_provenance_and_excludes_frames(self) -> None:
        registry = PersistentIdentityRegistry(max_history_per_identity=32)
        for index in range(8):
            detection = self._candidate(detection_id=str(index), yolo_track_id=str(index), timestamp=100.0 + index)
            registry.process_detections(now=100.0 + index, detections=[detection])

        payload = build_identity_snapshot_payload(
            registry=registry,
            now=200.0,
            freshness_window_seconds=3.0,
            max_identities=3,
            max_history_per_identity=2,
            max_historical_track_ids=2,
            max_payload_bytes=16_384,
        )

        self.assertEqual(payload["state"], "available")
        self.assertEqual(payload["identity_count"], 3)
        self.assertLessEqual(payload["payload_bytes"], 16_384)

        identities = payload.get("identities")
        self.assertIsInstance(identities, list)
        assert isinstance(identities, list)
        for identity in identities:
            self.assertIn("app_identity_id", identity)
            self.assertIn("status", identity)
            self.assertIn("generation", identity)
            self.assertIn("identity_version", identity)
            self.assertIn("observation_history", identity)
            self.assertIn("historical_yolo_track_ids", identity)
            self.assertNotIn("frame", identity)
            self.assertNotIn("image", identity)
            history = identity.get("observation_history")
            self.assertIsInstance(history, list)
            assert isinstance(history, list)
            self.assertLessEqual(len(history), 2)
            historical_track_ids = identity.get("historical_yolo_track_ids")
            self.assertIsInstance(historical_track_ids, list)
            assert isinstance(historical_track_ids, list)
            self.assertLessEqual(len(historical_track_ids), 2)
            if history:
                self.assertIn("detection_reference", history[-1])

    def test_size_cap_truncates_identities(self) -> None:
        registry = PersistentIdentityRegistry(max_history_per_identity=8)
        for index in range(6):
            registry.process_detections(
                now=300.0 + index,
                detections=[self._candidate(detection_id=f"d-{index}", yolo_track_id=f"t-{index}", timestamp=300.0 + index)],
            )

        payload = build_identity_snapshot_payload(
            registry=registry,
            now=350.0,
            freshness_window_seconds=3.0,
            max_identities=6,
            max_history_per_identity=4,
            max_historical_track_ids=4,
            max_payload_bytes=700,
        )

        self.assertLessEqual(payload["payload_bytes"], 700)
        self.assertTrue(payload["truncation"]["identities_truncated_by_size"])
        self.assertLess(payload["identity_count"], 6)

    def test_unavailable_payload_shape(self) -> None:
        payload = build_identity_snapshot_unavailable_payload(
            now=500.0,
            freshness_window_seconds=3.0,
            state="unavailable",
            reason="identity_registry_unavailable",
        )

        self.assertEqual(payload["state"], "unavailable")
        self.assertEqual(payload["reason"], "identity_registry_unavailable")
        self.assertEqual(payload["identity_count"], 0)

    def test_payload_generation_version_metadata_are_consistent(self) -> None:
        registry = PersistentIdentityRegistry(max_history_per_identity=16)
        registry.process_detections(
            now=700.0,
            detections=[self._candidate(detection_id="d-1", yolo_track_id="t-1", timestamp=700.0)],
        )
        registry.process_detections(
            now=701.0,
            detections=[self._candidate(detection_id="d-2", yolo_track_id="t-2", timestamp=701.0)],
        )

        payload = build_identity_snapshot_payload(
            registry=registry,
            now=702.0,
            freshness_window_seconds=3.0,
            max_identities=10,
            max_history_per_identity=10,
            max_historical_track_ids=10,
            max_payload_bytes=64_000,
        )

        source_generation = payload.get("source_generation")
        source_version = payload.get("source_version")
        self.assertIsInstance(source_generation, int)
        self.assertIsInstance(source_version, int)
        assert isinstance(source_generation, int)
        assert isinstance(source_version, int)

        identities = payload.get("identities")
        self.assertIsInstance(identities, list)
        assert isinstance(identities, list)
        for identity in identities:
            self.assertLessEqual(int(identity.get("generation", 0)), source_generation)
            self.assertGreaterEqual(int(identity.get("identity_version", 0)), 1)

    def test_publish_identity_snapshot_uses_atomic_replace(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 1000.0,
                "freshness_window_seconds": 3.0,
                "source_generation": 1,
                "source_version": 1,
                "identity_count": 0,
                "identities": [],
            }
            with patch("agents.dashboard.identity_snapshot_bridge.os.replace") as replace_mock:
                publish_identity_snapshot(path, payload)
                replace_mock.assert_called_once()

    def test_concurrent_updates_and_snapshot_capture(self) -> None:
        registry = PersistentIdentityRegistry(max_history_per_identity=20)
        stop_event = threading.Event()

        def _writer() -> None:
            index = 0
            while not stop_event.is_set():
                track_id = str(index % 5)
                detection = self._candidate(
                    detection_id=f"det-{index}",
                    yolo_track_id=track_id,
                    timestamp=600.0 + index,
                )
                registry.process_detections(now=600.0 + index, detections=[detection])
                index += 1

        worker = threading.Thread(target=_writer, daemon=True)
        worker.start()

        try:
            versions: list[int] = []
            for _ in range(25):
                payload = build_identity_snapshot_payload(
                    registry=registry,
                    now=time.time(),
                    freshness_window_seconds=3.0,
                    max_identities=5,
                    max_history_per_identity=5,
                    max_historical_track_ids=5,
                    max_payload_bytes=16_384,
                )
                source_version = payload.get("source_version")
                if isinstance(source_version, int):
                    versions.append(source_version)
                self.assertEqual(payload["state"], "available")
            self.assertTrue(versions)
            self.assertGreaterEqual(max(versions), min(versions))
        finally:
            stop_event.set()
            worker.join(timeout=1.0)


class TestIdentitySnapshotService(unittest.TestCase):
    def test_service_disabled_state_is_explicit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            result = load_identity_snapshot(path=path, enabled=False, now=1000.0)
            self.assertFalse(result["available"])
            self.assertEqual(result["state"], "disabled")

    def test_service_missing_stale_and_available_states(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"

            missing = load_identity_snapshot(path=path, enabled=True, now=1000.0)
            self.assertFalse(missing["available"])
            self.assertEqual(missing["state"], "missing")

            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 900.0,
                "freshness_window_seconds": 3.0,
                "source_generation": 2,
                "source_version": 4,
                "identity_count": 1,
                "identities": [{"app_identity_id": "app-1"}],
            }
            publish_identity_snapshot(path, payload)

            stale = load_identity_snapshot(path=path, enabled=True, now=905.5)
            self.assertFalse(stale["available"])
            self.assertEqual(stale["state"], "stale")

            fresh = load_identity_snapshot(path=path, enabled=True, now=902.0)
            self.assertTrue(fresh["available"])
            self.assertEqual(fresh["state"], "available")

    def test_cross_process_visibility_and_read_immutability(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 1000.0,
                "freshness_window_seconds": 5.0,
                "source_generation": 5,
                "source_version": 7,
                "identity_count": 1,
                "identities": [{"app_identity_id": "app-1", "status": "visible"}],
            }
            publish_identity_snapshot(path, payload)

            loaded = load_identity_snapshot(path=path, enabled=True, now=1001.0)
            self.assertTrue(loaded["available"])
            snapshot = loaded.get("snapshot")
            self.assertIsInstance(snapshot, dict)
            assert isinstance(snapshot, dict)
            self.assertEqual(snapshot.get("source_version"), 7)

            snapshot["source_version"] = 999
            snapshot_identities = snapshot.get("identities")
            if isinstance(snapshot_identities, list) and snapshot_identities:
                snapshot_identities[0]["app_identity_id"] = "mutated"

            reloaded = load_identity_snapshot(path=path, enabled=True, now=1001.0)
            self.assertEqual(reloaded["source_version"], 7)
            reloaded_snapshot = reloaded.get("snapshot")
            self.assertIsInstance(reloaded_snapshot, dict)
            assert isinstance(reloaded_snapshot, dict)
            identities = reloaded_snapshot.get("identities")
            self.assertIsInstance(identities, list)
            assert isinstance(identities, list)
            self.assertEqual(identities[0]["app_identity_id"], "app-1")

    def test_get_identity_status_distinguishes_provisional_confirmed_retired(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 1000.0,
                "freshness_window_seconds": 5.0,
                "source_generation": 9,
                "source_version": 12,
                "identity_count": 3,
                "identities": [
                    {
                        "app_identity_id": "app-provisional",
                        "status": "visible",
                        "provisional": True,
                        "generation": 9,
                        "identity_version": 2,
                        "updated_at": 999.8,
                        "observation_history": [],
                        "historical_yolo_track_ids": [],
                    },
                    {
                        "app_identity_id": "app-confirmed",
                        "status": "temporarily_lost",
                        "provisional": False,
                        "generation": 9,
                        "identity_version": 6,
                        "updated_at": 999.9,
                        "observation_history": [],
                        "historical_yolo_track_ids": [],
                    },
                    {
                        "app_identity_id": "app-retired",
                        "status": "retired",
                        "provisional": False,
                        "generation": 8,
                        "identity_version": 7,
                        "updated_at": 999.4,
                        "observation_history": [],
                        "historical_yolo_track_ids": [],
                    },
                ],
            }
            publish_identity_snapshot(path, payload)

            provisional = get_identity_status(app_identity_id="app-provisional", path=path, enabled=True, now=1001.0)
            confirmed = get_identity_status(app_identity_id="app-confirmed", path=path, enabled=True, now=1001.0)
            retired = get_identity_status(app_identity_id="app-retired", path=path, enabled=True, now=1001.0)

            self.assertEqual(provisional["identity"]["identity_state"], "provisional")
            self.assertEqual(confirmed["identity"]["identity_state"], "confirmed")
            self.assertEqual(confirmed["identity"]["lifecycle_status"], "temporarily_lost")
            self.assertEqual(retired["identity"]["identity_state"], "retired")

    def test_get_identity_history_is_bounded_and_includes_provenance(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 1000.0,
                "freshness_window_seconds": 5.0,
                "source_generation": 10,
                "source_version": 10,
                "identity_count": 1,
                "identities": [
                    {
                        "app_identity_id": "app-1",
                        "status": "temporarily_lost",
                        "provisional": True,
                        "generation": 10,
                        "identity_version": 4,
                        "updated_at": 1000.0,
                        "historical_yolo_track_ids": ["1", "2"],
                        "observation_history": [
                            {"generation": 8, "timestamp": 998.0, "detection_reference": "ref-a", "event": "visible"},
                            {"generation": 9, "timestamp": 999.0, "detection_reference": "ref-b", "event": "visible"},
                            {"generation": 10, "timestamp": 1000.0, "detection_reference": "ref-c", "event": "temporarily_lost"},
                        ],
                    }
                ],
            }
            publish_identity_snapshot(path, payload)

            result = get_identity_history(
                app_identity_id="app-1",
                max_entries=2,
                path=path,
                enabled=True,
                now=1001.0,
            )

            self.assertTrue(result["available"])
            self.assertEqual(result["identity"]["lifecycle_status"], "temporarily_lost")
            self.assertEqual(result["history_count"], 2)
            self.assertTrue(result["history_truncated"])
            self.assertEqual(result["history"][-1]["detection_reference"], "ref-c")
            self.assertEqual(result["provenance"]["resolution_mode"], "app_identity_id_exact_match")

    def test_find_identity_candidates_returns_ambiguous_without_auto_decision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 1000.0,
                "freshness_window_seconds": 5.0,
                "source_generation": 11,
                "source_version": 13,
                "identity_count": 2,
                "identities": [
                    {
                        "app_identity_id": "app-a",
                        "status": "visible",
                        "provisional": True,
                        "generation": 11,
                        "identity_version": 3,
                        "updated_at": 1000.0,
                        "detection_reference": "frame:11:idx:0:det:a:yolo:1",
                        "observation_history": [],
                        "historical_yolo_track_ids": ["1"],
                    },
                    {
                        "app_identity_id": "app-b",
                        "status": "temporarily_lost",
                        "provisional": False,
                        "generation": 10,
                        "identity_version": 8,
                        "updated_at": 999.5,
                        "detection_reference": "frame:10:idx:0:det:b:yolo:2",
                        "observation_history": [
                            {
                                "generation": 11,
                                "timestamp": 1000.0,
                                "event": "visible",
                                "detection_reference": "frame:11:idx:0:det:a:yolo:1",
                                "detection_id": "a",
                                "yolo_track_id": "2",
                            }
                        ],
                        "historical_yolo_track_ids": ["2"],
                    },
                ],
            }
            publish_identity_snapshot(path, payload)

            result = find_identity_candidates(
                detection_reference="frame:11:idx:0:det:a:yolo:1",
                max_candidates=8,
                path=path,
                enabled=True,
                now=1001.0,
            )

            self.assertTrue(result["available"])
            self.assertEqual(result["candidate_lookup_state"], "found")
            self.assertEqual(result["candidate_count"], 2)
            self.assertTrue(result["ambiguity"]["ambiguous"])
            self.assertEqual(result["identity_match_decision"], "not_decided")
            self.assertEqual(result["resolution_mode"], "exact_detection_reference_match_only")

    def test_retrieval_returns_explicit_unavailable_when_disabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"

            status_result = get_identity_status(app_identity_id="app-1", path=path, enabled=False, now=1000.0)
            history_result = get_identity_history(app_identity_id="app-1", path=path, enabled=False, now=1000.0)
            candidates_result = find_identity_candidates(
                detection_reference="frame:1:idx:0:det:x:yolo:y",
                path=path,
                enabled=False,
                now=1000.0,
            )

            self.assertEqual(status_result["state"], "disabled")
            self.assertEqual(history_result["state"], "disabled")
            self.assertEqual(candidates_result["state"], "disabled")
            self.assertFalse(status_result["mutation_authorized"])
            self.assertFalse(history_result["mutation_authorized"])
            self.assertFalse(candidates_result["mutation_authorized"])

    def test_stale_snapshot_does_not_authorize_retrieval_or_mutation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 900.0,
                "freshness_window_seconds": 2.0,
                "source_generation": 2,
                "source_version": 2,
                "identity_count": 1,
                "identities": [{"app_identity_id": "app-1", "status": "visible", "provisional": True}],
            }
            publish_identity_snapshot(path, payload)

            result = get_identity_status(app_identity_id="app-1", path=path, enabled=True, now=910.0)
            self.assertEqual(result["state"], "stale")
            self.assertFalse(result["available"])
            self.assertFalse(result["mutation_authorized"])

    def test_find_identity_candidates_resolves_evidence_detection_reference(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            refs_dir = evidence_root / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)

            canonical_reference = "fdet-1700000000-0-det-a-track-1"
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": 1000.0,
                "freshness_window_seconds": 5.0,
                "source_generation": 11,
                "source_version": 13,
                "identity_count": 1,
                "identities": [
                    {
                        "app_identity_id": "app-a",
                        "status": "visible",
                        "provisional": True,
                        "generation": 11,
                        "identity_version": 3,
                        "updated_at": 1000.0,
                        "detection_reference": canonical_reference,
                        "observation_history": [],
                        "historical_yolo_track_ids": ["1"],
                    }
                ],
            }
            publish_identity_snapshot(path, payload)

            evidence_ref = "evidence://detection/detref-1"
            (refs_dir / "detref-1.json").write_text(
                "{\"kind\":\"detection\",\"id\":\"detref-1\",\"reference\":\"evidence://detection/detref-1\",\"tracking_detection_reference\":\""
                + canonical_reference
                + "\"}",
                encoding="utf-8",
            )

            result = find_identity_candidates(
                detection_reference=evidence_ref,
                max_candidates=8,
                path=path,
                evidence_root_dir=evidence_root,
                enabled=True,
                now=1001.0,
            )

            self.assertTrue(result["available"])
            self.assertEqual(result["candidate_lookup_state"], "found")
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["reference_kind"], "evidence_detection_reference")
            self.assertEqual(result["resolved_detection_reference"], canonical_reference)

    def test_find_identity_candidates_rejects_unknown_reference_formats(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest_tracking_identity_snapshot.json"
            publish_identity_snapshot(
                path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": 1000.0,
                    "freshness_window_seconds": 5.0,
                    "source_generation": 1,
                    "source_version": 1,
                    "identity_count": 0,
                    "identities": [],
                },
            )

            result = find_identity_candidates(
                detection_reference="det-raw-identifier",
                path=path,
                enabled=True,
                now=1001.0,
            )

            self.assertFalse(result["available"])
            self.assertEqual(result["state"], "invalid_request")
            self.assertEqual(result["candidate_lookup_state"], "invalid_request")
            self.assertEqual(result["error"], "unsupported_detection_reference_format")

    def test_find_identity_candidates_reports_missing_and_conflicting_aliases(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            refs_dir = evidence_root / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)

            publish_identity_snapshot(
                path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": 1000.0,
                    "freshness_window_seconds": 5.0,
                    "source_generation": 1,
                    "source_version": 1,
                    "identity_count": 0,
                    "identities": [],
                },
            )

            missing_result = find_identity_candidates(
                detection_reference="evidence://detection/missing-id",
                path=path,
                evidence_root_dir=evidence_root,
                enabled=True,
                now=1001.0,
            )
            self.assertFalse(missing_result["available"])
            self.assertEqual(missing_result["state"], "invalid_reference")
            self.assertEqual(missing_result["error"], "evidence_detection_reference_missing")

            (refs_dir / "detref-conflict.json").write_text(
                json.dumps(
                    {
                        "kind": "detection",
                        "id": "detref-conflict",
                        "reference": "evidence://detection/detref-conflict",
                        "tracking_detection_reference": "fdet-a",
                        "tracking_detection_reference_aliases": ["fdet-a", "fdet-b"],
                    }
                ),
                encoding="utf-8",
            )
            conflict_result = find_identity_candidates(
                detection_reference="evidence://detection/detref-conflict",
                path=path,
                evidence_root_dir=evidence_root,
                enabled=True,
                now=1001.0,
            )
            self.assertFalse(conflict_result["available"])
            self.assertEqual(conflict_result["state"], "conflicting_reference")
            self.assertEqual(
                conflict_result["error"],
                "evidence_detection_reference_mapping_conflict",
            )


if __name__ == "__main__":
    unittest.main()
