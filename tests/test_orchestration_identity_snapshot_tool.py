from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.orchestration.tools import identity_snapshot as identity_snapshot_tool


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestOrchestrationIdentitySnapshotTool(unittest.TestCase):
    def _build_snapshot(self, root: Path, *, identities: list[dict], created_at: float | None = None) -> Path:
        snapshot_path = root / "latest_tracking_identity_snapshot.json"
        snapshot_created_at = time.time() if created_at is None else float(created_at)
        payload = {
            "schema_version": "1.0",
            "state": "available",
            "snapshot_created_at": snapshot_created_at,
            "freshness_window_seconds": 3.0,
            "source_generation": 10,
            "source_version": 12,
            "identity_count": len(identities),
            "identities": identities,
        }
        _write_json(snapshot_path, payload)
        return snapshot_path

    def test_identity_lookup_with_history_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = self._build_snapshot(
                root,
                identities=[
                    {
                        "app_identity_id": "app-1",
                        "status": "visible",
                        "provisional": False,
                        "generation": 10,
                        "identity_version": 5,
                        "updated_at": 1000.5,
                        "detection_reference": "frame:1:idx:0:det:a:yolo:1",
                        "observation_history": [
                            {"timestamp": 999.0, "detection_reference": "frame:0:idx:0:det:a:yolo:1", "event": "visible"},
                            {"timestamp": 1000.0, "detection_reference": "frame:1:idx:0:det:a:yolo:1", "event": "visible"},
                        ],
                        "historical_yolo_track_ids": ["1"],
                    }
                ],
            )

            with patch.object(identity_snapshot_tool, "DEFAULT_SNAPSHOT_PATH", snapshot_path):
                result = identity_snapshot_tool.get_identity_snapshot(
                    app_identity_id="app-1",
                    include_history=True,
                    max_history_entries=1,
                    stale_after_seconds=1_000_000_000.0,
                    investigation_id="inv-1",
                    correlation_id="corr-1",
                )

        self.assertTrue(result["available"])
        self.assertEqual(result["identity_lookup_state"], "found")
        self.assertEqual(result["history_count"], 1)
        self.assertEqual(result["history"][-1]["detection_reference"], "frame:1:idx:0:det:a:yolo:1")
        self.assertTrue(result["compatibility"]["compatible"])
        self.assertEqual(result["investigation_id"], "inv-1")
        self.assertEqual(result["correlation_id"], "corr-1")
        self.assertFalse(result["mutation_authorized"])

    def test_candidate_lookup_preserves_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = self._build_snapshot(
                root,
                identities=[
                    {
                        "app_identity_id": "app-1",
                        "status": "visible",
                        "provisional": False,
                        "generation": 10,
                        "identity_version": 1,
                        "updated_at": 1000.0,
                        "detection_reference": "frame:11:idx:0:det:a:yolo:1",
                        "observation_history": [],
                        "historical_yolo_track_ids": [],
                    },
                    {
                        "app_identity_id": "app-2",
                        "status": "temporarily_lost",
                        "provisional": False,
                        "generation": 10,
                        "identity_version": 2,
                        "updated_at": 1001.0,
                        "detection_reference": "frame:11:idx:0:det:a:yolo:1",
                        "observation_history": [],
                        "historical_yolo_track_ids": [],
                    },
                ],
            )

            with patch.object(identity_snapshot_tool, "DEFAULT_SNAPSHOT_PATH", snapshot_path):
                result = identity_snapshot_tool.get_identity_snapshot(
                    detection_reference="frame:11:idx:0:det:a:yolo:1",
                    max_candidates=8,
                    stale_after_seconds=1_000_000_000.0,
                )

        self.assertTrue(result["available"])
        self.assertEqual(result["candidate_lookup_state"], "found")
        self.assertEqual(result["candidate_count"], 2)
        self.assertTrue(result["ambiguity"]["ambiguous"])
        self.assertEqual(result["identity_match_decision"], "not_decided")

    def test_snapshot_minimum_generation_version_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = self._build_snapshot(
                root,
                identities=[
                    {
                        "app_identity_id": "app-1",
                        "status": "visible",
                        "provisional": False,
                        "generation": 10,
                        "identity_version": 1,
                        "updated_at": 1000.0,
                        "detection_reference": "frame:1:idx:0:det:a:yolo:1",
                        "observation_history": [],
                        "historical_yolo_track_ids": [],
                    }
                ],
            )

            with patch.object(identity_snapshot_tool, "DEFAULT_SNAPSHOT_PATH", snapshot_path):
                result = identity_snapshot_tool.get_identity_snapshot(
                    app_identity_id="app-1",
                    min_source_generation=999,
                    min_source_version=999,
                )

        self.assertFalse(result["available"])
        self.assertEqual(result["state"], "incompatible_snapshot")
        self.assertEqual(result["identity_lookup_state"], "incompatible_snapshot")
        self.assertFalse(result["compatibility"]["compatible"])
        self.assertIn("snapshot_generation_too_old", result["compatibility"]["incompatibilities"])
        self.assertIn("snapshot_version_too_old", result["compatibility"]["incompatibilities"])

    def test_snapshot_stale_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = self._build_snapshot(
                root,
                identities=[
                    {
                        "app_identity_id": "app-1",
                        "status": "visible",
                        "provisional": False,
                        "generation": 10,
                        "identity_version": 1,
                        "updated_at": 1.0,
                        "detection_reference": "frame:1:idx:0:det:a:yolo:1",
                        "observation_history": [],
                        "historical_yolo_track_ids": [],
                    }
                ],
                created_at=1.0,
            )

            with patch.object(identity_snapshot_tool, "DEFAULT_SNAPSHOT_PATH", snapshot_path):
                result = identity_snapshot_tool.get_identity_snapshot(app_identity_id="app-1", stale_after_seconds=0.1)

        self.assertFalse(result["available"])
        self.assertEqual(result["state"], "stale")
        self.assertTrue(result["stale_snapshot"])


if __name__ == "__main__":
    unittest.main()
