from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.contracts import BoundingBox, DetectionCandidate
from agents.tracking.agent_tracking import LLMVisualTrackingAgent


class _CaptureCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self._content),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice], usage=None)


class _CaptureClient:
    def __init__(self, completions: _CaptureCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _make_detection(
    *,
    detection_id: str = "det-1",
    yolo_track_id: str = "track-1",
    detection_reference: str | None = "fdet-1000000-0-det-1-track-1",
) -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        timestamp=1000.0,
        yolo_track_id=yolo_track_id,
        label="person",
        detection_confidence=0.91,
        pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0),
        pixel_center_x=30.0,
        pixel_center_y=80.0,
        relative_depth=0.4,
        ground_x=1.0,
        ground_z=2.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
        app_identity_id=None,
        detection_reference=detection_reference,
    )


def _write_snapshot(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_detection_ref(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _extract_structured_payload(call_kwargs: dict[str, object]) -> dict:
    messages = call_kwargs["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    content = user_message["content"]
    assert isinstance(content, list)
    text_payload = content[0]["text"]
    marker = "Structured input:\n"
    start = str(text_payload).index(marker) + len(marker)
    return json.loads(str(text_payload)[start:])


class TestTrackingIdentityEvidenceProposals(unittest.TestCase):
    def test_feature_disabled_by_default_payload_excludes_identity_evidence(self) -> None:
        completions = _CaptureCompletions(
            '{"proposals":[],"uncertainty_flags":[],"supporting_evidence":[]}'
        )
        client = _CaptureClient(completions)
        agent = LLMVisualTrackingAgent(client=client)

        with patch.dict(
            os.environ,
            {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
            },
            clear=False,
        ):
            result = asyncio.run(
                agent.propose_associations(
                    frame_reference=None,
                    detections=[_make_detection()],
                    event_reasons=["possible_id_switch"],
                    request_timeout_seconds=1.0,
                )
            )

        self.assertTrue(result.available)
        self.assertEqual(len(completions.calls), 1)
        payload = _extract_structured_payload(completions.calls[0])
        self.assertNotIn("identity_candidate_evidence", payload)

    def test_identity_evidence_includes_current_and_temporarily_lost_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            now = time.time()
            _write_snapshot(
                snapshot_path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": now,
                    "freshness_window_seconds": 3600.0,
                    "source_generation": 12,
                    "source_version": 21,
                    "identity_count": 3,
                    "identities": [
                        {
                            "app_identity_id": "app-visible",
                            "status": "visible",
                            "provisional": False,
                            "generation": 12,
                            "identity_version": 4,
                            "updated_at": 1000.0,
                            "detection_id": "det-1",
                            "detection_reference": "fdet-1000000-0-det-1-track-1",
                            "yolo_track_id": "track-1",
                            "historical_yolo_track_ids": ["track-1"],
                            "observation_history": [],
                        },
                        {
                            "app_identity_id": "app-lost",
                            "status": "temporarily_lost",
                            "provisional": False,
                            "generation": 11,
                            "identity_version": 7,
                            "updated_at": 999.0,
                            "detection_id": "det-old",
                            "detection_reference": "fdet-999000-0-det-old-track-9",
                            "yolo_track_id": "track-9",
                            "historical_yolo_track_ids": ["track-9"],
                            "observation_history": [
                                {
                                    "generation": 12,
                                    "timestamp": 1000.0,
                                    "event": "visible",
                                    "detection_id": "det-1",
                                    "detection_reference": "fdet-1000000-0-det-1-track-1",
                                    "yolo_track_id": "track-9",
                                }
                            ],
                        },
                        {
                            "app_identity_id": "app-retired",
                            "status": "retired",
                            "provisional": False,
                            "generation": 5,
                            "identity_version": 2,
                            "updated_at": 900.0,
                            "detection_id": None,
                            "detection_reference": None,
                            "yolo_track_id": None,
                            "historical_yolo_track_ids": [],
                            "observation_history": [],
                        },
                    ],
                },
            )

            proposal_content = json.dumps(
                {
                    "proposals": [
                        {
                            "detection_id": "det-1",
                            "proposed_track_id": "track-1",
                            "association_confidence": 0.81,
                            "occlusion_likelihood": 0.2,
                            "id_switch_risk": 0.3,
                            "depth_conflict_flag": False,
                            "position_conflict_flag": False,
                            "uncertainties": ["crossing_event"],
                            "supporting_evidence": ["history_match"],
                            "detection_reference": "fdet-1000000-0-det-1-track-1",
                            "proposed_app_identity_id": "app-visible",
                            "rejected_app_identity_ids": ["app-lost"],
                            "identity_evidence_refs": [
                                "identity_snapshot:gen:12:ver:21",
                                "identity_candidate:app-visible",
                            ],
                            "identity_snapshot_generation": 12,
                            "identity_snapshot_version": 21,
                            "identity_resolution_status": "identity_proposed",
                        }
                    ],
                    "uncertainty_flags": [],
                    "supporting_evidence": ["identity_evidence_used"],
                }
            )
            completions = _CaptureCompletions(proposal_content)
            client = _CaptureClient(completions)
            agent = LLMVisualTrackingAgent(client=client)

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "1",
                "TRACKING_IDENTITY_SNAPSHOT_PATH": str(snapshot_path),
                "TRACKING_IDENTITY_EVIDENCE_ROOT_DIR": str(evidence_root),
                "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS": "3600",
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection()],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertTrue(result.available)
            self.assertEqual(len(result.proposals), 1)
            proposal = result.proposals[0]
            self.assertEqual(proposal.proposed_app_identity_id, "app-visible")
            self.assertEqual(proposal.identity_snapshot_generation, 12)
            self.assertEqual(proposal.identity_snapshot_version, 21)
            self.assertEqual(proposal.identity_resolution_status, "identity_proposed")
            self.assertEqual(proposal.rejected_app_identity_ids, ["app-lost"])

            payload = _extract_structured_payload(completions.calls[0])
            identity_evidence = payload.get("identity_candidate_evidence")
            self.assertIsInstance(identity_evidence, dict)
            assert isinstance(identity_evidence, dict)

            snapshot_meta = identity_evidence.get("snapshot")
            self.assertIsInstance(snapshot_meta, dict)
            assert isinstance(snapshot_meta, dict)
            self.assertTrue(snapshot_meta.get("available"))
            self.assertEqual(snapshot_meta.get("source_generation"), 12)
            self.assertEqual(snapshot_meta.get("source_version"), 21)

            identity_pool = identity_evidence.get("identity_pool")
            self.assertIsInstance(identity_pool, list)
            assert isinstance(identity_pool, list)
            app_ids = {item.get("app_identity_id") for item in identity_pool if isinstance(item, dict)}
            self.assertIn("app-visible", app_ids)
            self.assertIn("app-lost", app_ids)
            self.assertNotIn("app-retired", app_ids)

    def test_ambiguous_reference_is_marked_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            now = time.time()
            _write_snapshot(
                snapshot_path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": now,
                    "freshness_window_seconds": 3600.0,
                    "source_generation": 2,
                    "source_version": 3,
                    "identity_count": 2,
                    "identities": [
                        {
                            "app_identity_id": "app-a",
                            "status": "visible",
                            "provisional": False,
                            "generation": 2,
                            "identity_version": 1,
                            "updated_at": 1000.0,
                            "detection_id": "det-a",
                            "detection_reference": "fdet-dup",
                            "yolo_track_id": "track-a",
                            "historical_yolo_track_ids": ["track-a"],
                            "observation_history": [],
                        },
                        {
                            "app_identity_id": "app-b",
                            "status": "temporarily_lost",
                            "provisional": False,
                            "generation": 2,
                            "identity_version": 1,
                            "updated_at": 999.0,
                            "detection_id": "det-b",
                            "detection_reference": "fdet-other",
                            "yolo_track_id": "track-b",
                            "historical_yolo_track_ids": ["track-b"],
                            "observation_history": [
                                {
                                    "generation": 2,
                                    "timestamp": 999.5,
                                    "event": "visible",
                                    "detection_id": "det-a",
                                    "detection_reference": "fdet-dup",
                                    "yolo_track_id": "track-b",
                                }
                            ],
                        },
                    ],
                },
            )
            completions = _CaptureCompletions(
                '{"proposals":[],"uncertainty_flags":[],"supporting_evidence":[]}'
            )
            client = _CaptureClient(completions)
            agent = LLMVisualTrackingAgent(client=client)

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "1",
                "TRACKING_IDENTITY_SNAPSHOT_PATH": str(snapshot_path),
                "TRACKING_IDENTITY_EVIDENCE_ROOT_DIR": str(evidence_root),
                "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS": "3600",
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection(detection_reference="fdet-dup")],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            payload = _extract_structured_payload(completions.calls[0])
            evidence = payload["identity_candidate_evidence"]
            per_detection = evidence["per_detection"]
            self.assertEqual(len(per_detection), 1)
            item = per_detection[0]
            self.assertFalse(item["evidence_usable"])
            self.assertEqual(item["evidence_unusable_reason"], "candidate_reference_ambiguous")

    def test_unknown_proposed_identity_is_normalized_as_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            now = time.time()
            _write_snapshot(
                snapshot_path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": now,
                    "freshness_window_seconds": 3600.0,
                    "source_generation": 4,
                    "source_version": 9,
                    "identity_count": 1,
                    "identities": [
                        {
                            "app_identity_id": "app-known",
                            "status": "visible",
                            "provisional": False,
                            "generation": 4,
                            "identity_version": 2,
                            "updated_at": 1000.0,
                            "detection_id": "det-1",
                            "detection_reference": "fdet-1000000-0-det-1-track-1",
                            "yolo_track_id": "track-1",
                            "historical_yolo_track_ids": ["track-1"],
                            "observation_history": [],
                        }
                    ],
                },
            )

            proposal_content = json.dumps(
                {
                    "proposals": [
                        {
                            "detection_id": "det-1",
                            "proposed_track_id": "track-1",
                            "association_confidence": 0.82,
                            "occlusion_likelihood": 0.12,
                            "id_switch_risk": 0.19,
                            "depth_conflict_flag": False,
                            "position_conflict_flag": False,
                            "uncertainties": [],
                            "supporting_evidence": ["identity_candidate_match"],
                            "detection_reference": "fdet-1000000-0-det-1-track-1",
                            "proposed_app_identity_id": "app-unknown",
                            "rejected_app_identity_ids": ["app-known"],
                            "identity_evidence_refs": ["identity_snapshot:gen:4:ver:9"],
                            "identity_snapshot_generation": 4,
                            "identity_snapshot_version": 9,
                            "identity_resolution_status": "identity_proposed",
                        }
                    ],
                    "uncertainty_flags": [],
                    "supporting_evidence": [],
                }
            )
            completions = _CaptureCompletions(proposal_content)
            client = _CaptureClient(completions)
            agent = LLMVisualTrackingAgent(client=client)

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "1",
                "TRACKING_IDENTITY_SNAPSHOT_PATH": str(snapshot_path),
                "TRACKING_IDENTITY_EVIDENCE_ROOT_DIR": str(evidence_root),
                "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS": "3600",
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection()],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertTrue(result.available)
            self.assertEqual(len(result.proposals), 1)
            proposal = result.proposals[0]
            self.assertIsNone(proposal.proposed_app_identity_id)
            self.assertEqual(proposal.identity_resolution_status, "unresolved_unknown_app_identity_id")
            self.assertIn("identity_claim_unknown_app_identity_id", proposal.uncertainties)

    def test_new_detection_reference_with_missing_history_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            now = time.time()
            _write_snapshot(
                snapshot_path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": now,
                    "freshness_window_seconds": 3600.0,
                    "source_generation": 4,
                    "source_version": 8,
                    "identity_count": 1,
                    "identities": [
                        {
                            "app_identity_id": "app-old",
                            "status": "temporarily_lost",
                            "provisional": False,
                            "generation": 4,
                            "identity_version": 2,
                            "updated_at": 995.0,
                            "detection_id": "det-old",
                            "detection_reference": "fdet-old",
                            "yolo_track_id": "track-old",
                            "historical_yolo_track_ids": ["track-old"],
                            "observation_history": [],
                        }
                    ],
                },
            )
            completions = _CaptureCompletions(
                json.dumps(
                    {
                        "proposals": [
                            {
                                "detection_id": "det-new",
                                "proposed_track_id": None,
                                "association_confidence": 0.32,
                                "occlusion_likelihood": 0.66,
                                "id_switch_risk": 0.77,
                                "depth_conflict_flag": False,
                                "position_conflict_flag": False,
                                "uncertainties": ["insufficient_history"],
                                "supporting_evidence": [],
                                "identity_resolution_status": "inconclusive",
                            }
                        ],
                        "uncertainty_flags": [],
                        "supporting_evidence": [],
                    }
                )
            )
            client = _CaptureClient(completions)
            agent = LLMVisualTrackingAgent(client=client)

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "1",
                "TRACKING_IDENTITY_SNAPSHOT_PATH": str(snapshot_path),
                "TRACKING_IDENTITY_EVIDENCE_ROOT_DIR": str(evidence_root),
                "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS": "3600",
            }
            detection = _make_detection(
                detection_id="det-new",
                yolo_track_id="track-new",
                detection_reference="fdet-new-reference",
            )
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[detection],
                        event_reasons=["occlusion_reappearance"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertTrue(result.available)
            self.assertEqual(len(result.proposals), 1)
            self.assertEqual(result.proposals[0].detection_reference, "fdet-new-reference")
            payload = _extract_structured_payload(completions.calls[0])
            per_detection = payload["identity_candidate_evidence"]["per_detection"][0]
            self.assertEqual(per_detection["lookup"]["candidate_lookup_state"], "none")
            self.assertTrue(per_detection["evidence_usable"])

    def test_stale_snapshot_is_marked_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            _write_snapshot(
                snapshot_path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": 1.0,
                    "freshness_window_seconds": 0.1,
                    "source_generation": 1,
                    "source_version": 1,
                    "identity_count": 0,
                    "identities": [],
                },
            )
            completions = _CaptureCompletions(
                '{"proposals":[],"uncertainty_flags":[],"supporting_evidence":[]}'
            )
            client = _CaptureClient(completions)
            agent = LLMVisualTrackingAgent(client=client)

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "1",
                "TRACKING_IDENTITY_SNAPSHOT_PATH": str(snapshot_path),
                "TRACKING_IDENTITY_EVIDENCE_ROOT_DIR": str(evidence_root),
                "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS": "0.1",
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection()],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            payload = _extract_structured_payload(completions.calls[0])
            snapshot_meta = payload["identity_candidate_evidence"]["snapshot"]
            self.assertFalse(snapshot_meta["available"])
            self.assertEqual(snapshot_meta["state"], "stale")
            per_detection = payload["identity_candidate_evidence"]["per_detection"][0]
            self.assertFalse(per_detection["evidence_usable"])
            self.assertEqual(per_detection["evidence_unusable_reason"], "snapshot_stale")

    def test_conflicting_evidence_alias_is_marked_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            refs_dir = evidence_root / "refs"
            now = time.time()
            _write_snapshot(
                snapshot_path,
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": now,
                    "freshness_window_seconds": 3600.0,
                    "source_generation": 1,
                    "source_version": 1,
                    "identity_count": 0,
                    "identities": [],
                },
            )
            _write_detection_ref(
                refs_dir / "detref-conflict.json",
                {
                    "kind": "detection",
                    "id": "detref-conflict",
                    "reference": "evidence://detection/detref-conflict",
                    "tracking_detection_reference": "fdet-a",
                    "tracking_detection_reference_aliases": ["fdet-a", "fdet-b"],
                },
            )
            completions = _CaptureCompletions(
                '{"proposals":[],"uncertainty_flags":[],"supporting_evidence":[]}'
            )
            client = _CaptureClient(completions)
            agent = LLMVisualTrackingAgent(client=client)

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "1",
                "TRACKING_IDENTITY_SNAPSHOT_PATH": str(snapshot_path),
                "TRACKING_IDENTITY_EVIDENCE_ROOT_DIR": str(evidence_root),
                "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS": "3600",
            }
            with patch.dict(os.environ, env, clear=False):
                asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[
                            _make_detection(
                                detection_reference="evidence://detection/detref-conflict"
                            )
                        ],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            payload = _extract_structured_payload(completions.calls[0])
            per_detection = payload["identity_candidate_evidence"]["per_detection"][0]
            self.assertFalse(per_detection["evidence_usable"])
            self.assertEqual(
                per_detection["evidence_unusable_reason"],
                "evidence_detection_reference_mapping_conflict",
            )

    def test_inconclusive_identity_outcome_and_no_mutation_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_path = root / "latest_tracking_identity_snapshot.json"
            evidence_root = root / "tracking_evidence"
            now = time.time()
            payload = {
                "schema_version": "1.0",
                "state": "available",
                "snapshot_created_at": now,
                "freshness_window_seconds": 3600.0,
                "source_generation": 9,
                "source_version": 10,
                "identity_count": 0,
                "identities": [],
            }
            _write_snapshot(snapshot_path, payload)
            before_contents = snapshot_path.read_text(encoding="utf-8")

            completions = _CaptureCompletions(
                json.dumps(
                    {
                        "proposals": [
                            {
                                "detection_id": "det-1",
                                "proposed_track_id": None,
                                "association_confidence": 0.35,
                                "occlusion_likelihood": 0.7,
                                "id_switch_risk": 0.8,
                                "depth_conflict_flag": False,
                                "position_conflict_flag": False,
                                "uncertainties": ["insufficient_identity_evidence"],
                                "supporting_evidence": [],
                                "proposed_app_identity_id": None,
                                "rejected_app_identity_ids": [],
                                "identity_evidence_refs": ["identity_snapshot:gen:9:ver:10"],
                                "identity_snapshot_generation": 9,
                                "identity_snapshot_version": 10,
                                "identity_resolution_status": "inconclusive",
                            }
                        ],
                        "uncertainty_flags": [],
                        "supporting_evidence": [],
                    }
                )
            )
            client = _CaptureClient(completions)
            agent = LLMVisualTrackingAgent(client=client)

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "1",
                "TRACKING_IDENTITY_SNAPSHOT_PATH": str(snapshot_path),
                "TRACKING_IDENTITY_EVIDENCE_ROOT_DIR": str(evidence_root),
                "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS": "3600",
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection()],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertTrue(result.available)
            self.assertEqual(len(result.proposals), 1)
            proposal = result.proposals[0]
            self.assertEqual(proposal.detection_reference, "fdet-1000000-0-det-1-track-1")
            self.assertIsNone(proposal.proposed_app_identity_id)
            self.assertEqual(proposal.identity_resolution_status, "inconclusive")
            self.assertEqual(proposal.identity_snapshot_generation, 9)
            self.assertEqual(proposal.identity_snapshot_version, 10)
            self.assertIn("insufficient_identity_evidence", proposal.uncertainties)

            after_contents = snapshot_path.read_text(encoding="utf-8")
            self.assertEqual(before_contents, after_contents)


if __name__ == "__main__":
    unittest.main()
