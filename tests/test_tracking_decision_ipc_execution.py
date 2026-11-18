from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from agents.contracts import BoundingBox, DetectionCandidate
from agents.dashboard.state import SharedDashboardState
from agents.dashboard import tracking_loop as dashboard_tracking_loop
from agents.dashboard.tracking_loop import _build_prediction_detection_advisories
from agents.observation.tools import track_evidence_history as observation_track_evidence_history_tools
from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport
from agents.tracking.identity_registry import PersistentIdentityRegistry


def _decision(
    *,
    investigation_id: str,
    request_id: str,
    tool_name: str = "get_track_evidence_history",
    iteration_index: int = 1,
    arguments: dict | None = None,
) -> dict:
    if arguments is None:
        if tool_name == "get_identity_snapshot":
            arguments = {"app_identity_id": "app-1"}
        else:
            arguments = {"track_id": "track-1"}
    return {
        "action": (
            "request_identity_snapshot"
            if tool_name == "get_identity_snapshot"
            else "request_additional_tracking_evidence"
        ),
        "investigation_id": investigation_id,
        "iteration_index": iteration_index,
        "tool_requests": [
            {
                "request_id": request_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        ],
    }


def _transport(root: Path, *, role: str) -> TrackingDecisionIpcTransport:
    return TrackingDecisionIpcTransport(
        requests_path=root / "tracking_decision_tool_requests.jsonl",
        results_path=root / "tracking_decision_tool_results.jsonl",
        request_index_path=root / "latest_tracking_decision_request_index.json",
        result_index_path=root / "latest_tracking_decision_result_index.json",
        writer_role=role,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_minimal_track_evidence_store(root: Path) -> Path:
    evidence_root = root / "tracking_evidence"
    refs_dir = evidence_root / "refs"
    frames_dir = evidence_root / "frames"
    envelopes_dir = evidence_root / "envelopes"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_id = "frame-abc"
    frame_reference = f"evidence://frame/{frame_id}"
    frame_path = frames_dir / f"{frame_id}.jpg"
    frame_path.write_bytes(b"fake-jpeg-bytes")

    _write_json(
        refs_dir / f"{frame_id}.json",
        {
            "kind": "frame",
            "id": frame_id,
            "reference": frame_reference,
            "timestamp": 1000.0,
            "created_at": 1000.0,
            "expires_at": 9999999999.0,
            "status": "available",
            "frame_path": str(frame_path),
            "width": 32,
            "height": 24,
        },
    )

    det_ref_id = "detref-abc"
    detection_reference = f"evidence://detection/{det_ref_id}"
    _write_json(
        refs_dir / f"{det_ref_id}.json",
        {
            "kind": "detection",
            "id": det_ref_id,
            "reference": detection_reference,
            "timestamp": 1000.0,
            "created_at": 1000.0,
            "expires_at": 9999999999.0,
            "track_id": "track-1",
            "detection_id": "det-1",
            "frame_reference": frame_reference,
            "pixel_bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
        },
    )

    _write_json(
        envelopes_dir / "msg-1.json",
        {
            "message_id": "msg-1",
            "stored_at": 1001.0,
            "validation": {
                "valid": True,
                "errors": [],
                "warnings": [],
                "message_id": "msg-1",
                "track_id": "track-1",
            },
            "envelope": {
                "schema_version": "1.0",
                "message_id": "msg-1",
                "track_id": "track-1",
                "timestamp": 1000.0,
                "frame_id": "frame-1",
                "detection_id": "det-1",
                "pixel_bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                "pixel_center": {"x": 20.0, "y": 40.0},
                "relative_depth": 0.3,
                "ground_position": {"x": 1.0, "z": 2.0, "unit": "meters"},
                "tracking_confidence": None,
                "quality_flags": ["position_kind:calibrated_ground_plane"],
                "track_history": [
                    {
                        "observation_id": "obs-1",
                        "track_id": "track-1",
                        "timestamp": 1000.0,
                        "source": "tracking_frame",
                        "pixel_x": 20.0,
                        "pixel_y": 40.0,
                        "ground_point": {"x": 1.0, "z": 2.0, "unit": "meters"},
                        "confidence": 0.8,
                        "quality_flags": ["position_kind:calibrated_ground_plane"],
                    }
                ],
                "tracking_claims": {
                    "claimed_resolved_track_id": "track-1",
                    "association_status": "accepted",
                    "possible_id_switch": False,
                    "visibility_status": "unknown",
                    "claim_reasons": ["vlm_association_accepted"],
                    "identity_claim": {"app_identity_id": "identity-1"},
                },
                "frame_reference": frame_reference,
                "source_evidence_refs": [frame_reference, detection_reference],
                "uncertainty": [],
            },
        },
    )

    return evidence_root


def _candidate(*, detection_id: str, yolo_track_id: str, timestamp: float, ground_x: float, ground_z: float) -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        timestamp=timestamp,
        yolo_track_id=yolo_track_id,
        label="person",
        detection_confidence=0.92,
        pixel_bbox=BoundingBox(x1=10.0, y1=15.0, x2=60.0, y2=110.0),
        pixel_center_x=100.0,
        pixel_center_y=80.0,
        relative_depth=0.3,
        ground_x=ground_x,
        ground_z=ground_z,
        ground_coordinate_unit="meters",
        position_kind="calibrated_ground_plane",
        motion_history=[],
    )


def _seed_temporarily_lost_identity() -> tuple[PersistentIdentityRegistry, str]:
    registry = PersistentIdentityRegistry()
    registry.process_detections(
        now=1000.0,
        detections=[
            _candidate(
                detection_id="seed",
                yolo_track_id="track-src",
                timestamp=1000.0,
                ground_x=2.0,
                ground_z=3.0,
            )
        ],
    )
    app_identity_id = registry.get_identity_for_yolo_track("track-src")
    assert isinstance(app_identity_id, str)
    registry.process_detections(now=1000.1, detections=[])
    return registry, app_identity_id


class _TrackEvidenceToolSession:
    def __init__(self, *, evidence_root: Path, advisory_store_path: Path) -> None:
        self._evidence_root = evidence_root
        self._advisory_store_path = advisory_store_path

    async def call_tool(self, tool_name: str, arguments: dict) -> SimpleNamespace:
        if tool_name != "get_track_evidence_history":
            raise RuntimeError("unsupported_tool")

        with (
            patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", self._evidence_root),
            patch.object(observation_track_evidence_history_tools, "ADVISORY_STORE_PATH", self._advisory_store_path),
        ):
            payload = observation_track_evidence_history_tools.get_track_evidence_history(**arguments)
        return SimpleNamespace(content=[SimpleNamespace(json=payload)])


def _dashboard_submit_worker(root_path: str) -> None:
    root = Path(root_path)
    transport = _transport(root, role="dashboard")
    transport.dashboard_submit_decision_request(
        decision=_decision(investigation_id="inv-proc-1", request_id="req-proc-1"),
        now_timestamp=1000.0,
    )


def _orchestration_fixture_worker(root_path: str) -> None:
    root = Path(root_path)
    transport = _transport(root, role="orchestration")
    transport.run_fixture_orchestration_cycle(now_timestamp=1000.1, max_requests=4)


class _FakeSession:
    def __init__(self, *, payload: dict | None = None, sleep_seconds: float = 0.0) -> None:
        self._payload = payload if payload is not None else {"available": True}
        self._sleep_seconds = max(0.0, float(sleep_seconds))
        self.call_count = 0
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name: str, arguments: dict) -> SimpleNamespace:
        self.call_count += 1
        self.calls.append((tool_name, dict(arguments)))
        if self._sleep_seconds > 0:
            await asyncio.sleep(self._sleep_seconds)
        item = SimpleNamespace(json=dict(self._payload))
        return SimpleNamespace(content=[item])


class TestTrackingDecisionIpcProcessIntegration(unittest.TestCase):
    def test_two_process_fixture_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = mp.get_context("spawn")

            dashboard_process = context.Process(target=_dashboard_submit_worker, args=(str(root),))
            dashboard_process.start()
            dashboard_process.join(timeout=10)
            self.assertEqual(dashboard_process.exitcode, 0)

            orchestration_process = context.Process(target=_orchestration_fixture_worker, args=(str(root),))
            orchestration_process.start()
            orchestration_process.join(timeout=10)
            self.assertEqual(orchestration_process.exitcode, 0)

            dashboard_transport = _transport(root, role="dashboard")
            polled = dashboard_transport.dashboard_poll_results(now_timestamp=1000.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["request_id"], "req-proc-1")
            self.assertEqual(polled["results"][0]["status"], "ok")


class TestTrackingDecisionIpcWriterOwnership(unittest.TestCase):
    def test_dashboard_role_cannot_publish_orchestration_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _transport(Path(temp_dir), role="dashboard")
            response = transport.orchestration_publish_fixture_result(
                request_event={
                    "request_id": "req-x",
                    "investigation_id": "inv-x",
                    "tool_name": "get_track_evidence_history",
                    "iteration_index": 1,
                }
            )
            self.assertTrue(response["rejected"])
            self.assertEqual(response["reason"], "writer_role_violation")

    def test_orchestration_role_cannot_submit_dashboard_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _transport(Path(temp_dir), role="orchestration")
            response = transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-x", request_id="req-x"),
                now_timestamp=1000.0,
            )
            self.assertTrue(response["rejected"])
            self.assertEqual(response["reason"], "writer_role_violation")


class TestTrackingDecisionIpcBoundedMcpExecution(unittest.IsolatedAsyncioTestCase):
    async def test_enforces_one_mcp_call_per_investigation_per_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True, "state": "recorded_observation"})
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-1", request_id="req-1"),
                now_timestamp=now,
            )
            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-1", request_id="req-2"),
                now_timestamp=now + 0.01,
            )

            summary = await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            self.assertEqual(session.call_count, 1)
            self.assertEqual(summary["pending_count"], 2)

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 1.0)
            self.assertEqual(polled["result_count"], 2)
            statuses = sorted(result["status"] for result in polled["results"])
            self.assertEqual(statuses, ["ok", "rejected"])

    async def test_rejects_invalid_tool_arguments_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True})
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(
                    investigation_id="inv-invalid",
                    request_id="req-invalid",
                    tool_name="get_track_evidence_history",
                    arguments={},
                ),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            self.assertEqual(session.call_count, 0)
            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.1)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "rejected")
            self.assertEqual(polled["results"][0]["error_code"], "invalid_tool_arguments")

    async def test_timeout_is_persisted_without_blocking_future_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            slow_session = _FakeSession(payload={"available": True}, sleep_seconds=0.05)
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-timeout", request_id="req-timeout"),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=slow_session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.001,
            )

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "timeout")

    async def test_enforces_one_mcp_call_per_investigation_across_cycles_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True, "state": "recorded_observation"})
            now = time.time()

            first_submit = dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-once", request_id="req-1"),
                now_timestamp=now,
            )
            self.assertTrue(first_submit["accepted"])

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )
            self.assertEqual(session.call_count, 1)

            second_submit = dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-once", request_id="req-2"),
                now_timestamp=now + 0.1,
            )
            self.assertTrue(second_submit["rejected"])
            self.assertEqual(second_submit["reason"], "tool_budget_exhausted")

            restarted_orchestration_transport = _transport(root, role="orchestration")
            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=restarted_orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )
            self.assertEqual(session.call_count, 1)

    async def test_lookup_results_for_request_does_not_consume_dashboard_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True, "state": "recorded_observation"})
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-lookup", request_id="req-lookup"),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            lookup = dashboard_transport.lookup_results_for_request(
                investigation_id="inv-lookup",
                request_id="req-lookup",
                tool_name="get_track_evidence_history",
                now_timestamp=now + 0.1,
            )
            self.assertTrue(lookup["available"])
            self.assertEqual(lookup["result_count"], 1)
            self.assertIsInstance(lookup.get("latest"), dict)

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "ok")

    async def test_tool_response_payload_preserves_advisory_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(
                payload={
                    "available": True,
                    "state": "recorded_observation",
                    "prediction_detection_advisory_evidence": {
                        "available": True,
                        "state": "advisory_matched",
                        "advisory_count": 1,
                        "matched_count": 1,
                        "ambiguous_count": 0,
                        "advisories": [{"decision": "matched", "app_identity_id": "identity-1"}],
                        "provenance": {
                            "identity_verification_authorized": False,
                            "identity_mutation_authorized": False,
                        },
                    },
                    "mutation_authorized": False,
                }
            )
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-advisory", request_id="req-advisory"),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.2)
            self.assertEqual(polled["result_count"], 1)
            result = polled["results"][0]
            self.assertEqual(result["status"], "ok")
            payload = result["result_payload"]
            advisory = payload["prediction_detection_advisory_evidence"]
            self.assertTrue(advisory["available"])
            self.assertEqual(advisory["advisory_count"], 1)
            self.assertFalse(advisory["provenance"]["identity_verification_authorized"])
            self.assertFalse(advisory["provenance"]["identity_mutation_authorized"])
            self.assertFalse(payload["mutation_authorized"])

    async def test_restart_stale_advisory_artifact_is_rejected_in_dispatch_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = _build_minimal_track_evidence_store(root)
            advisory_store_path = root / "latest_tracking_prediction_detection_advisories.json"
            _write_json(
                advisory_store_path,
                {
                    "schema_version": "1.0",
                    "updated_at": 1000.0,
                    "available": True,
                    "summary": {"available": True},
                    "advisories": [
                        {
                            "decision": "matched",
                            "classification": "prediction_detection_advisory",
                            "reason": None,
                            "app_identity_id": "identity-1",
                            "candidate_yolo_track_id": "track-1",
                            "source_yolo_track_id": "track-src",
                            "match_metric": "pixel_distance",
                            "match_distance": 10.0,
                            "match_threshold": 120.0,
                            "normalized_score": 0.08,
                            "predicted_for_timestamp": 1000.0,
                            "expires_at": 1000.5,
                            "advisory_created_at": 1000.0,
                        }
                    ],
                },
            )

            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _TrackEvidenceToolSession(
                evidence_root=evidence_root,
                advisory_store_path=advisory_store_path,
            )
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(
                    investigation_id="inv-restart-stale",
                    request_id="req-restart-stale",
                    arguments={
                        "track_id": "track-1",
                        "now_timestamp": 1002.0,
                    },
                ),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.2)
            self.assertEqual(polled["result_count"], 1)
            result = polled["results"][0]
            self.assertEqual(result["status"], "ok")
            payload = result["result_payload"]
            advisory = payload["prediction_detection_advisory_evidence"]
            self.assertFalse(advisory["available"])
            self.assertEqual(advisory["state"], "no_current_advisory")
            self.assertEqual(advisory["advisory_count"], 0)
            self.assertGreaterEqual(advisory["rejected_count"], 1)
            self.assertFalse(payload["mutation_authorized"])

    async def test_occlusion_reappearance_advisory_flows_through_ipc_and_stays_non_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = _build_minimal_track_evidence_store(root)
            advisory_store_path = root / "latest_tracking_prediction_detection_advisories.json"

            registry, app_identity_id = _seed_temporarily_lost_identity()
            snapshot = registry.snapshot()
            state = SharedDashboardState()
            state.tracking_identity_registry_enabled = True
            state.tracking_identity_registry = registry
            state.calibration_status = "loaded"
            state.latest_tracking_prediction_evidence_by_identity = {
                app_identity_id: {
                    "app_identity_id": app_identity_id,
                    "source_yolo_track_id": "track-src",
                    "estimate_id": "est-1",
                    "source_generated_at": 1000.2,
                    "source_identity_registry_generation": int(snapshot["generation"]),
                    "source_identity_registry_version": int(snapshot["version"]),
                    "source_estimated_at": 1000.2,
                    "source_verification_timestamp": 1000.2,
                    "predicted_for_timestamp": 1000.4,
                    "expires_at": 1001.2,
                    "coordinate_system": "calibrated_ground_plane",
                    "ground_coordinate_mode": "calibrated_metric",
                    "coordinate_unit": "meters",
                    "estimated_pixel_position": {"x": 100.0, "y": 80.0},
                    "estimated_ground_position": {"x": 2.0, "z": 3.0, "unit": "meters"},
                    "uncertainty": ["brief_occlusion"],
                    "classification": "occlusion_candidate",
                    "source_tracking_message_ids": ["msg-1"],
                    "source_verification_observation_id": "obs-1",
                    "source_verification_message_id": "verify-1",
                    "validation": {"valid": True},
                    "source_feedback_expires_at": 1001.2,
                    "promoted_at": 1000.2,
                }
            }

            with patch(
                "agents.dashboard.tracking_loop.LATEST_TRACKING_PREDICTION_DETECTION_ADVISORIES_PATH",
                advisory_store_path,
            ):
                _build_prediction_detection_advisories(
                    state=state,
                    now=1000.4,
                    tracking_candidates=[
                        _candidate(
                            detection_id="reappear",
                            yolo_track_id="track-1",
                            timestamp=1000.4,
                            ground_x=2.1,
                            ground_z=3.1,
                        )
                    ],
                )

            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _TrackEvidenceToolSession(
                evidence_root=evidence_root,
                advisory_store_path=advisory_store_path,
            )
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(
                    investigation_id="inv-occlusion-e2e",
                    request_id="req-occlusion-e2e",
                    arguments={
                        "track_id": "track-1",
                        "now_timestamp": 1000.5,
                    },
                ),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.2)
            self.assertEqual(polled["result_count"], 1)
            result = polled["results"][0]
            self.assertEqual(result["status"], "ok")
            payload = result["result_payload"]
            advisory = payload["prediction_detection_advisory_evidence"]
            motion = payload["observation_motion_consistency_assessment"]

            self.assertTrue(advisory["available"])
            self.assertEqual(advisory["state"], "advisory_matched")
            self.assertEqual(advisory["advisory_count"], 1)
            self.assertEqual(advisory["matched_count"], 1)
            self.assertEqual(advisory["ambiguous_count"], 0)
            self.assertEqual(advisory["advisories"][0]["decision"], "matched")
            self.assertEqual(advisory["advisories"][0]["classification"], "prediction_detection_advisory")
            self.assertEqual(advisory["advisories"][0]["app_identity_id"], app_identity_id)
            self.assertEqual(advisory["advisories"][0]["candidate_yolo_track_id"], "track-1")

            self.assertFalse(motion["available"])
            self.assertEqual(motion["motion_consistency"], "insufficient_evidence")
            self.assertEqual(motion["state"], "insufficient_history")
            self.assertIn("obs-1", motion["supporting_observation_ids"])
            self.assertIn("insufficient_history", motion["uncertainty"])
            self.assertFalse(motion["provenance"]["identity_verification_authorized"])
            self.assertFalse(motion["provenance"]["identity_mutation_authorized"])

            self.assertFalse(payload["mutation_authorized"])
            self.assertFalse(advisory["provenance"]["identity_verification_authorized"])
            self.assertFalse(advisory["provenance"]["identity_mutation_authorized"])

            lookup = dashboard_transport.lookup_results_for_request(
                investigation_id="inv-occlusion-e2e",
                request_id="req-occlusion-e2e",
                tool_name="get_track_evidence_history",
                now_timestamp=now + 0.25,
            )
            self.assertEqual(lookup["result_count"], 1)
            lookup_payload = lookup["results"][0]["result_payload"]
            self.assertEqual(lookup_payload.get("message_count"), 1)

            assessment_payload = {
                "available": True,
                "state": "observation_follow_up_completed",
                "track_id": "track-1",
                "summary": "candidate identity continuity supported but conflicting cues remain",
                "supporting_observation_ids": ["obs-1", "obs-occlusion-reappear"],
                "conflicting_evidence": ["occlusion_overlap_candidate"],
                "uncertainties": ["identity_ambiguity", "occlusion_overlap"],
                "needs_more_observations": True,
                "decision": {
                    "action": "inconclusive",
                    "reason": "conflicting_identity_evidence",
                },
                "provenance": {
                    "source": "observation_follow_up_llm",
                    "identity_verification_authorized": False,
                    "identity_mutation_authorized": False,
                },
            }
            publish_assessment = dashboard_transport.dashboard_register_observation_follow_up_assessment(
                investigation_id="inv-occlusion-e2e",
                request_id="req-occlusion-e2e",
                tool_name="get_track_evidence_history",
                idempotency_key="inv-occlusion-e2e:req-occlusion-e2e:get_track_evidence_history",
                track_id="track-1",
                assessment=assessment_payload,
                now_timestamp=now + 0.3,
            )
            self.assertTrue(publish_assessment["published"])

            tracking_state = SharedDashboardState()
            tracking_state.tracking_decision_ipc_enabled = True
            tracking_state.tracking_decision_ipc_transport = _transport(root, role="dashboard")
            tracking_state.tracking_decision_investigations["inv-occlusion-e2e"] = {
                "investigation_id": "inv-occlusion-e2e",
                "request_id": "req-occlusion-e2e",
                "tool_name": "get_track_evidence_history",
                "correlation_id": "corr-occlusion-e2e",
                "initial_decision": {
                    "action": "request_additional_tracking_evidence",
                    "investigation_id": "inv-occlusion-e2e",
                },
                "evidence_references": {
                    "tracking_message_id": "msg-1",
                    "frame_reference": "evidence://frame/frame-abc",
                },
                "tool_calls": 1,
                "iterations": 1,
                "terminal": False,
                "follow_up_submitted": False,
                "follow_up_completed": False,
                "result_status": "ok",
                "result_payload": dict(lookup_payload),
                "result_key": "inv-occlusion-e2e:req-occlusion-e2e:get_track_evidence_history",
                "updated_at": now + 0.35,
            }

            captured_follow_up_contexts: list[dict[str, Any]] = []

            def _capture_submit(**kwargs):
                follow_up = kwargs.get("follow_up_context")
                if isinstance(follow_up, dict):
                    captured_follow_up_contexts.append(dict(follow_up))
                return True

            with patch.object(
                dashboard_tracking_loop,
                "submit_tracking_runtime_candidates",
                side_effect=_capture_submit,
            ):
                dashboard_tracking_loop._consume_tracking_decision_results_if_enabled(
                    state=tracking_state,
                    tracking_runtime=SimpleNamespace(),
                    frame=SimpleNamespace(shape=(10, 10, 3)),
                    tracking_candidates=[],
                    now=now + 0.4,
                )

            self.assertEqual(len(captured_follow_up_contexts), 1)
            follow_up_context = captured_follow_up_contexts[0]
            self.assertEqual(follow_up_context.get("investigation_id"), "inv-occlusion-e2e")
            self.assertEqual(follow_up_context.get("request_id"), "req-occlusion-e2e")
            self.assertEqual(
                follow_up_context.get("idempotency_key"),
                "inv-occlusion-e2e:req-occlusion-e2e:get_track_evidence_history:observation_follow_up_assessment",
            )
            self.assertEqual(follow_up_context.get("correlation_id"), "corr-occlusion-e2e")

            follow_up_mcp_result = follow_up_context.get("mcp_result")
            self.assertIsInstance(follow_up_mcp_result, dict)
            assert isinstance(follow_up_mcp_result, dict)

            follow_up_advisory = follow_up_mcp_result.get("prediction_detection_advisory_evidence")
            self.assertIsInstance(follow_up_advisory, dict)
            assert isinstance(follow_up_advisory, dict)
            self.assertTrue(follow_up_advisory.get("available"))
            self.assertFalse(follow_up_advisory["provenance"]["identity_verification_authorized"])
            self.assertFalse(follow_up_advisory["provenance"]["identity_mutation_authorized"])

            follow_up_assessment = follow_up_mcp_result.get("observation_llm_follow_up_assessment")
            self.assertIsInstance(follow_up_assessment, dict)
            assert isinstance(follow_up_assessment, dict)
            self.assertEqual(follow_up_assessment.get("decision", {}).get("action"), "inconclusive")
            self.assertIn("obs-occlusion-reappear", follow_up_assessment.get("supporting_observation_ids", []))
            self.assertIn("occlusion_overlap_candidate", follow_up_assessment.get("conflicting_evidence", []))
            self.assertIn("identity_ambiguity", follow_up_assessment.get("uncertainties", []))

            follow_up_assessment_provenance = follow_up_mcp_result.get(
                "observation_llm_follow_up_assessment_provenance",
                {},
            )
            self.assertFalse(follow_up_assessment_provenance.get("identity_verification_authorized", True))
            self.assertFalse(follow_up_assessment_provenance.get("identity_mutation_authorized", True))
            self.assertFalse(follow_up_mcp_result.get("mutation_authorized", True))

            self.assertIsNone(registry.get_identity_for_yolo_track("track-1"))
            identity_record = registry.get_record(app_identity_id)
            self.assertIsNotNone(identity_record)
            assert identity_record is not None
            self.assertEqual(identity_record.status, "temporarily_lost")


class TestDecisionIpcNonBlockingLoop(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_movement_does_not_block_on_slow_decision_cycle(self) -> None:
        tracking_calls = 0

        async def run_tracking_cycle(_session, _address) -> None:
            nonlocal tracking_calls
            tracking_calls += 1
            if tracking_calls >= 3:
                raise KeyboardInterrupt()

        async def run_agent_ops_pipeline_fn(*_args, **_kwargs) -> dict:
            return {
                "agent_ops_enabled": False,
                "agent_ops_summary_enabled": False,
                "depth_summary_enabled": False,
                "completed_cycle_increment": 0,
            }

        async def slow_decision_cycle() -> dict[str, int]:
            await asyncio.sleep(0.2)
            return {"published_count": 0}

        started_at = time.perf_counter()
        with self.assertRaises(KeyboardInterrupt):
            await runtime_flow.monitor_movement(
                session=SimpleNamespace(),
                tracking_address=("127.0.0.1", 5005),
                analysis_interval_seconds=0.0,
                enable_agent_ops_heartbeat=False,
                agent_ops_interval_seconds=999.0,
                agent_ops_max_tracks=2,
                agent_ops_summary_every_cycles=10,
                agent_ops_summary_window_size=10,
                handoff_degraded_warn_rate=0.1,
                handoff_degraded_critical_rate=0.3,
                depth_coverage_summary_enabled=False,
                depth_coverage_summary_every_cycles=10,
                depth_coverage_sample_limit=10,
                run_tracking_cycle_fn=run_tracking_cycle,
                run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
                emit=lambda _message: None,
                decision_ipc_fixture_cycle_fn=slow_decision_cycle,
            )
        elapsed = time.perf_counter() - started_at

        self.assertGreaterEqual(tracking_calls, 3)
        self.assertLess(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
