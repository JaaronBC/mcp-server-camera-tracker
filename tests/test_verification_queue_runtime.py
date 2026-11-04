from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import sys

if "mcp" not in sys.modules:
    mcp_stub = types.ModuleType("mcp")

    class _StubClientSession:
        pass

    mcp_stub.ClientSession = _StubClientSession
    sys.modules["mcp"] = mcp_stub

import agents.orchestration.runtime.verification_queue as verification_queue_runtime

from agents.orchestration.runtime.verification_queue import (
    VerificationQueueManager,
    VerificationQueueRequest,
    extract_submission_decisions_from_cycle_payload,
)
from agents.tracking.evidence_protection import EvidenceProtectionLock


class _FakeItem:
    def __init__(self, payload: dict):
        self.json = payload


class _FakeResult:
    def __init__(self, payload: dict):
        self.content = [_FakeItem(payload)]


class _FakeSession:
    def __init__(self, handler):
        self._handler = handler

    async def call_tool(self, tool_name, arguments):
        return await self._handler(tool_name, arguments)


def _envelope_payload(*, message_id: str, track_id: str, timestamp: float) -> dict:
    frame_id = "frame-1"
    frame_reference = "evidence://frame/frame-1"
    return {
        "schema_version": "1.0",
        "message_id": message_id,
        "track_id": track_id,
        "timestamp": timestamp,
        "tracking_claims": {
            "claimed_resolved_track_id": track_id,
            "association_status": "accepted",
            "possible_id_switch": False,
            "visibility_status": "visible",
            "claim_reasons": ["test"],
        },
        "frame_id": frame_id,
        "detection_id": "det-1",
        "frame_reference": frame_reference,
        "source_evidence_refs": [frame_reference],
        "uncertainty": [],
        "quality_flags": ["position_kind:calibrated_ground_plane"],
        "track_history": [],
    }


def _write_validated_envelope(root: Path, *, message_id: str, track_id: str, timestamp: float) -> None:
    frame_id = "frame-1"
    frame_reference = "evidence://frame/frame-1"

    frames = root / "frames"
    refs = root / "refs"
    frames.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)

    frame_path = frames / f"{frame_id}.jpg"
    frame_path.write_bytes(b"jpeg")
    (refs / f"{frame_id}.json").write_text(
        json.dumps(
            {
                "kind": "frame",
                "id": frame_id,
                "reference": frame_reference,
                "timestamp": float(timestamp),
                "created_at": float(timestamp),
                "expires_at": float(time.time() + 3600.0),
                "status": "available",
                "frame_path": str(frame_path),
                "width": 16,
                "height": 16,
            }
        ),
        encoding="utf-8",
    )

    envelopes = root / "envelopes"
    envelopes.mkdir(parents=True, exist_ok=True)
    payload = {
        "message_id": message_id,
        "stored_at": timestamp,
        "validation": {"valid": True, "errors": [], "warnings": []},
        "envelope": _envelope_payload(message_id=message_id, track_id=track_id, timestamp=timestamp),
    }
    (envelopes / f"{message_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _verification_envelope(*, message_id: str, track_id: str, timestamp: float) -> dict:
    return {
        "schema_version": "1.0",
        "observation_id": f"verify-{message_id}",
        "referenced_track_id": track_id,
        "referenced_message_id": message_id,
        "timestamp": timestamp,
        "source_evidence_refs": ["evidence://frame/frame-1"],
        "verification_status": "supported",
        "visibility_status": "visible",
        "identity_consistency": "consistent",
        "possible_id_switch": False,
        "supporting_evidence_refs": [],
        "contradicting_evidence_refs": [],
        "uncertainty": [],
        "limitations": [],
        "confidence": 0.8,
        "visual_content_inspected": True,
    }


class TestVerificationQueueRuntime(unittest.IsolatedAsyncioTestCase):
    def test_extract_submission_decisions_requires_production_origin(self) -> None:
        payload = {
            "events": [
                {
                    "event_origin": "startup_activation",
                    "task_id": "t-1",
                    "track_id": "sim-1",
                    "timestamp": 1000.0,
                    "evidence_requested": True,
                }
            ]
        }

        decisions = extract_submission_decisions_from_cycle_payload(payload)
        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision.event_origin, "startup_activation")
        self.assertFalse(decision.submission_eligibility)
        self.assertEqual(decision.skip_reason, "startup_activation_event")
        self.assertIsNone(decision.queue_request)

    def test_extract_submission_decisions_with_production_origin_queues(self) -> None:
        payload = {
            "events": [
                {
                    "event_origin": "production_monitor",
                    "event_id": "evt-1",
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "evidence_requested": True,
                    "evidence_message_id": "msg-1",
                }
            ]
        }

        decisions = extract_submission_decisions_from_cycle_payload(payload)
        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision.event_origin, "production_monitor")
        self.assertTrue(decision.submission_eligibility)
        self.assertIsNotNone(decision.queue_request)
        self.assertEqual(decision.queue_request.source, "production_monitor")
        self.assertEqual(decision.queue_request.message_id, "msg-1")

    async def test_submission_boundary_records_queued_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-1", track_id="track-1", timestamp=1000.0)

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, run_id="run-a")
            queued = manager.submit_request(
                VerificationQueueRequest(
                    request_id="req-boundary",
                    track_id="track-1",
                    message_id="msg-1",
                    source="test",
                    requested_at=1001.0,
                )
            )
            self.assertTrue(queued["accepted"])
            self.assertEqual(manager.get_recent_results(10), [])

            submissions = manager.get_recent_submissions(10)
            self.assertEqual(len(submissions), 1)
            self.assertEqual(submissions[0]["request_id"], "req-boundary")
            self.assertEqual(submissions[0]["status"], "queued")
            self.assertTrue(submissions[0]["accepted"])
            self.assertEqual(submissions[0]["run_id"], "run-a")

    async def test_submit_request_retries_brief_lease_lock_contention_within_default_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(
                evidence_root,
                message_id="msg-lock-retry",
                track_id="track-1",
                timestamp=1000.0,
            )

            contention_lock = EvidenceProtectionLock(evidence_root)
            self.assertTrue(contention_lock.acquire(timeout_seconds=0.0))

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)

            release_thread = threading.Thread(
                target=lambda: (time.sleep(0.03), contention_lock.release()),
                daemon=True,
            )
            release_thread.start()
            try:
                response = manager.submit_request(
                    VerificationQueueRequest(
                        request_id="req-lock-retry",
                        track_id="track-1",
                        message_id="msg-lock-retry",
                        source="test",
                        requested_at=1001.0,
                    )
                )
            finally:
                release_thread.join(timeout=0.5)
                contention_lock.release()

            self.assertTrue(response["accepted"])
            self.assertEqual(response["status"], "queued")

            submissions = manager.get_recent_submissions(10)
            self.assertEqual(len(submissions), 1)
            self.assertEqual(submissions[0]["request_id"], "req-lock-retry")
            self.assertTrue(submissions[0]["accepted"])

    async def test_submission_boundary_rejection_records_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            manager = VerificationQueueManager(evidence_root_dir=evidence_root)

            rejected = manager.submit_request(
                VerificationQueueRequest(
                    request_id="req-invalid",
                    track_id="",
                    message_id="msg-1",
                    source="test",
                    requested_at=1001.0,
                )
            )
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["status"], "invalid_track_id")

            submissions = manager.get_recent_submissions(10)
            self.assertEqual(len(submissions), 1)
            self.assertEqual(submissions[0]["request_id"], "req-invalid")
            self.assertEqual(submissions[0]["status"], "invalid_track_id")
            self.assertEqual(submissions[0]["reason"], "invalid_track_id")

    async def test_successful_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-1", track_id="track-1", timestamp=1000.0)

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id="msg-1",
                                track_id="track-1",
                                timestamp=1000.5,
                            ),
                        }
                    )
                raise RuntimeError("unexpected tool")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                accepted = manager.submit_request(
                    VerificationQueueRequest(
                        request_id="req-1",
                        track_id="track-1",
                        message_id="msg-1",
                        source="test",
                        requested_at=1001.0,
                    )
                )
                self.assertTrue(accepted["accepted"])
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                results = manager.get_recent_results(1)
                self.assertEqual(results[0]["status"], "completed")
                self.assertEqual(results[0]["validation_outcome"], "accepted")
                self.assertIn("uncertainty", results[0]["details"])
                self.assertIn("limitations", results[0]["details"])
                self.assertEqual(results[0]["details"]["uncertainty"], [])
                self.assertEqual(results[0]["details"]["limitations"], [])
                self.assertIn("possible_id_switch", results[0]["details"])
                self.assertIn("confidence", results[0]["details"])
                self.assertIn("visual_content_inspected", results[0]["details"])
                self.assertFalse(results[0]["details"]["possible_id_switch"])
                self.assertEqual(results[0]["details"]["confidence"], 0.8)
                self.assertTrue(results[0]["details"]["visual_content_inspected"])
            finally:
                await manager.stop()

    async def test_operator_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-timeout", track_id="track-1", timestamp=1000.0)

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    await asyncio.sleep(0.2)
                    return _FakeResult({"available": True})
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(
                evidence_root_dir=evidence_root,
                request_timeout_seconds=0.05,
            )
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(
                    VerificationQueueRequest("req-timeout", "track-1", "msg-timeout", "test", 1001.0)
                )
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                result = manager.get_recent_results(1)[0]
                self.assertEqual(result["status"], "timed_out")
                self.assertEqual(result["error_code"], "operator_timeout")
            finally:
                await manager.stop()

    async def test_queue_saturation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-a", track_id="track-1", timestamp=1000.0)
            _write_validated_envelope(evidence_root, message_id="msg-b", track_id="track-1", timestamp=1001.0)
            manager = VerificationQueueManager(evidence_root_dir=evidence_root, queue_capacity=1)

            first = manager.submit_request(VerificationQueueRequest("req-a", "track-1", "msg-a", "test", 0.0))
            second = manager.submit_request(VerificationQueueRequest("req-b", "track-1", "msg-b", "test", 0.0))
            self.assertTrue(first["accepted"])
            self.assertFalse(second["accepted"])
            self.assertEqual(second["status"], "queue_saturated")

    async def test_duplicate_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="dup-msg", track_id="track-1", timestamp=1000.0)
            manager = VerificationQueueManager(evidence_root_dir=evidence_root, queue_capacity=4)

            first = manager.submit_request(VerificationQueueRequest("req-1", "track-1", "dup-msg", "test", 0.0))
            duplicate = manager.submit_request(VerificationQueueRequest("req-2", "track-1", "dup-msg", "test", 0.0))
            self.assertTrue(first["accepted"])
            self.assertFalse(duplicate["accepted"])
            self.assertEqual(duplicate["status"], "duplicate_inflight")

    async def test_pending_precheck_retries_and_failed_frame_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-pending", track_id="track-1", timestamp=1000.0)
            _write_validated_envelope(evidence_root, message_id="msg-failed", track_id="track-1", timestamp=1000.1)
            _write_validated_envelope(evidence_root, message_id="msg-pending-timeout", track_id="track-1", timestamp=1000.2)

            precheck_calls: dict[str, int] = {"msg-pending": 0, "msg-failed": 0, "msg-pending-timeout": 0}

            async def handler(tool_name, arguments):
                message_id = str(arguments.get("message_id", ""))
                if tool_name == "get_tracking_visual_evidence":
                    if message_id in precheck_calls:
                        precheck_calls[message_id] += 1
                    if message_id == "msg-pending-timeout":
                        return _FakeResult(
                            {
                                "available": False,
                                "error_code": "evidence_unavailable",
                                "details": {
                                    "upstream_code": "observation_evidence_preflight_failed",
                                    "gate_errors": ["pending_visual_evidence"],
                                },
                            }
                        )
                    if message_id == "msg-pending" and precheck_calls[message_id] == 1:
                        return _FakeResult(
                            {
                                "available": False,
                                "error_code": "evidence_unavailable",
                                "details": {
                                    "upstream_code": "observation_evidence_preflight_failed",
                                    "gate_errors": ["pending_visual_evidence"],
                                },
                            }
                        )
                    if message_id == "msg-pending":
                        return _FakeResult({"available": True, "track_id": "track-1"})
                    return _FakeResult(
                        {
                            "available": False,
                            "error_code": "evidence_unavailable",
                            "details": {
                                "upstream_code": "frame_reference_unresolvable",
                                "gate_errors": ["missing_visual_evidence"],
                            },
                        }
                    )
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id=message_id,
                                track_id="track-1",
                                timestamp=1000.5,
                            ),
                        }
                    )
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, max_concurrency=1)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(
                    VerificationQueueRequest("req-pending", "track-1", "msg-pending", "test", 0.0)
                )
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)

                manager.submit_request(
                    VerificationQueueRequest("req-failed", "track-1", "msg-failed", "test", 0.0)
                )
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)

                manager.submit_request(
                    VerificationQueueRequest("req-pending-timeout", "track-1", "msg-pending-timeout", "test", 0.0)
                )
                await asyncio.wait_for(manager._queue.join(), timeout=2.0)

                results = {
                    item["request_id"]: item
                    for item in manager.get_recent_results(10)
                }
                pending_result = results["req-pending"]
                failed_result = results["req-failed"]
                pending_timeout_result = results["req-pending-timeout"]

                self.assertEqual(pending_result["status"], "completed")
                self.assertEqual(failed_result["status"], "rejected")
                self.assertEqual(pending_timeout_result["status"], "rejected")
                self.assertEqual(failed_result["validation_outcome"], "evidence_unavailable")
                self.assertEqual(failed_result["error_code"], "missing_or_expired_evidence")
                self.assertEqual(failed_result["details"].get("upstream_code"), "frame_reference_unresolvable")
                self.assertEqual(pending_timeout_result["validation_outcome"], "evidence_unavailable")
                self.assertEqual(pending_timeout_result["error_code"], "missing_or_expired_evidence")
                self.assertEqual(
                    pending_timeout_result["details"].get("upstream_code"),
                    "observation_evidence_preflight_failed",
                )
                self.assertIn("pending_visual_evidence", pending_timeout_result["details"].get("gate_errors", []))
                self.assertGreaterEqual(int(pending_timeout_result["details"].get("attempts", 0)), 2)

                self.assertEqual(precheck_calls["msg-pending"], 2)
                self.assertEqual(precheck_calls["msg-failed"], 1)
                self.assertGreaterEqual(precheck_calls["msg-pending-timeout"], 2)
            finally:
                await manager.stop()

    async def test_historical_result_marked_stale_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-old", track_id="track-1", timestamp=1000.0)
            _write_validated_envelope(evidence_root, message_id="msg-new", track_id="track-1", timestamp=1001.0)

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id="msg-old",
                                track_id="track-1",
                                timestamp=1001.5,
                            ),
                        }
                    )
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(VerificationQueueRequest("req-historical", "track-1", "msg-old", "test", 0.0))
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                result = manager.get_recent_results(1)[0]
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["stale_for_current_decisions"])
                self.assertEqual(result["current_applicability"], "historical")
            finally:
                await manager.stop()

    async def test_mismatched_response_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-1", track_id="track-1", timestamp=1000.0)

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id="msg-other",
                                track_id="track-1",
                                timestamp=1000.5,
                            ),
                        }
                    )
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(VerificationQueueRequest("req-mismatch", "track-1", "msg-1", "test", 0.0))
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                result = manager.get_recent_results(1)[0]
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["error_code"], "message_id_mismatch")
            finally:
                await manager.stop()

    async def test_provider_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-provider", track_id="track-1", timestamp=1000.0)

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult({"available": False, "error_code": "provider_failure", "error": "upstream"})
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(VerificationQueueRequest("req-provider", "track-1", "msg-provider", "test", 0.0))
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                result = manager.get_recent_results(1)[0]
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["validation_outcome"], "provider_failure")
            finally:
                await manager.stop()

    async def test_cancellation_and_clean_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-cancel", track_id="track-1", timestamp=1000.0)

            begin = asyncio.Event()

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    begin.set()
                    await asyncio.sleep(5)
                    return _FakeResult({"available": True, "track_id": "track-1"})
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, max_concurrency=1)
            await manager.start(session=_FakeSession(handler))
            manager.submit_request(VerificationQueueRequest("req-cancel", "track-1", "msg-cancel", "test", 0.0))
            await asyncio.wait_for(begin.wait(), timeout=1.0)
            await manager.stop()

            results = manager.get_recent_results(10)
            self.assertTrue(any(item["status"] in {"cancelled", "rejected", "failed"} for item in results))
            self.assertEqual(len(manager._workers), 0)
            self.assertEqual(len(manager._inflight_message_ids), 0)

    async def test_track_only_request_resolves_once_before_enqueue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-1", track_id="track-1", timestamp=1000.0)
            _write_validated_envelope(evidence_root, message_id="msg-2", track_id="track-1", timestamp=1001.0)

            seen_message_ids: list[str] = []

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    seen_message_ids.append(arguments["message_id"])
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    seen_message_ids.append(arguments["message_id"])
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id=arguments["message_id"],
                                track_id="track-1",
                                timestamp=1001.5,
                            ),
                        }
                    )
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            queued = manager.submit_request(
                VerificationQueueRequest(
                    request_id="req-track-only",
                    track_id="track-1",
                    message_id=None,
                    source="test",
                    requested_at=0.0,
                )
            )
            self.assertTrue(queued["accepted"])
            self.assertEqual(queued["message_id"], "msg-2")

            _write_validated_envelope(evidence_root, message_id="msg-3", track_id="track-1", timestamp=1002.0)

            await manager.start(session=_FakeSession(handler))
            try:
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                self.assertIn("msg-2", seen_message_ids)
                self.assertNotIn("msg-3", seen_message_ids)
            finally:
                await manager.stop()

    async def test_successful_verification_result_serializes_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-run", track_id="track-1", timestamp=1000.0)

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id="msg-run",
                                track_id="track-1",
                                timestamp=1000.5,
                            ),
                        }
                    )
                raise RuntimeError("unexpected tool")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, run_id="run-a")
            await manager.start(session=_FakeSession(handler))
            try:
                accepted = manager.submit_request(
                    VerificationQueueRequest(
                        request_id="req-run-id",
                        track_id="track-1",
                        message_id="msg-run",
                        source="test",
                        requested_at=1001.0,
                    )
                )
                self.assertTrue(accepted["accepted"])
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                results = manager.get_recent_results(1)
                self.assertEqual(results[0]["run_id"], "run-a")
            finally:
                await manager.stop()


    def _lease_path(self, evidence_root: Path, request_id: str) -> Path:
        return evidence_root / "protection" / "leases" / f"{request_id}.json"

    async def test_invalid_lease_payload_releases_acquired_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-invalid-payload", track_id="track-1", timestamp=1000.0)

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)

            original_acquire = verification_queue_runtime.acquire_verification_evidence_lease
            original_release = verification_queue_runtime.release_verification_evidence_lease
            release_calls: list[str] = []

            def _fake_acquire(*, evidence_root_dir, request_id, message_id, track_id, now_timestamp, lease_ttl_seconds, max_protected_frames, max_active_leases, lock_timeout_seconds):
                return {"acquired": True, "status": "acquired", "lease": "invalid"}

            def _fake_release(*, evidence_root_dir, request_id, lock_timeout_seconds):
                release_calls.append(str(request_id))
                return {"released": True, "status": "released"}

            verification_queue_runtime.acquire_verification_evidence_lease = _fake_acquire
            verification_queue_runtime.release_verification_evidence_lease = _fake_release
            try:
                response = manager.submit_request(
                    VerificationQueueRequest(
                        request_id="req-invalid-payload",
                        track_id="track-1",
                        message_id="msg-invalid-payload",
                        source="test",
                        requested_at=0.0,
                    )
                )
            finally:
                verification_queue_runtime.acquire_verification_evidence_lease = original_acquire
                verification_queue_runtime.release_verification_evidence_lease = original_release

            self.assertFalse(response["accepted"])
            self.assertEqual(response["status"], "lease_payload_invalid")
            self.assertEqual(release_calls, ["req-invalid-payload"])
            self.assertNotIn("req-invalid-payload", manager._leases_by_request_id)

    async def test_transient_release_lock_unavailable_retries_then_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            manager = VerificationQueueManager(evidence_root_dir=evidence_root)

            original_release = verification_queue_runtime.release_verification_evidence_lease
            release_calls: list[str] = []

            def _fake_release(*, evidence_root_dir, request_id, lock_timeout_seconds):
                release_calls.append(str(request_id))
                if len(release_calls) == 1:
                    return {"released": False, "status": "lease_lock_unavailable"}
                return {"released": True, "status": "released"}

            verification_queue_runtime.release_verification_evidence_lease = _fake_release
            try:
                manager._leases_by_request_id["req-retry-release"] = {"request_id": "req-retry-release"}

                first_released = manager._release_request_lease("req-retry-release")
                self.assertFalse(first_released)
                self.assertIn("req-retry-release", manager._leases_by_request_id)
                self.assertIn("req-retry-release", manager._pending_lease_release_request_ids)

                manager._retry_pending_lease_releases(max_passes=1)

                self.assertNotIn("req-retry-release", manager._leases_by_request_id)
                self.assertNotIn("req-retry-release", manager._pending_lease_release_request_ids)
                self.assertNotIn("req-retry-release", manager._lease_release_retry_attempts_by_request_id)
                self.assertEqual(release_calls, ["req-retry-release", "req-retry-release"])
            finally:
                verification_queue_runtime.release_verification_evidence_lease = original_release

    async def test_retry_pending_release_attempts_all_bounded_passes_until_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            manager = VerificationQueueManager(evidence_root_dir=evidence_root)

            original_release = verification_queue_runtime.release_verification_evidence_lease
            release_calls: list[str] = []

            def _fake_release(*, evidence_root_dir, request_id, lock_timeout_seconds):
                release_calls.append(str(request_id))
                if len(release_calls) <= 3:
                    return {"released": False, "status": "lease_lock_unavailable"}
                return {"released": True, "status": "released"}

            verification_queue_runtime.release_verification_evidence_lease = _fake_release
            try:
                manager._leases_by_request_id["req-retry-many"] = {"request_id": "req-retry-many"}
                manager._pending_lease_release_request_ids.add("req-retry-many")

                manager._retry_pending_lease_releases(max_passes=4)

                self.assertEqual(
                    release_calls,
                    ["req-retry-many", "req-retry-many", "req-retry-many", "req-retry-many"],
                )
                self.assertNotIn("req-retry-many", manager._pending_lease_release_request_ids)
                self.assertNotIn("req-retry-many", manager._leases_by_request_id)
                self.assertNotIn("req-retry-many", manager._lease_release_retry_attempts_by_request_id)
            finally:
                verification_queue_runtime.release_verification_evidence_lease = original_release

    async def test_enqueue_failure_rolls_back_acquired_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-rollback", track_id="track-1", timestamp=1000.0)

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)

            def _raise_queue_full(_request: VerificationQueueRequest) -> None:
                raise asyncio.QueueFull()

            manager._queue.put_nowait = _raise_queue_full  # type: ignore[method-assign]

            response = manager.submit_request(
                VerificationQueueRequest(
                    request_id="req-rollback",
                    track_id="track-1",
                    message_id="msg-rollback",
                    source="test",
                    requested_at=0.0,
                )
            )
            self.assertFalse(response["accepted"])
            self.assertEqual(response["status"], "queue_enqueue_failed")
            self.assertFalse(self._lease_path(evidence_root, "req-rollback").is_file())

    async def test_releases_lease_on_completion_and_shutdown_cancellation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-complete", track_id="track-1", timestamp=1000.0)
            _write_validated_envelope(evidence_root, message_id="msg-shutdown", track_id="track-1", timestamp=1001.0)

            block_precheck = asyncio.Event()
            allow_precheck = asyncio.Event()

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    message_id = str(arguments.get("message_id", ""))
                    if message_id == "msg-shutdown":
                        block_precheck.set()
                        await allow_precheck.wait()
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id=str(arguments.get("message_id", "")),
                                track_id="track-1",
                                timestamp=1002.0,
                            ),
                        }
                    )
                raise RuntimeError("unexpected tool")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, max_concurrency=1)
            await manager.start(session=_FakeSession(handler))
            try:
                accepted = manager.submit_request(
                    VerificationQueueRequest("req-complete", "track-1", "msg-complete", "test", 0.0)
                )
                self.assertTrue(accepted["accepted"])
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                self.assertFalse(self._lease_path(evidence_root, "req-complete").is_file())

                accepted_shutdown = manager.submit_request(
                    VerificationQueueRequest("req-shutdown", "track-1", "msg-shutdown", "test", 0.0)
                )
                self.assertTrue(accepted_shutdown["accepted"])
                await asyncio.wait_for(block_precheck.wait(), timeout=1.0)

                await manager.stop()
                self.assertFalse(self._lease_path(evidence_root, "req-shutdown").is_file())
            finally:
                allow_precheck.set()


    async def test_real_evidence_generation_drives_real_queue_precheck(self) -> None:
        import numpy as np
        import time

        from agents.contracts import BoundingBox, DetectionCandidate
        from agents.coordinator import AgentCoordinator
        from agents.observation.evidence_consumption import (
            ObservationEvidenceConsumptionError,
            ObservationEvidenceRequest,
            consume_tracking_evidence_by_message_id,
        )
        from agents.tracking.evidence_packaging import TrackingEvidencePackager
        from agents.tracking.evidence_store import TrackingEvidenceResolver, TrackingEvidenceStore

        def _detection(track_id: str, detection_id: str, timestamp: float) -> DetectionCandidate:
            return DetectionCandidate(
                detection_id=detection_id,
                timestamp=timestamp,
                yolo_track_id=track_id,
                label='person',
                detection_confidence=0.93,
                pixel_bbox=BoundingBox(x1=100.0, y1=120.0, x2=180.0, y2=260.0),
                pixel_center_x=140.0,
                pixel_center_y=190.0,
                relative_depth=0.4,
                ground_x=1.5,
                ground_z=5.0,
                ground_coordinate_unit='meters',
                position_kind='calibrated_ground_plane',
                motion_history=[],
            )

        def _wait_for_frame_available(resolver: TrackingEvidenceResolver, frame_reference: str, timeout_seconds: float = 1.5) -> bool:
            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                resolved = resolver.resolve_reference(frame_reference)
                if bool(resolved.get('valid', False)):
                    return True
                time.sleep(0.01)
            return False

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / 'tracking_evidence'

            coordinator = AgentCoordinator()
            store = TrackingEvidenceStore(evidence_root)
            resolver = TrackingEvidenceResolver(evidence_root)
            packager = TrackingEvidencePackager(coordinator=coordinator, evidence_store=store)

            track_id = 'track-precheck'
            timestamp = time.time()
            frame = np.zeros((24, 24, 3), dtype=np.uint8)
            frame_record = store.register_frame(
                frame=frame,
                timestamp=timestamp,
                width=24,
                height=24,
            )

            packaged = packager.package_detection(
                detection=_detection(track_id, 'det-precheck', timestamp),
                timestamp=timestamp,
                frame_evidence=frame_record,
                update_payload={
                    'detection_id': 'det-precheck',
                    'resolved_track_id': track_id,
                    'accepted': True,
                    'rejection_reasons': [],
                },
            )

            self.assertTrue(packaged.validation.get('valid', False))
            self.assertEqual(packaged.envelope.track_id, track_id)
            self.assertEqual(packaged.envelope.frame_reference, frame_record.frame_reference)
            self.assertTrue(any(ref.startswith('evidence://detection/') for ref in packaged.envelope.source_evidence_refs))
            self.assertTrue(_wait_for_frame_available(resolver, frame_record.frame_reference))

            verify_calls: list[str] = []

            async def handler(tool_name, arguments):
                if tool_name == 'get_tracking_visual_evidence':
                    message_id = str(arguments.get('message_id', ''))
                    try:
                        package = consume_tracking_evidence_by_message_id(
                            ObservationEvidenceRequest(message_id=message_id, require_visual_evidence=True),
                            evidence_root_dir=evidence_root,
                        )
                        payload = {
                            'available': True,
                            'message_id': package.message_id,
                            'track_id': package.envelope.track_id,
                        }
                    except ObservationEvidenceConsumptionError as error:
                        payload = {
                            'available': False,
                            'error_code': 'evidence_unavailable',
                            'details': {
                                'upstream_code': error.code,
                                'gate_errors': error.details.get('gate_errors') if isinstance(error.details, dict) else None,
                                'message_id': message_id,
                            },
                        }
                    return _FakeResult(payload)

                if tool_name == 'verify_tracking_visual_evidence':
                    message_id = str(arguments.get('message_id', ''))
                    verify_calls.append(message_id)
                    return _FakeResult(
                        {
                            'available': False,
                            'error_code': 'provider_failure',
                            'error': 'provider intentionally mocked in test',
                        }
                    )

                raise RuntimeError('unexpected tool')

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                accepted = manager.submit_request(
                    VerificationQueueRequest(
                        request_id='req-real-precheck',
                        track_id=track_id,
                        message_id=packaged.envelope.message_id,
                        source='test',
                        requested_at=timestamp + 0.1,
                    )
                )
                self.assertTrue(accepted['accepted'])

                await asyncio.wait_for(manager._queue.join(), timeout=2.0)
                result = manager.get_recent_results(1)[0]

                self.assertEqual(verify_calls, [packaged.envelope.message_id])
                self.assertEqual(result['request_id'], 'req-real-precheck')
                self.assertEqual(result['track_id'], track_id)
                self.assertEqual(result['evidence_message_id'], packaged.envelope.message_id)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['validation_outcome'], 'provider_failure')
                self.assertEqual(result['error_code'], 'provider_failure')
            finally:
                await manager.stop()
                resolver.shutdown()
                store.shutdown()
