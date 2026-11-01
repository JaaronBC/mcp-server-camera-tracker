from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agents.orchestration.runtime.verification_queue import (
    VerificationQueueManager,
    VerificationQueueRequest,
)


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
        "frame_id": "frame-1",
        "detection_id": "det-1",
        "source_evidence_refs": ["evidence://frame/frame-1"],
        "uncertainty": [],
        "quality_flags": ["position_kind:calibrated_ground_plane"],
        "track_history": [],
    }


def _write_validated_envelope(root: Path, *, message_id: str, track_id: str, timestamp: float) -> None:
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

    async def test_expired_or_missing_evidence_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-expired", track_id="track-1", timestamp=1000.0)

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": False, "error_code": "expired_reference"})
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(
                    VerificationQueueRequest("req-expired", "track-1", "msg-expired", "test", 0.0)
                )
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                result = manager.get_recent_results(1)[0]
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["validation_outcome"], "evidence_unavailable")
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
