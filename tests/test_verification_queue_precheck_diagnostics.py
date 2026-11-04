from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from mcp.types import CallToolResult, TextContent

from agents.orchestration.runtime.verification_queue import (
    VerificationQueueManager,
    VerificationQueueRequest,
)


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


class TestVerificationQueuePrecheckDiagnostics(unittest.IsolatedAsyncioTestCase):
    async def test_precheck_textcontent_json_dict_reaches_verify_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-textcontent", track_id="track-1", timestamp=1000.0)

            verify_called = False

            async def handler(tool_name, arguments):
                nonlocal verify_called
                if tool_name == "get_tracking_visual_evidence":
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=json.dumps(
                                    {
                                        "available": True,
                                        "message_id": "msg-textcontent",
                                        "track_id": "track-1",
                                    }
                                ),
                            )
                        ],
                        isError=False,
                    )
                if tool_name == "verify_tracking_visual_evidence":
                    verify_called = True

                    class _VerifyItem:
                        def __init__(self) -> None:
                            self.json = {
                                "available": False,
                                "error_code": "provider_failure",
                                "error": "provider intentionally mocked in test",
                            }

                    class _VerifyResult:
                        def __init__(self) -> None:
                            self.content = [_VerifyItem()]

                    return _VerifyResult()
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(VerificationQueueRequest("req-textcontent", "track-1", "msg-textcontent", "test", 0.0))
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                result = manager.get_recent_results(1)[0]

                self.assertTrue(verify_called)
                self.assertNotEqual(result["error_code"], "precheck_response_malformed")
                self.assertEqual(result["validation_outcome"], "provider_failure")
            finally:
                await manager.stop()

    async def test_precheck_malformed_response_is_diagnosed_distinctly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-malformed", track_id="track-1", timestamp=1000.0)

            class _MalformedItem:
                def __init__(self) -> None:
                    self.text = "not-json"

            class _MalformedResult:
                def __init__(self) -> None:
                    self.content = [_MalformedItem()]

            async def handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _MalformedResult()
                if tool_name == "verify_tracking_visual_evidence":
                    raise RuntimeError("verify should not be called for malformed precheck")
                raise RuntimeError("unexpected")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(VerificationQueueRequest("req-malformed", "track-1", "msg-malformed", "test", 0.0))
                await asyncio.wait_for(manager._queue.join(), timeout=1.0)
                result = manager.get_recent_results(1)[0]

                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["validation_outcome"], "evidence_unavailable")
                self.assertEqual(result["error_code"], "precheck_response_malformed")
                self.assertEqual(result["details"].get("precheck_failure_kind"), "precheck_response_malformed")
                self.assertIn("precheck_response_shape", result["details"])
                self.assertEqual(result["details"].get("precheck_content_item_count"), 1)
            finally:
                await manager.stop()


if __name__ == "__main__":
    unittest.main()
