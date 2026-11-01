from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.contracts import VisualVerificationEnvelope
from agents.observation.agent_observation import ObservationVisualVerificationError
from agents.observation.tools import evidence as observation_evidence_tools


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_envelope_payload(*, message_id: str, frame_reference: str, detection_reference: str) -> dict:
    return {
        "schema_version": "1.0",
        "message_id": message_id,
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
                "observation_id": "obs-a",
                "track_id": "track-1",
                "timestamp": 998.0,
                "source": "tracking_frame",
                "pixel_x": 19.0,
                "pixel_y": 39.0,
                "ground_point": {"x": 0.9, "z": 1.9, "unit": "meters"},
                "confidence": 0.7,
                "quality_flags": ["position_kind:calibrated_ground_plane"],
            },
            {
                "observation_id": "obs-b",
                "track_id": "track-1",
                "timestamp": 1000.0,
                "source": "tracking_frame",
                "pixel_x": 20.0,
                "pixel_y": 40.0,
                "ground_point": {"x": 1.0, "z": 2.0, "unit": "meters"},
                "confidence": 0.8,
                "quality_flags": ["position_kind:calibrated_ground_plane"],
            },
        ],
        "tracking_claims": {
            "claimed_resolved_track_id": "track-1",
            "association_status": "accepted",
            "possible_id_switch": False,
            "visibility_status": "unknown",
            "claim_reasons": ["vlm_association_accepted"],
        },
        "frame_reference": frame_reference,
        "source_evidence_refs": [frame_reference, detection_reference],
        "uncertainty": [],
    }


class _FakeVerificationResult:
    def __init__(self) -> None:
        self.envelope = VisualVerificationEnvelope(
            schema_version="1.0",
            observation_id="verify-1",
            referenced_track_id="track-1",
            referenced_message_id="msg-1",
            timestamp=1001.0,
            source_evidence_refs=["evidence://frame/frame-abc"],
            verification_status="supported",
            visibility_status="visible",
            identity_consistency="consistent",
            possible_id_switch=False,
            uncertainty=[],
            limitations=[],
            supporting_evidence_refs=["evidence://frame/frame-abc"],
            contradicting_evidence_refs=[],
            confidence=0.85,
            recommended_follow_up=None,
            visual_content_inspected=True,
        )
        self.validation = {"valid": True, "errors": [], "warnings": []}
        self.llm_meta = {
            "status": "ok",
            "model": "qwen-3.8-27b",
            "latency_ms": 25.0,
            "error": None,
            "error_code": None,
            "attempts": 1,
            "retry_count": 0,
        }
        self.preflight = {
            "preflight_passed": True,
            "visual_evidence_inspectable": True,
            "gate_errors": [],
        }
        self.measurement_provenance = {
            "ground_position": {"source": "calculated", "basis": "homography_calibration"}
        }


class TestObservationMcpEvidenceTools(unittest.TestCase):
    def _build_valid_store(self, root: Path, *, message_id: str = "msg-1") -> Path:
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
            envelopes_dir / f"{message_id}.json",
            {
                "message_id": message_id,
                "stored_at": 1001.0,
                "validation": {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "message_id": message_id,
                    "track_id": "track-1",
                },
                "envelope": _valid_envelope_payload(
                    message_id=message_id,
                    frame_reference=frame_reference,
                    detection_reference=detection_reference,
                ),
            },
        )

        return evidence_root

    def test_get_tracking_visual_evidence_success_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = self._build_valid_store(Path(temp_dir), message_id="msg-safe")
            with patch.object(observation_evidence_tools, "EVIDENCE_ROOT_DIR", evidence_root):
                result = observation_evidence_tools.get_tracking_visual_evidence("msg-safe")

        self.assertTrue(result["available"])
        self.assertEqual(result["message_id"], "msg-safe")
        self.assertIn("measurement_provenance", result)

        serialized = json.dumps(result)
        self.assertNotIn("image_bytes", serialized)
        self.assertNotIn("frame_path", serialized)
        self.assertNotIn("data:image/jpeg;base64", serialized)

    def test_get_tracking_visual_evidence_error_is_structured_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir) / "tracking_evidence"
            with patch.object(observation_evidence_tools, "EVIDENCE_ROOT_DIR", missing_root):
                result = observation_evidence_tools.get_tracking_visual_evidence("missing")

        self.assertFalse(result["available"])
        self.assertEqual(result["error_code"], "evidence_unavailable")
        self.assertIn("details", result)

        serialized = json.dumps(result)
        self.assertNotIn("envelope_path", serialized)
        self.assertNotIn("/Users/", serialized)

    def test_verify_tracking_visual_evidence_success_and_bounded_timeout(self) -> None:
        fake_result = _FakeVerificationResult()
        with patch.object(
            observation_evidence_tools,
            "verify_tracking_evidence_for_observation",
            return_value=fake_result,
        ) as verify_mock:
            result = observation_evidence_tools.verify_tracking_visual_evidence("msg-1")

        self.assertTrue(result["available"])
        self.assertEqual(result["envelope"]["referenced_message_id"], "msg-1")
        self.assertTrue(result["validation"]["valid"])
        verify_mock.assert_called_once()
        call_kwargs = verify_mock.call_args.kwargs
        self.assertEqual(
            call_kwargs["timeout_seconds"],
            observation_evidence_tools.DEFAULT_VISUAL_VERIFICATION_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            call_kwargs["retry_limit"],
            observation_evidence_tools.DEFAULT_VISUAL_VERIFICATION_RETRY_LIMIT,
        )

    def test_verify_tracking_visual_evidence_error_is_structured_and_sanitized(self) -> None:
        with patch.object(
            observation_evidence_tools,
            "verify_tracking_evidence_for_observation",
            side_effect=ObservationVisualVerificationError(
                "provider_failure",
                "provider failed",
                details={
                    "model": "qwen-3.8-27b",
                    "error": "RuntimeError:secret provider exception",
                    "upstream_details": {"envelope_path": "/tmp/secret.json"},
                    "gate_errors": ["expired_visual_evidence"],
                },
            ),
        ):
            result = observation_evidence_tools.verify_tracking_visual_evidence("msg-1")

        self.assertFalse(result["available"])
        self.assertEqual(result["error_code"], "provider_failure")
        serialized = json.dumps(result)
        self.assertNotIn("envelope_path", serialized)
        self.assertNotIn("secret provider exception", serialized)


class TestObservationMcpToolDiscoverability(unittest.IsolatedAsyncioTestCase):
    async def test_observation_tools_discoverable_via_list_tools(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        server_path = project_root / "server.py"

        params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_path)],
        )

        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tool_list = await session.list_tools()

        names = {tool.name for tool in tool_list.tools}

        self.assertIn("get_tracking_visual_evidence", names)
        self.assertIn("verify_tracking_visual_evidence", names)

        self.assertIn("simulate_observation_summary", names)
        self.assertIn("simulate_reasoning_assessment", names)
        self.assertIn("simulate_visual_tracking_association", names)
        self.assertIn("run_reasoning_cycle_and_log", names)


if __name__ == "__main__":
    unittest.main()
