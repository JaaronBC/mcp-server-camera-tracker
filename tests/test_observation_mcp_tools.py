from __future__ import annotations

import json
import os
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
from agents.observation.tools import track_evidence_history as observation_track_evidence_history_tools
from agents.orchestration.tools import control as orchestration_control


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

    def test_verify_tracking_visual_evidence_timeout_override_from_env(self) -> None:
        fake_result = _FakeVerificationResult()
        with patch.dict(
            os.environ,
            {observation_evidence_tools.VISUAL_VERIFICATION_TIMEOUT_ENV: "30"},
            clear=False,
        ):
            with patch.object(
                observation_evidence_tools,
                "verify_tracking_evidence_for_observation",
                return_value=fake_result,
            ) as verify_mock:
                result = observation_evidence_tools.verify_tracking_visual_evidence("msg-1")

        self.assertTrue(result["available"])
        verify_mock.assert_called_once()
        call_kwargs = verify_mock.call_args.kwargs
        self.assertEqual(call_kwargs["timeout_seconds"], 30.0)

    def test_verify_tracking_visual_evidence_invalid_timeout_env_falls_back(self) -> None:
        fake_result = _FakeVerificationResult()
        with patch.dict(
            os.environ,
            {observation_evidence_tools.VISUAL_VERIFICATION_TIMEOUT_ENV: "not-a-number"},
            clear=False,
        ):
            with patch.object(
                observation_evidence_tools,
                "verify_tracking_evidence_for_observation",
                return_value=fake_result,
            ) as verify_mock:
                result = observation_evidence_tools.verify_tracking_visual_evidence("msg-1")

        self.assertTrue(result["available"])
        verify_mock.assert_called_once()
        call_kwargs = verify_mock.call_args.kwargs
        self.assertEqual(
            call_kwargs["timeout_seconds"],
            observation_evidence_tools.DEFAULT_VISUAL_VERIFICATION_TIMEOUT_SECONDS,
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

    def test_verify_tracking_visual_evidence_unsupported_model_cannot_succeed(self) -> None:
        with patch.object(
            observation_evidence_tools,
            "verify_tracking_evidence_for_observation",
            side_effect=ObservationVisualVerificationError(
                "unsupported_vision_model",
                "unsupported",
                details={"model": "gpt-oss-120b"},
            ),
        ):
            result = observation_evidence_tools.verify_tracking_visual_evidence("msg-1")

        self.assertFalse(result["available"])
        self.assertEqual(result["error_code"], "unsupported_vision_model")

    def test_evidence_remains_retrievable_and_verifiable_after_dashboard_stop(self) -> None:
        fake_result = _FakeVerificationResult()
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = self._build_valid_store(Path(temp_dir), message_id="msg-post-stop")
            with patch.object(observation_evidence_tools, "EVIDENCE_ROOT_DIR", evidence_root):
                with (
                    patch(
                        "agents.orchestration.tools.control.stop_dashboard_process",
                        return_value="Dashboard demo stopped.",
                    ),
                    patch(
                        "agents.orchestration.tools.control.dashboard_process_status",
                        return_value={
                            "running": False,
                            "pid": None,
                            "return_code": None,
                            "script_path": "agents/dashboard/live_tracking_display.py",
                        },
                    ),
                ):
                    stop_result = orchestration_control.stop_dashboard_and_confirm(
                        timeout_seconds=0.5,
                        poll_interval_seconds=0.01,
                    )

                self.assertTrue(stop_result["stopped"])

                get_result = observation_evidence_tools.get_tracking_visual_evidence("msg-post-stop")
                self.assertTrue(get_result["available"])

                with patch.object(
                    observation_evidence_tools,
                    "verify_tracking_evidence_for_observation",
                    return_value=fake_result,
                ):
                    verify_result = observation_evidence_tools.verify_tracking_visual_evidence("msg-post-stop")

                self.assertTrue(verify_result["available"])
                self.assertEqual(verify_result["envelope"]["referenced_message_id"], "msg-1")

    def test_get_track_evidence_history_valid_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = self._build_valid_store(Path(temp_dir), message_id="msg-a")
            envelopes_dir = evidence_root / "envelopes"
            refs_dir = evidence_root / "refs"

            frame_id = "frame-def"
            frame_reference = f"evidence://frame/{frame_id}"
            frame_path = evidence_root / "frames" / f"{frame_id}.jpg"
            frame_path.write_bytes(b"fake-jpeg-bytes-2")
            _write_json(
                refs_dir / f"{frame_id}.json",
                {
                    "kind": "frame",
                    "id": frame_id,
                    "reference": frame_reference,
                    "timestamp": 1002.0,
                    "created_at": 1002.0,
                    "expires_at": 9999999999.0,
                    "status": "available",
                    "frame_path": str(frame_path),
                    "width": 32,
                    "height": 24,
                },
            )
            det_ref_id = "detref-def"
            detection_reference = f"evidence://detection/{det_ref_id}"
            _write_json(
                refs_dir / f"{det_ref_id}.json",
                {
                    "kind": "detection",
                    "id": det_ref_id,
                    "reference": detection_reference,
                    "timestamp": 1002.0,
                    "created_at": 1002.0,
                    "expires_at": 9999999999.0,
                    "track_id": "track-1",
                    "detection_id": "det-2",
                    "frame_reference": frame_reference,
                    "pixel_bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                },
            )
            _write_json(
                envelopes_dir / "msg-b.json",
                {
                    "message_id": "msg-b",
                    "stored_at": 1002.0,
                    "validation": {
                        "valid": True,
                        "errors": [],
                        "warnings": [],
                        "message_id": "msg-b",
                        "track_id": "track-1",
                    },
                    "envelope": _valid_envelope_payload(
                        message_id="msg-b",
                        frame_reference=frame_reference,
                        detection_reference=detection_reference,
                    ),
                },
            )

            with patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", evidence_root):
                result = observation_track_evidence_history_tools.get_track_evidence_history(
                    "track-1",
                    max_messages=1,
                    max_history_records=1,
                    max_scan_records=10,
                )

        self.assertTrue(result["available"])
        self.assertEqual(result["state"], "recorded_observation")
        self.assertEqual(result["message_count"], 1)
        self.assertLessEqual(result["work"]["scanned_records"], 10)
        self.assertEqual(len(result["messages"][0]["bounded_history"]), 1)

    def test_get_track_evidence_history_invalid_and_missing_track(self) -> None:
        invalid = observation_track_evidence_history_tools.get_track_evidence_history("   ")
        self.assertFalse(invalid["available"])
        self.assertEqual(invalid["state"], "invalid_request")

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = self._build_valid_store(Path(temp_dir), message_id="msg-c")
            with patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", evidence_root):
                missing = observation_track_evidence_history_tools.get_track_evidence_history("track-does-not-exist")

        self.assertTrue(missing["available"])
        self.assertEqual(missing["state"], "no_recorded_observation")
        self.assertEqual(missing["message_count"], 0)

    def test_get_track_evidence_history_expired_and_missing_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = self._build_valid_store(Path(temp_dir), message_id="msg-expired")
            refs_dir = evidence_root / "refs"

            frame_ref_path = refs_dir / "frame-abc.json"
            frame_payload = json.loads(frame_ref_path.read_text(encoding="utf-8"))
            frame_payload["expires_at"] = 0.0
            _write_json(frame_ref_path, frame_payload)

            detection_ref_path = refs_dir / "detref-abc.json"
            detection_ref_path.unlink(missing_ok=True)

            with patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", evidence_root):
                result = observation_track_evidence_history_tools.get_track_evidence_history(
                    "track-1",
                    now_timestamp=10_000.0,
                )

        self.assertTrue(result["available"])
        self.assertIn(result["state"], {"expired_evidence", "missing_evidence", "expired_or_missing", "recorded_observation"})
        self.assertEqual(result["message_count"], 1)
        summary = result["messages"][0]["reference_statuses"]["summary"]
        self.assertGreaterEqual(summary["expired_count"] + summary["missing_count"], 1)

    def test_get_track_evidence_history_unavailable_storage_and_bounded_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "missing-root"
            with patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", evidence_root):
                unavailable = observation_track_evidence_history_tools.get_track_evidence_history("track-1")

        self.assertFalse(unavailable["available"])
        self.assertEqual(unavailable["state"], "storage_unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = self._build_valid_store(Path(temp_dir), message_id="msg-limit")
            envelopes_dir = evidence_root / "envelopes"
            for index in range(25):
                payload = {
                    "message_id": f"other-{index}",
                    "stored_at": 1100.0 + index,
                    "validation": {
                        "valid": True,
                        "errors": [],
                        "warnings": [],
                        "message_id": f"other-{index}",
                        "track_id": "track-other",
                    },
                    "envelope": _valid_envelope_payload(
                        message_id=f"other-{index}",
                        frame_reference="evidence://frame/frame-abc",
                        detection_reference="evidence://detection/detref-abc",
                    ),
                }
                payload["envelope"]["track_id"] = "track-other"
                _write_json(envelopes_dir / f"other-{index}.json", payload)

            with patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", evidence_root):
                limited = observation_track_evidence_history_tools.get_track_evidence_history(
                    "track-1",
                    max_scan_records=3,
                )

        self.assertTrue(limited["available"])
        self.assertLessEqual(limited["work"]["scanned_records"], 3)
        self.assertTrue(limited["work"]["scan_limit_reached"])

    def test_get_track_evidence_history_includes_prediction_detection_advisories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            evidence_root = self._build_valid_store(root_path, message_id="msg-adv")
            advisory_store_path = root_path / "latest_tracking_prediction_detection_advisories.json"

            envelope_path = evidence_root / "envelopes" / "msg-adv.json"
            envelope_payload = json.loads(envelope_path.read_text(encoding="utf-8"))
            tracking_claims = envelope_payload["envelope"].get("tracking_claims", {})
            tracking_claims["identity_claim"] = {"app_identity_id": "identity-1"}
            envelope_payload["envelope"]["tracking_claims"] = tracking_claims
            _write_json(envelope_path, envelope_payload)

            _write_json(
                advisory_store_path,
                {
                    "schema_version": "1.0",
                    "updated_at": 1001.0,
                    "available": True,
                    "summary": {"available": True, "matched_count": 1, "ambiguous_count": 1},
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
                            "predicted_for_timestamp": 1001.0,
                            "expires_at": 1005.0,
                            "advisory_created_at": 1001.0,
                            "source_identity_registry_generation": 3,
                            "source_identity_registry_version": 9,
                            "source_tracking_message_ids": ["msg-adv"],
                            "source_verification_observation_id": "obs-1",
                            "source_verification_message_id": "verify-1",
                        },
                        {
                            "decision": "ambiguous",
                            "classification": "prediction_detection_advisory",
                            "reason": "identity_has_competing_candidates",
                            "app_identity_id": "identity-1",
                            "candidate_yolo_track_id": "track-other",
                            "source_yolo_track_id": "track-src-2",
                            "match_metric": "pixel_distance",
                            "match_distance": 15.0,
                            "match_threshold": 120.0,
                            "normalized_score": 0.12,
                            "predicted_for_timestamp": 1001.0,
                            "expires_at": 1005.0,
                            "advisory_created_at": 1001.0,
                            "source_identity_registry_generation": 3,
                            "source_identity_registry_version": 9,
                            "source_tracking_message_ids": ["msg-adv"],
                            "source_verification_observation_id": "obs-2",
                            "source_verification_message_id": "verify-2",
                        },
                        {"invalid": True},
                    ],
                },
            )

            with (
                patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", evidence_root),
                patch.object(observation_track_evidence_history_tools, "ADVISORY_STORE_PATH", advisory_store_path),
            ):
                result = observation_track_evidence_history_tools.get_track_evidence_history(
                    "track-1",
                    now_timestamp=1001.1,
                )

        advisory = result["prediction_detection_advisory_evidence"]
        motion = result["observation_motion_consistency_assessment"]
        self.assertTrue(result["available"])
        self.assertFalse(result["mutation_authorized"])
        self.assertTrue(advisory["available"])
        self.assertEqual(advisory["advisory_count"], 2)
        self.assertEqual(advisory["matched_count"], 1)
        self.assertEqual(advisory["ambiguous_count"], 1)
        self.assertGreaterEqual(advisory["rejected_count"], 1)
        decisions = {item["decision"] for item in advisory["advisories"]}
        self.assertEqual(decisions, {"matched", "ambiguous"})
        self.assertFalse(advisory["provenance"]["identity_verification_authorized"])
        self.assertFalse(advisory["provenance"]["identity_mutation_authorized"])
        self.assertTrue(motion["available"])
        self.assertEqual(motion["motion_consistency"], "ambiguous")
        self.assertIn("advisory_ambiguity", motion["uncertainty"])
        self.assertFalse(motion["provenance"]["identity_verification_authorized"])
        self.assertFalse(motion["provenance"]["identity_mutation_authorized"])

    def test_get_track_evidence_history_rejects_stale_or_mismatched_advisories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            evidence_root = self._build_valid_store(root_path, message_id="msg-stale")
            advisory_store_path = root_path / "latest_tracking_prediction_detection_advisories.json"

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
                        },
                        {
                            "decision": "matched",
                            "classification": "prediction_detection_advisory",
                            "reason": None,
                            "app_identity_id": "identity-other",
                            "candidate_yolo_track_id": "track-other",
                            "source_yolo_track_id": "track-other-src",
                            "match_metric": "pixel_distance",
                            "match_distance": 9.0,
                            "match_threshold": 120.0,
                            "normalized_score": 0.07,
                            "predicted_for_timestamp": 1001.0,
                            "expires_at": 1005.0,
                            "advisory_created_at": 1001.0,
                        },
                    ],
                },
            )

            with (
                patch.object(observation_track_evidence_history_tools, "EVIDENCE_ROOT_DIR", evidence_root),
                patch.object(observation_track_evidence_history_tools, "ADVISORY_STORE_PATH", advisory_store_path),
            ):
                result = observation_track_evidence_history_tools.get_track_evidence_history(
                    "track-1",
                    now_timestamp=1002.0,
                )

        advisory = result["prediction_detection_advisory_evidence"]
        motion = result["observation_motion_consistency_assessment"]
        self.assertTrue(result["available"])
        self.assertFalse(advisory["available"])
        self.assertEqual(advisory["state"], "no_current_advisory")
        self.assertEqual(advisory["advisory_count"], 0)
        self.assertGreaterEqual(advisory["rejected_count"], 2)
        self.assertTrue(motion["available"])
        self.assertEqual(motion["motion_consistency"], "consistent")
        self.assertFalse(motion["provenance"]["identity_mutation_authorized"])


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
        self.assertIn("get_track_evidence_history", names)
        self.assertIn("get_identity_snapshot", names)

        self.assertIn("simulate_observation_summary", names)
        self.assertIn("simulate_reasoning_assessment", names)
        self.assertIn("simulate_visual_tracking_association", names)
        self.assertIn("run_reasoning_cycle_and_log", names)


if __name__ == "__main__":
    unittest.main()
