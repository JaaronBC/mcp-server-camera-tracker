from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agents.observation.agent_observation import (
    ObservationVisualVerificationError,
    ObservationVisualVerificationRequest,
    verify_tracking_evidence_for_observation,
)


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


class TestObservationVisualVerification(unittest.TestCase):
    async def _async_return(self, payload: dict) -> dict:
        await asyncio.sleep(0)
        return payload

    def _build_valid_store(self, root: Path, *, message_id: str = "msg-1") -> tuple[Path, str, str]:
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
                "expires_at": 2000.0,
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
                "expires_at": 2000.0,
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

        return evidence_root, frame_reference, detection_reference

    def _payload(self, *, status: str, identity_consistency: str) -> dict:
        return {
            "verification_status": status,
            "visibility_status": "visible",
            "identity_consistency": identity_consistency,
            "possible_id_switch": identity_consistency == "inconsistent",
            "confidence": 0.82,
            "uncertainty": [],
            "limitations": [],
            "supporting_evidence_refs": ["evidence://frame/frame-abc"],
            "contradicting_evidence_refs": [],
            "recommended_follow_up": None,
            "visual_content_inspected": True,
        }

    def test_supported_visual_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-supported")
            result = verify_tracking_evidence_for_observation(
                ObservationVisualVerificationRequest(message_id="msg-supported", now_timestamp=1001.0),
                evidence_root_dir=evidence_root,
                invoke_fn=lambda **_kwargs: self._async_return(
                    self._payload(status="supported", identity_consistency="consistent")
                ),
                model="qwen-3.8-27b",
                retry_limit=0,
            )

        self.assertEqual(result.envelope.verification_status, "supported")
        self.assertEqual(result.envelope.identity_consistency, "consistent")
        self.assertTrue(result.validation["valid"])
        self.assertTrue(result.envelope.visual_content_inspected)

    def test_contradicted_visual_verification(self) -> None:
        payload = self._payload(status="contradicted", identity_consistency="inconsistent")
        payload["supporting_evidence_refs"] = []
        payload["contradicting_evidence_refs"] = ["evidence://frame/frame-abc"]

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-contradicted")
            result = verify_tracking_evidence_for_observation(
                ObservationVisualVerificationRequest(message_id="msg-contradicted", now_timestamp=1001.0),
                evidence_root_dir=evidence_root,
                invoke_fn=lambda **_kwargs: self._async_return(payload),
                model="qwen-3.8-27b",
                retry_limit=0,
            )

        self.assertEqual(result.envelope.verification_status, "contradicted")
        self.assertEqual(result.envelope.identity_consistency, "inconsistent")
        self.assertTrue(result.envelope.possible_id_switch)
        self.assertTrue(result.validation["valid"])

    def test_inconclusive_visual_verification(self) -> None:
        payload = self._payload(status="inconclusive", identity_consistency="unknown")
        payload["visibility_status"] = "occluded"
        payload["supporting_evidence_refs"] = []
        payload["contradicting_evidence_refs"] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-inconclusive")
            result = verify_tracking_evidence_for_observation(
                ObservationVisualVerificationRequest(message_id="msg-inconclusive", now_timestamp=1001.0),
                evidence_root_dir=evidence_root,
                invoke_fn=lambda **_kwargs: self._async_return(payload),
                model="qwen-3.8-27b",
                retry_limit=0,
            )

        self.assertEqual(result.envelope.verification_status, "inconclusive")
        self.assertEqual(result.envelope.visibility_status, "occluded")
        self.assertTrue(result.validation["valid"])

    def test_missing_or_expired_evidence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            with self.assertRaises(ObservationVisualVerificationError) as missing_error:
                verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="missing", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    invoke_fn=lambda **_kwargs: self._async_return(
                        self._payload(status="supported", identity_consistency="consistent")
                    ),
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

            evidence_root, frame_reference, _ = self._build_valid_store(Path(temp_dir), message_id="msg-expired")
            frame_id = frame_reference.split("/")[-1]
            _write_json(
                evidence_root / "refs" / f"{frame_id}.json",
                {
                    "kind": "frame",
                    "id": frame_id,
                    "reference": frame_reference,
                    "timestamp": 1000.0,
                    "created_at": 1000.0,
                    "expires_at": 1000.5,
                    "status": "available",
                    "frame_path": str(evidence_root / "frames" / f"{frame_id}.jpg"),
                },
            )

            with self.assertRaises(ObservationVisualVerificationError) as expired_error:
                verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-expired", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    invoke_fn=lambda **_kwargs: self._async_return(
                        self._payload(status="supported", identity_consistency="consistent")
                    ),
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

        self.assertEqual(missing_error.exception.code, "evidence_unavailable")
        self.assertEqual(expired_error.exception.code, "evidence_unavailable")

    def test_unsupported_vision_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-unsupported")
            with self.assertRaises(ObservationVisualVerificationError) as context:
                verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-unsupported", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    invoke_fn=lambda **_kwargs: self._async_return(
                        self._payload(status="supported", identity_consistency="consistent")
                    ),
                    model="gpt-oss-120b",
                    retry_limit=0,
                )

        self.assertEqual(context.exception.code, "unsupported_vision_model")

    def test_malformed_response_and_provider_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-bad")

            async def _bad_payload(**_kwargs):
                return {"verification_status": "supported"}

            with self.assertRaises(ObservationVisualVerificationError) as malformed_error:
                verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-bad", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    invoke_fn=_bad_payload,
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

            async def _raise_provider(**_kwargs):
                raise RuntimeError("provider down")

            with self.assertRaises(ObservationVisualVerificationError) as provider_error:
                verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-bad", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    invoke_fn=_raise_provider,
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

        self.assertEqual(malformed_error.exception.code, "malformed_verification_response")
        self.assertEqual(provider_error.exception.code, "provider_failure")

    def test_validation_rejection_is_raised(self) -> None:
        payload = self._payload(status="supported", identity_consistency="consistent")
        payload["supporting_evidence_refs"] = ["unknown://ref"]

        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-reject")
            with self.assertRaises(ObservationVisualVerificationError) as context:
                verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-reject", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    invoke_fn=lambda **_kwargs: self._async_return(payload),
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

        self.assertEqual(context.exception.code, "coordinator_validation_rejected")


if __name__ == "__main__":
    unittest.main()
