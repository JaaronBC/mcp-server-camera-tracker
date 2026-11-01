from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.observation.evidence_consumption import (
    ObservationEvidenceConsumptionError,
    ObservationEvidenceRequest,
    consume_tracking_evidence_by_message_id,
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


class TestObservationEvidenceConsumption(unittest.TestCase):
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

    def test_successful_exact_message_retrieval_and_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-1")

            package = consume_tracking_evidence_by_message_id(
                ObservationEvidenceRequest(message_id="msg-1", now_timestamp=1001.0, max_history_records=1),
                evidence_root_dir=evidence_root,
            )

            self.assertEqual(package.message_id, "msg-1")
            self.assertEqual(package.envelope.message_id, "msg-1")
            self.assertTrue(package.preflight["preflight_passed"])
            self.assertIsNotNone(package.image_evidence)
            assert package.image_evidence is not None
            self.assertEqual(package.image_evidence.image_bytes, b"fake-jpeg-bytes")
            self.assertEqual(len(package.bounded_history), 1)
            self.assertEqual(package.bounded_history[0].observation_id, "obs-b")
            self.assertIn("ground_position", package.measurement_provenance)

    def test_missing_envelope_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            with self.assertRaises(ObservationEvidenceConsumptionError) as context:
                consume_tracking_evidence_by_message_id(
                    ObservationEvidenceRequest(message_id="missing"),
                    evidence_root_dir=evidence_root,
                )
            self.assertEqual(context.exception.code, "envelope_not_found")

    def test_invalid_stored_validation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, frame_reference, detection_reference = self._build_valid_store(Path(temp_dir), message_id="msg-invalid")
            _write_json(
                evidence_root / "envelopes" / "msg-invalid.json",
                {
                    "message_id": "msg-invalid",
                    "stored_at": 1001.0,
                    "validation": {"valid": False, "errors": ["forced"]},
                    "envelope": _valid_envelope_payload(
                        message_id="msg-invalid",
                        frame_reference=frame_reference,
                        detection_reference=detection_reference,
                    ),
                },
            )

            with self.assertRaises(ObservationEvidenceConsumptionError) as context:
                consume_tracking_evidence_by_message_id(
                    ObservationEvidenceRequest(message_id="msg-invalid"),
                    evidence_root_dir=evidence_root,
                )
            self.assertEqual(context.exception.code, "envelope_validation_failed")

    def test_expired_reference_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-expired")
            with self.assertRaises(ObservationEvidenceConsumptionError) as context:
                consume_tracking_evidence_by_message_id(
                    ObservationEvidenceRequest(message_id="msg-expired", now_timestamp=2500.0),
                    evidence_root_dir=evidence_root,
                )
            self.assertEqual(context.exception.code, "observation_evidence_preflight_failed")
            self.assertIn("expired_visual_evidence", context.exception.details.get("gate_errors", []))

    def test_pending_reference_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, frame_reference, detection_reference = self._build_valid_store(Path(temp_dir), message_id="msg-pending")
            frame_id = frame_reference.split("/")[-1]
            _write_json(
                evidence_root / "refs" / f"{frame_id}.json",
                {
                    "kind": "frame",
                    "id": frame_id,
                    "reference": frame_reference,
                    "timestamp": 1000.0,
                    "created_at": 1000.0,
                    "expires_at": 2000.0,
                    "status": "pending",
                    "frame_path": str(evidence_root / "frames" / f"{frame_id}.jpg"),
                },
            )

            with self.assertRaises(ObservationEvidenceConsumptionError) as context:
                consume_tracking_evidence_by_message_id(
                    ObservationEvidenceRequest(message_id="msg-pending", now_timestamp=1100.0),
                    evidence_root_dir=evidence_root,
                )
            self.assertEqual(context.exception.code, "observation_evidence_preflight_failed")
            self.assertIn("pending_visual_evidence", context.exception.details.get("gate_errors", []))

    def test_mismatched_detection_linkage_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, detection_reference = self._build_valid_store(Path(temp_dir), message_id="msg-mismatch")
            det_id = detection_reference.split("/")[-1]
            _write_json(
                evidence_root / "refs" / f"{det_id}.json",
                {
                    "kind": "detection",
                    "id": det_id,
                    "reference": detection_reference,
                    "timestamp": 1000.0,
                    "created_at": 1000.0,
                    "expires_at": 2000.0,
                    "track_id": "track-x",
                    "detection_id": "det-1",
                    "frame_reference": "evidence://frame/frame-abc",
                    "pixel_bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
                },
            )

            with self.assertRaises(ObservationEvidenceConsumptionError) as context:
                consume_tracking_evidence_by_message_id(
                    ObservationEvidenceRequest(message_id="msg-mismatch", now_timestamp=1100.0),
                    evidence_root_dir=evidence_root,
                )
            self.assertEqual(context.exception.code, "observation_evidence_preflight_failed")
            self.assertIn("unresolvable_evidence_references", context.exception.details.get("gate_errors", []))

    def test_cross_process_accessibility_from_disk_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root, _, _ = self._build_valid_store(Path(temp_dir), message_id="msg-cross")
            package = consume_tracking_evidence_by_message_id(
                ObservationEvidenceRequest(message_id="msg-cross", now_timestamp=1100.0),
                evidence_root_dir=evidence_root,
            )
            self.assertTrue(package.preflight["preflight_passed"])


if __name__ == "__main__":
    unittest.main()
