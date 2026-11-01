from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from agents.contracts import BoundingBox, GroundPoint, TrackingClaims, TrackingEvidenceEnvelope
from agents.observation.evidence_preflight import (
    build_measurement_provenance,
    preflight_tracking_evidence_for_observation,
)
from agents.tracking.evidence_store import TrackingEvidenceResolver


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestObservationEvidencePreflight(unittest.TestCase):
    def _envelope(self, *, frame_reference: str, source_refs: list[str], detection_id: str = "det-1") -> TrackingEvidenceEnvelope:
        return TrackingEvidenceEnvelope(
            schema_version="1.0",
            message_id="msg-1",
            track_id="track-1",
            timestamp=1000.0,
            tracking_claims=TrackingClaims(
                claimed_resolved_track_id="track-1",
                association_status="accepted",
                possible_id_switch=False,
                visibility_status="unknown",
                claim_reasons=["vlm_association_accepted"],
            ),
            frame_id="frame-1",
            detection_id=detection_id,
            pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0),
            relative_depth=0.3,
            ground_position=GroundPoint(x=1.0, z=2.0, unit="meters"),
            tracking_confidence=None,
            quality_flags=["position_kind:calibrated_ground_plane"],
            frame_reference=frame_reference,
            source_evidence_refs=source_refs,
            uncertainty=[],
        )

    def test_pending_reference_blocks_visual_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frame_id = "frame-pending"
            frame_reference = f"evidence://frame/{frame_id}"

            _write_json(
                refs / f"{frame_id}.json",
                {
                    "kind": "frame",
                    "id": frame_id,
                    "reference": frame_reference,
                    "created_at": 1000.0,
                    "expires_at": 2000.0,
                    "status": "pending",
                    "frame_path": str(root / "frames" / f"{frame_id}.jpg"),
                },
            )

            det_id = "detref-pending"
            det_ref = f"evidence://detection/{det_id}"
            _write_json(
                refs / f"{det_id}.json",
                {
                    "kind": "detection",
                    "id": det_id,
                    "reference": det_ref,
                    "created_at": 1000.0,
                    "expires_at": 2000.0,
                    "track_id": "track-1",
                    "detection_id": "det-1",
                    "frame_reference": frame_reference,
                },
            )

            resolver = TrackingEvidenceResolver(root)
            envelope = self._envelope(frame_reference=frame_reference, source_refs=[det_ref])

            result = preflight_tracking_evidence_for_observation(
                envelope,
                resolver=resolver,
                now_timestamp=1100.0,
            )

            self.assertFalse(result["preflight_passed"])
            self.assertFalse(result["visual_evidence_inspectable"])
            self.assertIn("pending_visual_evidence", result["gate_errors"])

    def test_expired_reference_never_substitutes_with_other_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frames = root / "frames"
            frames.mkdir(parents=True, exist_ok=True)

            old_frame_id = "frame-old"
            old_ref = f"evidence://frame/{old_frame_id}"
            (frames / f"{old_frame_id}.jpg").write_bytes(b"x")
            _write_json(
                refs / f"{old_frame_id}.json",
                {
                    "kind": "frame",
                    "id": old_frame_id,
                    "reference": old_ref,
                    "created_at": 900.0,
                    "expires_at": 950.0,
                    "status": "available",
                    "frame_path": str(frames / f"{old_frame_id}.jpg"),
                },
            )

            new_frame_id = "frame-new"
            new_ref = f"evidence://frame/{new_frame_id}"
            (frames / f"{new_frame_id}.jpg").write_bytes(b"x")
            _write_json(
                refs / f"{new_frame_id}.json",
                {
                    "kind": "frame",
                    "id": new_frame_id,
                    "reference": new_ref,
                    "created_at": 1000.0,
                    "expires_at": 2000.0,
                    "status": "available",
                    "frame_path": str(frames / f"{new_frame_id}.jpg"),
                },
            )

            det_id = "detref-old"
            det_ref = f"evidence://detection/{det_id}"
            _write_json(
                refs / f"{det_id}.json",
                {
                    "kind": "detection",
                    "id": det_id,
                    "reference": det_ref,
                    "created_at": 900.0,
                    "expires_at": 950.0,
                    "track_id": "track-1",
                    "detection_id": "det-1",
                    "frame_reference": old_ref,
                },
            )

            resolver = TrackingEvidenceResolver(root)
            envelope = self._envelope(frame_reference=old_ref, source_refs=[det_ref])

            result = preflight_tracking_evidence_for_observation(
                envelope,
                resolver=resolver,
                now_timestamp=1000.0,
            )

            self.assertFalse(result["preflight_passed"])
            self.assertIn("expired_visual_evidence", result["gate_errors"])
            self.assertEqual(
                result["evidence_reference_validation"]["no_substitution_policy"],
                "strict_reference_only",
            )

    def test_linkage_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "tracking_evidence"
            refs = root / "refs"
            frames = root / "frames"
            frames.mkdir(parents=True, exist_ok=True)

            frame_id = "frame-ok"
            frame_ref = f"evidence://frame/{frame_id}"
            (frames / f"{frame_id}.jpg").write_bytes(b"frame")
            _write_json(
                refs / f"{frame_id}.json",
                {
                    "kind": "frame",
                    "id": frame_id,
                    "reference": frame_ref,
                    "created_at": 1000.0,
                    "expires_at": 2000.0,
                    "status": "available",
                    "frame_path": str(frames / f"{frame_id}.jpg"),
                },
            )

            det_id = "detref-ok"
            det_ref = f"evidence://detection/{det_id}"
            _write_json(
                refs / f"{det_id}.json",
                {
                    "kind": "detection",
                    "id": det_id,
                    "reference": det_ref,
                    "created_at": 1000.0,
                    "expires_at": 2000.0,
                    "track_id": "track-other",
                    "detection_id": "det-1",
                    "frame_reference": frame_ref,
                },
            )

            resolver = TrackingEvidenceResolver(root)
            envelope = self._envelope(frame_reference=frame_ref, source_refs=[det_ref])
            result = preflight_tracking_evidence_for_observation(
                envelope,
                resolver=resolver,
                now_timestamp=1100.0,
            )

            self.assertFalse(result["preflight_passed"])
            self.assertIn("unresolvable_evidence_references", result["gate_errors"])

    def test_provenance_labels_calibrated_vs_relative(self) -> None:
        calibrated = TrackingEvidenceEnvelope(
            schema_version="1.0",
            message_id="msg-cal",
            track_id="track-1",
            timestamp=1000.0,
            tracking_claims=TrackingClaims(
                claimed_resolved_track_id="track-1",
                association_status="accepted",
                possible_id_switch=False,
                visibility_status="unknown",
                claim_reasons=["ok"],
            ),
            pixel_bbox=BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0),
            relative_depth=0.5,
            ground_position=GroundPoint(x=1.0, z=2.0, unit="meters"),
            quality_flags=["position_kind:calibrated_ground_plane"],
            source_evidence_refs=[],
        )

        relative = TrackingEvidenceEnvelope(
            schema_version="1.0",
            message_id="msg-rel",
            track_id="track-2",
            timestamp=1000.0,
            tracking_claims=TrackingClaims(
                claimed_resolved_track_id="track-2",
                association_status="accepted",
                possible_id_switch=False,
                visibility_status="unknown",
                claim_reasons=["ok"],
            ),
            pixel_bbox=BoundingBox(x1=1.0, y1=2.0, x2=3.0, y2=4.0),
            relative_depth=0.5,
            ground_position=GroundPoint(x=1.0, z=2.0, unit="relative_units"),
            quality_flags=["position_kind:relative_ground_estimate"],
            source_evidence_refs=[],
        )

        calibrated_p = build_measurement_provenance(calibrated)
        relative_p = build_measurement_provenance(relative)

        self.assertEqual(calibrated_p["ground_position"]["source"], "calculated")
        self.assertEqual(calibrated_p["ground_position"]["basis"], "homography_calibration")
        self.assertEqual(relative_p["ground_position"]["source"], "inferred")
        self.assertEqual(relative_p["ground_position"]["basis"], "relative_or_fallback_estimate")
        self.assertEqual(calibrated_p["relative_depth"]["source"], "inferred")


if __name__ == "__main__":
    unittest.main()
