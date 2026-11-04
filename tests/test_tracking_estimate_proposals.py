from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.contracts import GroundPoint, TrackingClaims, TrackingEvidenceEnvelope, TrackingObservationRecord
from agents.coordinator import AgentCoordinator
from agents.orchestration.runtime.verification_feedback import VerificationFeedbackPublisher
from agents.tracking.estimate_proposals import TrackingEstimateProposalEngine
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer
from agents.tracking.tracking_types import GroundPosition, PixelPosition, RadarTrackSnapshot


def _history_record(
    *,
    observation_id: str,
    track_id: str,
    timestamp: float,
    pixel_x: float,
    pixel_y: float,
    ground_x: float | None = None,
    ground_z: float | None = None,
    ground_unit: str | None = None,
    calibrated: bool = False,
) -> TrackingObservationRecord:
    ground_point = None
    if ground_x is not None and ground_z is not None and isinstance(ground_unit, str):
        ground_point = GroundPoint(x=ground_x, z=ground_z, unit=ground_unit)

    quality_flags = ["calibrated_ground_plane"] if calibrated else ["relative_ground_estimate"]
    return TrackingObservationRecord(
        observation_id=observation_id,
        track_id=track_id,
        timestamp=timestamp,
        source="tracking_frame",
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        ground_point=ground_point,
        confidence=0.9,
        quality_flags=quality_flags,
    )


def _envelope(
    *,
    message_id: str,
    track_id: str,
    timestamp: float,
    pixel_center: PixelPosition | None,
    ground_point: GroundPoint | None,
    position_kind: str,
    history: list[TrackingObservationRecord],
) -> TrackingEvidenceEnvelope:
    return TrackingEvidenceEnvelope(
        schema_version="1.0",
        message_id=message_id,
        track_id=track_id,
        timestamp=timestamp,
        tracking_claims=TrackingClaims(
            claimed_resolved_track_id=track_id,
            association_status="accepted",
            possible_id_switch=False,
            visibility_status="visible",
            claim_reasons=["test"],
        ),
        uncertainty=[],
        frame_id="frame-1",
        detection_id="det-1",
        pixel_center=pixel_center,
        ground_position=ground_point,
        quality_flags=[f"position_kind:{position_kind}"],
        track_history=history,
        frame_reference="evidence://frame/frame-1",
        source_evidence_refs=["evidence://frame/frame-1"],
    )


def _feedback_record(
    *,
    track_id: str,
    message_id: str,
    visibility_status: str = "occluded",
    identity_consistency: str = "consistent",
    verification_status: str = "supported",
    possible_id_switch: bool = False,
    visual_content_inspected: bool | None = True,
    contradicting_refs: list[str] | None = None,
    uncertainty: list[str] | None = None,
    verification_timestamp: float = 1000.2,
) -> dict:
    return {
        "schema_version": "1.0",
        "feedback_id": f"obsfeedback-{message_id}",
        "verification_request_id": f"req-{message_id}",
        "track_id": track_id,
        "evidence_message_id": message_id,
        "verification_observation_id": f"verify-{message_id}",
        "verification_message_id": message_id,
        "verification_timestamp": verification_timestamp,
        "source": "reasoning_cycle",
        "current_applicability": "current",
        "historical_observation": False,
        "stale_for_current_decisions": False,
        "coordinator_valid": True,
        "validation_outcome": "accepted",
        "submitted_at": 1000.0,
        "started_at": 1000.1,
        "finished_at": 1000.2,
        "published_at": 1000.3,
        "expires_at": 1010.0,
        "provenance": {"publisher": "verification_queue", "source": "reasoning_cycle"},
        "details": {
            "verification_status": verification_status,
            "identity_consistency": identity_consistency,
            "visibility_status": visibility_status,
            "possible_id_switch": possible_id_switch,
            "confidence": 0.9,
            "visual_content_inspected": visual_content_inspected,
            "source_evidence_refs": ["evidence://frame/frame-1"],
            "supporting_evidence_refs": ["evidence://frame/frame-1"],
            "contradicting_evidence_refs": ([] if contradicting_refs is None else contradicting_refs),
            "uncertainty": (["brief_occlusion"] if uncertainty is None else uncertainty),
            "limitations": ["single_view"],
        },
    }


def _write_envelope_record(root: Path, envelope: TrackingEvidenceEnvelope) -> None:
    envelopes = root / "envelopes"
    envelopes.mkdir(parents=True, exist_ok=True)
    payload = {
        "message_id": envelope.message_id,
        "stored_at": envelope.timestamp,
        "validation": {"valid": True, "errors": [], "warnings": []},
        "envelope": envelope.to_dict(),
    }
    (envelopes / f"{envelope.message_id}.json").write_text(json.dumps(payload), encoding="utf-8")


class TestTrackingEstimateProposals(unittest.TestCase):
    def test_short_occlusion_generates_valid_pixel_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-1",
                track_id="track-1",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=110.0, y=55.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-1", timestamp=999.5, pixel_x=100.0, pixel_y=50.0),
                    _history_record(observation_id="obs-2", track_id="track-1", timestamp=1000.0, pixel_x=110.0, pixel_y=55.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
            )
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-1", message_id="msg-1")],
                current_tracks={"track-1": object()},
                now_timestamp=1001.0,
            )

            self.assertEqual(result["accepted_count"], 1)
            proposal = result["accepted"][0]["proposal"]
            self.assertEqual(proposal["coordinate_system"], "relative_image_plane")
            self.assertEqual(proposal["ground_coordinate_mode"], "none")
            self.assertEqual(proposal["estimation_method"], "constant_velocity_extrapolation")

    def test_missing_detection_can_trigger_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-2",
                track_id="track-2",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=210.0, y=80.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-a", track_id="track-2", timestamp=999.5, pixel_x=200.0, pixel_y=70.0),
                    _history_record(observation_id="obs-b", track_id="track-2", timestamp=1000.0, pixel_x=210.0, pixel_y=80.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-2", message_id="msg-2")],
                current_tracks={},
                now_timestamp=1001.0,
            )
            self.assertEqual(result["accepted_count"], 1)

    def test_valid_calibrated_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-3",
                track_id="track-3",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=310.0, y=150.0),
                ground_point=GroundPoint(x=2.0, z=4.0, unit="meters"),
                position_kind="calibrated_ground_plane",
                history=[
                    _history_record(
                        observation_id="obs-c1",
                        track_id="track-3",
                        timestamp=999.5,
                        pixel_x=300.0,
                        pixel_y=140.0,
                        ground_x=1.0,
                        ground_z=3.0,
                        ground_unit="meters",
                        calibrated=True,
                    ),
                    _history_record(
                        observation_id="obs-c2",
                        track_id="track-3",
                        timestamp=1000.0,
                        pixel_x=310.0,
                        pixel_y=150.0,
                        ground_x=2.0,
                        ground_z=4.0,
                        ground_unit="meters",
                        calibrated=True,
                    ),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-3", message_id="msg-3")],
                current_tracks={"track-3": object()},
                now_timestamp=1001.0,
            )

            self.assertEqual(result["accepted_count"], 1)
            proposal = result["accepted"][0]["proposal"]
            self.assertEqual(proposal["coordinate_system"], "calibrated_ground_plane")
            self.assertEqual(proposal["ground_coordinate_mode"], "calibrated_metric")
            self.assertIsNotNone(proposal["estimated_ground_position"])

    def test_insufficient_history_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-4",
                track_id="track-4",
                timestamp=1000.0,
                pixel_center=None,
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-4", timestamp=999.5, pixel_x=100.0, pixel_y=50.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-4", message_id="msg-4")],
                current_tracks={"track-4": object()},
                now_timestamp=1001.0,
            )

            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "insufficient_or_invalid_history_or_horizon" for item in result["rejected"]))

    def test_stale_evidence_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-5",
                track_id="track-5",
                timestamp=900.0,
                pixel_center=PixelPosition(x=10.0, y=10.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-5", timestamp=899.5, pixel_x=8.0, pixel_y=9.0),
                    _history_record(observation_id="obs-2", track_id="track-5", timestamp=900.0, pixel_x=10.0, pixel_y=10.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-5", message_id="msg-5")],
                current_tracks={"track-5": object()},
                now_timestamp=1001.0,
            )
            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "stale_evidence" for item in result["rejected"]))

    def test_identity_conflict_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-6",
                track_id="track-6",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=10.0, y=10.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-6", timestamp=999.5, pixel_x=8.0, pixel_y=9.0),
                    _history_record(observation_id="obs-2", track_id="track-6", timestamp=1000.0, pixel_x=10.0, pixel_y=10.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(track_id="track-6", message_id="msg-6", identity_consistency="inconsistent")
                ],
                current_tracks={"track-6": object()},
                now_timestamp=1001.0,
            )
            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "identity_not_consistent" for item in result["rejected"]))

    def test_contradictory_findings_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-7",
                track_id="track-7",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=10.0, y=10.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-7", timestamp=999.5, pixel_x=8.0, pixel_y=9.0),
                    _history_record(observation_id="obs-2", track_id="track-7", timestamp=1000.0, pixel_x=10.0, pixel_y=10.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(track_id="track-7", message_id="msg-7", contradicting_refs=["evidence://frame/frame-1"])
                ],
                current_tracks={"track-7": object()},
                now_timestamp=1001.0,
            )
            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "contradicting_evidence_present" for item in result["rejected"]))

    def test_invalid_coordinates_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-8",
                track_id="track-8",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=10.0, y=10.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-8", timestamp=999.5, pixel_x=float("nan"), pixel_y=9.0),
                    _history_record(observation_id="obs-2", track_id="track-8", timestamp=1000.0, pixel_x=10.0, pixel_y=10.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-8", message_id="msg-8")],
                current_tracks={"track-8": object()},
                now_timestamp=1001.0,
            )
            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "insufficient_or_invalid_history_or_horizon" for item in result["rejected"]))

    def test_delayed_feedback_uses_correct_target_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-11",
                track_id="track-11",
                timestamp=1001.0,
                pixel_center=PixelPosition(x=10.0, y=5.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-11", timestamp=1000.0, pixel_x=0.0, pixel_y=0.0),
                    _history_record(observation_id="obs-2", track_id="track-11", timestamp=1001.0, pixel_x=10.0, pixel_y=5.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
                prediction_horizon_seconds=0.5,
                max_effective_horizon_seconds=2.0,
            )
            result = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-11",
                        message_id="msg-11",
                        verification_timestamp=1001.2,
                    )
                ],
                current_tracks={"track-11": object()},
                now_timestamp=1001.5,
            )

            self.assertEqual(result["accepted_count"], 1)
            proposal = result["accepted"][0]["proposal"]
            predicted = proposal["estimated_pixel_center"]
            self.assertIsNotNone(predicted)
            assert isinstance(predicted, dict)
            self.assertAlmostEqual(predicted["x"], 20.0, places=6)
            self.assertAlmostEqual(predicted["y"], 10.0, places=6)
            self.assertAlmostEqual(proposal["predicted_for_timestamp"], 1002.0, places=6)

    def test_excessively_delayed_feedback_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-12",
                track_id="track-12",
                timestamp=1001.0,
                pixel_center=PixelPosition(x=10.0, y=5.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-12", timestamp=1000.0, pixel_x=0.0, pixel_y=0.0),
                    _history_record(observation_id="obs-2", track_id="track-12", timestamp=1001.0, pixel_x=10.0, pixel_y=5.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
                prediction_horizon_seconds=0.5,
                max_effective_horizon_seconds=1.5,
                max_evidence_age_seconds=10.0,
            )
            result = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-12",
                        message_id="msg-12",
                        verification_timestamp=1001.2,
                    )
                ],
                current_tracks={"track-12": object()},
                now_timestamp=1005.0,
            )

            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "insufficient_or_invalid_history_or_horizon" for item in result["rejected"]))

    def test_invalid_measurement_timestamps_abstain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-13",
                track_id="track-13",
                timestamp=1001.0,
                pixel_center=PixelPosition(x=10.0, y=5.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-13", timestamp=1001.0, pixel_x=0.0, pixel_y=0.0),
                    _history_record(observation_id="obs-2", track_id="track-13", timestamp=1001.0, pixel_x=10.0, pixel_y=5.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
                prediction_horizon_seconds=0.5,
                max_effective_horizon_seconds=2.0,
            )
            result = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-13",
                        message_id="msg-13",
                        verification_timestamp=1001.05,
                    )
                ],
                current_tracks={"track-13": object()},
                now_timestamp=1001.1,
            )

            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "insufficient_or_invalid_history_or_horizon" for item in result["rejected"]))

    def test_repeated_polling_does_not_generate_duplicate_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path = root / "operator_tracking_feedback.json"
            publisher = VerificationFeedbackPublisher(artifact_path=artifact_path)
            consumer = TrackingObservationFeedbackConsumer(artifact_path=artifact_path)

            envelope = _envelope(
                message_id="msg-14",
                track_id="track-14",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=110.0, y=55.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-14", timestamp=999.5, pixel_x=100.0, pixel_y=50.0),
                    _history_record(observation_id="obs-2", track_id="track-14", timestamp=1000.0, pixel_x=110.0, pixel_y=55.0),
                ],
            )
            _write_envelope_record(root, envelope)

            publisher.publish_from_results(
                [
                    {
                        "request_id": "req-14",
                        "source": "reasoning_cycle",
                        "track_id": "track-14",
                        "evidence_message_id": "msg-14",
                        "status": "completed",
                        "validation_outcome": "accepted",
                        "submitted_at": 1000.0,
                        "started_at": 1000.1,
                        "finished_at": 1000.2,
                        "current_applicability": "current",
                        "stale_for_current_decisions": False,
                        "coordinator_valid": True,
                        "error_code": None,
                        "error": None,
                        "details": {
                            "verification_observation_id": "verify-msg-14",
                            "verification_referenced_message_id": "msg-14",
                            "verification_timestamp": 1000.2,
                            "verification_status": "supported",
                            "identity_consistency": "consistent",
                            "visibility_status": "occluded",
                            "possible_id_switch": False,
                            "confidence": 0.9,
                            "visual_content_inspected": True,
                            "source_evidence_refs": ["evidence://frame/frame-1"],
                            "supporting_evidence_refs": ["evidence://frame/frame-1"],
                            "contradicting_evidence_refs": [],
                            "uncertainty": ["brief_occlusion"],
                            "limitations": ["single_view"],
                        },
                    }
                ],
                now_timestamp=1001.0,
            )

            first_feedback = consumer.poll(
                current_track_ids={"track-14"},
                latest_message_id_by_track={"track-14": "msg-14"},
                now_timestamp=1001.1,
            )
            second_feedback = consumer.poll(
                current_track_ids={"track-14"},
                latest_message_id_by_track={"track-14": "msg-14"},
                now_timestamp=1001.2,
            )

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            first_result = engine.generate(
                accepted_feedback_records=first_feedback.get("accepted", []),
                current_tracks={"track-14": object()},
                now_timestamp=1001.1,
            )
            second_result = engine.generate(
                accepted_feedback_records=second_feedback.get("accepted", []),
                current_tracks={"track-14": object()},
                now_timestamp=1001.2,
            )

            self.assertEqual(first_result["accepted_count"], 1)
            self.assertEqual(second_result["accepted_count"], 0)

    def test_coordinator_rejection_is_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-9",
                track_id="track-9",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=110.0, y=55.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-9", timestamp=999.5, pixel_x=100.0, pixel_y=50.0),
                    _history_record(observation_id="obs-2", track_id="track-9", timestamp=1000.0, pixel_x=110.0, pixel_y=55.0),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
                prediction_horizon_seconds=4.0,
                max_effective_horizon_seconds=10.0,
            )
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-9", message_id="msg-9")],
                current_tracks={"track-9": object()},
                now_timestamp=1001.0,
            )

            self.assertEqual(result["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "coordinator_rejected" for item in result["rejected"]))

    def test_no_mutation_of_measured_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-10",
                track_id="track-10",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=110.0, y=55.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(observation_id="obs-1", track_id="track-10", timestamp=999.5, pixel_x=100.0, pixel_y=50.0),
                    _history_record(observation_id="obs-2", track_id="track-10", timestamp=1000.0, pixel_x=110.0, pixel_y=55.0),
                ],
            )
            _write_envelope_record(root, envelope)

            track_snapshot = RadarTrackSnapshot(
                track_id="track-10",
                timestamp=1001.0,
                pixel_position=PixelPosition(x=110.0, y=55.0),
                ground_position=GroundPosition(x=1.0, z=2.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=["position_kind:relative_ground_estimate"],
                confidence=0.8,
                relative_depth=0.4,
            )
            before = track_snapshot.to_dict()

            engine = TrackingEstimateProposalEngine(coordinator=AgentCoordinator(), evidence_root_dir=root)
            result = engine.generate(
                accepted_feedback_records=[_feedback_record(track_id="track-10", message_id="msg-10")],
                current_tracks={"track-10": track_snapshot},
                now_timestamp=1001.0,
            )

            self.assertEqual(result["accepted_count"], 1)
            self.assertEqual(track_snapshot.to_dict(), before)


    def test_verification_age_boundary_accepts_at_threshold_and_rejects_after(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-boundary-verification",
                track_id="track-boundary-verification",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=110.0, y=55.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(
                        observation_id="obs-1",
                        track_id="track-boundary-verification",
                        timestamp=999.5,
                        pixel_x=100.0,
                        pixel_y=50.0,
                    ),
                    _history_record(
                        observation_id="obs-2",
                        track_id="track-boundary-verification",
                        timestamp=1000.0,
                        pixel_x=110.0,
                        pixel_y=55.0,
                    ),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
                prediction_horizon_seconds=0.5,
                max_effective_horizon_seconds=10.0,
                max_verification_age_seconds=5.0,
                max_evidence_age_seconds=30.0,
            )

            at_boundary = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-boundary-verification",
                        message_id="msg-boundary-verification",
                        verification_timestamp=1000.0,
                    )
                ],
                current_tracks={"track-boundary-verification": object()},
                now_timestamp=1005.0,
            )
            self.assertEqual(at_boundary["accepted_count"], 1)

            over_boundary = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-boundary-verification",
                        message_id="msg-boundary-verification",
                        verification_timestamp=1000.0,
                    )
                ],
                current_tracks={"track-boundary-verification": object()},
                now_timestamp=1005.001,
            )
            self.assertEqual(over_boundary["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "coordinator_rejected" for item in over_boundary["rejected"]))
            stale_errors = [
                err
                for item in over_boundary["rejected"]
                for err in item.get("errors", [])
            ]
            self.assertIn("stale_verification", stale_errors)

    def test_evidence_age_boundary_accepts_at_threshold_and_rejects_after(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-boundary-evidence",
                track_id="track-boundary-evidence",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=110.0, y=55.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(
                        observation_id="obs-1",
                        track_id="track-boundary-evidence",
                        timestamp=999.5,
                        pixel_x=100.0,
                        pixel_y=50.0,
                    ),
                    _history_record(
                        observation_id="obs-2",
                        track_id="track-boundary-evidence",
                        timestamp=1000.0,
                        pixel_x=110.0,
                        pixel_y=55.0,
                    ),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
                prediction_horizon_seconds=0.5,
                max_effective_horizon_seconds=10.0,
                max_verification_age_seconds=30.0,
                max_evidence_age_seconds=5.0,
            )

            at_boundary = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-boundary-evidence",
                        message_id="msg-boundary-evidence",
                        verification_timestamp=1004.9,
                    )
                ],
                current_tracks={"track-boundary-evidence": object()},
                now_timestamp=1005.0,
            )
            self.assertEqual(at_boundary["accepted_count"], 1)

            over_boundary = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-boundary-evidence",
                        message_id="msg-boundary-evidence",
                        verification_timestamp=1004.9,
                    )
                ],
                current_tracks={"track-boundary-evidence": object()},
                now_timestamp=1005.001,
            )
            self.assertEqual(over_boundary["accepted_count"], 0)
            self.assertTrue(any(item.get("reason") == "stale_evidence" for item in over_boundary["rejected"]))

    def test_effective_horizon_boundary_accepts_at_threshold_and_rejects_after(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            envelope = _envelope(
                message_id="msg-boundary-horizon",
                track_id="track-boundary-horizon",
                timestamp=1000.0,
                pixel_center=PixelPosition(x=110.0, y=55.0),
                ground_point=None,
                position_kind="relative_ground_estimate",
                history=[
                    _history_record(
                        observation_id="obs-1",
                        track_id="track-boundary-horizon",
                        timestamp=999.5,
                        pixel_x=100.0,
                        pixel_y=50.0,
                    ),
                    _history_record(
                        observation_id="obs-2",
                        track_id="track-boundary-horizon",
                        timestamp=1000.0,
                        pixel_x=110.0,
                        pixel_y=55.0,
                    ),
                ],
            )
            _write_envelope_record(root, envelope)

            engine = TrackingEstimateProposalEngine(
                coordinator=AgentCoordinator(),
                evidence_root_dir=root,
                prediction_horizon_seconds=0.5,
                max_effective_horizon_seconds=2.0,
                max_verification_age_seconds=30.0,
                max_evidence_age_seconds=30.0,
            )

            at_boundary = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-boundary-horizon",
                        message_id="msg-boundary-horizon",
                        verification_timestamp=1000.0,
                    )
                ],
                current_tracks={"track-boundary-horizon": object()},
                now_timestamp=1001.5,
            )
            self.assertEqual(at_boundary["accepted_count"], 1)

            over_boundary = engine.generate(
                accepted_feedback_records=[
                    _feedback_record(
                        track_id="track-boundary-horizon",
                        message_id="msg-boundary-horizon",
                        verification_timestamp=1000.0,
                    )
                ],
                current_tracks={"track-boundary-horizon": object()},
                now_timestamp=1001.501,
            )
            self.assertEqual(over_boundary["accepted_count"], 0)
            self.assertTrue(
                any(
                    item.get("reason") == "insufficient_or_invalid_history_or_horizon"
                    for item in over_boundary["rejected"]
                )
            )


if __name__ == "__main__":
    unittest.main()
