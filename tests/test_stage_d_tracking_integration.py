from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from agents.contracts import BoundingBox, DetectionCandidate
from agents.coordinator import AgentCoordinator
from agents.dashboard.state import SharedDashboardState
from agents.dashboard.tracking_loop import (
    _consume_tracking_observation_feedback,
    _generate_tracking_estimate_proposals,
)
from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.verification_feedback import VerificationFeedbackPublisher
from agents.orchestration.runtime.verification_queue import (
    VerificationQueueManager,
    VerificationQueueRequest,
)
from agents.tracking.evidence_packaging import TrackingEvidencePackager
from agents.tracking.evidence_store import TrackingEvidenceStore
from agents.tracking.estimate_proposals import TrackingEstimateProposalEngine
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer
from agents.tracking.tracking_types import GroundPosition, PixelPosition, RadarTrackSnapshot


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


def _detection(track_id: str, detection_id: str, timestamp: float) -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        timestamp=timestamp,
        yolo_track_id=track_id,
        label="person",
        detection_confidence=0.94,
        pixel_bbox=BoundingBox(x1=100.0, y1=80.0, x2=180.0, y2=220.0),
        pixel_center_x=140.0,
        pixel_center_y=150.0,
        relative_depth=0.42,
        ground_x=1.2,
        ground_z=4.8,
        ground_coordinate_unit="meters",
        position_kind="calibrated_ground_plane",
        motion_history=[],
    )


def _verification_envelope(
    *,
    message_id: str,
    track_id: str,
    timestamp: float,
    source_evidence_refs: list[str],
    identity_consistency: str = "consistent",
    possible_id_switch: bool = False,
    contradicting_refs: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "1.0",
        "observation_id": f"verify-{message_id}",
        "referenced_track_id": track_id,
        "referenced_message_id": message_id,
        "timestamp": timestamp,
        "source_evidence_refs": list(source_evidence_refs),
        "verification_status": "supported",
        "visibility_status": "occluded",
        "identity_consistency": identity_consistency,
        "possible_id_switch": possible_id_switch,
        "supporting_evidence_refs": list(source_evidence_refs),
        "contradicting_evidence_refs": [] if contradicting_refs is None else list(contradicting_refs),
        "uncertainty": ["brief_occlusion"],
        "limitations": ["single_view"],
        "confidence": 0.9,
        "visual_content_inspected": True,
    }


class TestStageDTrackingIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_complete_pipeline_happy_path_with_timing_and_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            feedback_artifact_path = root / "operator_tracking_feedback.json"

            coordinator = AgentCoordinator()
            evidence_store = TrackingEvidenceStore(evidence_root)
            packager = TrackingEvidencePackager(
                coordinator=coordinator,
                evidence_store=evidence_store,
                max_history_per_track=20,
            )

            track_id = "track-1"
            frame = np.zeros((24, 24, 3), dtype=np.uint8)
            frame_record = evidence_store.register_frame(
                frame=frame,
                timestamp=1000.0,
                width=24,
                height=24,
            )

            t0 = time.perf_counter()
            packaged_a = packager.package_detection(
                detection=_detection(track_id, "det-1", 1000.0),
                timestamp=1000.0,
                frame_evidence=frame_record,
                update_payload={
                    "detection_id": "det-1",
                    "resolved_track_id": track_id,
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            packaged_b = packager.package_detection(
                detection=_detection(track_id, "det-2", 1000.5),
                timestamp=1000.5,
                frame_evidence=frame_record,
                update_payload={
                    "detection_id": "det-2",
                    "resolved_track_id": track_id,
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            evidence_write_ms = (time.perf_counter() - t0) * 1000.0

            self.assertTrue(packaged_a.validation.get("valid", False))
            self.assertTrue(packaged_b.validation.get("valid", False))

            target_message_id = packaged_b.envelope.message_id
            verify_timestamp = 1000.7

            async def queue_handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": track_id})
                if tool_name == "verify_tracking_visual_evidence":
                    self.assertEqual(arguments.get("message_id"), target_message_id)
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id=target_message_id,
                                track_id=track_id,
                                timestamp=verify_timestamp,
                                source_evidence_refs=list(packaged_b.envelope.source_evidence_refs),
                            ),
                        }
                    )
                raise RuntimeError(f"unexpected tool: {tool_name}")

            manager = VerificationQueueManager(
                evidence_root_dir=evidence_root,
                request_timeout_seconds=2.0,
            )
            await manager.start(session=_FakeSession(queue_handler))
            try:
                accepted = manager.submit_request(
                    VerificationQueueRequest(
                        request_id="req-1",
                        track_id=track_id,
                        message_id=target_message_id,
                        source="stage_d_test",
                        requested_at=1000.8,
                    )
                )
                self.assertTrue(accepted.get("accepted", False))
                await asyncio.wait_for(manager._queue.join(), timeout=2.0)

                records = manager.get_recent_results(20)
                completed = next(item for item in records if item["status"] == "completed")
                self.assertEqual(completed["evidence_message_id"], target_message_id)
                self.assertTrue(completed["coordinator_valid"])

                publisher = VerificationFeedbackPublisher(
                    artifact_path=feedback_artifact_path,
                    max_records=64,
                    record_ttl_seconds=45.0,
                )
                publish_result = publisher.publish_from_results(records, now_timestamp=1001.0)
                self.assertGreaterEqual(publish_result["published"], 1)

                timing_consumer = TrackingObservationFeedbackConsumer(artifact_path=feedback_artifact_path)
                poll_start = time.perf_counter()
                feedback_summary = timing_consumer.poll(
                    current_track_ids={track_id},
                    latest_message_id_by_track={track_id: target_message_id},
                    now_timestamp=1001.1,
                )
                feedback_read_ms = (time.perf_counter() - poll_start) * 1000.0
                self.assertEqual(feedback_summary["accepted_count"], 1)

                consumer = TrackingObservationFeedbackConsumer(artifact_path=feedback_artifact_path)

                state = SharedDashboardState()
                state.tracking_feedback_consumer = consumer
                state.tracking_estimate_engine = TrackingEstimateProposalEngine(
                    coordinator=coordinator,
                    evidence_root_dir=evidence_root,
                    prediction_horizon_seconds=0.5,
                    proposal_ttl_seconds=1.0,
                    max_feedback_items=32,
                    max_history_points=12,
                    max_verification_age_seconds=5.0,
                    max_evidence_age_seconds=5.0,
                    max_effective_horizon_seconds=2.0,
                )
                state.latest_tracking_evidence_message_by_track[track_id] = target_message_id

                measured_track = RadarTrackSnapshot(
                    track_id=track_id,
                    timestamp=1001.1,
                    pixel_position=PixelPosition(x=140.0, y=150.0),
                    ground_position=GroundPosition(x=1.2, z=4.8),
                    ground_coordinate_unit="meters",
                    position_kind="calibrated_ground_plane",
                    quality_flags=["position_kind:calibrated_ground_plane"],
                    confidence=0.92,
                    relative_depth=0.42,
                )
                measured_before = measured_track.to_dict()
                radar_state_before = {
                    "active_tracks": {track_id: measured_before},
                    "last_observation_time": state.last_observation_time,
                }

                _consume_tracking_observation_feedback(
                    state=state,
                    now=1001.15,
                    tracks={track_id: measured_track},
                )
                first_feedback = state.latest_tracking_observation_feedback
                assert isinstance(first_feedback, dict)
                self.assertEqual(first_feedback.get("accepted_count"), 1)

                _consume_tracking_observation_feedback(
                    state=state,
                    now=1001.2,
                    tracks={},
                )
                self.assertEqual(
                    state.latest_tracking_lost_evidence_message_by_track.get(track_id),
                    target_message_id,
                )

                state.latest_tracking_observation_feedback = first_feedback

                estimate_start = time.perf_counter()
                _generate_tracking_estimate_proposals(
                    state=state,
                    now=1001.3,
                    tracks={},
                )
                estimate_validation_ms = (time.perf_counter() - estimate_start) * 1000.0

                proposal_summary = state.latest_tracking_estimate_proposals
                assert isinstance(proposal_summary, dict)
                self.assertEqual(proposal_summary.get("accepted_count"), 1)

                accepted_proposal = proposal_summary["accepted"][0]
                proposal_payload = accepted_proposal["proposal"]
                validation_payload = accepted_proposal["validation"]

                self.assertEqual(proposal_payload["track_id"], track_id)
                self.assertEqual(proposal_payload["source_tracking_message_ids"], [target_message_id])
                self.assertEqual(proposal_payload["source_verification_message_id"], target_message_id)
                self.assertEqual(proposal_payload["source_verification_observation_id"], f"verify-{target_message_id}")
                self.assertTrue(validation_payload["valid"])
                self.assertEqual(validation_payload["linked_tracking_message_ids"], [target_message_id])
                self.assertEqual(validation_payload["linked_verification_message_id"], target_message_id)

                self.assertEqual(measured_track.to_dict(), measured_before)
                self.assertEqual(radar_state_before["active_tracks"][track_id], measured_before)
                self.assertEqual(state.active_tracks, {})

                print(
                    "STAGE_D_TIMING_MS "
                    f"evidence_write={evidence_write_ms:.3f} "
                    f"feedback_read={feedback_read_ms:.3f} "
                    f"estimate_validation={estimate_validation_ms:.3f}"
                )
            finally:
                await manager.stop()
                evidence_store.shutdown()

    async def test_negative_path_integration_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"
            feedback_artifact_path = root / "operator_tracking_feedback.json"

            coordinator = AgentCoordinator()
            evidence_store = TrackingEvidenceStore(evidence_root)
            packager = TrackingEvidencePackager(coordinator=coordinator, evidence_store=evidence_store)

            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame_ref = evidence_store.register_frame(frame=frame, timestamp=2000.0, width=16, height=16)
            old_packaged = packager.package_detection(
                detection=_detection("track-matrix", "det-old", 2000.0),
                timestamp=2000.0,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-old",
                    "resolved_track_id": "track-matrix",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            new_packaged = packager.package_detection(
                detection=_detection("track-matrix", "det-new", 2001.0),
                timestamp=2001.0,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-new",
                    "resolved_track_id": "track-matrix",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            invalid_packaged = packager.package_detection(
                detection=_detection("track-matrix", "det-invalid", 2001.1),
                timestamp=2001.1,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-invalid",
                    "resolved_track_id": "track-matrix",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            self.assertTrue(old_packaged.validation.get("valid", False))
            self.assertTrue(new_packaged.validation.get("valid", False))
            self.assertTrue(invalid_packaged.validation.get("valid", False))

            async def handler(tool_name, arguments):
                message_id = str(arguments.get("message_id", ""))
                if tool_name == "get_tracking_visual_evidence":
                    if message_id == "expired-msg":
                        return _FakeResult({"available": False, "error_code": "expired_reference"})
                    return _FakeResult({"available": True, "track_id": "track-matrix"})
                if tool_name == "verify_tracking_visual_evidence":
                    if message_id == "timeout-msg":
                        await asyncio.sleep(0.2)
                        return _FakeResult({"available": True})
                    if message_id == "failure-msg":
                        return _FakeResult({"available": False, "error_code": "provider_failure", "error": "upstream"})
                    if message_id == invalid_packaged.envelope.message_id:
                        return _FakeResult(
                            {
                                "available": True,
                                "envelope": _verification_envelope(
                                    message_id=message_id,
                                    track_id="track-matrix",
                                    timestamp=2001.2,
                                    source_evidence_refs=["evidence://frame/not-in-evidence"],
                                ),
                            }
                        )
                    if message_id == "budget-msg":
                        return _FakeResult(
                            {
                                "available": False,
                                "error_code": "budget_exhausted",
                                "error": "provider budget exhausted",
                            }
                        )
                    if message_id in {old_packaged.envelope.message_id, new_packaged.envelope.message_id}:
                        return _FakeResult(
                            {
                                "available": True,
                                "envelope": _verification_envelope(
                                    message_id=message_id,
                                    track_id="track-matrix",
                                    timestamp=2001.2,
                                    source_evidence_refs=list(
                                        old_packaged.envelope.source_evidence_refs
                                        if message_id == old_packaged.envelope.message_id
                                        else new_packaged.envelope.source_evidence_refs
                                    ),
                                ),
                            }
                        )
                    raise RuntimeError("unexpected message id")
                raise RuntimeError(f"unexpected tool: {tool_name}")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, request_timeout_seconds=0.05)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(
                    VerificationQueueRequest("req-old", "track-matrix", old_packaged.envelope.message_id, "stage_d", 2001.3)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-new", "track-matrix", new_packaged.envelope.message_id, "stage_d", 2001.4)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-expired", "track-matrix", "expired-msg", "stage_d", 2001.5)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-timeout", "track-matrix", "timeout-msg", "stage_d", 2001.6)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-failure", "track-matrix", "failure-msg", "stage_d", 2001.7)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-invalid", "track-matrix", invalid_packaged.envelope.message_id, "stage_d", 2001.8)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-budget", "track-matrix", "budget-msg", "stage_d", 2001.9)
                )
                await asyncio.wait_for(manager._queue.join(), timeout=4.0)

                records = manager.get_recent_results(100)

                old_record = next(item for item in records if item["request_id"] == "req-old")
                new_record = next(item for item in records if item["request_id"] == "req-new")
                expired_record = next(item for item in records if item["request_id"] == "req-expired")
                timeout_record = next(item for item in records if item["request_id"] == "req-timeout")
                failure_record = next(item for item in records if item["request_id"] == "req-failure")
                invalid_record = next(item for item in records if item["request_id"] == "req-invalid")
                budget_record = next(item for item in records if item["request_id"] == "req-budget")

                self.assertEqual(old_record["status"], "completed")
                self.assertEqual(old_record["current_applicability"], "historical")
                self.assertTrue(old_record["stale_for_current_decisions"])
                self.assertEqual(new_record["status"], "completed")
                self.assertEqual(expired_record["validation_outcome"], "lease_acquire_failed")
                self.assertEqual(timeout_record["status"], "rejected")
                self.assertEqual(timeout_record["validation_outcome"], "lease_acquire_failed")
                self.assertEqual(failure_record["status"], "rejected")
                self.assertEqual(failure_record["validation_outcome"], "lease_acquire_failed")
                self.assertEqual(invalid_record["status"], "rejected")
                self.assertEqual(invalid_record["validation_outcome"], "coordinator_validation_failed")
                self.assertEqual(invalid_record["error_code"], "coordinator_validation_failed")
                self.assertEqual(budget_record["status"], "rejected")
                self.assertEqual(budget_record["validation_outcome"], "lease_acquire_failed")

                publisher = VerificationFeedbackPublisher(
                    artifact_path=feedback_artifact_path,
                    max_records=128,
                    record_ttl_seconds=1.0,
                )
                publish_stats = publisher.publish_from_results(records, now_timestamp=2002.0)
                self.assertGreaterEqual(publish_stats["published"], 2)

                consumer = TrackingObservationFeedbackConsumer(artifact_path=feedback_artifact_path)
                poll_first = consumer.poll(
                    current_track_ids={"track-matrix"},
                    latest_message_id_by_track={"track-matrix": new_packaged.envelope.message_id},
                    now_timestamp=2002.1,
                )
                self.assertGreaterEqual(poll_first["accepted_count"], 2)

                historical_feedback = [
                    item for item in poll_first["accepted"]
                    if item.get("current_applicability") == "historical"
                ]
                self.assertGreaterEqual(len(historical_feedback), 1)

                poll_second = consumer.poll(
                    current_track_ids={"track-matrix"},
                    latest_message_id_by_track={"track-matrix": new_packaged.envelope.message_id},
                    now_timestamp=2002.2,
                )
                self.assertEqual(poll_second["accepted_count"], 0)

                restarted_consumer = TrackingObservationFeedbackConsumer(artifact_path=feedback_artifact_path)
                poll_restart = restarted_consumer.poll(
                    current_track_ids={"track-matrix"},
                    latest_message_id_by_track={"track-matrix": new_packaged.envelope.message_id},
                    now_timestamp=2002.3,
                )
                self.assertGreaterEqual(poll_restart["accepted_count"], 2)

                feedback_payload = json.loads(feedback_artifact_path.read_text(encoding="utf-8"))
                feedback_artifact_path.write_text("{\"schema_version\":\"1.0\",\"records\":[", encoding="utf-8")
                malformed_poll = TrackingObservationFeedbackConsumer(artifact_path=feedback_artifact_path).poll(
                    current_track_ids={"track-matrix"},
                    latest_message_id_by_track={"track-matrix": new_packaged.envelope.message_id},
                    now_timestamp=2002.4,
                )
                self.assertEqual(malformed_poll["accepted_count"], 0)
                feedback_artifact_path.write_text(json.dumps(feedback_payload), encoding="utf-8")

                expired_poll = TrackingObservationFeedbackConsumer(artifact_path=feedback_artifact_path).poll(
                    current_track_ids={"track-matrix"},
                    latest_message_id_by_track={"track-matrix": new_packaged.envelope.message_id},
                    now_timestamp=2004.0,
                )
                self.assertEqual(expired_poll["accepted_count"], 0)

                state = SharedDashboardState()
                state.tracking_feedback_consumer = TrackingObservationFeedbackConsumer(artifact_path=feedback_artifact_path)
                state.tracking_missing_track_grace_seconds = 0.1
                state.latest_tracking_evidence_message_by_track = {
                    "track-matrix": new_packaged.envelope.message_id,
                }
                unrelated_measured_track = RadarTrackSnapshot(
                    track_id="track-unrelated",
                    timestamp=2002.45,
                    pixel_position=PixelPosition(x=15.0, y=18.0),
                    ground_position=GroundPosition(x=0.2, z=1.8),
                    ground_coordinate_unit="meters",
                    position_kind="calibrated_ground_plane",
                    quality_flags=["position_kind:calibrated_ground_plane"],
                    confidence=0.75,
                    relative_depth=0.25,
                )
                state.active_tracks = {"track-unrelated": unrelated_measured_track}
                unrelated_track_before = unrelated_measured_track.to_dict()
                radar_state_before = {
                    "active_tracks": {
                        "track-unrelated": unrelated_track_before,
                    },
                    "last_observation_time": state.last_observation_time,
                }

                _consume_tracking_observation_feedback(state=state, now=2002.5, tracks={})
                self.assertIn("track-matrix", state.latest_tracking_lost_evidence_message_by_track)

                _consume_tracking_observation_feedback(state=state, now=2002.55, tracks={"track-matrix": object()})
                self.assertNotIn("track-matrix", state.latest_tracking_lost_evidence_message_by_track)

                state.latest_tracking_evidence_message_by_track = {
                    "track-matrix": new_packaged.envelope.message_id,
                }
                _consume_tracking_observation_feedback(state=state, now=2002.6, tracks={})
                _consume_tracking_observation_feedback(state=state, now=2003.0, tracks={})
                self.assertNotIn("track-matrix", state.latest_tracking_lost_evidence_message_by_track)
                self.assertEqual(unrelated_measured_track.to_dict(), unrelated_track_before)
                self.assertEqual(
                    state.active_tracks["track-unrelated"].to_dict(),
                    radar_state_before["active_tracks"]["track-unrelated"],
                )
                self.assertEqual(state.last_observation_time, radar_state_before["last_observation_time"])

                estimator = TrackingEstimateProposalEngine(
                    coordinator=coordinator,
                    evidence_root_dir=evidence_root,
                    prediction_horizon_seconds=0.5,
                    max_effective_horizon_seconds=2.0,
                    max_evidence_age_seconds=10.0,
                )

                measured_track = RadarTrackSnapshot(
                    track_id="track-matrix",
                    timestamp=2002.5,
                    pixel_position=PixelPosition(x=140.0, y=150.0),
                    ground_position=GroundPosition(x=1.2, z=4.8),
                    ground_coordinate_unit="meters",
                    position_kind="calibrated_ground_plane",
                    quality_flags=["position_kind:calibrated_ground_plane"],
                    confidence=0.9,
                    relative_depth=0.4,
                )
                measured_before = measured_track.to_dict()

                base_feedback = {
                    "schema_version": "1.0",
                    "feedback_id": "obsfeedback-stage-d",
                    "verification_request_id": "req-stage-d",
                    "track_id": "track-matrix",
                    "evidence_message_id": new_packaged.envelope.message_id,
                    "verification_observation_id": f"verify-{new_packaged.envelope.message_id}",
                    "verification_message_id": new_packaged.envelope.message_id,
                    "verification_timestamp": 2001.2,
                    "source": "stage_d",
                    "current_applicability": "current",
                    "historical_observation": False,
                    "stale_for_current_decisions": False,
                    "coordinator_valid": True,
                    "validation_outcome": "accepted",
                    "submitted_at": 2001.0,
                    "started_at": 2001.1,
                    "finished_at": 2001.2,
                    "published_at": 2002.0,
                    "expires_at": 2100.0,
                    "provenance": {"publisher": "verification_queue", "source": "stage_d"},
                    "details": {
                        "verification_status": "supported",
                        "identity_consistency": "consistent",
                        "visibility_status": "occluded",
                        "possible_id_switch": False,
                        "confidence": 0.9,
                        "visual_content_inspected": True,
                        "source_evidence_refs": list(new_packaged.envelope.source_evidence_refs),
                        "supporting_evidence_refs": list(new_packaged.envelope.source_evidence_refs),
                        "contradicting_evidence_refs": [],
                        "uncertainty": ["brief_occlusion"],
                        "limitations": ["single_view"],
                    },
                }

                identity_bad = dict(base_feedback)
                identity_bad["details"] = dict(base_feedback["details"])
                identity_bad["details"]["identity_consistency"] = "inconsistent"

                id_switch_bad = dict(base_feedback)
                id_switch_bad["details"] = dict(base_feedback["details"])
                id_switch_bad["details"]["possible_id_switch"] = True

                contradiction_bad = dict(base_feedback)
                contradiction_bad["details"] = dict(base_feedback["details"])
                contradiction_bad["details"]["contradicting_evidence_refs"] = ["evidence://frame/frame-1"]

                missing_evidence = dict(base_feedback)
                missing_evidence["evidence_message_id"] = "missing-envelope"
                missing_evidence["verification_message_id"] = "missing-envelope"
                missing_evidence["verification_observation_id"] = "verify-missing-envelope"

                matrix_result = estimator.generate(
                    accepted_feedback_records=[
                        identity_bad,
                        id_switch_bad,
                        contradiction_bad,
                        missing_evidence,
                    ],
                    current_tracks={"track-matrix": measured_track},
                    now_timestamp=2002.8,
                )
                self.assertEqual(matrix_result["accepted_count"], 0)
                reasons = {item.get("reason") for item in matrix_result["rejected"]}
                self.assertIn("identity_not_consistent", reasons)
                self.assertIn("possible_id_switch", reasons)
                self.assertIn("contradicting_evidence_present", reasons)
                self.assertIn("trusted_evidence_unavailable", reasons)

                delayed_feedback = dict(base_feedback)
                delayed_feedback["feedback_id"] = "obsfeedback-stage-d-delayed"
                delayed_feedback["verification_request_id"] = "req-stage-d-delayed"
                delayed_feedback["verification_timestamp"] = 2002.0
                delayed_feedback["details"] = dict(base_feedback["details"])
                stale_verification_result = TrackingEstimateProposalEngine(
                    coordinator=coordinator,
                    evidence_root_dir=evidence_root,
                    prediction_horizon_seconds=0.5,
                    max_effective_horizon_seconds=30.0,
                    max_evidence_age_seconds=30.0,
                    max_verification_age_seconds=5.0,
                ).generate(
                    accepted_feedback_records=[delayed_feedback],
                    current_tracks={"track-matrix": measured_track},
                    now_timestamp=2015.0,
                )
                self.assertEqual(stale_verification_result["accepted_count"], 0)
                stale_reasons = {item.get("reason") for item in stale_verification_result["rejected"]}
                self.assertIn("coordinator_rejected", stale_reasons)
                stale_errors = [
                    err
                    for item in stale_verification_result["rejected"]
                    for err in item.get("errors", [])
                ]
                self.assertIn("stale_verification", stale_errors)
                self.assertEqual(measured_track.to_dict(), measured_before)
            finally:
                await manager.stop()
                evidence_store.shutdown()

    async def test_tracking_iterations_continue_while_verification_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            store = TrackingEvidenceStore(evidence_root)
            packager = TrackingEvidencePackager(
                coordinator=AgentCoordinator(),
                evidence_store=store,
                max_history_per_track=10,
            )

            frame = np.zeros((12, 12, 3), dtype=np.uint8)
            frame_ref = store.register_frame(frame=frame, timestamp=3000.0, width=12, height=12)
            packaged = packager.package_detection(
                detection=_detection("track-pending", "det-pending", 3000.0),
                timestamp=3000.0,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-pending",
                    "resolved_track_id": "track-pending",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            self.assertTrue(packaged.validation.get("valid", False))

            verify_started = asyncio.Event()
            allow_verify_finish = asyncio.Event()
            tracking_calls = 0
            feedback_publish_calls = 0
            verification_waiting = False
            tracking_calls_while_verification_pending = 0

            async def handler(tool_name, arguments):
                nonlocal verification_waiting
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-pending"})
                if tool_name == "verify_tracking_visual_evidence":
                    verify_started.set()
                    verification_waiting = True
                    await allow_verify_finish.wait()
                    verification_waiting = False
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id=str(arguments["message_id"]),
                                track_id="track-pending",
                                timestamp=3000.5,
                                source_evidence_refs=list(packaged.envelope.source_evidence_refs),
                            ),
                        }
                    )
                raise RuntimeError(f"unexpected tool {tool_name}")

            async def run_tracking_cycle(_session, _address):
                nonlocal tracking_calls, tracking_calls_while_verification_pending
                tracking_calls += 1
                if verification_waiting and not allow_verify_finish.is_set():
                    tracking_calls_while_verification_pending += 1
                if tracking_calls == 3:
                    allow_verify_finish.set()
                if tracking_calls >= 6:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            async def run_agent_ops_pipeline_fn(_session, **_kwargs):
                return {
                    "agent_ops_enabled": True,
                    "agent_ops_summary_enabled": False,
                    "depth_summary_enabled": False,
                    "completed_cycle_increment": 1,
                    "cycle_payload": {
                        "events": [
                            {
                                "track_id": "track-pending",
                                "evidence_requested": True,
                            }
                        ]
                    },
                }

            def publish_feedback_stub(_results):
                nonlocal feedback_publish_calls
                feedback_publish_calls += 1
                return {"published": 0, "malformed": 0, "skipped": 0, "retained": 0}

            manager = VerificationQueueManager(
                evidence_root_dir=evidence_root,
                request_timeout_seconds=2.0,
            )
            await manager.start(session=_FakeSession(handler))
            try:
                with self.assertRaises(KeyboardInterrupt):
                    await runtime_flow.monitor_movement(
                        _FakeSession(handler),
                        tracking_address=("127.0.0.1", 5005),
                        analysis_interval_seconds=0,
                        enable_agent_ops_heartbeat=True,
                        agent_ops_interval_seconds=0,
                        agent_ops_max_tracks=5,
                        agent_ops_summary_every_cycles=999,
                        agent_ops_summary_window_size=10,
                        handoff_degraded_warn_rate=0.1,
                        handoff_degraded_critical_rate=0.3,
                        depth_coverage_summary_enabled=False,
                        depth_coverage_summary_every_cycles=999,
                        depth_coverage_sample_limit=10,
                        run_tracking_cycle_fn=run_tracking_cycle,
                        run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
                        emit=lambda _msg: None,
                        verification_manager=manager,
                        verification_feedback_publish_fn=publish_feedback_stub,
                    )

                await asyncio.wait_for(manager._queue.join(), timeout=2.0)
                results = manager.get_recent_results(10)
                self.assertTrue(verify_started.is_set())
                self.assertGreaterEqual(tracking_calls_while_verification_pending, 1)
                self.assertGreaterEqual(tracking_calls, 5)
                self.assertGreaterEqual(feedback_publish_calls, 1)
                self.assertTrue(any(item.get("status") == "completed" for item in results))
            finally:
                await manager.stop()
                store.shutdown()


    async def test_provider_paths_with_valid_evidence_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = root / "tracking_evidence"

            coordinator = AgentCoordinator()
            evidence_store = TrackingEvidenceStore(evidence_root)
            packager = TrackingEvidencePackager(coordinator=coordinator, evidence_store=evidence_store)

            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame_ref = evidence_store.register_frame(frame=frame, timestamp=2100.0, width=16, height=16)

            success_packaged = packager.package_detection(
                detection=_detection("track-provider", "det-success", 2100.0),
                timestamp=2100.0,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-success",
                    "resolved_track_id": "track-provider",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            timeout_packaged = packager.package_detection(
                detection=_detection("track-provider", "det-timeout", 2100.1),
                timestamp=2100.1,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-timeout",
                    "resolved_track_id": "track-provider",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            failure_packaged = packager.package_detection(
                detection=_detection("track-provider", "det-failure", 2100.2),
                timestamp=2100.2,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-failure",
                    "resolved_track_id": "track-provider",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )
            budget_packaged = packager.package_detection(
                detection=_detection("track-provider", "det-budget", 2100.3),
                timestamp=2100.3,
                frame_evidence=frame_ref,
                update_payload={
                    "detection_id": "det-budget",
                    "resolved_track_id": "track-provider",
                    "accepted": True,
                    "rejection_reasons": [],
                },
            )

            self.assertTrue(success_packaged.validation.get("valid", False))
            self.assertTrue(timeout_packaged.validation.get("valid", False))
            self.assertTrue(failure_packaged.validation.get("valid", False))
            self.assertTrue(budget_packaged.validation.get("valid", False))

            async def handler(tool_name, arguments):
                message_id = str(arguments.get("message_id", ""))
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-provider"})

                if tool_name == "verify_tracking_visual_evidence":
                    if message_id == timeout_packaged.envelope.message_id:
                        await asyncio.sleep(0.2)
                        return _FakeResult({"available": True})
                    if message_id == failure_packaged.envelope.message_id:
                        return _FakeResult({"available": False, "error_code": "provider_failure", "error": "upstream"})
                    if message_id == budget_packaged.envelope.message_id:
                        return _FakeResult({"available": False, "error_code": "budget_exhausted", "error": "provider budget exhausted"})
                    if message_id == success_packaged.envelope.message_id:
                        return _FakeResult(
                            {
                                "available": True,
                                "envelope": _verification_envelope(
                                    message_id=message_id,
                                    track_id="track-provider",
                                    timestamp=2100.4,
                                    source_evidence_refs=list(success_packaged.envelope.source_evidence_refs),
                                ),
                            }
                        )
                    raise RuntimeError("unexpected message id")

                raise RuntimeError(f"unexpected tool: {tool_name}")

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, request_timeout_seconds=0.05)
            await manager.start(session=_FakeSession(handler))
            try:
                manager.submit_request(
                    VerificationQueueRequest("req-provider-success", "track-provider", success_packaged.envelope.message_id, "stage_d", 2100.5)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-provider-timeout", "track-provider", timeout_packaged.envelope.message_id, "stage_d", 2100.6)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-provider-failure", "track-provider", failure_packaged.envelope.message_id, "stage_d", 2100.7)
                )
                manager.submit_request(
                    VerificationQueueRequest("req-provider-budget", "track-provider", budget_packaged.envelope.message_id, "stage_d", 2100.8)
                )
                await asyncio.wait_for(manager._queue.join(), timeout=4.0)

                records = manager.get_recent_results(20)
                success_record = next(item for item in records if item["request_id"] == "req-provider-success")
                timeout_record = next(item for item in records if item["request_id"] == "req-provider-timeout")
                failure_record = next(item for item in records if item["request_id"] == "req-provider-failure")
                budget_record = next(item for item in records if item["request_id"] == "req-provider-budget")

                self.assertEqual(success_record["status"], "completed")
                self.assertEqual(timeout_record["status"], "timed_out")
                self.assertEqual(timeout_record["validation_outcome"], "operator_timeout")
                self.assertEqual(failure_record["status"], "failed")
                self.assertEqual(failure_record["validation_outcome"], "provider_failure")
                self.assertEqual(failure_record["error_code"], "provider_failure")
                self.assertEqual(budget_record["status"], "failed")
                self.assertEqual(budget_record["validation_outcome"], "provider_failure")
                self.assertEqual(budget_record["error_code"], "budget_exhausted")
            finally:
                await manager.stop()
                evidence_store.shutdown()


if __name__ == "__main__":
    unittest.main()
