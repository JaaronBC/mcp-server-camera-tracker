from __future__ import annotations

import asyncio
import json
import unittest

from agents.contracts import BoundingBox, DetectionCandidate
from tools import tracking_agent_phase1_validation as phase1


class TestTrackingAgentPhase1ObservationCollaboration(unittest.TestCase):
    def _make_detection(self) -> DetectionCandidate:
        return DetectionCandidate(
            detection_id="det-amb-1",
            timestamp=1000.0,
            yolo_track_id="track-B",
            label="person",
            detection_confidence=0.81,
            pixel_bbox=BoundingBox(x1=42.0, y1=44.0, x2=136.0, y2=162.0),
            pixel_center_x=89.0,
            pixel_center_y=162.0,
            relative_depth=0.52,
            ground_x=1.4,
            ground_z=3.8,
            ground_coordinate_unit="relative_units",
            position_kind="relative_ground_estimate",
            motion_history=[
                {"history_index": 0, "ground_x": 0.7, "ground_z": 4.5},
                {"history_index": 1, "ground_x": 0.95, "ground_z": 4.2},
                {"history_index": 2, "ground_x": 1.2, "ground_z": 4.0},
            ],
        )

    def _make_scenario_metadata(self) -> dict:
        return {
            "scenario": phase1.SCENARIO_YOLO_ID_SWAP,
            phase1.INDEPENDENT_TRACK_HISTORY_KEY: phase1._build_independent_track_history_evidence(1000.0),
        }

    def _make_detection_candidate(
        self,
        *,
        detection_id: str,
        yolo_track_id: str,
        relative_depth: float,
        ground_x: float,
        ground_z: float,
        confidence: float = 0.81,
    ) -> DetectionCandidate:
        return DetectionCandidate(
            detection_id=detection_id,
            timestamp=1000.0,
            yolo_track_id=yolo_track_id,
            label="person",
            detection_confidence=confidence,
            pixel_bbox=BoundingBox(x1=42.0, y1=44.0, x2=136.0, y2=162.0),
            pixel_center_x=89.0,
            pixel_center_y=162.0,
            relative_depth=relative_depth,
            ground_x=ground_x,
            ground_z=ground_z,
            ground_coordinate_unit="relative_units",
            position_kind="relative_ground_estimate",
            motion_history=[
                {"history_index": 0, "ground_x": ground_x - 0.28, "ground_z": ground_z + 0.45},
                {"history_index": 1, "ground_x": ground_x - 0.16, "ground_z": ground_z + 0.22},
                {"history_index": 2, "ground_x": ground_x, "ground_z": ground_z},
            ],
        )

    def _history_record(
        self,
        *,
        track_id: str,
        depth_median: float,
        last_ground_x: float,
        last_ground_z: float,
    ) -> dict:
        return {
            "track_id": track_id,
            "history_window_seconds": 1.8,
            "history_sample_count": 3,
            "recent_ground_path": [
                {"sample_age_s": 1.4, "ground_x": last_ground_x - 0.30, "ground_z": last_ground_z + 0.50},
                {"sample_age_s": 0.9, "ground_x": last_ground_x - 0.18, "ground_z": last_ground_z + 0.24},
                {"sample_age_s": 0.3, "ground_x": last_ground_x, "ground_z": last_ground_z},
            ],
            "recent_relative_depth": {"median": depth_median, "range": [depth_median - 0.02, depth_median + 0.02]},
            "evidence_timestamp": 999.8,
            "source": "independent_track_memory",
        }

    def _scenario_metadata_with_tracks(self, scenario_name: str, records: list[dict]) -> dict:
        return {
            "scenario": scenario_name,
            phase1.INDEPENDENT_TRACK_HISTORY_KEY: records,
        }

    def _supported_verdict(self) -> dict:
        return {
            "verification_status": "supported",
            "visibility_status": "visible",
            "identity_consistency": "consistent",
            "possible_id_switch": False,
            "confidence": 0.91,
            "uncertainty": [],
            "limitations": [],
            "supporting_evidence_refs": ["evidence://motion"],
            "contradicting_evidence_refs": [],
            "recommended_follow_up": None,
            "visual_content_inspected": True,
        }

    def test_build_observation_payload_includes_raw_measurements_history_and_proposal(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        tracking_proposal = {
            "detection_id": "det-amb-1",
            "proposed_track_id": "track-A",
            "association_confidence": 0.72,
            "depth_conflict_flag": True,
            "position_conflict_flag": False,
            "uncertainties": ["possible_id_switch"],
            "supporting_evidence": ["motion_continuity", "depth_consistency"],
        }

        payload = phase1._build_observation_verification_payload(
            detection=detection,
            tracking_proposal=tracking_proposal,
            scenario_metadata=scenario_metadata,
            event_reasons=["possible_id_switch"],
        )

        self.assertEqual(payload["tracking_proposal"]["proposed_track_id"], "track-A")
        self.assertEqual(payload["detection_measurements"]["detection_id"], "det-amb-1")
        self.assertEqual(payload["detection_measurements"]["relative_depth"], 0.52)
        self.assertEqual(
            payload["independent_history"]["for_proposed_track"]["track_id"],
            "track-A",
        )

    def test_deterministic_audit_flags_depth_contradiction_without_overwriting_verdict(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        tracking_proposal = {
            "detection_id": "det-amb-1",
            "proposed_track_id": "track-A",
            "association_confidence": 0.72,
            "depth_conflict_flag": True,
            "position_conflict_flag": False,
            "uncertainties": ["possible_id_switch"],
            "supporting_evidence": ["motion_continuity", "depth_consistency"],
        }

        audit = phase1._run_deterministic_observation_audit(
            detection=detection,
            tracking_proposal=tracking_proposal,
            scenario_metadata=scenario_metadata,
        )

        self.assertIn(
            "depth_conflict_flag_and_depth_consistency_claimed",
            audit["audit_flags"],
        )
        self.assertTrue(audit["measurement_checks"]["depth_conflict_flag"])
        self.assertTrue(audit["measurement_checks"]["depth_consistency_claimed"])

    def test_collaboration_diagnostic_keeps_observation_verdict_separate_from_audit(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.72,
                "occlusion_likelihood": 0.65,
                "id_switch_risk": 0.75,
                "depth_conflict_flag": True,
                "position_conflict_flag": False,
                "uncertainties": ["possible_id_switch", "crossing_targets"],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-B",
                "accepted": False,
                "rejection_reasons": ["depth_conflict_with_reliable_measurement"],
                "uncertainties": ["possible_id_switch"],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        ]

        async def _mock_live_like_invoker(**kwargs):
            self.assertIn("user_payload", kwargs)
            return {
                "verification_status": "contradicted",
                "visibility_status": "visible",
                "identity_consistency": "inconsistent",
                "possible_id_switch": True,
                "confidence": 0.79,
                "uncertainty": ["crossing_targets"],
                "limitations": ["single_frame_view"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": ["evidence://detection/demo"],
                "recommended_follow_up": "collect_additional_frames",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_mock_live_like_invoker,
            )
        )

        self.assertTrue(result["enabled"])
        self.assertEqual(result["mode"], phase1.OBSERVATION_VERIFICATION_MODE_MOCK)

        per_detection = result["per_detection"]["det-amb-1"]
        self.assertEqual(
            per_detection["observation_agent_verdict"]["verification_status"],
            "contradicted",
        )
        self.assertEqual(
            per_detection["observation_agent_invocation"]["status"],
            "ok",
        )
        self.assertIn(
            "depth_conflict_flag_and_depth_consistency_claimed",
            per_detection["deterministic_audit"]["audit_flags"],
        )
        self.assertFalse(per_detection["coordinator_decision"]["accepted"])

    def test_observation_invoke_default_optional_controls_are_none(self) -> None:
        captured: dict = {}

        async def _capturing_invoker(**kwargs):
            captured.update(kwargs)
            return {
                "verification_status": "supported",
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.9,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._invoke_observation_agent_verdict(
                mode=phase1.OBSERVATION_VERIFICATION_MODE_LIVE,
                observation_model="qwen-3.8-27b",
                observation_system_prompt=phase1.DEFAULT_OBSERVATION_VISUAL_SYSTEM_PROMPT,
                user_payload={"task": "demo"},
                frame_reference="data:image/jpeg;base64,abc",
                timeout_seconds=6.0,
                invoker=_capturing_invoker,
            )
        )

        self.assertEqual(result["invocation"]["status"], "ok")
        self.assertIn("max_output_tokens", captured)
        self.assertIsNone(captured["max_output_tokens"])
        self.assertIn("reasoning_effort", captured)
        self.assertIsNone(captured["reasoning_effort"])

    def test_observation_invoke_forwards_explicit_optional_controls(self) -> None:
        captured: dict = {}

        async def _capturing_invoker(**kwargs):
            captured.update(kwargs)
            return {
                "verification_status": "supported",
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.9,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._invoke_observation_agent_verdict(
                mode=phase1.OBSERVATION_VERIFICATION_MODE_LIVE,
                observation_model="qwen-3.8-27b",
                observation_system_prompt=phase1.DEFAULT_OBSERVATION_VISUAL_SYSTEM_PROMPT,
                user_payload={"task": "demo"},
                frame_reference="data:image/jpeg;base64,abc",
                timeout_seconds=6.0,
                max_output_tokens=768,
                reasoning_effort="none",
                invoker=_capturing_invoker,
            )
        )

        self.assertEqual(result["invocation"]["status"], "ok")
        self.assertEqual(captured.get("max_output_tokens"), 768)
        self.assertEqual(captured.get("reasoning_effort"), "none")

    def test_adjudication_supported_with_consistent_evidence_is_accepted(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.9,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.2,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        ]

        async def _supported_invoker(**_kwargs):
            return {
                "verification_status": "supported",
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.92,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": ["evidence://motion"],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_supported_invoker,
            )
        )

        adjudication = result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]
        self.assertEqual(adjudication["classification"], "accepted")

    def test_adjudication_contradicted_with_motion_mismatch_is_rejected(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.75,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.7,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["motion_trend_mismatch"],
                "supporting_evidence": ["depth_consistency"],
            }
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["depth_consistency"],
            }
        ]

        async def _contradicted_invoker(**_kwargs):
            return {
                "verification_status": "contradicted",
                "visibility_status": "visible",
                "identity_consistency": "inconsistent",
                "possible_id_switch": True,
                "confidence": 0.84,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": ["evidence://motion"],
                "recommended_follow_up": "review_association",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_contradicted_invoker,
            )
        )

        adjudication = result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]
        self.assertEqual(adjudication["classification"], "rejected")
        self.assertIn("deterministic_motion_inconsistent", adjudication["reason_codes"])

    def test_adjudication_inconclusive_with_motion_mismatch_is_unresolved(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.75,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.7,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["motion_trend_mismatch"],
                "supporting_evidence": ["depth_consistency"],
            }
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["depth_consistency"],
            }
        ]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.4,
                "uncertainty": [],
                "limitations": ["insufficient_visual_detail"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        adjudication = result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]
        self.assertEqual(adjudication["classification"], "unresolved")
        self.assertIn("observation_inconclusive", adjudication["reason_codes"])

    def test_adjudication_low_confidence_is_unresolved(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.5,
                "occlusion_likelihood": 0.25,
                "id_switch_risk": 0.3,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": ["low_association_confidence"],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["low_association_confidence"],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        ]

        async def _supported_invoker(**_kwargs):
            return {
                "verification_status": "supported",
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.9,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": ["evidence://motion"],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_supported_invoker,
            )
        )

        adjudication = result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]
        self.assertEqual(adjudication["classification"], "unresolved")
        self.assertIn("provisional_low_confidence", adjudication["reason_codes"])

    def test_adjudication_honors_existing_coordinator_rejection(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.9,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.2,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-B",
                "accepted": False,
                "rejection_reasons": ["depth_conflict_with_reliable_measurement"],
                "uncertainties": ["possible_id_switch"],
                "supporting_evidence": ["motion_continuity"],
            }
        ]

        async def _supported_invoker(**_kwargs):
            return {
                "verification_status": "supported",
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.9,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": ["evidence://motion"],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_supported_invoker,
            )
        )

        adjudication = result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]
        self.assertEqual(adjudication["classification"], "rejected")
        self.assertIn("coordinator_rejected", adjudication["reason_codes"])

    def test_adjudication_flags_conflicting_assignments_to_same_track(self) -> None:
        detections, _, scenario_metadata = phase1._build_detection_scenario(
            1000.0,
            phase1.SCENARIO_YOLO_ID_SWAP,
        )
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.9,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.2,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
            "det-amb-2": {
                "detection_id": "det-amb-2",
                "proposed_track_id": "track-A",
                "association_confidence": 0.8,
                "occlusion_likelihood": 0.3,
                "id_switch_risk": 0.35,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["depth_consistency"],
            },
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
            {
                "detection_id": "det-amb-2",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["depth_consistency"],
            },
        ]

        async def _supported_invoker(**_kwargs):
            return {
                "verification_status": "supported",
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.9,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": ["evidence://motion"],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_supported_invoker,
            )
        )

        summary = result["summary"]
        self.assertIn("track-A", summary["conflicting_proposed_track_ids"])
        self.assertEqual(
            result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]["classification"],
            "unresolved",
        )

    def test_adjudication_with_missing_observation_verdict_is_unresolved(self) -> None:
        detection = self._make_detection()
        scenario_metadata = self._make_scenario_metadata()
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.9,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.2,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            }
        ]

        async def _malformed_invoker(**_kwargs):
            return {
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.9,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_malformed_invoker,
            )
        )

        adjudication = result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]
        self.assertEqual(adjudication["classification"], "unresolved")
        self.assertIn("observation_verification_missing_or_malformed", adjudication["reason_codes"])

    def test_id_swap_scenario_classifies_non_justified_proposals_unresolved_or_rejected(self) -> None:
        detections, _, scenario_metadata = phase1._build_detection_scenario(
            1000.0,
            phase1.SCENARIO_YOLO_ID_SWAP,
        )
        proposals = {
            "det-amb-1": {
                "detection_id": "det-amb-1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.45,
                "occlusion_likelihood": 0.65,
                "id_switch_risk": 0.75,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets", "motion_trend_mismatch"],
                "supporting_evidence": ["depth_consistency", "appearance_match"],
            },
            "det-amb-2": {
                "detection_id": "det-amb-2",
                "proposed_track_id": "track-A",
                "association_confidence": 0.45,
                "occlusion_likelihood": 0.65,
                "id_switch_risk": 0.75,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets", "motion_trend_mismatch"],
                "supporting_evidence": ["depth_consistency", "appearance_match"],
            },
        }
        updates = [
            {
                "detection_id": "det-amb-1",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets", "motion_trend_mismatch"],
                "supporting_evidence": ["depth_consistency", "appearance_match"],
            },
            {
                "detection_id": "det-amb-2",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets", "motion_trend_mismatch"],
                "supporting_evidence": ["depth_consistency", "appearance_match"],
            },
        ]

        responses = {
            "det-amb-1": {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.4,
                "uncertainty": [],
                "limitations": ["insufficient_visual_detail"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            },
            "det-amb-2": {
                "verification_status": "contradicted",
                "visibility_status": "not_visible",
                "identity_consistency": "contradicted",
                "possible_id_switch": True,
                "confidence": 0.85,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": ["evidence://motion"],
                "recommended_follow_up": "review_association",
                "visual_content_inspected": True,
            },
        }

        async def _mixed_invoker(**kwargs):
            user_payload = kwargs["user_payload"]
            detection_payload = user_payload["detection_measurements"]
            detection_id = detection_payload["detection_id"]
            return responses[detection_id]

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_mixed_invoker,
            )
        )

        det1_classification = result["per_detection"]["det-amb-1"]["diagnostic_adjudication"]["classification"]
        det2_classification = result["per_detection"]["det-amb-2"]["diagnostic_adjudication"]["classification"]
        self.assertIn(det1_classification, {"unresolved", "rejected"})
        self.assertIn(det2_classification, {"unresolved", "rejected"})
        self.assertNotEqual(det1_classification, "accepted")
        self.assertNotEqual(det2_classification, "accepted")

    def test_adjudication_confidence_threshold_boundaries(self) -> None:
        detections = [
            self._make_detection_candidate(
                detection_id="det-low",
                yolo_track_id="track-A",
                relative_depth=0.50,
                ground_x=1.2,
                ground_z=4.0,
            ),
            self._make_detection_candidate(
                detection_id="det-edge",
                yolo_track_id="track-B",
                relative_depth=0.48,
                ground_x=2.0,
                ground_z=4.3,
            ),
            self._make_detection_candidate(
                detection_id="det-high",
                yolo_track_id="track-C",
                relative_depth=0.57,
                ground_x=3.1,
                ground_z=3.7,
            ),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [
                self._history_record(track_id="track-A", depth_median=0.50, last_ground_x=1.2, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.48, last_ground_x=2.0, last_ground_z=4.3),
                self._history_record(track_id="track-C", depth_median=0.57, last_ground_x=3.1, last_ground_z=3.7),
            ],
        )

        proposals = {
            "det-low": {
                "detection_id": "det-low",
                "proposed_track_id": "track-A",
                "association_confidence": 0.54,
                "occlusion_likelihood": 0.15,
                "id_switch_risk": 0.18,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": ["borderline_confidence"],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
            "det-edge": {
                "detection_id": "det-edge",
                "proposed_track_id": "track-B",
                "association_confidence": 0.55,
                "occlusion_likelihood": 0.12,
                "id_switch_risk": 0.16,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
            "det-high": {
                "detection_id": "det-high",
                "proposed_track_id": "track-C",
                "association_confidence": 0.56,
                "occlusion_likelihood": 0.1,
                "id_switch_risk": 0.14,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
        }
        updates = [
            {
                "detection_id": "det-low",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
            {
                "detection_id": "det-edge",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
            {
                "detection_id": "det-high",
                "resolved_track_id": "track-C",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
        ]

        async def _supported_invoker(**_kwargs):
            return self._supported_verdict()

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_supported_invoker,
            )
        )

        self.assertEqual(
            result["per_detection"]["det-low"]["diagnostic_adjudication"]["classification"],
            "unresolved",
        )
        self.assertIn(
            "provisional_low_confidence",
            result["per_detection"]["det-low"]["diagnostic_adjudication"]["reason_codes"],
        )
        self.assertEqual(
            result["per_detection"]["det-edge"]["diagnostic_adjudication"]["classification"],
            "accepted",
        )
        self.assertEqual(
            result["per_detection"]["det-high"]["diagnostic_adjudication"]["classification"],
            "accepted",
        )
        self.assertEqual(result["summary"]["diagnostic_adjudication_counts"]["accepted"], 2)
        self.assertEqual(result["summary"]["diagnostic_adjudication_counts"]["unresolved"], 1)

    def test_conflicting_assignments_cannot_both_be_accepted(self) -> None:
        detections = [
            self._make_detection_candidate(
                detection_id="det-1",
                yolo_track_id="track-A",
                relative_depth=0.51,
                ground_x=1.1,
                ground_z=3.9,
            ),
            self._make_detection_candidate(
                detection_id="det-2",
                yolo_track_id="track-B",
                relative_depth=0.52,
                ground_x=1.14,
                ground_z=3.95,
            ),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [self._history_record(track_id="track-A", depth_median=0.51, last_ground_x=1.1, last_ground_z=3.9)],
        )
        proposals = {
            "det-1": {
                "detection_id": "det-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.88,
                "occlusion_likelihood": 0.3,
                "id_switch_risk": 0.45,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
            "det-2": {
                "detection_id": "det-2",
                "proposed_track_id": "track-A",
                "association_confidence": 0.86,
                "occlusion_likelihood": 0.35,
                "id_switch_risk": 0.5,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
        }
        updates = [
            {
                "detection_id": "det-1",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
            {
                "detection_id": "det-2",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
        ]

        async def _supported_invoker(**_kwargs):
            return self._supported_verdict()

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_supported_invoker,
            )
        )

        self.assertIn("track-A", result["summary"]["conflicting_proposed_track_ids"])
        det1 = result["per_detection"]["det-1"]["diagnostic_adjudication"]
        det2 = result["per_detection"]["det-2"]["diagnostic_adjudication"]
        self.assertNotEqual(det1["classification"], "accepted")
        self.assertNotEqual(det2["classification"], "accepted")
        self.assertIn("conflicting_assignment_to_same_track", det1["reason_codes"])
        self.assertIn("conflicting_assignment_to_same_track", det2["reason_codes"])
        self.assertEqual(result["summary"]["diagnostic_adjudication_counts"]["accepted"], 0)
        self.assertTrue(result["per_detection"]["det-1"]["coordinator_decision"]["accepted"])
        self.assertTrue(result["per_detection"]["det-2"]["coordinator_decision"]["accepted"])

    def test_three_target_replay_independent_classification_and_evidence_retention(self) -> None:
        detections = [
            self._make_detection_candidate(
                detection_id="det-3a",
                yolo_track_id="track-A",
                relative_depth=0.50,
                ground_x=1.0,
                ground_z=4.2,
            ),
            self._make_detection_candidate(
                detection_id="det-3b",
                yolo_track_id="track-C",
                relative_depth=0.61,
                ground_x=2.8,
                ground_z=3.5,
            ),
            self._make_detection_candidate(
                detection_id="det-3c",
                yolo_track_id="track-B",
                relative_depth=0.44,
                ground_x=2.0,
                ground_z=4.5,
            ),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.50, last_ground_x=1.0, last_ground_z=4.2),
                self._history_record(track_id="track-B", depth_median=0.44, last_ground_x=2.0, last_ground_z=4.5),
                self._history_record(track_id="track-C", depth_median=0.52, last_ground_x=1.7, last_ground_z=4.7),
            ],
        )
        proposals = {
            "det-3a": {
                "detection_id": "det-3a",
                "proposed_track_id": "track-A",
                "association_confidence": 0.88,
                "occlusion_likelihood": 0.1,
                "id_switch_risk": 0.1,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
            "det-3b": {
                "detection_id": "det-3b",
                "proposed_track_id": "track-B",
                "association_confidence": 0.79,
                "occlusion_likelihood": 0.45,
                "id_switch_risk": 0.72,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets", "motion_trend_mismatch"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [
                    {"track_id": "track-C", "association_confidence": 0.63},
                    {"track_id": "track-B", "association_confidence": 0.79},
                ],
            },
            "det-3c": {
                "detection_id": "det-3c",
                "proposed_track_id": "track-B",
                "association_confidence": 0.58,
                "occlusion_likelihood": 0.7,
                "id_switch_risk": 0.65,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["occluded_target", "ambiguous_overlap"],
                "supporting_evidence": ["depth_consistency"],
                "alternative_track_candidates": [
                    {"track_id": "track-C", "association_confidence": 0.57},
                ],
            },
        }
        updates = [
            {
                "detection_id": "det-3a",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
            {
                "detection_id": "det-3b",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            },
            {
                "detection_id": "det-3c",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["occluded_target"],
                "supporting_evidence": ["depth_consistency"],
            },
        ]

        verdicts = {
            "det-3a": self._supported_verdict(),
            "det-3b": {
                "verification_status": "contradicted",
                "visibility_status": "visible",
                "identity_consistency": "inconsistent",
                "possible_id_switch": True,
                "confidence": 0.86,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": ["evidence://motion"],
                "recommended_follow_up": "review_association",
                "visual_content_inspected": True,
            },
            "det-3c": {
                "verification_status": "inconclusive",
                "visibility_status": "not_visible",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.4,
                "uncertainty": ["occlusion"],
                "limitations": ["target_occluded"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            },
        }

        async def _verdict_invoker(**kwargs):
            detection_id = kwargs["user_payload"]["detection_measurements"]["detection_id"]
            return verdicts[detection_id]

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch", "crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_verdict_invoker,
            )
        )

        det_a = result["per_detection"]["det-3a"]["diagnostic_adjudication"]
        det_b = result["per_detection"]["det-3b"]["diagnostic_adjudication"]
        det_c = result["per_detection"]["det-3c"]["diagnostic_adjudication"]
        self.assertEqual(det_a["classification"], "accepted")
        self.assertEqual(det_b["classification"], "rejected")
        self.assertEqual(det_c["classification"], "unresolved")
        self.assertTrue(det_b["reason_codes"])
        self.assertTrue(det_c["reason_codes"])

        retained_alternatives_b = result["per_detection"]["det-3b"]["tracking_proposal"]["alternative_track_candidates"]
        retained_alternatives_c = result["per_detection"]["det-3c"]["tracking_proposal"]["alternative_track_candidates"]
        self.assertEqual(len(retained_alternatives_b), 2)
        self.assertEqual(retained_alternatives_c[0]["track_id"], "track-C")

        self.assertEqual(result["per_detection"]["det-3b"]["coordinator_decision"]["resolved_track_id"], "track-B")
        self.assertEqual(result["per_detection"]["det-3c"]["coordinator_decision"]["resolved_track_id"], "track-B")
        self.assertNotIn("replacement_track_id", det_b)
        self.assertNotIn("replacement_track_id", det_c)

    def test_five_target_replay_group_conflict_safety_and_coordinator_unchanged(self) -> None:
        detections = [
            self._make_detection_candidate(
                detection_id="det-5a",
                yolo_track_id="track-A",
                relative_depth=0.50,
                ground_x=1.0,
                ground_z=4.2,
            ),
            self._make_detection_candidate(
                detection_id="det-5b",
                yolo_track_id="track-C",
                relative_depth=0.61,
                ground_x=2.8,
                ground_z=3.5,
            ),
            self._make_detection_candidate(
                detection_id="det-5c",
                yolo_track_id="track-B",
                relative_depth=0.45,
                ground_x=2.0,
                ground_z=4.5,
            ),
            self._make_detection_candidate(
                detection_id="det-5d",
                yolo_track_id="track-D",
                relative_depth=0.38,
                ground_x=3.3,
                ground_z=4.9,
            ),
            self._make_detection_candidate(
                detection_id="det-5e",
                yolo_track_id="track-E",
                relative_depth=0.70,
                ground_x=0.7,
                ground_z=3.3,
            ),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.50, last_ground_x=1.0, last_ground_z=4.2),
                self._history_record(track_id="track-B", depth_median=0.44, last_ground_x=2.0, last_ground_z=4.5),
                self._history_record(track_id="track-C", depth_median=0.52, last_ground_x=1.7, last_ground_z=4.7),
                self._history_record(track_id="track-D", depth_median=0.39, last_ground_x=3.3, last_ground_z=4.9),
                self._history_record(track_id="track-E", depth_median=0.70, last_ground_x=0.7, last_ground_z=3.3),
            ],
        )

        proposals = {
            "det-5a": {
                "detection_id": "det-5a",
                "proposed_track_id": "track-A",
                "association_confidence": 0.9,
                "occlusion_likelihood": 0.1,
                "id_switch_risk": 0.1,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
            "det-5b": {
                "detection_id": "det-5b",
                "proposed_track_id": "track-B",
                "association_confidence": 0.82,
                "occlusion_likelihood": 0.4,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [
                    {"track_id": "track-C", "association_confidence": 0.74},
                ],
            },
            "det-5c": {
                "detection_id": "det-5c",
                "proposed_track_id": "track-C",
                "association_confidence": 0.77,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.72,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets", "ambiguous_overlap"],
                "supporting_evidence": ["depth_consistency"],
                "alternative_track_candidates": [
                    {"track_id": "track-B", "association_confidence": 0.74},
                ],
            },
            "det-5d": {
                "detection_id": "det-5d",
                "proposed_track_id": "track-C",
                "association_confidence": 0.8,
                "occlusion_likelihood": 0.65,
                "id_switch_risk": 0.6,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": ["ambiguous_overlap"],
                "supporting_evidence": ["motion_continuity", "depth_consistency"],
            },
            "det-5e": {
                "detection_id": "det-5e",
                "proposed_track_id": "track-E",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.75,
                "id_switch_risk": 0.5,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": ["partial_occlusion"],
                "supporting_evidence": ["depth_consistency"],
                "alternative_track_candidates": [
                    {"track_id": "track-D", "association_confidence": 0.5},
                ],
            },
        }
        updates = [
            {
                "detection_id": "det-5a",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": [],
                "supporting_evidence": ["motion_continuity"],
            },
            {
                "detection_id": "det-5b",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            },
            {
                "detection_id": "det-5c",
                "resolved_track_id": "track-C",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["depth_consistency"],
            },
            {
                "detection_id": "det-5d",
                "resolved_track_id": "track-C",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["ambiguous_overlap"],
                "supporting_evidence": ["motion_continuity"],
            },
            {
                "detection_id": "det-5e",
                "resolved_track_id": "track-E",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["partial_occlusion"],
                "supporting_evidence": ["depth_consistency"],
            },
        ]

        verdicts = {
            "det-5a": self._supported_verdict(),
            "det-5b": {
                "verification_status": "contradicted",
                "visibility_status": "visible",
                "identity_consistency": "inconsistent",
                "possible_id_switch": True,
                "confidence": 0.87,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": ["evidence://motion"],
                "recommended_follow_up": "review_association",
                "visual_content_inspected": True,
            },
            "det-5c": self._supported_verdict(),
            "det-5d": {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.43,
                "uncertainty": ["overlap"],
                "limitations": ["crowded_scene"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            },
            "det-5e": {
                "verification_status": "supported",
                "visibility_status": "not_visible",
                "identity_consistency": "unknown",
                "possible_id_switch": False,
                "confidence": 0.52,
                "uncertainty": ["occluded"],
                "limitations": ["target_occluded"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            },
        }

        async def _verdict_invoker(**kwargs):
            detection_id = kwargs["user_payload"]["detection_measurements"]["detection_id"]
            return verdicts[detection_id]

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["possible_id_switch", "crossing_targets", "occlusion"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_verdict_invoker,
            )
        )

        self.assertIn("track-C", result["summary"]["conflicting_proposed_track_ids"])
        classifications = {
            detection_id: payload["diagnostic_adjudication"]["classification"]
            for detection_id, payload in result["per_detection"].items()
        }
        self.assertEqual(classifications["det-5a"], "accepted")
        self.assertEqual(classifications["det-5b"], "rejected")
        self.assertEqual(classifications["det-5c"], "unresolved")
        self.assertEqual(classifications["det-5d"], "unresolved")
        self.assertEqual(classifications["det-5e"], "unresolved")
        self.assertNotEqual(classifications["det-5c"], "accepted")
        self.assertNotEqual(classifications["det-5d"], "accepted")

        for update in updates:
            detection_id = update["detection_id"]
            self.assertEqual(
                result["per_detection"][detection_id]["coordinator_decision"],
                update,
            )

        self.assertTrue(
            result["per_detection"]["det-5e"]["tracking_proposal"]["alternative_track_candidates"]
        )
        self.assertIn(
            "provisional_low_confidence",
            result["per_detection"]["det-5e"]["diagnostic_adjudication"]["reason_codes"],
        )
        self.assertNotIn(
            "multi_source_consistent_support",
            result["per_detection"]["det-5e"]["diagnostic_adjudication"]["reason_codes"],
        )

    def test_recovery_two_target_id_swap_with_sufficient_history(self) -> None:
        detections = [
            self._make_detection_candidate(
                detection_id="det-r1",
                yolo_track_id="track-B",
                relative_depth=0.5,
                ground_x=1.0,
                ground_z=4.0,
            ),
            self._make_detection_candidate(
                detection_id="det-r2",
                yolo_track_id="track-A",
                relative_depth=0.45,
                ground_x=2.0,
                ground_z=4.4,
            ),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
            ],
        )
        proposals = {
            "det-r1": {
                "detection_id": "det-r1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.5,
                "occlusion_likelihood": 0.4,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.78}],
            },
            "det-r2": {
                "detection_id": "det-r2",
                "proposed_track_id": "track-A",
                "association_confidence": 0.5,
                "occlusion_likelihood": 0.35,
                "id_switch_risk": 0.75,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-B", "association_confidence": 0.77}],
            },
        }
        updates = [
            {
                "detection_id": "det-r1",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            },
            {
                "detection_id": "det-r2",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            },
        ]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.45,
                "uncertainty": [],
                "limitations": ["crossing"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        recovery_summary = result["summary"]["diagnostic_alternative_identity_recovery"]
        self.assertEqual(recovery_summary["proposed_corrections_count"], 2)
        self.assertEqual(set(recovery_summary["correction_detection_ids"]), {"det-r1", "det-r2"})
        self.assertEqual(
            result["per_detection"]["det-r1"]["diagnostic_alternative_identity_recovery"]["proposed_correction"]["to_track_id"],
            "track-A",
        )
        self.assertEqual(
            result["per_detection"]["det-r2"]["diagnostic_alternative_identity_recovery"]["proposed_correction"]["to_track_id"],
            "track-B",
        )

    def test_recovery_two_target_crossing_with_insufficient_evidence_remains_unresolved(self) -> None:
        detections = [
            self._make_detection_candidate(
                detection_id="det-u1",
                yolo_track_id="track-B",
                relative_depth=0.5,
                ground_x=1.0,
                ground_z=4.0,
            ),
            self._make_detection_candidate(
                detection_id="det-u2",
                yolo_track_id="track-A",
                relative_depth=0.45,
                ground_x=2.0,
                ground_z=4.4,
            ),
        ]
        stale_a = self._history_record(track_id="track-A", depth_median=0.8, last_ground_x=3.5, last_ground_z=2.0)
        stale_b = self._history_record(track_id="track-B", depth_median=0.2, last_ground_x=0.2, last_ground_z=6.0)
        stale_a["evidence_timestamp"] = 990.0
        stale_b["evidence_timestamp"] = 989.0
        scenario_metadata = self._scenario_metadata_with_tracks(phase1.SCENARIO_YOLO_ID_SWAP, [stale_a, stale_b])

        proposals = {
            "det-u1": {
                "detection_id": "det-u1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.55,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.73}],
            },
            "det-u2": {
                "detection_id": "det-u2",
                "proposed_track_id": "track-A",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.55,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-B", "association_confidence": 0.72}],
            },
        }
        updates = [
            {
                "detection_id": "det-u1",
                "resolved_track_id": "track-B",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            },
            {
                "detection_id": "det-u2",
                "resolved_track_id": "track-A",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            },
        ]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.4,
                "uncertainty": [],
                "limitations": ["insufficient_detail"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        recovery_summary = result["summary"]["diagnostic_alternative_identity_recovery"]
        self.assertEqual(recovery_summary["proposed_corrections_count"], 0)
        self.assertEqual(recovery_summary["unresolved_count"], 2)

    def test_recovery_three_target_identity_permutation(self) -> None:
        detections = [
            self._make_detection_candidate(detection_id="det-p1", yolo_track_id="track-B", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
            self._make_detection_candidate(detection_id="det-p2", yolo_track_id="track-C", relative_depth=0.45, ground_x=2.0, ground_z=4.4),
            self._make_detection_candidate(detection_id="det-p3", yolo_track_id="track-A", relative_depth=0.62, ground_x=3.0, ground_z=3.6),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
                self._history_record(track_id="track-C", depth_median=0.62, last_ground_x=3.0, last_ground_z=3.6),
            ],
        )
        proposals = {
            "det-p1": {
                "detection_id": "det-p1", "proposed_track_id": "track-B", "association_confidence": 0.53,
                "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False,
                "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.78}],
            },
            "det-p2": {
                "detection_id": "det-p2", "proposed_track_id": "track-C", "association_confidence": 0.53,
                "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False,
                "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-B", "association_confidence": 0.76}],
            },
            "det-p3": {
                "detection_id": "det-p3", "proposed_track_id": "track-A", "association_confidence": 0.53,
                "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False,
                "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-C", "association_confidence": 0.75}],
            },
        }
        updates = [
            {"detection_id": "det-p1", "resolved_track_id": "track-B", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-p2", "resolved_track_id": "track-C", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-p3", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
        ]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.43, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        self.assertEqual(result["summary"]["diagnostic_alternative_identity_recovery"]["proposed_corrections_count"], 3)

    def test_recovery_five_target_competing_candidates_group_safe(self) -> None:
        detections = [
            self._make_detection_candidate(detection_id="det-q1", yolo_track_id="track-B", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
            self._make_detection_candidate(detection_id="det-q2", yolo_track_id="track-A", relative_depth=0.45, ground_x=2.0, ground_z=4.4),
            self._make_detection_candidate(detection_id="det-q3", yolo_track_id="track-C", relative_depth=0.62, ground_x=3.0, ground_z=3.6),
            self._make_detection_candidate(detection_id="det-q4", yolo_track_id="track-D", relative_depth=0.38, ground_x=3.3, ground_z=4.9),
            self._make_detection_candidate(detection_id="det-q5", yolo_track_id="track-E", relative_depth=0.71, ground_x=0.7, ground_z=3.3),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
                self._history_record(track_id="track-C", depth_median=0.62, last_ground_x=3.0, last_ground_z=3.6),
                self._history_record(track_id="track-D", depth_median=0.38, last_ground_x=3.3, last_ground_z=4.9),
                self._history_record(track_id="track-E", depth_median=0.71, last_ground_x=0.7, last_ground_z=3.3),
            ],
        )
        proposals = {
            "det-q1": {"detection_id": "det-q1", "proposed_track_id": "track-B", "association_confidence": 0.52, "occlusion_likelihood": 0.4, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}]},
            "det-q2": {"detection_id": "det-q2", "proposed_track_id": "track-A", "association_confidence": 0.52, "occlusion_likelihood": 0.4, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-B", "association_confidence": 0.8}]},
            "det-q3": {"detection_id": "det-q3", "proposed_track_id": "track-C", "association_confidence": 0.52, "occlusion_likelihood": 0.6, "id_switch_risk": 0.7, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["ambiguous_overlap"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-D", "association_confidence": 0.72}]},
            "det-q4": {"detection_id": "det-q4", "proposed_track_id": "track-D", "association_confidence": 0.52, "occlusion_likelihood": 0.6, "id_switch_risk": 0.7, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["ambiguous_overlap"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-D", "association_confidence": 0.72}]},
            "det-q5": {"detection_id": "det-q5", "proposed_track_id": "track-E", "association_confidence": 0.51, "occlusion_likelihood": 0.7, "id_switch_risk": 0.6, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["occluded"], "supporting_evidence": ["depth_consistency"]},
        }
        updates = [
            {"detection_id": "det-q1", "resolved_track_id": "track-B", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-q2", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-q3", "resolved_track_id": "track-C", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-q4", "resolved_track_id": "track-D", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-q5", "resolved_track_id": "track-E", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["depth_consistency"]},
        ]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.42, "uncertainty": [], "limitations": ["crowded_scene"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets", "occlusion"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        recovery_summary = result["summary"]["diagnostic_alternative_identity_recovery"]
        self.assertFalse(recovery_summary["duplicate_assignment_detected"])
        self.assertTrue(recovery_summary["group_consistency_enforced"])

    def test_recovery_missing_or_occluded_target_stays_unresolved(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-m1",
            yolo_track_id="track-Z",
            relative_depth=0.6,
            ground_x=5.0,
            ground_z=8.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(phase1.SCENARIO_BASELINE, [])
        proposals = {
            "det-m1": {
                "detection_id": "det-m1",
                "proposed_track_id": "track-Z",
                "association_confidence": 0.5,
                "occlusion_likelihood": 0.95,
                "id_switch_risk": 0.7,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["occluded_target"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-Y", "association_confidence": 0.6}],
            }
        }
        updates = [
            {
                "detection_id": "det-m1",
                "resolved_track_id": "track-Z",
                "accepted": True,
                "rejection_reasons": [],
                "uncertainties": ["occluded_target"],
                "supporting_evidence": ["appearance_match"],
            }
        ]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "not_visible", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.33, "uncertainty": ["occluded"], "limitations": ["target_occluded"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["occluded_target"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        recovery = result["per_detection"]["det-m1"]["diagnostic_alternative_identity_recovery"]
        self.assertEqual(recovery["recommended_outcome"], "unresolved")
        self.assertIsNone(recovery["proposed_correction"])

    def test_recovery_conflicting_candidates_same_identity_does_not_duplicate_assignment(self) -> None:
        detections = [
            self._make_detection_candidate(detection_id="det-c1", yolo_track_id="track-A", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
            self._make_detection_candidate(detection_id="det-c2", yolo_track_id="track-B", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [self._history_record(track_id="track-X", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)],
        )
        proposals = {
            "det-c1": {
                "detection_id": "det-c1", "proposed_track_id": "track-A", "association_confidence": 0.5,
                "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-X", "association_confidence": 0.8}],
            },
            "det-c2": {
                "detection_id": "det-c2", "proposed_track_id": "track-B", "association_confidence": 0.5,
                "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-X", "association_confidence": 0.8}],
            },
        }
        updates = [
            {"detection_id": "det-c1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-c2", "resolved_track_id": "track-B", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
        ]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        recovery_summary = result["summary"]["diagnostic_alternative_identity_recovery"]
        self.assertEqual(recovery_summary["proposed_corrections_count"], 1)
        self.assertEqual(recovery_summary["unresolved_count"], 1)
        self.assertFalse(recovery_summary["duplicate_assignment_detected"])

    def test_recovery_depth_contradiction_blocks_motion_supported_candidate(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-d1",
            yolo_track_id="track-A",
            relative_depth=0.2,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [self._history_record(track_id="track-X", depth_median=0.8, last_ground_x=1.0, last_ground_z=4.0)],
        )
        proposals = {
            "det-d1": {
                "detection_id": "det-d1", "proposed_track_id": "track-A", "association_confidence": 0.5,
                "occlusion_likelihood": 0.4, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-X", "association_confidence": 0.81}],
            }
        }
        updates = [{"detection_id": "det-d1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.41, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        recovery = result["per_detection"]["det-d1"]["diagnostic_alternative_identity_recovery"]
        self.assertEqual(recovery["recommended_outcome"], "unresolved")
        reasons = recovery["candidate_evaluations"][1]["reason_codes"]
        self.assertIn("depth_contradicted_by_history", reasons)

    def test_recovery_missing_stale_and_malformed_history_measurements(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-h1",
            yolo_track_id="track-A",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        stale_record = self._history_record(track_id="track-stale", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)
        stale_record["evidence_timestamp"] = 990.0
        malformed_record = {
            "track_id": "track-bad",
            "history_window_seconds": 1.8,
            "history_sample_count": 1,
            "recent_ground_path": [{"sample_age_s": 0.2, "ground_x": None, "ground_z": "bad"}],
            "recent_relative_depth": {"median": "nan"},
            "evidence_timestamp": None,
            "source": "independent_track_memory",
        }
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [stale_record, malformed_record],
        )
        proposals = {
            "det-h1": {
                "detection_id": "det-h1", "proposed_track_id": "track-A", "association_confidence": 0.5,
                "occlusion_likelihood": 0.4, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [
                    {"track_id": "track-missing", "association_confidence": 0.8},
                    {"track_id": "track-stale", "association_confidence": 0.79},
                    {"track_id": "track-bad", "association_confidence": 0.78},
                ],
            }
        }
        updates = [{"detection_id": "det-h1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _inconclusive_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.39, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_inconclusive_invoker,
            )
        )

        recovery = result["per_detection"]["det-h1"]["diagnostic_alternative_identity_recovery"]
        self.assertEqual(recovery["recommended_outcome"], "unresolved")
        statuses = {item["history_status"] for item in recovery["candidate_evaluations"]}
        self.assertIn("missing", statuses)
        self.assertIn("stale", statuses)
        self.assertIn("malformed", statuses)

    def test_reconsideration_successful_two_target_id_swap_correction(self) -> None:
        detections = [
            self._make_detection_candidate(detection_id="det-rs1", yolo_track_id="track-B", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
            self._make_detection_candidate(detection_id="det-rs2", yolo_track_id="track-A", relative_depth=0.45, ground_x=2.0, ground_z=4.4),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
            ],
        )
        proposals = {
            "det-rs1": {"detection_id": "det-rs1", "proposed_track_id": "track-B", "association_confidence": 0.5, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}]},
            "det-rs2": {"detection_id": "det-rs2", "proposed_track_id": "track-A", "association_confidence": 0.5, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-B", "association_confidence": 0.8}]},
        }
        updates = [
            {"detection_id": "det-rs1", "resolved_track_id": "track-B", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-rs2", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
        ]

        async def _tracking_reconsideration_invoker(**kwargs):
            targets = kwargs["user_payload"]["targets"]
            self.assertEqual(len(targets), 2)
            return {
                "declined": False,
                "revisions": [
                    {"detection_id": "det-rs1", "proposed_track_id": "track-A", "association_confidence": 0.83, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_motion_depth_alignment"]},
                    {"detection_id": "det-rs2", "proposed_track_id": "track-B", "association_confidence": 0.82, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_motion_depth_alignment"]},
                ],
            }

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["proposed_track_id"] in {"track-A", "track-B"} and proposal["association_confidence"] >= 0.8:
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                tracking_reconsideration_model="qwen-3.8-27b",
                tracking_reconsideration_timeout_seconds=10.0,
                tracking_reconsideration_max_output_tokens=768,
                tracking_reconsideration_reasoning_effort="none",
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        self.assertEqual(result["summary"]["diagnostic_adjudication_counts"]["unresolved"], 2)
        self.assertEqual(result["summary"]["diagnostic_final_adjudication_counts"]["accepted"], 2)
        self.assertEqual(result["per_detection"]["det-rs1"]["identity_origin"], "revised_tracking")
        self.assertEqual(result["per_detection"]["det-rs2"]["identity_origin"], "revised_tracking")

    def test_reconsideration_observation_rejects_incorrect_revision(self) -> None:
        detection = self._make_detection_candidate(detection_id="det-rj1", yolo_track_id="track-A", relative_depth=0.5, ground_x=1.0, ground_z=4.0)
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)],
        )
        proposals = {
            "det-rj1": {"detection_id": "det-rj1", "proposed_track_id": "track-A", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.7, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-Z", "association_confidence": 0.8}]}
        }
        updates = [{"detection_id": "det-rj1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {"declined": False, "revisions": [{"detection_id": "det-rj1", "proposed_track_id": "track-Z", "association_confidence": 0.85, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["weak_signal"]}]}

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["proposed_track_id"] == "track-Z":
                return {
                    "verification_status": "contradicted", "visibility_status": "visible", "identity_consistency": "inconsistent",
                    "possible_id_switch": True, "confidence": 0.86, "uncertainty": [], "limitations": [],
                    "supporting_evidence_refs": [], "contradicting_evidence_refs": ["evidence://motion"], "recommended_follow_up": "review_association",
                    "visual_content_inspected": True,
                }
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        self.assertEqual(result["per_detection"]["det-rj1"]["diagnostic_adjudication_final"]["classification"], "rejected")

    def test_reconsideration_three_target_identity_permutation(self) -> None:
        detections = [
            self._make_detection_candidate(detection_id="det-rp1", yolo_track_id="track-B", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
            self._make_detection_candidate(detection_id="det-rp2", yolo_track_id="track-C", relative_depth=0.45, ground_x=2.0, ground_z=4.4),
            self._make_detection_candidate(detection_id="det-rp3", yolo_track_id="track-A", relative_depth=0.62, ground_x=3.0, ground_z=3.6),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
                self._history_record(track_id="track-C", depth_median=0.62, last_ground_x=3.0, last_ground_z=3.6),
            ],
        )
        proposals = {
            "det-rp1": {"detection_id": "det-rp1", "proposed_track_id": "track-B", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}]},
            "det-rp2": {"detection_id": "det-rp2", "proposed_track_id": "track-C", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-B", "association_confidence": 0.8}]},
            "det-rp3": {"detection_id": "det-rp3", "proposed_track_id": "track-A", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-C", "association_confidence": 0.8}]},
        }
        updates = [
            {"detection_id": "det-rp1", "resolved_track_id": "track-B", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-rp2", "resolved_track_id": "track-C", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
            {"detection_id": "det-rp3", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
        ]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {
                "declined": False,
                "revisions": [
                    {"detection_id": "det-rp1", "proposed_track_id": "track-A", "association_confidence": 0.82, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]},
                    {"detection_id": "det-rp2", "proposed_track_id": "track-B", "association_confidence": 0.81, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]},
                    {"detection_id": "det-rp3", "proposed_track_id": "track-C", "association_confidence": 0.8, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]},
                ],
            }

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["association_confidence"] >= 0.8:
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        self.assertEqual(result["summary"]["diagnostic_final_adjudication_counts"]["accepted"], 3)

    def test_reconsideration_five_target_competing_associations(self) -> None:
        detections = [
            self._make_detection_candidate(detection_id="det-rf1", yolo_track_id="track-B", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
            self._make_detection_candidate(detection_id="det-rf2", yolo_track_id="track-A", relative_depth=0.45, ground_x=2.0, ground_z=4.4),
            self._make_detection_candidate(detection_id="det-rf3", yolo_track_id="track-C", relative_depth=0.62, ground_x=3.0, ground_z=3.6),
            self._make_detection_candidate(detection_id="det-rf4", yolo_track_id="track-D", relative_depth=0.38, ground_x=3.3, ground_z=4.9),
            self._make_detection_candidate(detection_id="det-rf5", yolo_track_id="track-E", relative_depth=0.71, ground_x=0.7, ground_z=3.3),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
                self._history_record(track_id="track-C", depth_median=0.62, last_ground_x=3.0, last_ground_z=3.6),
                self._history_record(track_id="track-D", depth_median=0.38, last_ground_x=3.3, last_ground_z=4.9),
                self._history_record(track_id="track-E", depth_median=0.71, last_ground_x=0.7, last_ground_z=3.3),
            ],
        )
        proposals = {
            f"det-rf{i}": {"detection_id": f"det-rf{i}", "proposed_track_id": track, "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"]}
            for i, track in [(1, "track-B"), (2, "track-A"), (3, "track-C"), (4, "track-D"), (5, "track-E")]
        }
        proposals["det-rf1"]["alternative_track_candidates"] = [{"track_id": "track-A", "association_confidence": 0.8}]
        proposals["det-rf2"]["alternative_track_candidates"] = [{"track_id": "track-B", "association_confidence": 0.8}]
        updates = [
            {"detection_id": f"det-rf{i}", "resolved_track_id": track, "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}
            for i, track in [(1, "track-B"), (2, "track-A"), (3, "track-C"), (4, "track-D"), (5, "track-E")]
        ]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {
                "declined": False,
                "revisions": [
                    {"detection_id": "det-rf1", "proposed_track_id": "track-A", "association_confidence": 0.82, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]},
                    {"detection_id": "det-rf2", "proposed_track_id": "track-B", "association_confidence": 0.81, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]},
                ],
            }

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["association_confidence"] >= 0.8:
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        self.assertGreaterEqual(result["summary"]["tracking_reconsideration"]["revised_detection_count"], 2)

    def test_reconsideration_enforces_global_uniqueness_with_non_reconsidered_accepted_identity(self) -> None:
        detections = [
            self._make_detection_candidate(detection_id="det-ru1", yolo_track_id="track-A", relative_depth=0.5, ground_x=1.0, ground_z=4.0),
            self._make_detection_candidate(detection_id="det-ru2", yolo_track_id="track-B", relative_depth=0.45, ground_x=2.0, ground_z=4.4),
        ]
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
            ],
        )
        proposals = {
            "det-ru1": {"detection_id": "det-ru1", "proposed_track_id": "track-A", "association_confidence": 0.9, "occlusion_likelihood": 0.1, "id_switch_risk": 0.1, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["motion_continuity", "depth_consistency"]},
            "det-ru2": {"detection_id": "det-ru2", "proposed_track_id": "track-B", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}]},
        }
        updates = [
            {"detection_id": "det-ru1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["motion_continuity"]},
            {"detection_id": "det-ru2", "resolved_track_id": "track-B", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]},
        ]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {"declined": False, "revisions": [{"detection_id": "det-ru2", "proposed_track_id": "track-A", "association_confidence": 0.81, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]}]}

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["detection_id"] == "det-ru1":
                return self._supported_verdict()
            if proposal["proposed_track_id"] == "track-A" and proposal["association_confidence"] >= 0.8:
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=detections,
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        self.assertIn("track-A", result["summary"]["conflicting_final_assigned_track_ids"])
        self.assertEqual(result["per_detection"]["det-ru1"]["diagnostic_adjudication_final"]["classification"], "unresolved")
        self.assertEqual(result["per_detection"]["det-ru2"]["diagnostic_adjudication_final"]["classification"], "unresolved")

    def test_reconsideration_missing_or_stale_history_stays_unresolved(self) -> None:
        detection = self._make_detection_candidate(detection_id="det-rh1", yolo_track_id="track-A", relative_depth=0.5, ground_x=1.0, ground_z=4.0)
        stale_history = self._history_record(track_id="track-Z", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)
        stale_history["evidence_timestamp"] = 990.0
        scenario_metadata = self._scenario_metadata_with_tracks(phase1.SCENARIO_BASELINE, [stale_history])
        proposals = {"det-rh1": {"detection_id": "det-rh1", "proposed_track_id": "track-A", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-Q", "association_confidence": 0.8}]}}
        updates = [{"detection_id": "det-rh1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {"declined": False, "revisions": [{"detection_id": "det-rh1", "proposed_track_id": "track-Q", "association_confidence": 0.81, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]}]}

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["proposed_track_id"] == "track-Q":
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        self.assertEqual(result["per_detection"]["det-rh1"]["diagnostic_adjudication_final"]["classification"], "unresolved")

    def test_reconsideration_tracking_declines_to_revise(self) -> None:
        detection = self._make_detection_candidate(detection_id="det-rd1", yolo_track_id="track-A", relative_depth=0.5, ground_x=1.0, ground_z=4.0)
        scenario_metadata = self._scenario_metadata_with_tracks(phase1.SCENARIO_BASELINE, [self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)])
        proposals = {"det-rd1": {"detection_id": "det-rd1", "proposed_track_id": "track-A", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"]}}
        updates = [{"detection_id": "det-rd1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {"declined": True, "decline_reason": "insufficient_evidence", "revisions": []}

        async def _observation_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        reconsideration = result["per_detection"]["det-rd1"]["tracking_reconsideration"]
        self.assertTrue(reconsideration["considered"])
        self.assertFalse(reconsideration["revised"])
        self.assertEqual(result["per_detection"]["det-rd1"]["identity_origin"], "initial_tracking")

    def test_reconsideration_tracking_timeout_or_malformed_response(self) -> None:
        detection = self._make_detection_candidate(detection_id="det-rt1", yolo_track_id="track-A", relative_depth=0.5, ground_x=1.0, ground_z=4.0)
        scenario_metadata = self._scenario_metadata_with_tracks(phase1.SCENARIO_BASELINE, [self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)])
        proposals = {"det-rt1": {"detection_id": "det-rt1", "proposed_track_id": "track-A", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"]}}
        updates = [{"detection_id": "det-rt1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _timeout_reconsideration_invoker(**_kwargs):
            raise TimeoutError("reconsideration timeout")

        async def _observation_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        timeout_result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection], proposal_by_detection=proposals, updates=updates,
                scenario_metadata=scenario_metadata, event_reasons=["crossing_targets"], frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK, observation_model="qwen-3.8-27b", observation_timeout_seconds=6.0,
                invoker=_observation_invoker, reconsideration_invoker=_timeout_reconsideration_invoker,
            )
        )
        self.assertEqual(timeout_result["summary"]["tracking_reconsideration"]["invocation"]["status"], "error")

        async def _malformed_reconsideration_invoker(**_kwargs):
            return {"declined": False}

        malformed_result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection], proposal_by_detection=proposals, updates=updates,
                scenario_metadata=scenario_metadata, event_reasons=["crossing_targets"], frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK, observation_model="qwen-3.8-27b", observation_timeout_seconds=6.0,
                invoker=_observation_invoker, reconsideration_invoker=_malformed_reconsideration_invoker,
            )
        )
        self.assertEqual(malformed_result["summary"]["tracking_reconsideration"]["invocation"]["status"], "error")

    def test_reconsideration_observation_timeout_or_malformed_round2(self) -> None:
        detection = self._make_detection_candidate(detection_id="det-ro1", yolo_track_id="track-A", relative_depth=0.5, ground_x=1.0, ground_z=4.0)
        scenario_metadata = self._scenario_metadata_with_tracks(phase1.SCENARIO_BASELINE, [self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)])
        proposals = {"det-ro1": {"detection_id": "det-ro1", "proposed_track_id": "track-A", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"]}}
        updates = [{"detection_id": "det-ro1", "resolved_track_id": "track-A", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {"declined": False, "revisions": [{"detection_id": "det-ro1", "proposed_track_id": "track-A", "association_confidence": 0.81, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]}]}

        call_count = {"value": 0}

        async def _observation_timeout_invoker(**_kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return {
                    "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                    "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                    "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                    "visual_content_inspected": True,
                }
            raise TimeoutError("observation round2 timeout")

        timeout_result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection], proposal_by_detection=proposals, updates=updates,
                scenario_metadata=scenario_metadata, event_reasons=["crossing_targets"], frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK, observation_model="qwen-3.8-27b", observation_timeout_seconds=6.0,
                invoker=_observation_timeout_invoker, reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )
        self.assertEqual(timeout_result["per_detection"]["det-ro1"]["diagnostic_adjudication_final"]["classification"], "unresolved")

        async def _observation_malformed_invoker(**_kwargs):
            return {"verification_status": "supported"}

        malformed_result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection], proposal_by_detection=proposals, updates=updates,
                scenario_metadata=scenario_metadata, event_reasons=["crossing_targets"], frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK, observation_model="qwen-3.8-27b", observation_timeout_seconds=6.0,
                invoker=_observation_malformed_invoker, reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )
        self.assertEqual(malformed_result["per_detection"]["det-ro1"]["diagnostic_adjudication_final"]["classification"], "unresolved")

    def test_reconsideration_preserves_original_diagnostic_results(self) -> None:
        detection = self._make_detection_candidate(detection_id="det-rp0", yolo_track_id="track-B", relative_depth=0.5, ground_x=1.0, ground_z=4.0)
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
            ],
        )
        proposals = {"det-rp0": {"detection_id": "det-rp0", "proposed_track_id": "track-B", "association_confidence": 0.52, "occlusion_likelihood": 0.5, "id_switch_risk": 0.8, "depth_conflict_flag": False, "position_conflict_flag": True, "uncertainties": ["crossing_targets"], "supporting_evidence": ["appearance_match"], "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}]}}
        updates = [{"detection_id": "det-rp0", "resolved_track_id": "track-B", "accepted": True, "rejection_reasons": [], "uncertainties": [], "supporting_evidence": ["appearance_match"]}]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {"declined": False, "revisions": [{"detection_id": "det-rp0", "proposed_track_id": "track-A", "association_confidence": 0.82, "depth_conflict_flag": False, "position_conflict_flag": False, "uncertainties": [], "supporting_evidence": ["history_match"]}]}

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["association_confidence"] >= 0.8:
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive", "visibility_status": "partial", "identity_consistency": "unknown",
                "possible_id_switch": True, "confidence": 0.4, "uncertainty": [], "limitations": ["crossing"],
                "supporting_evidence_refs": [], "contradicting_evidence_refs": [], "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection], proposal_by_detection=proposals, updates=updates,
                scenario_metadata=scenario_metadata, event_reasons=["crossing_targets"], frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK, observation_model="qwen-3.8-27b", observation_timeout_seconds=6.0,
                invoker=_observation_invoker, reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        self.assertEqual(result["per_detection"]["det-rp0"]["diagnostic_adjudication"]["classification"], "unresolved")
        self.assertEqual(result["per_detection"]["det-rp0"]["diagnostic_adjudication_final"]["classification"], "accepted")

    def test_reconsideration_disabled_by_default_mode_off(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-off-1",
            yolo_track_id="track-A",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)],
        )
        proposals = {
            "det-off-1": {
                "detection_id": "det-off-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            }
        }
        updates = [{
            "detection_id": "det-off-1",
            "resolved_track_id": "track-A",
            "accepted": True,
            "rejection_reasons": [],
            "uncertainties": [],
            "supporting_evidence": ["appearance_match"],
        }]

        async def _observation_invoker(**_kwargs):
            return {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.4,
                "uncertainty": [],
                "limitations": ["crossing"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        async def _should_not_be_called(**_kwargs):
            raise AssertionError("reconsideration invoker should not be called when mode=off")

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_should_not_be_called,
                tracking_reconsideration_mode=phase1.TRACKING_RECONSIDERATION_MODE_OFF,
            )
        )

        invocation = result["summary"]["tracking_reconsideration"]["invocation"]
        self.assertEqual(invocation["status"], "skipped")
        self.assertEqual(invocation["mode"], "off")

    def test_reconsideration_live_mode_uses_real_path_with_mocked_live_invoker(self) -> None:
        from unittest.mock import patch

        detection = self._make_detection_candidate(
            detection_id="det-live-1",
            yolo_track_id="track-B",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
            ],
        )
        proposals = {
            "det-live-1": {
                "detection_id": "det-live-1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}],
            }
        }
        updates = [{
            "detection_id": "det-live-1",
            "resolved_track_id": "track-B",
            "accepted": True,
            "rejection_reasons": [],
            "uncertainties": [],
            "supporting_evidence": ["appearance_match"],
        }]

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["association_confidence"] >= 0.8:
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.4,
                "uncertainty": [],
                "limitations": ["crossing"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        live_called = {"value": False}

        async def _mock_live_reconsideration(**kwargs):
            live_called["value"] = True
            targets = kwargs["user_payload"]["targets"]
            self.assertEqual(len(targets), 1)
            return {
                "declined": False,
                "revisions": [
                    {
                        "detection_id": "det-live-1",
                        "proposed_track_id": "track-A",
                        "association_confidence": 0.82,
                        "depth_conflict_flag": False,
                        "position_conflict_flag": False,
                        "uncertainties": [],
                        "supporting_evidence": ["history_match"],
                    }
                ],
            }

        with patch("tools.tracking_agent_phase1_validation._invoke_tracking_reconsideration_live", _mock_live_reconsideration):
            result = asyncio.run(
                phase1._run_observation_collaboration_diagnostic(
                    detections=[detection],
                    proposal_by_detection=proposals,
                    updates=updates,
                    scenario_metadata=scenario_metadata,
                    event_reasons=["crossing_targets"],
                    frame_reference="data:image/jpeg;base64,abc",
                    mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                    observation_model="qwen-3.8-27b",
                    observation_timeout_seconds=6.0,
                    invoker=_observation_invoker,
                    tracking_reconsideration_mode=phase1.TRACKING_RECONSIDERATION_MODE_LIVE,
                    provider_request_budget={"max_total": 5, "consumed": 0, "remaining": 5},
                )
            )

        self.assertTrue(live_called["value"])
        self.assertEqual(result["per_detection"]["det-live-1"]["identity_origin"], "revised_tracking")

    def test_reconsideration_payload_excludes_ground_truth_and_deterministic_recovery_answers(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-pay-1",
            yolo_track_id="track-B",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = {
            "scenario": phase1.SCENARIO_YOLO_ID_SWAP,
            phase1.INDEPENDENT_TRACK_HISTORY_KEY: [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
            ],
            "expected_correct_associations": {"det-pay-1": "track-A"},
        }
        per_detection = {
            "det-pay-1": {
                "tracking_proposal": {
                    "detection_id": "det-pay-1",
                    "proposed_track_id": "track-B",
                    "association_confidence": 0.52,
                    "occlusion_likelihood": 0.5,
                    "id_switch_risk": 0.8,
                    "depth_conflict_flag": False,
                    "position_conflict_flag": True,
                    "uncertainties": ["crossing_targets"],
                    "supporting_evidence": ["appearance_match"],
                    "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}],
                },
                "observation_agent_verdict": {
                    "verification_status": "inconclusive",
                    "supporting_evidence_refs": [],
                    "contradicting_evidence_refs": ["evidence://motion"],
                },
                "diagnostic_adjudication": {"classification": "unresolved"},
                "diagnostic_alternative_identity_recovery": {
                    "recommended_outcome": "correction_proposed",
                    "proposed_correction": {"to_track_id": "track-A"},
                },
            }
        }

        payload = phase1._build_tracking_reconsideration_payload(
            eligible_detection_ids=["det-pay-1"],
            detections_by_id={"det-pay-1": detection},
            per_detection=per_detection,
            scenario_metadata=scenario_metadata,
            event_reasons=["crossing_targets"],
        )

        payload_text = json.dumps(payload, sort_keys=True)
        self.assertIn("original_tracking_proposal", payload_text)
        self.assertIn("original_observation_feedback", payload_text)
        self.assertIn("independent_track_history_evidence", payload_text)
        self.assertNotIn("expected_correct_associations", payload_text)
        self.assertNotIn("diagnostic_alternative_identity_recovery", payload_text)
        self.assertNotIn("proposed_correction", payload_text)

    def test_provider_request_limits_fail_closed_for_live_modes(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-lim-1",
            yolo_track_id="track-A",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)],
        )
        proposals = {
            "det-lim-1": {
                "detection_id": "det-lim-1",
                "proposed_track_id": "track-A",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            }
        }
        updates = [{
            "detection_id": "det-lim-1",
            "resolved_track_id": "track-A",
            "accepted": True,
            "rejection_reasons": [],
            "uncertainties": [],
            "supporting_evidence": ["appearance_match"],
        }]

        async def _never_called_observation_live(**_kwargs):
            raise AssertionError("observation live invoker should not run when provider budget exhausted")

        async def _never_called_tracking_live(**_kwargs):
            raise AssertionError("tracking reconsideration live invoker should not run when provider budget exhausted")

        from unittest.mock import patch

        with patch("tools.tracking_agent_phase1_validation._invoke_tracking_reconsideration_live", _never_called_tracking_live):
            result = asyncio.run(
                phase1._run_observation_collaboration_diagnostic(
                    detections=[detection],
                    proposal_by_detection=proposals,
                    updates=updates,
                    scenario_metadata=scenario_metadata,
                    event_reasons=["crossing_targets"],
                    frame_reference="data:image/jpeg;base64,abc",
                    mode=phase1.OBSERVATION_VERIFICATION_MODE_LIVE,
                    observation_model="qwen-3.8-27b",
                    observation_timeout_seconds=6.0,
                    invoker=_never_called_observation_live,
                    tracking_reconsideration_mode=phase1.TRACKING_RECONSIDERATION_MODE_LIVE,
                    provider_request_budget={"max_total": 1, "consumed": 1, "remaining": 0},
                )
            )

        det_payload = result["per_detection"]["det-lim-1"]
        self.assertEqual(det_payload["observation_agent_invocation"]["error"], "provider_request_limit_exhausted")
        self.assertEqual(result["summary"]["tracking_reconsideration"]["invocation"]["error"], "provider_request_limit_exhausted")

    def test_provenance_separation_and_coordinator_behavior_unchanged_after_reconsideration(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-prov-1",
            yolo_track_id="track-B",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
            ],
        )
        proposals = {
            "det-prov-1": {
                "detection_id": "det-prov-1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}],
            }
        }
        updates = [{
            "detection_id": "det-prov-1",
            "resolved_track_id": "track-B",
            "accepted": True,
            "rejection_reasons": [],
            "uncertainties": [],
            "supporting_evidence": ["appearance_match"],
        }]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {
                "declined": False,
                "revisions": [
                    {
                        "detection_id": "det-prov-1",
                        "proposed_track_id": "track-A",
                        "association_confidence": 0.82,
                        "depth_conflict_flag": False,
                        "position_conflict_flag": False,
                        "uncertainties": [],
                        "supporting_evidence": ["history_match"],
                    }
                ],
            }

        async def _observation_invoker(**kwargs):
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["association_confidence"] >= 0.8:
                return self._supported_verdict()
            return {
                "verification_status": "inconclusive",
                "visibility_status": "partial",
                "identity_consistency": "unknown",
                "possible_id_switch": True,
                "confidence": 0.4,
                "uncertainty": [],
                "limitations": ["crossing"],
                "supporting_evidence_refs": [],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": "collect_more_context",
                "visual_content_inspected": True,
            }

        result = asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
            )
        )

        det_payload = result["per_detection"]["det-prov-1"]
        self.assertEqual(det_payload["coordinator_decision"], updates[0])
        self.assertEqual(det_payload["tracking_reconsideration"]["coordinator_decision_round2"]["resolved_track_id"], "track-A")
        provenance = det_payload["evidence_provenance"]
        self.assertIn("tracking_llm_proposal", provenance)
        self.assertIn("observation_llm_verdict", provenance)
        self.assertIn("observation_llm_verdict_round2", provenance)

    def test_provider_call_telemetry_captures_rounds_and_usage(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-tel-1",
            yolo_track_id="track-B",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_YOLO_ID_SWAP,
            [
                self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0),
                self._history_record(track_id="track-B", depth_median=0.45, last_ground_x=2.0, last_ground_z=4.4),
            ],
        )
        proposals = {
            "det-tel-1": {
                "detection_id": "det-tel-1",
                "proposed_track_id": "track-B",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
                "alternative_track_candidates": [{"track_id": "track-A", "association_confidence": 0.8}],
            }
        }
        updates = [{
            "detection_id": "det-tel-1",
            "resolved_track_id": "track-B",
            "accepted": True,
            "rejection_reasons": [],
            "uncertainties": [],
            "supporting_evidence": ["appearance_match"],
        }]

        async def _tracking_reconsideration_invoker(**_kwargs):
            return {
                "declined": False,
                "revisions": [
                    {
                        "detection_id": "det-tel-1",
                        "proposed_track_id": "track-A",
                        "association_confidence": 0.82,
                        "depth_conflict_flag": False,
                        "position_conflict_flag": False,
                        "uncertainties": [],
                        "supporting_evidence": ["history_match"],
                    }
                ],
                phase1.DIAGNOSTIC_PROVIDER_CALL_META_KEY: {
                    "token_usage": {"prompt_tokens": 31, "completion_tokens": 11, "total_tokens": 42},
                    "usage_unavailable_reason": None,
                },
            }

        observation_call_counter = {"count": 0}

        async def _observation_invoker(**kwargs):
            observation_call_counter["count"] += 1
            proposal = kwargs["user_payload"]["tracking_proposal"]
            if proposal["association_confidence"] >= 0.8:
                payload = self._supported_verdict()
            else:
                payload = {
                    "verification_status": "inconclusive",
                    "visibility_status": "partial",
                    "identity_consistency": "unknown",
                    "possible_id_switch": True,
                    "confidence": 0.4,
                    "uncertainty": [],
                    "limitations": ["crossing"],
                    "supporting_evidence_refs": [],
                    "contradicting_evidence_refs": [],
                    "recommended_follow_up": "collect_more_context",
                    "visual_content_inspected": True,
                }
            payload[phase1.DIAGNOSTIC_PROVIDER_CALL_META_KEY] = {
                "token_usage": {
                    "prompt_tokens": 10 + observation_call_counter["count"],
                    "completion_tokens": 5,
                    "total_tokens": 15 + observation_call_counter["count"],
                },
                "usage_unavailable_reason": None,
            }
            return payload

        telemetry: list[dict] = []
        asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker,
                reconsideration_invoker=_tracking_reconsideration_invoker,
                provider_call_telemetry=telemetry,
            )
        )

        self.assertEqual(len(telemetry), 3)
        rounds = [entry["round"] for entry in telemetry]
        self.assertEqual(rounds, ["observation_round1", "tracking_reconsideration", "observation_round2"])
        self.assertTrue(all(entry["attempted"] for entry in telemetry))
        self.assertTrue(all(entry["usage_available"] for entry in telemetry))
        self.assertTrue(all(isinstance(entry["total_tokens"], int) and entry["total_tokens"] > 0 for entry in telemetry))

    def test_provider_call_telemetry_marks_usage_unavailable_without_zeroing(self) -> None:
        detection = self._make_detection_candidate(
            detection_id="det-tel-2",
            yolo_track_id="track-A",
            relative_depth=0.5,
            ground_x=1.0,
            ground_z=4.0,
        )
        scenario_metadata = self._scenario_metadata_with_tracks(
            phase1.SCENARIO_BASELINE,
            [self._history_record(track_id="track-A", depth_median=0.5, last_ground_x=1.0, last_ground_z=4.0)],
        )
        proposals = {
            "det-tel-2": {
                "detection_id": "det-tel-2",
                "proposed_track_id": "track-A",
                "association_confidence": 0.52,
                "occlusion_likelihood": 0.5,
                "id_switch_risk": 0.8,
                "depth_conflict_flag": False,
                "position_conflict_flag": True,
                "uncertainties": ["crossing_targets"],
                "supporting_evidence": ["appearance_match"],
            }
        }
        updates = [{
            "detection_id": "det-tel-2",
            "resolved_track_id": "track-A",
            "accepted": True,
            "rejection_reasons": [],
            "uncertainties": [],
            "supporting_evidence": ["appearance_match"],
        }]

        async def _observation_invoker_without_usage(**_kwargs):
            return {
                "verification_status": "supported",
                "visibility_status": "visible",
                "identity_consistency": "consistent",
                "possible_id_switch": False,
                "confidence": 0.92,
                "uncertainty": [],
                "limitations": [],
                "supporting_evidence_refs": ["evidence://motion"],
                "contradicting_evidence_refs": [],
                "recommended_follow_up": None,
                "visual_content_inspected": True,
            }

        telemetry: list[dict] = []
        asyncio.run(
            phase1._run_observation_collaboration_diagnostic(
                detections=[detection],
                proposal_by_detection=proposals,
                updates=updates,
                scenario_metadata=scenario_metadata,
                event_reasons=["crossing_targets"],
                frame_reference="data:image/jpeg;base64,abc",
                mode=phase1.OBSERVATION_VERIFICATION_MODE_MOCK,
                observation_model="qwen-3.8-27b",
                observation_timeout_seconds=6.0,
                invoker=_observation_invoker_without_usage,
                tracking_reconsideration_mode=phase1.TRACKING_RECONSIDERATION_MODE_OFF,
                provider_call_telemetry=telemetry,
            )
        )

        self.assertEqual(len(telemetry), 1)
        entry = telemetry[0]
        self.assertFalse(entry["usage_available"])
        self.assertIsNone(entry["input_tokens"])
        self.assertIsNone(entry["output_tokens"])
        self.assertIsNone(entry["total_tokens"])
        self.assertEqual(entry["usage_unavailable_reason"], "usage_not_exposed_by_invoker")

    def test_persisted_budget_preflight_shortfall_and_success(self) -> None:
        from unittest.mock import patch

        exhausted_state = {
            "state_path": "captures/provider_call_budget_state.json",
            "available": True,
            "error": None,
            "limits": {
                "total_limit": 3,
                "per_role_limits": {
                    "tracking_vlm": 1,
                    "observation_visual": 2,
                },
            },
            "totals": {
                "attempted": 3,
                "per_role_attempted": {
                    "tracking_vlm": 1,
                    "observation_visual": 2,
                },
            },
        }

        with patch("tools.tracking_agent_phase1_validation._read_persisted_provider_budget_state", return_value=exhausted_state):
            preflight = phase1._compute_persisted_budget_preflight(
                detection_count=2,
                observation_verification_mode=phase1.OBSERVATION_VERIFICATION_MODE_LIVE,
                tracking_reconsideration_mode=phase1.TRACKING_RECONSIDERATION_MODE_LIVE,
            )

        self.assertFalse(preflight["sufficient"])
        self.assertEqual(preflight["required"]["tracking_vlm"], 2)
        self.assertEqual(preflight["required"]["observation_visual"], 4)
        self.assertEqual(preflight["required"]["total"], 6)
        self.assertEqual(preflight["shortfall"]["tracking_vlm"], 2)
        self.assertEqual(preflight["shortfall"]["observation_visual"], 4)
        self.assertEqual(preflight["shortfall"]["total"], 6)

        sufficient_state = {
            "state_path": "captures/provider_call_budget_state.json",
            "available": True,
            "error": None,
            "limits": {
                "total_limit": 20,
                "per_role_limits": {
                    "tracking_vlm": 8,
                    "observation_visual": 12,
                },
            },
            "totals": {
                "attempted": 3,
                "per_role_attempted": {
                    "tracking_vlm": 1,
                    "observation_visual": 2,
                },
            },
        }

        with patch("tools.tracking_agent_phase1_validation._read_persisted_provider_budget_state", return_value=sufficient_state):
            preflight_ok = phase1._compute_persisted_budget_preflight(
                detection_count=2,
                observation_verification_mode=phase1.OBSERVATION_VERIFICATION_MODE_LIVE,
                tracking_reconsideration_mode=phase1.TRACKING_RECONSIDERATION_MODE_LIVE,
            )

        self.assertTrue(preflight_ok["sufficient"])
        self.assertEqual(preflight_ok["shortfall"]["tracking_vlm"], 0)
        self.assertEqual(preflight_ok["shortfall"]["observation_visual"], 0)
        self.assertEqual(preflight_ok["shortfall"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
