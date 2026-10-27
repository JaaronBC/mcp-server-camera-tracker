# Handles test dashboard visual tracking behavior.
from __future__ import annotations

import unittest
from unittest.mock import patch

from tools import dashboardVisualTracking
from tools.agent_contracts import VisualAssociationProposal
from tools.visual_tracking_agent import VisualTrackingAgentResult


class FakeAgent:
    def __init__(self, model: str):
        # Performs init logic.
        self.model = model

    async def propose_associations(self, frame_reference, detections, event_reasons=None, request_timeout_seconds=1.0):
        # Performs propose associations logic.
        proposal = VisualAssociationProposal(
            detection_id=detections[0].detection_id,
            proposed_track_id="vlm-track-1",
            association_confidence=0.88,
            occlusion_likelihood=0.2,
            id_switch_risk=0.3,
            depth_conflict_flag=False,
            position_conflict_flag=False,
            uncertainties=[],
            supporting_evidence=["appearance_match"],
        )
        return VisualTrackingAgentResult(
            available=True,
            proposals=[proposal],
            uncertainty_flags=[],
            supporting_evidence=["appearance_match"],
            latency_ms=12.0,
        )


class TestDashboardVisualTracking(unittest.TestCase):
    def test_simulate_visual_tracking_association_returns_validated_updates(self) -> None:
        # Performs test simulate visual tracking association returns validated updates logic.
        detections = [
            {
                "detection_id": "det-1",
                "timestamp": 1000.0,
                "yolo_track_id": "yolo-1",
                "label": "person",
                "detection_confidence": 0.9,
                "pixel_bbox": {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 30.0},
                "pixel_center_x": 15.0,
                "pixel_center_y": 30.0,
                "relative_depth": None,
                "ground_x": 1.0,
                "ground_z": 2.0,
                "ground_coordinate_unit": "relative_units",
                "position_kind": "relative_ground_estimate",
            }
        ]

        with patch("tools.dashboardVisualTracking.LLMVisualTrackingAgent", FakeAgent):
            result = dashboardVisualTracking.simulate_visual_tracking_association(
                detections=detections,
                frame_reference="data:image/jpeg;base64,abc",
                event_reasons=["possible_id_switch"],
                timeout_seconds=1.0,
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["proposal_count"], 1)
        updates = result["observation"]["updates"]
        self.assertEqual(updates[0]["resolved_track_id"], "vlm-track-1")
        self.assertTrue(updates[0]["accepted"])


if __name__ == "__main__":
    unittest.main()
