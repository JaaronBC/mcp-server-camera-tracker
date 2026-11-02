# Handles test visual tracking runtime behavior.
from __future__ import annotations

import asyncio
import time
import unittest

from agents.contracts import BoundingBox, DetectionCandidate, VisualAssociationProposal
from agents.coordinator import AgentCoordinator
from agents.tracking.agent_tracking import AsyncVisualTrackingRuntime, VisualTrackingAgentResult


class FakeVisualTrackingAgent:
    def __init__(self, result: VisualTrackingAgentResult, delay_seconds: float = 0.0, model: str = "qwen-3.8-27b"):
        # Performs init logic.
        self._result = result
        self._delay_seconds = delay_seconds
        self.model = model

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult:
        # Performs propose associations logic.
        if self._delay_seconds > 0:
            await asyncio.sleep(self._delay_seconds)
        return self._result


class SequenceVisualTrackingAgent:
    def __init__(self, results: list[VisualTrackingAgentResult], model: str = "qwen-3.8-27b") -> None:
        self._results = list(results)
        self._index = 0
        self.model = model

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult:
        if self._index >= len(self._results):
            return self._results[-1]
        value = self._results[self._index]
        self._index += 1
        return value


def make_detection(relative_depth: float | None = None) -> DetectionCandidate:
    # Performs make detection logic.
    return DetectionCandidate(
        detection_id="det-1",
        timestamp=1000.0,
        yolo_track_id="yolo-7",
        label="person",
        detection_confidence=0.91,
        pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0),
        pixel_center_x=30.0,
        pixel_center_y=80.0,
        relative_depth=relative_depth,
        ground_x=1.0,
        ground_z=2.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
    )


class TestVisualTrackingRuntime(unittest.TestCase):
    def test_vlm_assisted_association_is_applied(self) -> None:
        # Performs test vlm assisted association is applied logic.
        proposal = VisualAssociationProposal(
            detection_id="det-1",
            proposed_track_id="vlm-track-2",
            association_confidence=0.86,
            occlusion_likelihood=0.1,
            id_switch_risk=0.2,
            depth_conflict_flag=False,
            position_conflict_flag=False,
            uncertainties=[],
            supporting_evidence=["appearance_match"],
        )
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[proposal],
                uncertainty_flags=[],
                supporting_evidence=["appearance_match"],
                latency_ms=8.4,
            )
        )

        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()

        started = runtime.submit_if_needed(
            now=time.time(),
            frame_reference="data:image/jpeg;base64,abc",
            detections=[detection],
            event_reasons=["possible_id_switch"],
        )
        self.assertTrue(started)

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.vlm_model, "qwen-3.8-27b")
        self.assertEqual(observation.updates[0].resolved_track_id, "vlm-track-2")
        self.assertTrue(observation.updates[0].accepted)

    def test_depth_conflict_is_rejected_and_falls_back(self) -> None:
        # Performs test depth conflict is rejected and falls back logic.
        proposal = VisualAssociationProposal(
            detection_id="det-1",
            proposed_track_id="vlm-track-2",
            association_confidence=0.91,
            occlusion_likelihood=0.3,
            id_switch_risk=0.7,
            depth_conflict_flag=True,
            position_conflict_flag=False,
            uncertainties=[],
            supporting_evidence=["ambiguous_motion"],
        )
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[proposal],
                uncertainty_flags=[],
                supporting_evidence=["ambiguous_motion"],
                latency_ms=5.0,
            )
        )

        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection(relative_depth=4.2)
        runtime.submit_if_needed(
            now=time.time(),
            frame_reference="data:image/jpeg;base64,abc",
            detections=[detection],
            event_reasons=["depth_conflict"],
        )

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertFalse(observation.updates[0].accepted)
        self.assertEqual(observation.updates[0].resolved_track_id, "yolo-7")
        self.assertIn("depth_conflict_with_reliable_measurement", observation.updates[0].rejection_reasons)

    def test_tracking_continues_when_vlm_is_slow(self) -> None:
        # Performs test tracking continues when vlm is slow logic.
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["slow_response"],
                supporting_evidence=[],
                latency_ms=1200.0,
            ),
            delay_seconds=0.2,
        )

        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()

        started_at = time.perf_counter()
        started = runtime.submit_if_needed(
            now=time.time(),
            frame_reference="data:image/jpeg;base64,abc",
            detections=[detection],
            event_reasons=["periodic_visual_continuity_check"],
        )
        elapsed = time.perf_counter() - started_at

        self.assertTrue(started)
        self.assertLess(elapsed, 0.05)
        self.assertTrue(runtime.is_busy())

        runtime.shutdown()

    def test_tracking_continues_when_vlm_unavailable(self) -> None:
        # Performs test tracking continues when vlm unavailable logic.
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_unavailable"],
                supporting_evidence=[],
                latency_ms=None,
                error="vlm_unavailable",
            )
        )

        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()
        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["low_detection_confidence"],
        )

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertIn("vlm_unavailable", observation.uncertainties)
        self.assertEqual(observation.updates[0].resolved_track_id, "yolo-7")
        self.assertIn("no_vlm_proposal_used", observation.updates[0].uncertainties)

    def test_completed_result_is_harvested_before_next_submit(self) -> None:
        first = VisualTrackingAgentResult(
            available=False,
            proposals=[],
            uncertainty_flags=["vlm_api_key_missing"],
            supporting_evidence=[],
            latency_ms=None,
            error="vlm_api_key_missing",
        )
        second = VisualTrackingAgentResult(
            available=True,
            proposals=[],
            uncertainty_flags=["no_detections"],
            supporting_evidence=[],
            latency_ms=3.2,
        )
        agent = SequenceVisualTrackingAgent([first, second])
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()

        started_first = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["first"],
        )
        self.assertTrue(started_first)

        deadline = time.time() + 1.0
        while runtime.is_busy() and time.time() < deadline:
            time.sleep(0.01)

        started_second = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["second"],
        )
        self.assertTrue(started_second)

        first_observation = runtime.poll_latest_observation()
        self.assertIsNotNone(first_observation)
        assert first_observation is not None
        self.assertIn("vlm_api_key_missing", first_observation.uncertainties)

        second_observation = None
        deadline = time.time() + 1.0
        while second_observation is None and time.time() < deadline:
            second_observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(second_observation)

    def test_pending_request_is_not_replaced(self) -> None:
        slow_agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=[],
                supporting_evidence=[],
                latency_ms=20.0,
            ),
            delay_seconds=0.2,
        )
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), slow_agent)
        detection = make_detection()

        started = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["initial"],
        )
        self.assertTrue(started)

        started_second = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["second"],
        )
        self.assertFalse(started_second)
        runtime.shutdown()

    def test_observation_consumed_exactly_once(self) -> None:
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_unavailable"],
                supporting_evidence=[],
                latency_ms=None,
                error="vlm_unavailable",
            )
        )
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()

        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["once"],
        )

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        self.assertIsNotNone(observation)
        self.assertIsNone(runtime.poll_latest_observation())
        runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
