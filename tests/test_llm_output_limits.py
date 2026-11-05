from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import llm_call
from agents.contracts import BoundingBox, DetectionCandidate
from agents.observation import agent_observation
from agents.reasoning import agent_reasoning
from agents.tracking.agent_tracking import LLMVisualTrackingAgent


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response: object) -> None:
        self._completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self._completions)

    def close(self):
        return None


def _make_response(*, content: str, finish_reason: str = "stop") -> object:
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(choices=[choice], usage=None)


def _make_detection() -> DetectionCandidate:
    return DetectionCandidate(
        detection_id="det-1",
        timestamp=1000.0,
        yolo_track_id="track-1",
        label="person",
        detection_confidence=0.9,
        pixel_bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=20.0),
        pixel_center_x=5.0,
        pixel_center_y=10.0,
        relative_depth=0.4,
        ground_x=1.0,
        ground_z=2.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
    )


class TestLlmOutputLimits(unittest.TestCase):
    def test_classify_llm_error_output_truncated(self) -> None:
        code = llm_call.classify_llm_error(RuntimeError("llm_output_truncated"))
        self.assertEqual(code, "output_truncated")

    def test_resolve_max_output_tokens_from_env(self) -> None:
        with patch.dict(os.environ, {"TEST_MAX_OUTPUT": "321"}, clear=False):
            resolved = llm_call.resolve_max_output_tokens("TEST_MAX_OUTPUT", 128)
        self.assertEqual(resolved, 321)

    def test_resolve_max_output_tokens_falls_back_on_invalid(self) -> None:
        with patch.dict(os.environ, {"TEST_MAX_OUTPUT": "invalid"}, clear=False):
            resolved = llm_call.resolve_max_output_tokens("TEST_MAX_OUTPUT", 128)
        self.assertEqual(resolved, 128)

    def test_observation_invoke_raises_for_truncated_output(self) -> None:
        fake_client = _FakeClient(_make_response(content='{"summary":"x"}', finish_reason="length"))

        with patch("agents.observation.agent_observation._create_openai_client", return_value=fake_client):
            with patch("agents.observation.agent_observation.provider_call_budget.reserve_or_raise", return_value=object()):
                with patch("agents.observation.agent_observation.provider_call_budget.finalize_provider_call_attempt"):
                    with self.assertRaisesRegex(RuntimeError, "llm_output_truncated"):
                        asyncio.run(
                            agent_observation.invoke_llm_json_response(
                                model="gpt-oss-120b",
                                system_prompt="test",
                                user_payload={"track_id": "track-1"},
                                timeout_seconds=1.0,
                            )
                        )

    def test_reasoning_invoke_raises_for_truncated_output(self) -> None:
        fake_client = _FakeClient(_make_response(content='{"summary":"x"}', finish_reason="length"))

        with patch("agents.reasoning.agent_reasoning._create_openai_client", return_value=fake_client):
            with patch("agents.reasoning.agent_reasoning.provider_call_budget.reserve_or_raise", return_value=object()):
                with patch("agents.reasoning.agent_reasoning.provider_call_budget.finalize_provider_call_attempt"):
                    with self.assertRaisesRegex(RuntimeError, "llm_output_truncated"):
                        asyncio.run(
                            agent_reasoning.invoke_llm_json_response(
                                model="gpt-oss-120b",
                                system_prompt="test",
                                user_payload={"track_id": "track-1"},
                                timeout_seconds=1.0,
                            )
                        )

    def test_tracking_returns_output_truncated_error(self) -> None:
        fake_client = _FakeClient(_make_response(content='{"proposals":[]}', finish_reason="length"))
        agent = LLMVisualTrackingAgent(client=fake_client)

        with patch("agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise", return_value=object()):
            with patch("agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt"):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection()],
                        request_timeout_seconds=1.0,
                    )
                )

        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_output_truncated")
        self.assertIn("vlm_output_truncated", result.uncertainty_flags)


if __name__ == "__main__":
    unittest.main()