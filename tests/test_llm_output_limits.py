from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from openai import APIConnectionError, AuthenticationError, BadRequestError, RateLimitError

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
        if isinstance(self._response, Exception):
            raise self._response
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


def _make_status_error(status_code: int, code: str, *, error_type: str = "invalid_request_error") -> Exception:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx2.Response(
        status_code=status_code,
        request=request,
        json={
            "code": code,
            "type": error_type,
            "message": "sensitive details should not be logged",
        },
    )
    if status_code == 400:
        return BadRequestError("bad request", response=response, body=response.json())
    if status_code == 401:
        return AuthenticationError("unauthorized", response=response, body=response.json())
    if status_code == 429:
        return RateLimitError("rate limited", response=response, body=response.json())
    raise ValueError(f"unsupported status code: {status_code}")


def _make_connection_error() -> Exception:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    return APIConnectionError(message="connection failed", request=request)


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

    def test_tracking_accepts_missing_keys_as_backward_compatible(self) -> None:
        fake_client = _FakeClient(_make_response(content='{"proposals":[]}'))
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

        self.assertTrue(result.available)
        self.assertIsNone(result.error)
        self.assertEqual(result.proposals, [])
        self.assertEqual(result.supporting_evidence, [])
        self.assertIn("schema_incomplete_response", result.uncertainty_flags)
        self.assertNotIn("vlm_invalid_schema_response", result.uncertainty_flags)

    def test_tracking_accepts_incorrect_top_level_types_as_backward_compatible(self) -> None:
        fake_client = _FakeClient(
            _make_response(
                content=(
                    '{"proposals":{},"uncertainty_flags":"bad","supporting_evidence":null}'
                )
            )
        )
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

        self.assertTrue(result.available)
        self.assertIsNone(result.error)
        self.assertEqual(result.proposals, [])
        self.assertEqual(result.supporting_evidence, [])
        self.assertEqual(result.uncertainty_flags, ["schema_incomplete_response"])

    def test_tracking_marks_complete_schema_response(self) -> None:
        fake_client = _FakeClient(
            _make_response(
                content=(
                    '{"proposals":[{"detection_id":"det-1","proposed_track_id":"track-1",'
                    '"association_confidence":0.9,"occlusion_likelihood":0.1,"id_switch_risk":0.2,'
                    '"depth_conflict_flag":false,"position_conflict_flag":false,'
                    '"uncertainties":[],"supporting_evidence":["appearance_match"]}],'
                    '"uncertainty_flags":[],"supporting_evidence":["frame_consistent"]}'
                )
            )
        )
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

        self.assertTrue(result.available)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.proposals[0].detection_id, "det-1")
        self.assertEqual(result.supporting_evidence, ["frame_consistent"])
        self.assertIn("schema_complete_response", result.uncertainty_flags)
        self.assertNotIn("schema_incomplete_response", result.uncertainty_flags)

    def test_tracking_runtime_error_telemetry_http_400(self) -> None:
        fake_client = _FakeClient(_make_status_error(400, "invalid_image_data"))
        agent = LLMVisualTrackingAgent(client=fake_client)
        captured_events: list[tuple[tuple[object, ...], dict[str, object]]] = []

        with patch("agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise", return_value=object()):
            with patch("agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt"):
                with patch(
                    "agents.tracking.agent_tracking.latency_instrumentation.emit_event",
                    side_effect=lambda *a, **k: captured_events.append((a, k)),
                ):
                    result = asyncio.run(
                        agent.propose_associations(
                            frame_reference=None,
                            detections=[_make_detection()],
                            request_timeout_seconds=1.0,
                        )
                    )

        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_unavailable")
        completed = captured_events[-1][1]
        self.assertEqual(completed.get("status"), "error")
        self.assertEqual(completed.get("error_code"), "runtime_error")
        self.assertEqual(completed.get("sdk_exception_class"), "BadRequestError")
        self.assertEqual(completed.get("http_status_code"), 400)
        self.assertEqual(completed.get("provider_error_code"), "invalid_image_data")
        self.assertNotIn("message", completed)

    def test_tracking_runtime_error_telemetry_http_401(self) -> None:
        fake_client = _FakeClient(_make_status_error(401, "invalid_api_key"))
        agent = LLMVisualTrackingAgent(client=fake_client)
        captured_events: list[tuple[tuple[object, ...], dict[str, object]]] = []

        with patch("agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise", return_value=object()):
            with patch("agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt"):
                with patch(
                    "agents.tracking.agent_tracking.latency_instrumentation.emit_event",
                    side_effect=lambda *a, **k: captured_events.append((a, k)),
                ):
                    result = asyncio.run(
                        agent.propose_associations(
                            frame_reference=None,
                            detections=[_make_detection()],
                            request_timeout_seconds=1.0,
                        )
                    )

        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_unavailable")
        completed = captured_events[-1][1]
        self.assertEqual(completed.get("sdk_exception_class"), "AuthenticationError")
        self.assertEqual(completed.get("http_status_code"), 401)
        self.assertEqual(completed.get("provider_error_code"), "invalid_api_key")

    def test_tracking_runtime_error_telemetry_http_429(self) -> None:
        fake_client = _FakeClient(_make_status_error(429, "rate_limit_exceeded"))
        agent = LLMVisualTrackingAgent(client=fake_client)
        captured_events: list[tuple[tuple[object, ...], dict[str, object]]] = []

        with patch("agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise", return_value=object()):
            with patch("agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt"):
                with patch(
                    "agents.tracking.agent_tracking.latency_instrumentation.emit_event",
                    side_effect=lambda *a, **k: captured_events.append((a, k)),
                ):
                    result = asyncio.run(
                        agent.propose_associations(
                            frame_reference=None,
                            detections=[_make_detection()],
                            request_timeout_seconds=1.0,
                        )
                    )

        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_unavailable")
        completed = captured_events[-1][1]
        self.assertEqual(completed.get("sdk_exception_class"), "RateLimitError")
        self.assertEqual(completed.get("http_status_code"), 429)
        self.assertEqual(completed.get("provider_error_code"), "rate_limit_exceeded")

    def test_tracking_runtime_error_telemetry_connection_error(self) -> None:
        fake_client = _FakeClient(_make_connection_error())
        agent = LLMVisualTrackingAgent(client=fake_client)
        captured_events: list[tuple[tuple[object, ...], dict[str, object]]] = []

        with patch("agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise", return_value=object()):
            with patch("agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt"):
                with patch(
                    "agents.tracking.agent_tracking.latency_instrumentation.emit_event",
                    side_effect=lambda *a, **k: captured_events.append((a, k)),
                ):
                    result = asyncio.run(
                        agent.propose_associations(
                            frame_reference=None,
                            detections=[_make_detection()],
                            request_timeout_seconds=1.0,
                        )
                    )

        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_unavailable")
        completed = captured_events[-1][1]
        self.assertEqual(completed.get("sdk_exception_class"), "APIConnectionError")
        self.assertNotIn("http_status_code", completed)
        self.assertNotIn("provider_error_code", completed)

    def test_tracking_success_event_unchanged(self) -> None:
        fake_client = _FakeClient(_make_response(content='{"proposals":[],"uncertainty_flags":[],"supporting_evidence":[]}'))
        agent = LLMVisualTrackingAgent(client=fake_client)
        captured_events: list[tuple[tuple[object, ...], dict[str, object]]] = []

        with patch("agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise", return_value=object()):
            with patch("agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt"):
                with patch(
                    "agents.tracking.agent_tracking.latency_instrumentation.emit_event",
                    side_effect=lambda *a, **k: captured_events.append((a, k)),
                ):
                    result = asyncio.run(
                        agent.propose_associations(
                            frame_reference=None,
                            detections=[_make_detection()],
                            request_timeout_seconds=1.0,
                        )
                    )

        self.assertTrue(result.available)
        completed = captured_events[-1][1]
        self.assertEqual(completed.get("status"), "ok")
        self.assertNotIn("sdk_exception_class", completed)
        self.assertNotIn("http_status_code", completed)
        self.assertNotIn("provider_error_code", completed)

    def test_tracking_runtime_error_telemetry_extraction_failure_does_not_interrupt(self) -> None:
        fake_client = _FakeClient(_make_connection_error())
        agent = LLMVisualTrackingAgent(client=fake_client)

        with patch("agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise", return_value=object()):
            with patch("agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt"):
                with patch("agents.tracking.agent_tracking._extract_exception_telemetry_fields", side_effect=RuntimeError("boom")):
                    result = asyncio.run(
                        agent.propose_associations(
                            frame_reference=None,
                            detections=[_make_detection()],
                            request_timeout_seconds=1.0,
                        )
                    )

        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_unavailable")


if __name__ == "__main__":
    unittest.main()