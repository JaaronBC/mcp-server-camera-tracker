# Handles test visual tracking runtime behavior.
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.contracts import BoundingBox, DetectionCandidate, VisualAssociationProposal
from agents.coordinator import AgentCoordinator
from agents.tracking import agent_tracking as tracking_module
from agents.tracking.agent_tracking import AsyncVisualTrackingRuntime, LLMVisualTrackingAgent, VisualTrackingAgentResult


class FakeVisualTrackingAgent:
    def __init__(self, result: VisualTrackingAgentResult, delay_seconds: float = 0.0, model: str = "qwen-3.8-27b"):
        # Performs init logic.
        self._result = result
        self._delay_seconds = delay_seconds
        self.model = model
        self.call_count = 0
        self.received_event_reasons: list[list[str]] = []

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult:
        # Performs propose associations logic.
        self.call_count += 1
        self.received_event_reasons.append(list(event_reasons or []))
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


class _SlowCompletions:
    async def create(self, **kwargs):
        await asyncio.sleep(0.05)
        return SimpleNamespace(choices=[])


class _SlowClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_SlowCompletions())


class _CapturingCompletions:
    def __init__(self) -> None:
        self.call_count = 0
        self.last_kwargs: dict[str, object] = {}

    async def create(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = dict(kwargs)

        class _Message:
            content = (
                '{"proposals":[{"detection_id":"det-1","proposed_track_id":"track-1",'
                '"association_confidence":0.81,"occlusion_likelihood":0.2,"id_switch_risk":0.3,'
                '"depth_conflict_flag":false,"position_conflict_flag":false,'
                '"uncertainties":[],"supporting_evidence":[]}],'
                '"uncertainty_flags":[],"supporting_evidence":[]}'
            )

        class _Choice:
            message = _Message()
            finish_reason = "stop"

        return SimpleNamespace(choices=[_Choice()])


class _CapturingClient:
    def __init__(self, completions: _CapturingCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


class _LoopTrackingCompletions:
    def __init__(self, *, sleep_seconds: float = 0.0, fail_on_call_number: int | None = None) -> None:
        self.call_count = 0
        self._sleep_seconds = sleep_seconds
        self._fail_on_call_number = fail_on_call_number
        self.loop_ids: list[int] = []
        self.follow_up_markers: list[bool] = []
        self.payload_texts: list[str] = []
        self._active_calls = 0
        self.max_active_calls = 0

    async def create(self, **kwargs):
        self.call_count += 1
        self._active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            self.loop_ids.append(id(asyncio.get_running_loop()))
            messages = kwargs.get("messages")
            text_payload = ""
            if isinstance(messages, list) and len(messages) >= 2 and isinstance(messages[1], dict):
                content = messages[1].get("content")
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    text = content[0].get("text")
                    if isinstance(text, str):
                        text_payload = text
            self.payload_texts.append(text_payload)
            self.follow_up_markers.append('"investigation_follow_up"' in text_payload)

            if self._fail_on_call_number is not None and self.call_count == self._fail_on_call_number:
                raise RuntimeError("synthetic_runtime_error")
            if self._sleep_seconds > 0:
                await asyncio.sleep(self._sleep_seconds)

            class _Message:
                content = '{"proposals":[],"uncertainty_flags":[],"supporting_evidence":[]}'

            class _Choice:
                message = _Message()
                finish_reason = "stop"

            return SimpleNamespace(choices=[_Choice()], usage=None)
        finally:
            self._active_calls = max(0, self._active_calls - 1)


class _LoopTrackingClient:
    def __init__(self, completions: _LoopTrackingCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.close_calls = 0
        self.close_loop_ids: list[int] = []

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_loop_ids.append(id(asyncio.get_running_loop()))


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


def _wait_for_observation(runtime: AsyncVisualTrackingRuntime, timeout_seconds: float = 1.0):
    observation = None
    deadline = time.time() + timeout_seconds
    while observation is None and time.time() < deadline:
        observation = runtime.poll_latest_observation()
        time.sleep(0.01)
    return observation


class TestVisualTrackingRuntime(unittest.TestCase):
    def test_tracking_skill_disabled_by_default_uses_hardcoded_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill_path = temp_path / "skills" / "object-identity-tracking" / "runtime.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("SKILL_PROMPT_MARKER", encoding="utf-8")

            with patch.object(tracking_module, "DEFAULT_TRACKING_SKILL_PATH", skill_path), patch.dict(
                os.environ, {"TRACKING_SKILL_ENABLED": "0"}, clear=False
            ):
                loaded = tracking_module.load_tracking_agent_skill()

        self.assertEqual(loaded, tracking_module.TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK)

    def test_tracking_skill_enabled_appends_skill_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill_path = temp_path / "skills" / "object-identity-tracking" / "runtime.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("SKILL_PROMPT_MARKER", encoding="utf-8")

            with patch.object(tracking_module, "DEFAULT_TRACKING_SKILL_PATH", skill_path), patch.dict(
                os.environ, {"TRACKING_SKILL_ENABLED": "1"}, clear=False
            ):
                loaded = tracking_module.load_tracking_agent_skill()

        self.assertIn(tracking_module.TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK, loaded)
        self.assertIn("SKILL_PROMPT_MARKER", loaded)

    def test_tracking_skill_enabled_does_not_duplicate_base_prompt_when_skill_matches_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill_path = temp_path / "skills" / "object-identity-tracking" / "runtime.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text(tracking_module.TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK, encoding="utf-8")

            with patch.object(tracking_module, "DEFAULT_TRACKING_SKILL_PATH", skill_path), patch.dict(
                os.environ, {"TRACKING_SKILL_ENABLED": "1"}, clear=False
            ):
                loaded = tracking_module.load_tracking_agent_skill()

        self.assertEqual(loaded, tracking_module.TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK)

    def test_tracking_skill_override_path_is_resolved_from_project_root(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRACKING_SKILL_PATH": "skills/object-identity-tracking/runtime.md",
            },
            clear=False,
        ):
            resolved = tracking_module._resolve_tracking_skill_path()

        expected = (tracking_module.PROJECT_ROOT / "skills" / "object-identity-tracking" / "runtime.md").resolve()
        self.assertEqual(resolved, expected)

    def test_tracking_skill_default_path_resolution(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            resolved = tracking_module._resolve_tracking_skill_path()
        self.assertEqual(resolved, tracking_module.DEFAULT_TRACKING_SKILL_PATH)

    def test_tracking_skill_invalid_file_falls_back_to_hardcoded_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            invalid_skill_path = temp_path / "invalid-skill.md"
            invalid_skill_path.write_bytes(b"\xff\xfe\xfa")

            with patch.dict(
                os.environ,
                {"TRACKING_SKILL_ENABLED": "1", "TRACKING_SKILL_PATH": str(invalid_skill_path)},
                clear=False,
            ):
                loaded = tracking_module.load_tracking_agent_skill()

        self.assertEqual(loaded, tracking_module.TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK)

    def test_tracking_skill_explicit_system_prompt_override_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill_path = temp_path / "skills" / "object-identity-tracking" / "runtime.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("SKILL_PROMPT_MARKER", encoding="utf-8")

            with patch.object(tracking_module, "DEFAULT_TRACKING_SKILL_PATH", skill_path), patch.dict(
                os.environ, {"TRACKING_SKILL_ENABLED": "1"}, clear=False
            ):
                agent = LLMVisualTrackingAgent(client=_SlowClient(), system_prompt="OVERRIDE_TRACKING_PROMPT")

        self.assertEqual(agent._system_prompt, "OVERRIDE_TRACKING_PROMPT")

    def test_tracking_skill_schema_instruction_authority_is_retained_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill_path = temp_path / "skills" / "object-identity-tracking" / "runtime.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("SKILL_PROMPT_MARKER", encoding="utf-8")

            with patch.object(tracking_module, "DEFAULT_TRACKING_SKILL_PATH", skill_path), patch.dict(
                os.environ, {"TRACKING_SKILL_ENABLED": "1"}, clear=False
            ):
                loaded = tracking_module.load_tracking_agent_skill()

        self.assertIn(
            "follow the existing JSON-only and response-schema requirements above",
            loaded,
        )

    def test_tracking_skill_does_not_change_provider_invocation_count(self) -> None:
        completions = _CapturingCompletions()
        client = _CapturingClient(completions)
        detection = make_detection()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            skill_path = temp_path / "skills" / "object-identity-tracking" / "runtime.md"
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            skill_path.write_text("SKILL_PROMPT_MARKER", encoding="utf-8")

            with patch.object(tracking_module, "DEFAULT_TRACKING_SKILL_PATH", skill_path), patch.dict(
                os.environ,
                {
                    "TRACKING_SKILL_ENABLED": "1",
                    "PROVIDER_CALL_BUDGET_ENABLED": "0",
                },
                clear=False,
            ):
                agent = LLMVisualTrackingAgent(client=client)
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[detection],
                        event_reasons=["count_check"],
                        request_timeout_seconds=0.2,
                    )
                )

        self.assertTrue(result.available)
        self.assertEqual(completions.call_count, 1)

    def test_prompt_includes_decision_contract_actions_and_evidence_guidance(self) -> None:
        completions = _CapturingCompletions()
        client = _CapturingClient(completions)
        detection = make_detection()

        with patch.dict(
            os.environ,
            {
                "TRACKING_SKILL_ENABLED": "0",
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
            },
            clear=False,
        ):
            agent = LLMVisualTrackingAgent(client=client)
            result = asyncio.run(
                agent.propose_associations(
                    frame_reference=None,
                    detections=[detection],
                    event_reasons=["count_check"],
                    request_timeout_seconds=0.2,
                )
            )

        self.assertTrue(result.available)
        messages = completions.last_kwargs.get("messages")
        self.assertIsInstance(messages, list)
        assert isinstance(messages, list)
        user_message = messages[1]
        content_list = user_message.get("content")
        self.assertIsInstance(content_list, list)
        assert isinstance(content_list, list)
        prompt_text = str(content_list[0].get("text", ""))
        self.assertIn("top-level decision object", prompt_text)
        self.assertIn("request_identity_snapshot", prompt_text)
        self.assertIn("request_additional_tracking_evidence", prompt_text)
        self.assertIn("propose_identity_association", prompt_text)
        self.assertIn("inconclusive", prompt_text)
        self.assertIn("do not guess", prompt_text)

    def test_compact_prompt_includes_exact_allowlisted_tool_names_and_mapping(self) -> None:
        completions = _CapturingCompletions()
        client = _CapturingClient(completions)
        detection = make_detection()

        with patch.dict(
            os.environ,
            {
                "TRACKING_SKILL_ENABLED": "0",
                "PROVIDER_CALL_BUDGET_ENABLED": "0",
                "TRACKING_AGENT_COMPACT_DECISION_MODE": "1",
            },
            clear=False,
        ):
            agent = LLMVisualTrackingAgent(client=client)
            result = asyncio.run(
                agent.propose_associations(
                    frame_reference=None,
                    detections=[detection],
                    event_reasons=["count_check"],
                    request_timeout_seconds=0.2,
                )
            )

        self.assertTrue(result.available)
        messages = completions.last_kwargs.get("messages")
        self.assertIsInstance(messages, list)
        assert isinstance(messages, list)
        user_message = messages[1]
        content_list = user_message.get("content")
        self.assertIsInstance(content_list, list)
        assert isinstance(content_list, list)
        prompt_text = str(content_list[0].get("text", ""))
        self.assertIn("Semantic-first decision contract: provide action, target selector, concise reason/uncertainty, and optional evidence_bounds.", prompt_text)
        self.assertIn("Runtime binds investigation_id, request_id, correlation_id, canonical tool_name, and canonical tool_requests structure.", prompt_text)
        self.assertIn("If you include explicit tool_name/tool/tool_request, it must exactly match action mapping or the decision is rejected.", prompt_text)
        self.assertIn(
            "Tool mapping is strict: request_identity_snapshot -> get_identity_snapshot; request_additional_tracking_evidence -> get_track_evidence_history.",
            prompt_text,
        )
        self.assertIn(
            "If including provenance, it must be a flat object with non-empty string keys and scalar values only (string, number, boolean, or null); do not use nested objects or arrays.",
            prompt_text,
        )
        self.assertIn("evidence_bounds must be an object when present.", prompt_text)
        self.assertIn(
            "For request_identity_snapshot, provide exactly one selector: app_identity_id or detection_reference.",
            prompt_text,
        )
        self.assertIn(
            "For request_additional_tracking_evidence, provide track_id.",
            prompt_text,
        )

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

    def test_tracking_continues_when_vlm_is_slow_with_decision_metadata(self) -> None:
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["slow_response"],
                supporting_evidence=[],
                latency_ms=1200.0,
                decision={
                    "action": "request_additional_tracking_evidence",
                    "investigation_id": "inv-slow-1",
                    "tool_requests": [
                        {
                            "request_id": "req-slow-1",
                            "tool_name": "get_track_evidence_history",
                            "arguments": {"track_id": "track-1"},
                        }
                    ],
                },
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

    def test_periodic_only_reasons_respect_minimum_interval(self) -> None:
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["periodic"],
                supporting_evidence=[],
                latency_ms=5.0,
            )
        )
        runtime = AsyncVisualTrackingRuntime(
            AgentCoordinator(),
            agent,
            min_invoke_interval_seconds=0.5,
        )
        detection = make_detection()
        baseline = time.time()

        started_first = runtime.submit_if_needed(
            now=baseline,
            frame_reference=None,
            detections=[detection],
            event_reasons=["periodic_visual_continuity_check"],
        )
        self.assertTrue(started_first)

        deadline = time.time() + 1.0
        while runtime.is_busy() and time.time() < deadline:
            time.sleep(0.01)

        started_second = runtime.submit_if_needed(
            now=baseline + 0.1,
            frame_reference=None,
            detections=[detection],
            event_reasons=["periodic_visual_continuity_check"],
        )
        self.assertFalse(started_second)

        started_third = runtime.submit_if_needed(
            now=baseline + 0.7,
            frame_reference=None,
            detections=[detection],
            event_reasons=["periodic_visual_continuity_check"],
        )
        self.assertTrue(started_third)
        runtime.shutdown()

    def test_event_driven_reasons_can_bypass_minimum_interval(self) -> None:
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["event"],
                supporting_evidence=[],
                latency_ms=5.0,
            )
        )
        runtime = AsyncVisualTrackingRuntime(
            AgentCoordinator(),
            agent,
            min_invoke_interval_seconds=5.0,
        )
        detection = make_detection()
        baseline = time.time()

        started_first = runtime.submit_if_needed(
            now=baseline,
            frame_reference=None,
            detections=[detection],
            event_reasons=["periodic_visual_continuity_check"],
        )
        self.assertTrue(started_first)

        deadline = time.time() + 1.0
        while runtime.is_busy() and time.time() < deadline:
            time.sleep(0.01)

        started_second = runtime.submit_if_needed(
            now=baseline + 0.1,
            frame_reference=None,
            detections=[detection],
            event_reasons=["possible_id_switch", "periodic_visual_continuity_check"],
        )
        self.assertTrue(started_second)
        runtime.shutdown()

    def test_repeated_periodic_submissions_do_not_trigger_extra_calls(self) -> None:
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["periodic"],
                supporting_evidence=[],
                latency_ms=5.0,
            )
        )
        runtime = AsyncVisualTrackingRuntime(
            AgentCoordinator(),
            agent,
            min_invoke_interval_seconds=0.5,
        )
        detection = make_detection()
        baseline = time.time()

        started_first = runtime.submit_if_needed(
            now=baseline,
            frame_reference=None,
            detections=[detection],
            event_reasons=["periodic_visual_continuity_check"],
        )
        self.assertTrue(started_first)

        deadline = time.time() + 1.0
        while runtime.is_busy() and time.time() < deadline:
            time.sleep(0.01)

        for offset in (0.1, 0.2, 0.3, 0.4):
            started = runtime.submit_if_needed(
                now=baseline + offset,
                frame_reference=None,
                detections=[detection],
                event_reasons=["periodic_visual_continuity_check"],
            )
            self.assertFalse(started)

        self.assertEqual(agent.call_count, 1)
        runtime.shutdown()

    def test_priority_event_does_not_replace_pending_request(self) -> None:
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

        started_first = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["periodic_visual_continuity_check"],
        )
        self.assertTrue(started_first)

        started_second = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["possible_id_switch"],
        )
        self.assertFalse(started_second)
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

    def test_success_observation_includes_ids(self) -> None:
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
                llm_correlation_id="corr-success",
            )
        )
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()
        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["id_check"],
        )

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.llm_correlation_id, "corr-success")
        self.assertIsInstance(observation.observation_id, str)
        self.assertTrue(bool(observation.observation_id))

    def test_runtime_observation_preserves_decision_metadata(self) -> None:
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["needs_more_evidence"],
                supporting_evidence=[],
                latency_ms=10.0,
                llm_correlation_id="corr-decision",
                decision={
                    "action": "request_identity_snapshot",
                    "investigation_id": "inv-1",
                    "tool_requests": [
                        {
                            "request_id": "req-1",
                            "tool_name": "get_identity_snapshot",
                            "arguments": {"app_identity_id": "app-1"},
                        }
                    ],
                },
            )
        )
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()
        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["possible_id_switch"],
        )

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertIsInstance(observation.decision, dict)
        assert isinstance(observation.decision, dict)
        self.assertEqual(observation.decision.get("action"), "request_identity_snapshot")
        self.assertEqual(observation.to_dict().get("decision", {}).get("investigation_id"), "inv-1")

    def test_failed_observation_includes_ids(self) -> None:
        agent = FakeVisualTrackingAgent(
            VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_unavailable"],
                supporting_evidence=[],
                latency_ms=None,
                error="vlm_unavailable",
                llm_correlation_id="corr-fail",
            )
        )
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent)
        detection = make_detection()
        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["error_check"],
        )

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(observation.llm_correlation_id, "corr-fail")
        self.assertIn("vlm_unavailable", observation.uncertainties)
        self.assertIsInstance(observation.observation_id, str)
        self.assertTrue(bool(observation.observation_id))

    def test_timeout_observation_includes_ids(self) -> None:
        runtime = AsyncVisualTrackingRuntime(
            AgentCoordinator(),
            LLMVisualTrackingAgent(client=_SlowClient()),
        )
        detection = make_detection()
        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["timeout_check"],
            request_timeout_seconds=0.001,
        )

        observation = None
        deadline = time.time() + 1.0
        while observation is None and time.time() < deadline:
            observation = runtime.poll_latest_observation()
            time.sleep(0.01)

        runtime.shutdown()
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertIn("vlm_timeout", observation.uncertainties)
        self.assertIsInstance(observation.llm_correlation_id, str)
        self.assertTrue(bool(observation.llm_correlation_id))
        self.assertIsInstance(observation.observation_id, str)
        self.assertTrue(bool(observation.observation_id))

    def test_runtime_reuses_single_worker_event_loop_for_initial_and_followup_calls(self) -> None:
        completions = _LoopTrackingCompletions()
        client = _LoopTrackingClient(completions)
        agent = LLMVisualTrackingAgent(client=client)
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent, min_invoke_interval_seconds=0.0)
        detection = make_detection()

        started_first = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["possible_id_switch"],
        )
        self.assertTrue(started_first)
        first_observation = _wait_for_observation(runtime)
        self.assertIsNotNone(first_observation)

        started_second = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["possible_id_switch"],
            follow_up_context={
                "investigation_id": "inv-runtime-loop",
                "request_id": "req-runtime-loop",
            },
        )
        self.assertTrue(started_second)
        second_observation = _wait_for_observation(runtime)
        runtime.shutdown()

        self.assertIsNotNone(second_observation)
        self.assertEqual(completions.call_count, 2)
        self.assertEqual(len(completions.loop_ids), 2)
        self.assertEqual(len(set(completions.loop_ids)), 1)
        self.assertEqual(completions.follow_up_markers, [False, True])

    def test_runtime_does_not_issue_concurrent_provider_calls(self) -> None:
        completions = _LoopTrackingCompletions(sleep_seconds=0.05)
        client = _LoopTrackingClient(completions)
        agent = LLMVisualTrackingAgent(client=client)
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent, min_invoke_interval_seconds=0.0)
        detection = make_detection()

        started_first = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["initial"],
        )
        self.assertTrue(started_first)

        started_second = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["second"],
        )
        self.assertFalse(started_second)
        self.assertIsNotNone(_wait_for_observation(runtime))
        runtime.shutdown()

        self.assertEqual(completions.call_count, 1)
        self.assertEqual(completions.max_active_calls, 1)

    def test_runtime_surfaces_runtime_error_as_vlm_unavailable_observation(self) -> None:
        completions = _LoopTrackingCompletions(fail_on_call_number=2)
        client = _LoopTrackingClient(completions)
        agent = LLMVisualTrackingAgent(client=client)
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent, min_invoke_interval_seconds=0.0)
        detection = make_detection()

        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["initial"],
        )
        self.assertIsNotNone(_wait_for_observation(runtime))

        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["follow_up"],
            follow_up_context={
                "investigation_id": "inv-runtime-error",
                "request_id": "req-runtime-error",
            },
        )
        observation = _wait_for_observation(runtime)
        runtime.shutdown()

        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertIn("vlm_unavailable", observation.uncertainties)

    def test_runtime_shutdown_closes_cached_client_on_worker_loop(self) -> None:
        completions = _LoopTrackingCompletions()
        client = _LoopTrackingClient(completions)
        agent = LLMVisualTrackingAgent(client=client)
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent, min_invoke_interval_seconds=0.0)
        detection = make_detection()

        runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["initial"],
        )
        self.assertIsNotNone(_wait_for_observation(runtime))
        runtime.shutdown()

        self.assertEqual(client.close_calls, 1)
        self.assertEqual(len(client.close_loop_ids), 1)
        self.assertEqual(len(completions.loop_ids), 1)
        self.assertEqual(client.close_loop_ids[0], completions.loop_ids[0])
        self.assertIsNone(agent._client)

    def test_runtime_does_not_submit_after_shutdown(self) -> None:
        completions = _LoopTrackingCompletions()
        client = _LoopTrackingClient(completions)
        agent = LLMVisualTrackingAgent(client=client)
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent, min_invoke_interval_seconds=0.0)
        detection = make_detection()

        runtime.shutdown()
        started = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["should_not_run"],
        )

        self.assertFalse(started)
        self.assertEqual(completions.call_count, 0)

    def test_second_provider_payload_contains_trusted_mcp_follow_up_context(self) -> None:
        completions = _LoopTrackingCompletions()
        client = _LoopTrackingClient(completions)
        agent = LLMVisualTrackingAgent(client=client)
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent, min_invoke_interval_seconds=0.0)
        detection = make_detection()

        first_started = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["initial"],
        )
        self.assertTrue(first_started)
        self.assertIsNotNone(_wait_for_observation(runtime))

        follow_up_context = {
            "investigation_id": "inv-synth-followup-1",
            "request_id": "req-synth-followup-1",
            "idempotency_key": "inv-synth-followup-1:req-synth-followup-1:get_track_evidence_history",
            "result_status": "ok",
            "tool_name": "get_track_evidence_history",
            "correlation_id": "corr-synth-followup-1",
            "initial_decision": {
                "action": "request_additional_tracking_evidence",
                "investigation_id": "inv-synth-followup-1",
            },
            "evidence_references": {
                "frame_reference": "evidence://frame/f-1",
                "detection_reference": "evidence://detection/d-1",
            },
            "mcp_result": {
                "available": True,
                "state": "no_recorded_observation",
                "track_id": "track-1",
                "messages": [],
            },
        }

        second_started = runtime.submit_if_needed(
            now=time.time(),
            frame_reference=None,
            detections=[detection],
            event_reasons=["tracking_decision_follow_up"],
            follow_up_context=follow_up_context,
        )
        self.assertTrue(second_started)
        self.assertIsNotNone(_wait_for_observation(runtime))
        runtime.shutdown()

        self.assertEqual(completions.call_count, 2)
        self.assertEqual(completions.follow_up_markers, [False, True])
        second_payload_text = completions.payload_texts[1]
        self.assertIn('"investigation_follow_up"', second_payload_text)
        self.assertIn('"investigation_id": "inv-synth-followup-1"', second_payload_text)
        self.assertIn('"request_id": "req-synth-followup-1"', second_payload_text)
        self.assertIn('"correlation_id": "corr-synth-followup-1"', second_payload_text)
        self.assertIn('"mcp_result"', second_payload_text)
        self.assertIn('"tool_name": "get_track_evidence_history"', second_payload_text)

    def test_runtime_emits_observation_published_diagnostic_with_runtime_markers(self) -> None:
        completions = _LoopTrackingCompletions()
        client = _LoopTrackingClient(completions)
        agent = LLMVisualTrackingAgent(client=client)
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), agent, min_invoke_interval_seconds=0.0)
        detection = make_detection()

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_vlm_attempt_diagnostics.jsonl"
            with patch.dict(
                os.environ,
                {
                    "TRACKING_VLM_ATTEMPT_DIAGNOSTICS_PATH": str(diagnostics_path),
                    "PROVIDER_CALL_BUDGET_ENABLED": "0",
                    "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
                },
                clear=False,
            ):
                started = runtime.submit_if_needed(
                    now=time.time(),
                    frame_reference=None,
                    detections=[detection],
                    event_reasons=["initial"],
                )
                self.assertTrue(started)
                self.assertIsNotNone(_wait_for_observation(runtime))
                runtime.shutdown()

            records = [
                json.loads(line)
                for line in diagnostics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        observation_records = [
            record
            for record in records
            if record.get("event_type") == "tracking_vlm_observation_published"
        ]
        attempt_records = [
            record
            for record in records
            if record.get("event_type") == "tracking_vlm_attempt_completed"
        ]
        self.assertTrue(attempt_records)
        self.assertTrue(observation_records)
        attempt_record = attempt_records[-1]
        observation_record = observation_records[-1]

        self.assertEqual(observation_record.get("attempt_id"), attempt_record.get("attempt_id"))
        self.assertEqual(observation_record.get("llm_correlation_id"), attempt_record.get("llm_correlation_id"))

        attempt_timing = attempt_record.get("timing_markers")
        self.assertIsInstance(attempt_timing, dict)
        assert isinstance(attempt_timing, dict)

        timing_markers = observation_records[-1].get("timing_markers")
        self.assertIsInstance(timing_markers, dict)
        assert isinstance(timing_markers, dict)
        runtime_candidate_submitted_at = timing_markers.get("runtime_candidate_submitted_at")
        runtime_worker_started_at = timing_markers.get("runtime_worker_started_at")
        provider_request_started_at = timing_markers.get("provider_request_started_at")
        provider_response_completed_at = timing_markers.get("provider_response_completed_at")
        observation_published_at = timing_markers.get("observation_published_at")

        self.assertIsInstance(runtime_candidate_submitted_at, float)
        self.assertIsInstance(runtime_worker_started_at, float)
        self.assertIsInstance(provider_request_started_at, float)
        self.assertIsInstance(provider_response_completed_at, float)
        self.assertIsInstance(observation_published_at, float)
        self.assertIsNone(timing_markers.get("provider_timeout_at"))

        assert isinstance(runtime_candidate_submitted_at, float)
        assert isinstance(runtime_worker_started_at, float)
        assert isinstance(provider_request_started_at, float)
        assert isinstance(provider_response_completed_at, float)
        assert isinstance(observation_published_at, float)
        self.assertLessEqual(runtime_candidate_submitted_at, runtime_worker_started_at)
        self.assertLessEqual(runtime_worker_started_at, provider_request_started_at)
        self.assertLessEqual(provider_request_started_at, provider_response_completed_at)
        self.assertLessEqual(provider_response_completed_at, observation_published_at)

        self.assertEqual(timing_markers.get("provider_request_started_at"), attempt_timing.get("provider_request_started_at"))
        self.assertEqual(
            timing_markers.get("provider_response_completed_at"),
            attempt_timing.get("provider_response_completed_at"),
        )


if __name__ == "__main__":
    unittest.main()
