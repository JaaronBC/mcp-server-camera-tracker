from __future__ import annotations

import copy
import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.contracts import BoundingBox, DetectionCandidate, TrackingAgentObservation, VisualAssociationUpdate
from agents.coordinator import AgentCoordinator
from agents.dashboard.state import SharedDashboardState
from agents.dashboard import tracking_loop
from agents.tracking.agent_tracking import AsyncVisualTrackingRuntime, VisualTrackingAgentResult
from agents.tracking.tracking_types import GroundPosition, RadarTrackSnapshot


class _FakeSubmitTransport:
    def __init__(self, accepted: bool = True, duplicate: bool = False) -> None:
        self.accepted = accepted
        self.duplicate = duplicate
        self.calls: list[dict] = []

    def dashboard_submit_decision_request(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "accepted": self.accepted,
            "duplicate": self.duplicate,
            "rejected": False,
            "reason": None,
        }


class _FakeResultTransport:
    def __init__(self, results: list[dict]) -> None:
        self._results = list(results)

    def dashboard_poll_results(self, now_timestamp: float | None = None) -> dict:
        return {
            "available": True,
            "result_count": len(self._results),
            "results": [copy.deepcopy(item) for item in self._results],
        }


class _FakeRuntime:
    def __init__(self, observation: TrackingAgentObservation | None = None) -> None:
        self._observation = observation

    def poll_latest_observation(self):
        observation = self._observation
        self._observation = None
        return observation


class _ScriptedRuntimeAgent:
    def __init__(self, scripted_results: list[VisualTrackingAgentResult]) -> None:
        self.model = "qwen-3.8-27b"
        self._scripted_results = list(scripted_results)
        self._index = 0
        self.call_started_at: list[float] = []
        self.call_follow_up_contexts: list[dict | None] = []

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        follow_up_context: dict[str, object] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult:
        self.call_started_at.append(time.time())
        self.call_follow_up_contexts.append(dict(follow_up_context) if isinstance(follow_up_context, dict) else None)

        if self._index >= len(self._scripted_results):
            return self._scripted_results[-1]
        result = self._scripted_results[self._index]
        self._index += 1
        return result


def _track_snapshot() -> RadarTrackSnapshot:
    return RadarTrackSnapshot(
        track_id="det-1",
        timestamp=1000.0,
        pixel_position=None,
        ground_position=GroundPosition(x=1.0, z=2.0),
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        quality_flags=[],
        confidence=0.9,
    )


def _detection_candidate(*, detection_id: str = "det-1") -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        timestamp=1000.0,
        yolo_track_id="yolo-7",
        label="person",
        detection_confidence=0.93,
        pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=90.0),
        pixel_center_x=30.0,
        pixel_center_y=90.0,
        relative_depth=0.4,
        ground_x=1.0,
        ground_z=2.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
    )


def _wait_for_observation(runtime: AsyncVisualTrackingRuntime, timeout_seconds: float = 1.0):
    deadline = time.time() + timeout_seconds
    observation = None
    while time.time() < deadline:
        observation = runtime.poll_latest_observation()
        if observation is not None:
            return observation
        time.sleep(0.01)
    return observation


def _wait_until_runtime_idle(runtime: AsyncVisualTrackingRuntime, timeout_seconds: float = 1.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not runtime.is_busy():
            return True
        time.sleep(0.01)
    return False


def _base_investigation_state() -> dict:
    return {
        "investigation_id": "inv-1",
        "request_id": "req-1",
        "tool_name": "get_track_evidence_history",
        "correlation_id": "corr-1",
        "initial_decision": {
            "action": "request_additional_tracking_evidence",
            "investigation_id": "inv-1",
            "correlation_id": "corr-1",
            "evidence_references": {
                "frame_reference": "evidence://frame/f-1",
                "detection_reference": "evidence://detection/d-1",
                "tracking_message_id": "trkmsg-1",
            },
            "tool_requests": [
                {
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "arguments": {"track_id": "track-1"},
                }
            ],
        },
        "evidence_references": {
            "frame_reference": "evidence://frame/f-1",
            "detection_reference": "evidence://detection/d-1",
            "tracking_message_id": "trkmsg-1",
        },
        "tool_calls": 1,
        "iterations": 1,
        "terminal": False,
        "follow_up_submitted": False,
        "follow_up_completed": False,
        "result_key": "rk-1",
        "result_payload": {"available": True},
    }


def _follow_up_observation(*, with_llm_proposal: bool, decision: dict | None = None) -> TrackingAgentObservation:
    update_uncertainties = [] if with_llm_proposal else ["no_vlm_proposal_used"]
    update = VisualAssociationUpdate(
        detection_id="det-1",
        resolved_track_id="track-merged",
        accepted=True,
        rejection_reasons=[],
        uncertainties=update_uncertainties,
        supporting_evidence=["evidence_ref"],
    )
    return TrackingAgentObservation(
        timestamp=1001.0,
        source_agent="visual_tracking",
        vlm_model="qwen-3.8-27b",
        proposal_latency_ms=10.0,
        frame_reference=None,
        updates=[update],
        uncertainties=[
            "schema_complete_response",
            "followup_investigation_id:inv-1",
            "followup_request_id:req-1",
            "followup_result_key:rk-1",
            "followup_result_status:ok",
        ],
        observation_id="obs-followup",
        llm_correlation_id="corr-followup",
        decision=decision,
    )


def _initial_decision_observation(*, investigation_id: str = "inv-2") -> TrackingAgentObservation:
    return TrackingAgentObservation(
        timestamp=1001.5,
        source_agent="visual_tracking",
        vlm_model="qwen-3.8-27b",
        proposal_latency_ms=10.0,
        frame_reference=None,
        updates=[],
        uncertainties=["schema_complete_response"],
        observation_id="obs-initial-new",
        llm_correlation_id="corr-initial-new",
        decision={
            "action": "request_identity_snapshot",
            "investigation_id": investigation_id,
            "correlation_id": "corr-new",
            "tool_requests": [
                {
                    "request_id": "req-new",
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"detection_reference": "fdet-1", "max_candidates": 4},
                }
            ],
        },
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


class TestTrackingDecisionFollowUp(unittest.TestCase):
    def test_submit_registers_initial_investigation(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_ipc_transport = transport

        observation = types.SimpleNamespace(
            decision={
                "action": "request_additional_tracking_evidence",
                "investigation_id": "inv-1",
                "correlation_id": "corr-1",
                "evidence_references": {"frame_reference": "evidence://frame/f-1"},
                "tool_requests": [
                    {
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {"track_id": "track-1"},
                    }
                ],
            },
            llm_correlation_id="corr-llm-1",
            uncertainties=[],
        )

        tracking_loop._submit_tracking_decision_request_if_enabled(
            state=state,
            tracking_agent_observation=observation,
            now=1000.0,
        )

        self.assertEqual(len(transport.calls), 1)
        self.assertIn("inv-1", state.tracking_decision_investigations)
        investigation = state.tracking_decision_investigations["inv-1"]
        self.assertEqual(investigation["request_id"], "req-1")
        self.assertFalse(investigation["terminal"])

    def test_successful_result_schedules_follow_up_with_continuity(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-1",
                    "result_payload": {"available": True, "messages": [{"message_id": "m-1"}]},
                }
            ]
        )
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        captured_context: dict[str, object] = {}

        def _capture_submit(**kwargs):
            follow_up = kwargs.get("follow_up_context")
            if isinstance(follow_up, dict):
                captured_context.update(follow_up)
            return True

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", side_effect=_capture_submit):
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

        self.assertEqual(captured_context.get("investigation_id"), "inv-1")
        self.assertEqual(captured_context.get("request_id"), "req-1")
        self.assertEqual(captured_context.get("idempotency_key"), "rk-1")
        self.assertEqual(captured_context.get("correlation_id"), "corr-1")
        self.assertIsInstance(captured_context.get("initial_decision"), dict)
        self.assertIsInstance(captured_context.get("evidence_references"), dict)
        self.assertIsInstance(captured_context.get("mcp_result"), dict)

    def test_follow_up_context_includes_observation_llm_assessment_payload(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-1",
                    "result_payload": {
                        "available": True,
                        "messages": [{"message_id": "m-1"}],
                        "observation_llm_follow_up_assessment": {
                            "available": True,
                            "state": "observation_follow_up_completed",
                            "uncertainties": ["occlusion_overlap"],
                            "needs_more_observations": True,
                            "decision": {"action": "inconclusive", "reason": "ambiguous_motion"},
                            "supporting_observation_ids": ["obs-9"],
                            "provenance": {
                                "source": "observation_follow_up_llm",
                                "identity_verification_authorized": False,
                                "identity_mutation_authorized": False,
                            },
                        },
                    },
                }
            ]
        )
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        captured_context: dict[str, object] = {}

        def _capture_submit(**kwargs):
            follow_up = kwargs.get("follow_up_context")
            if isinstance(follow_up, dict):
                captured_context.update(follow_up)
            return True

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", side_effect=_capture_submit):
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

        mcp_result = captured_context.get("mcp_result")
        self.assertIsInstance(mcp_result, dict)
        assert isinstance(mcp_result, dict)
        assessment = mcp_result.get("observation_llm_follow_up_assessment")
        self.assertIsInstance(assessment, dict)
        assert isinstance(assessment, dict)
        self.assertEqual(assessment.get("state"), "observation_follow_up_completed")
        self.assertIn("occlusion_overlap", assessment.get("uncertainties", []))
        self.assertEqual(assessment.get("decision", {}).get("action"), "inconclusive")
        self.assertFalse(assessment.get("provenance", {}).get("identity_verification_authorized", True))
        self.assertFalse(assessment.get("provenance", {}).get("identity_mutation_authorized", True))

    def test_late_observation_assessment_is_processed_after_mcp_result_consumed(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        captured_contexts: list[dict[str, object]] = []

        def _capture_submit(**kwargs):
            follow_up = kwargs.get("follow_up_context")
            if isinstance(follow_up, dict):
                captured_contexts.append(dict(follow_up))
            return len(captured_contexts) >= 2

        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "delivery_kind": "mcp_result",
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-1",
                    "result_payload": {"available": True, "messages": [{"message_id": "m-1"}]},
                }
            ]
        )

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", side_effect=_capture_submit):
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

            state.tracking_decision_ipc_transport = _FakeResultTransport(
                [
                    {
                        "delivery_kind": "observation_follow_up_assessment",
                        "investigation_id": "inv-1",
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "status": "ok",
                        "idempotency_key": "rk-1:observation_follow_up_assessment",
                        "result_payload": {
                            "observation_llm_follow_up_assessment": {
                                "available": True,
                                "state": "observation_follow_up_completed",
                                "supporting_observation_ids": ["obs-2"],
                                "uncertainties": ["occlusion_overlap"],
                                "provenance": {
                                    "identity_verification_authorized": False,
                                    "identity_mutation_authorized": False,
                                },
                            }
                        },
                    }
                ]
            )
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.1,
            )

        self.assertEqual(len(captured_contexts), 2)
        late_payload = captured_contexts[1].get("mcp_result")
        self.assertIsInstance(late_payload, dict)
        assert isinstance(late_payload, dict)
        assessment = late_payload.get("observation_llm_follow_up_assessment")
        self.assertIsInstance(assessment, dict)
        assert isinstance(assessment, dict)
        self.assertIn("obs-2", assessment.get("supporting_observation_ids", []))
        self.assertIn("occlusion_overlap", assessment.get("uncertainties", []))
        self.assertFalse(assessment.get("provenance", {}).get("identity_mutation_authorized", True))

    def test_late_assessment_does_not_resubmit_when_follow_up_already_submitted(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            state.tracking_decision_ipc_transport = _FakeResultTransport(
                [
                    {
                        "delivery_kind": "mcp_result",
                        "investigation_id": "inv-1",
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "status": "ok",
                        "idempotency_key": "rk-1",
                        "result_payload": {"available": True},
                    }
                ]
            )
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

            state.tracking_decision_ipc_transport = _FakeResultTransport(
                [
                    {
                        "delivery_kind": "observation_follow_up_assessment",
                        "investigation_id": "inv-1",
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "status": "ok",
                        "idempotency_key": "rk-1:observation_follow_up_assessment",
                        "result_payload": {
                            "observation_llm_follow_up_assessment": {
                                "available": True,
                                "state": "observation_follow_up_completed",
                                "provenance": {
                                    "identity_verification_authorized": False,
                                    "identity_mutation_authorized": False,
                                },
                            }
                        },
                    }
                ]
            )
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.1,
            )

        self.assertEqual(submit_mock.call_count, 1)

    def test_late_assessment_after_investigation_termination_is_ignored(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        investigation = state.tracking_decision_investigations["inv-1"]
        investigation["terminal"] = True
        investigation["follow_up_completed"] = True

        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "delivery_kind": "observation_follow_up_assessment",
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-1:observation_follow_up_assessment",
                    "result_payload": {
                        "observation_llm_follow_up_assessment": {
                            "available": True,
                            "state": "observation_follow_up_completed",
                            "uncertainties": ["late_assessment"],
                            "provenance": {
                                "identity_verification_authorized": False,
                                "identity_mutation_authorized": False,
                            },
                        }
                    },
                }
            ]
        )

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

        submit_mock.assert_not_called()

    def test_failed_follow_up_submission_retains_pending_recovery_behavior(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "delivery_kind": "mcp_result",
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-1",
                    "result_payload": {"available": True, "messages": [{"message_id": "m-1"}]},
                }
            ]
        )

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False) as first_submit_mock:
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

        self.assertEqual(first_submit_mock.call_count, 1)
        investigation = state.tracking_decision_investigations["inv-1"]
        self.assertTrue(investigation.get("follow_up_submit_pending", False))
        self.assertFalse(investigation.get("follow_up_submitted", False))

        captured: dict[str, object] = {}

        def _capture_submit(**kwargs):
            follow_up = kwargs.get("follow_up_context")
            if isinstance(follow_up, dict):
                captured.update(follow_up)
            return True

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", side_effect=_capture_submit) as retry_submit_mock:
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=_FakeRuntime(),
                now=1001.1,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[{"track_id": "track-new"}],
                selected_event_reasons=["new_tracking_candidate"],
            )

        self.assertEqual(retry_submit_mock.call_count, 1)
        self.assertEqual(captured.get("investigation_id"), "inv-1")
        self.assertEqual(captured.get("idempotency_key"), "rk-1")

    def test_duplicate_and_mismatched_assessments_do_not_trigger_additional_llm_calls(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            state.tracking_decision_ipc_transport = _FakeResultTransport(
                [
                    {
                        "delivery_kind": "mcp_result",
                        "investigation_id": "inv-1",
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "status": "ok",
                        "idempotency_key": "rk-1",
                        "result_payload": {"available": True},
                    }
                ]
            )
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

            state.tracking_decision_ipc_transport = _FakeResultTransport(
                [
                    {
                        "delivery_kind": "observation_follow_up_assessment",
                        "investigation_id": "inv-1",
                        "request_id": "req-mismatch",
                        "tool_name": "get_track_evidence_history",
                        "status": "ok",
                        "idempotency_key": "rk-1:observation_follow_up_assessment_mismatch",
                        "result_payload": {
                            "observation_llm_follow_up_assessment": {
                                "available": True,
                                "state": "observation_follow_up_completed",
                            }
                        },
                    },
                    {
                        "delivery_kind": "observation_follow_up_assessment",
                        "investigation_id": "inv-1",
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "status": "ok",
                        "idempotency_key": "rk-1",
                        "result_payload": {
                            "observation_llm_follow_up_assessment": {
                                "available": True,
                                "state": "observation_follow_up_completed",
                            }
                        },
                    },
                ]
            )
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.1,
            )

        self.assertEqual(submit_mock.call_count, 1)

    def test_pending_follow_up_is_prioritized_over_new_candidate_submission(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        investigation = state.tracking_decision_investigations["inv-1"]
        investigation["result_status"] = "ok"
        investigation["result_payload"] = {"available": True, "messages": [{"message_id": "m-1"}]}
        investigation["follow_up_completed"] = False
        investigation["follow_up_submitted"] = False
        investigation["updated_at"] = 1000.0

        captured: dict[str, object] = {}

        def _capture_submit(**kwargs):
            captured.update(kwargs)
            return True

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", side_effect=_capture_submit):
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=_FakeRuntime(),
                now=1001.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[{"track_id": "track-new"}],
                selected_event_reasons=["new_tracking_candidate"],
            )

        self.assertEqual(captured.get("event_reasons"), ["tracking_decision_follow_up"])
        self.assertIsInstance(captured.get("follow_up_context"), dict)
        follow_up_context = captured["follow_up_context"]
        self.assertIsInstance(follow_up_context, dict)
        self.assertEqual(follow_up_context.get("investigation_id"), "inv-1")
        self.assertEqual(follow_up_context.get("idempotency_key"), "rk-1")
        self.assertTrue(state.tracking_decision_investigations["inv-1"]["follow_up_submitted"])
        self.assertIn("rk-1", state.tracking_decision_followup_result_keys)

    def test_pending_follow_up_times_out_into_terminal_inconclusive(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        investigation = state.tracking_decision_investigations["inv-1"]
        investigation["result_status"] = "ok"
        investigation["follow_up_completed"] = False
        investigation["terminal"] = False
        investigation["updated_at"] = 1000.0

        tracking_loop._enforce_investigation_follow_up_timeout_locked(
            state,
            now=1000.0 + tracking_loop.FOLLOW_UP_TIMEOUT_GRACE_SECONDS + 0.1,
        )

        final_decision = state.tracking_decision_investigations["inv-1"]["final_decision"]
        self.assertTrue(state.tracking_decision_investigations["inv-1"]["terminal"])
        self.assertTrue(state.tracking_decision_investigations["inv-1"]["follow_up_completed"])
        self.assertEqual(final_decision["action"], "inconclusive")
        self.assertEqual(final_decision["reason"], "follow_up_timeout")

    def test_pending_investigation_without_result_suppresses_new_candidate_submission(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        investigation = state.tracking_decision_investigations["inv-1"]
        investigation["result_status"] = None
        investigation["result_payload"] = None
        investigation["follow_up_submitted"] = False
        investigation["follow_up_completed"] = False

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=_FakeRuntime(),
                now=1001.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[{"track_id": "track-a"}],
                selected_event_reasons=["new_tracking_candidate"],
            )
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=_FakeRuntime(),
                now=1001.1,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[{"track_id": "track-b"}],
                selected_event_reasons=["new_tracking_candidate"],
            )

        submit_mock.assert_not_called()

    def test_non_followup_decision_response_registers_new_investigation_while_prior_open(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        runtime = _FakeRuntime(_initial_decision_observation(investigation_id="inv-2"))

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
            observation = tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=runtime,
                now=1002.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[],
                selected_event_reasons=[],
            )

        tracking_loop._submit_tracking_decision_request_if_enabled(
            state=state,
            tracking_agent_observation=observation,
            now=1002.0,
        )

        self.assertIn("inv-1", state.tracking_decision_investigations)
        self.assertNotIn("inv-2", state.tracking_decision_investigations)

    def test_missing_mcp_result_times_out_to_terminal_inconclusive(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        investigation = state.tracking_decision_investigations["inv-1"]
        investigation["result_status"] = None
        investigation["terminal"] = False
        investigation["follow_up_completed"] = False
        investigation["updated_at"] = 1000.0

        tracking_loop._enforce_investigation_follow_up_timeout_locked(
            state,
            now=1000.0 + tracking_loop.FOLLOW_UP_TIMEOUT_GRACE_SECONDS + 0.1,
        )

        final_decision = state.tracking_decision_investigations["inv-1"]["final_decision"]
        self.assertTrue(state.tracking_decision_investigations["inv-1"]["terminal"])
        self.assertTrue(state.tracking_decision_investigations["inv-1"]["follow_up_completed"])
        self.assertEqual(final_decision["action"], "inconclusive")
        self.assertEqual(final_decision["reason"], "mcp_result_timeout")

    def test_follow_up_completion_with_normalized_proposal_marks_terminal(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        runtime = _FakeRuntime(_follow_up_observation(with_llm_proposal=True))

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=runtime,
                now=1002.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[],
                selected_event_reasons=[],
            )

        investigation = state.tracking_decision_investigations["inv-1"]
        self.assertTrue(investigation["terminal"])
        self.assertEqual(investigation["final_decision"]["action"], "propose_identity_association")

    def test_follow_up_completion_without_supported_association_is_inconclusive(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        runtime = _FakeRuntime(_follow_up_observation(with_llm_proposal=False))

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=runtime,
                now=1002.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[],
                selected_event_reasons=[],
            )

        investigation = state.tracking_decision_investigations["inv-1"]
        self.assertTrue(investigation["terminal"])
        self.assertEqual(investigation["final_decision"]["action"], "inconclusive")
        self.assertEqual(investigation["final_decision"]["reason"], "follow_up_no_supported_association")

    def test_follow_up_attempting_tool_request_is_forced_inconclusive(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        follow_up_decision = {
            "action": "request_identity_snapshot",
            "investigation_id": "inv-1",
            "tool_requests": [
                {
                    "request_id": "req-2",
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                }
            ],
        }
        runtime = _FakeRuntime(_follow_up_observation(with_llm_proposal=True, decision=follow_up_decision))

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=runtime,
                now=1002.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[],
                selected_event_reasons=[],
            )

        investigation = state.tracking_decision_investigations["inv-1"]
        self.assertTrue(investigation["terminal"])
        self.assertEqual(investigation["final_decision"]["action"], "inconclusive")
        self.assertEqual(investigation["final_decision"]["reason"], "follow_up_tool_request_not_allowed")

    def test_terminal_follow_up_inconclusive_is_persisted_without_mcp_request(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_ipc_transport = transport
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        decision = {
            "action": "inconclusive",
            "investigation_id": "inv-1",
            "correlation_id": "corr-followup",
            "reason": "insufficient_evidence",
            "uncertainty": ["needs_more_evidence"],
            "evidence_references": {
                "frame_reference": "evidence://frame/f-1",
                "tracking_message_id": "trkmsg-1",
            },
        }
        runtime = _FakeRuntime(_follow_up_observation(with_llm_proposal=False, decision=decision))

        with tempfile.TemporaryDirectory() as temp_dir:
            terminal_path = Path(temp_dir) / "tracking_terminal_follow_up_decisions.jsonl"
            with patch.object(
                tracking_loop,
                "TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH",
                terminal_path,
            ), patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=1002.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )

            rows = _read_jsonl(terminal_path)

        self.assertEqual(len(transport.calls), 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "accepted")
        self.assertEqual(rows[0]["decision"]["action"], "inconclusive")
        self.assertEqual(rows[0]["request_id"], "req-1")
        self.assertEqual(rows[0]["investigation_id"], "inv-1")
        self.assertEqual(rows[0]["evidence_references"]["frame_reference"], "evidence://frame/f-1")

    def test_terminal_follow_up_association_proposal_is_persisted_without_mutation(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        decision = {
            "action": "propose_identity_association",
            "investigation_id": "inv-1",
            "correlation_id": "corr-followup",
            "reason": "consistent_motion",
            "uncertainty": ["candidate_match"],
            "provenance": {"attempt_id": "attempt-followup-1"},
        }
        runtime = _FakeRuntime(_follow_up_observation(with_llm_proposal=True, decision=decision))

        with tempfile.TemporaryDirectory() as temp_dir:
            terminal_path = Path(temp_dir) / "tracking_terminal_follow_up_decisions.jsonl"
            with patch.object(
                tracking_loop,
                "TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH",
                terminal_path,
            ), patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=1002.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )
            rows = _read_jsonl(terminal_path)

        investigation = state.tracking_decision_investigations["inv-1"]
        self.assertTrue(investigation["terminal"])
        self.assertEqual(investigation["final_decision"]["action"], "propose_identity_association")
        self.assertEqual(rows[0]["decision"]["action"], "propose_identity_association")
        self.assertEqual(rows[0]["llm_attempt_id"], "attempt-followup-1")

    def test_terminal_follow_up_with_mismatched_request_is_rejected(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        observation = TrackingAgentObservation(
            timestamp=1001.0,
            source_agent="visual_tracking",
            vlm_model="qwen-3.8-27b",
            proposal_latency_ms=10.0,
            frame_reference=None,
            updates=[],
            uncertainties=[
                "schema_complete_response",
                "followup_investigation_id:inv-1",
                "followup_request_id:req-mismatch",
                "followup_result_key:rk-1",
                "followup_result_status:ok",
            ],
            observation_id="obs-mismatch",
            llm_correlation_id="corr-mismatch",
            decision={"action": "inconclusive", "investigation_id": "inv-1", "reason": "insufficient_evidence"},
        )
        runtime = _FakeRuntime(observation)

        with tempfile.TemporaryDirectory() as temp_dir:
            terminal_path = Path(temp_dir) / "tracking_terminal_follow_up_decisions.jsonl"
            with patch.object(
                tracking_loop,
                "TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH",
                terminal_path,
            ), patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=1002.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )
            rows = _read_jsonl(terminal_path)

        self.assertFalse(state.tracking_decision_investigations["inv-1"]["terminal"])
        self.assertEqual(rows[0]["status"], "rejected")
        self.assertEqual(rows[0]["reason"], "follow_up_context_mismatch")

    def test_invalid_terminal_action_is_normalized_and_persisted(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        decision = {
            "action": "unexpected_action",
            "investigation_id": "inv-1",
            "reason": "bad_action",
        }
        runtime = _FakeRuntime(_follow_up_observation(with_llm_proposal=False, decision=decision))

        with tempfile.TemporaryDirectory() as temp_dir:
            terminal_path = Path(temp_dir) / "tracking_terminal_follow_up_decisions.jsonl"
            with patch.object(
                tracking_loop,
                "TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH",
                terminal_path,
            ), patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=1002.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )
            rows = _read_jsonl(terminal_path)

        self.assertTrue(state.tracking_decision_investigations["inv-1"]["terminal"])
        self.assertEqual(state.tracking_decision_investigations["inv-1"]["final_decision"]["action"], "inconclusive")
        self.assertEqual(rows[0]["status"], "accepted")
        self.assertEqual(rows[0]["reason"], "terminal_follow_up_normalized_to_inconclusive")

    def test_duplicate_follow_up_publication_is_rejected(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        decision = {
            "action": "inconclusive",
            "investigation_id": "inv-1",
            "reason": "insufficient_evidence",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            terminal_path = Path(temp_dir) / "tracking_terminal_follow_up_decisions.jsonl"
            with patch.object(
                tracking_loop,
                "TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH",
                terminal_path,
            ), patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=_FakeRuntime(_follow_up_observation(with_llm_proposal=False, decision=decision)),
                    now=1002.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=_FakeRuntime(_follow_up_observation(with_llm_proposal=False, decision=decision)),
                    now=1002.1,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )
            rows = _read_jsonl(terminal_path)

        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0]["status"], "accepted")
        self.assertEqual(rows[1]["status"], "rejected")
        self.assertEqual(rows[1]["reason"], "late_follow_up_after_investigation_closed")

    def test_late_follow_up_after_closure_is_rejected(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        investigation = _base_investigation_state()
        investigation["terminal"] = True
        investigation["follow_up_completed"] = True
        investigation["terminal_follow_up_persisted"] = True
        state.tracking_decision_investigations["inv-1"] = investigation

        decision = {
            "action": "inconclusive",
            "investigation_id": "inv-1",
            "reason": "late",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            terminal_path = Path(temp_dir) / "tracking_terminal_follow_up_decisions.jsonl"
            with patch.object(
                tracking_loop,
                "TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH",
                terminal_path,
            ), patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=_FakeRuntime(_follow_up_observation(with_llm_proposal=False, decision=decision)),
                    now=1002.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )
            rows = _read_jsonl(terminal_path)

        self.assertEqual(rows[0]["status"], "rejected")
        self.assertEqual(rows[0]["reason"], "late_follow_up_after_investigation_closed")

    def test_terminal_follow_up_evidence_references_preserved(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        decision = {
            "action": "inconclusive",
            "investigation_id": "inv-1",
            "reason": "insufficient_evidence",
            "evidence_references": {
                "frame_reference": "evidence://frame/custom-f",
                "detection_reference": "evidence://detection/custom-d",
                "tracking_message_id": "trkmsg-custom",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            terminal_path = Path(temp_dir) / "tracking_terminal_follow_up_decisions.jsonl"
            with patch.object(
                tracking_loop,
                "TRACKING_TERMINAL_FOLLOW_UP_DECISIONS_PATH",
                terminal_path,
            ), patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=_FakeRuntime(_follow_up_observation(with_llm_proposal=False, decision=decision)),
                    now=1002.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=[],
                    selected_event_reasons=[],
                )
            rows = _read_jsonl(terminal_path)

        self.assertEqual(rows[0]["evidence_references"]["frame_reference"], "evidence://frame/custom-f")
        self.assertEqual(rows[0]["evidence_references"]["detection_reference"], "evidence://detection/custom-d")

    def test_failed_follow_up_releases_guard_for_next_initial_submission(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        follow_up_decision = {
            "action": "request_identity_snapshot",
            "investigation_id": "inv-1",
            "tool_requests": [
                {
                    "request_id": "req-2",
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                }
            ],
        }
        runtime = _FakeRuntime(_follow_up_observation(with_llm_proposal=True, decision=follow_up_decision))

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=runtime,
                now=1002.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[],
                selected_event_reasons=[],
            )

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=_FakeRuntime(),
                now=1002.1,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[{"track_id": "track-new"}],
                selected_event_reasons=["new_tracking_candidate"],
            )

        submit_mock.assert_called_once()

    def test_guard_active_still_allows_observation_update_application(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()
        investigation = state.tracking_decision_investigations["inv-1"]
        investigation["result_status"] = None
        investigation["terminal"] = False

        runtime = _FakeRuntime(_initial_decision_observation(investigation_id="inv-unrelated"))

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock, patch.object(
            tracking_loop,
            "apply_tracking_observation_update",
            return_value={"accepted": 0, "rejected": 0, "unmatched": 0, "fallback_used": 0, "llm_applied": 0},
        ) as update_mock:
            observation = tracking_loop._apply_tracking_agent_runtime_cycle(
                state=state,
                tracking_runtime=runtime,
                now=1002.0,
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracks={"det-1": _track_snapshot()},
                tracking_candidates=[{"track_id": "track-new"}],
                selected_event_reasons=["new_tracking_candidate"],
            )

        submit_mock.assert_not_called()
        update_mock.assert_called_once()
        self.assertIsNotNone(observation)

    def test_duplicate_replayed_result_does_not_resubmit_follow_up(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-1",
                    "result_payload": {"available": True},
                }
            ]
        )
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.1,
            )

        self.assertEqual(submit_mock.call_count, 1)

    def test_unsuccessful_results_are_terminal_inconclusive(self) -> None:
        for status in ["timeout", "unavailable", "stale", "incompatible", "expired", "terminal"]:
            with self.subTest(status=status):
                state = SharedDashboardState()
                state.tracking_decision_ipc_enabled = True
                state.tracking_decision_ipc_transport = _FakeResultTransport(
                    [
                        {
                            "investigation_id": "inv-1",
                            "request_id": "req-1",
                            "tool_name": "get_track_evidence_history",
                            "status": status,
                            "idempotency_key": f"rk-{status}",
                            "result_payload": {"available": False},
                        }
                    ]
                )
                state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

                with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
                    tracking_loop._consume_tracking_decision_results_if_enabled(
                        state=state,
                        tracking_runtime=_FakeRuntime(),
                        frame=types.SimpleNamespace(shape=(10, 10, 3)),
                        tracking_candidates=[],
                        now=1001.0,
                    )

                submit_mock.assert_not_called()
                final_decision = state.tracking_decision_investigations["inv-1"]["final_decision"]
                self.assertEqual(final_decision["action"], "inconclusive")
                self.assertEqual(final_decision["reason"], f"tool_result_{status}")

    def test_terminal_investigation_result_does_not_schedule_follow_up(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-1",
                    "result_payload": {"available": True},
                }
            ]
        )
        investigation_state = _base_investigation_state()
        investigation_state["terminal"] = True
        state.tracking_decision_investigations["inv-1"] = investigation_state

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

        submit_mock.assert_not_called()

    def test_investigation_request_mismatch_rejected(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "investigation_id": "inv-1",
                    "request_id": "req-other",
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "idempotency_key": "rk-mismatch",
                    "result_payload": {"available": True},
                }
            ]
        )
        state.tracking_decision_investigations["inv-1"] = _base_investigation_state()

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

        submit_mock.assert_not_called()
        summary = state.latest_tracking_decision_ipc_summary["results"]
        self.assertGreaterEqual(summary["mismatch_count"], 1)

    def test_identity_snapshot_generation_version_mismatch_forces_inconclusive(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeResultTransport(
            [
                {
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "tool_name": "get_identity_snapshot",
                    "status": "ok",
                    "idempotency_key": "rk-snapshot",
                    "result_payload": {
                        "available": True,
                        "source_generation": 1,
                        "source_version": 1,
                    },
                }
            ]
        )
        investigation = _base_investigation_state()
        investigation["tool_name"] = "get_identity_snapshot"
        investigation["initial_decision"] = {
            "action": "request_identity_snapshot",
            "investigation_id": "inv-1",
            "snapshot_constraints": {
                "min_source_generation": 5,
                "min_source_version": 5,
            },
            "tool_requests": [
                {
                    "request_id": "req-1",
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                }
            ],
        }
        state.tracking_decision_investigations["inv-1"] = investigation

        with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
            tracking_loop._consume_tracking_decision_results_if_enabled(
                state=state,
                tracking_runtime=_FakeRuntime(),
                frame=types.SimpleNamespace(shape=(10, 10, 3)),
                tracking_candidates=[],
                now=1001.0,
            )

        submit_mock.assert_not_called()
        final_decision = state.tracking_decision_investigations["inv-1"]["final_decision"]
        self.assertEqual(final_decision["action"], "inconclusive")
        self.assertEqual(final_decision["reason"], "snapshot_generation_mismatch")

    def test_legacy_observation_without_decision_remains_noop(self) -> None:
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)

        legacy_observation = types.SimpleNamespace(decision=None, llm_correlation_id="corr-legacy", uncertainties=[])
        tracking_loop._submit_tracking_decision_request_if_enabled(
            state=state,
            tracking_agent_observation=legacy_observation,
            now=1000.0,
        )

        self.assertEqual(state.tracking_decision_investigations, {})

    def test_runtime_cycle_prevents_second_generic_call_after_initial_decision_observation(self) -> None:
        first_decision_result = VisualTrackingAgentResult(
            available=True,
            proposals=[],
            uncertainty_flags=["schema_complete_response"],
            supporting_evidence=[],
            latency_ms=5.0,
            decision={
                "action": "request_identity_snapshot",
                "investigation_id": "inv-race-1",
                "correlation_id": "corr-race-1",
                "tool_requests": [
                    {
                        "request_id": "req-race-1",
                        "tool_name": "get_identity_snapshot",
                        "arguments": {"app_identity_id": "app-1"},
                    }
                ],
            },
        )
        generic_result = VisualTrackingAgentResult(
            available=True,
            proposals=[],
            uncertainty_flags=["schema_complete_response"],
            supporting_evidence=[],
            latency_ms=5.0,
            decision=None,
        )

        runtime_agent = _ScriptedRuntimeAgent([first_decision_result, generic_result])
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), runtime_agent, min_invoke_interval_seconds=0.0)
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        detections = [_detection_candidate()]

        try:
            with patch.object(tracking_loop, "encode_frame_as_data_url", return_value="data:image/jpeg;base64,synthetic"):
                first_observation = tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=1000.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=detections,
                    selected_event_reasons=["new_tracking_candidate"],
                )
                self.assertIsNone(first_observation)

                self.assertTrue(_wait_until_runtime_idle(runtime))

                self.assertEqual(len(runtime_agent.call_follow_up_contexts), 1)
                self.assertIsNone(runtime_agent.call_follow_up_contexts[0])
                self.assertEqual(state.tracking_decision_investigations, {})

                observation_with_initial_decision = tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=1001.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-1": _track_snapshot()},
                    tracking_candidates=detections,
                    selected_event_reasons=["new_tracking_candidate"],
                )

                self.assertIsNotNone(observation_with_initial_decision)
                self.assertIn("inv-race-1", state.tracking_decision_investigations)
                self.assertTrue(_wait_until_runtime_idle(runtime))
                self.assertEqual(
                    len(runtime_agent.call_follow_up_contexts),
                    1,
                    "Generic second provider invocation must be suppressed in the wait window",
                )
                self.assertIn("inv-race-1", state.tracking_decision_investigations)
        finally:
            runtime.shutdown()

    def test_runtime_wait_window_should_prevent_second_provider_start_until_mcp_result(self) -> None:
        first_decision_result = VisualTrackingAgentResult(
            available=True,
            proposals=[],
            uncertainty_flags=["schema_complete_response"],
            supporting_evidence=[],
            latency_ms=5.0,
            decision={
                "action": "request_identity_snapshot",
                "investigation_id": "inv-race-2",
                "correlation_id": "corr-race-2",
                "tool_requests": [
                    {
                        "request_id": "req-race-2",
                        "tool_name": "get_identity_snapshot",
                        "arguments": {"app_identity_id": "app-2"},
                    }
                ],
            },
        )
        neutral_result = VisualTrackingAgentResult(
            available=True,
            proposals=[],
            uncertainty_flags=["schema_complete_response"],
            supporting_evidence=[],
            latency_ms=5.0,
            decision=None,
        )

        follow_up_decision_result = VisualTrackingAgentResult(
            available=True,
            proposals=[],
            uncertainty_flags=["schema_complete_response"],
            supporting_evidence=[],
            latency_ms=5.0,
            decision={
                "action": "inconclusive",
                "reason": "insufficient_visual_evidence",
                "investigation_id": "inv-race-2",
            },
        )

        runtime_agent = _ScriptedRuntimeAgent([first_decision_result, neutral_result, follow_up_decision_result])
        runtime = AsyncVisualTrackingRuntime(AgentCoordinator(), runtime_agent, min_invoke_interval_seconds=0.0)
        state = SharedDashboardState()
        state.tracking_decision_ipc_enabled = True
        state.tracking_decision_ipc_transport = _FakeSubmitTransport(accepted=True)
        detections = [_detection_candidate(detection_id="det-race-2")]

        try:
            with patch.object(tracking_loop, "encode_frame_as_data_url", return_value="data:image/jpeg;base64,synthetic"):
                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=2000.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-race-2": _track_snapshot()},
                    tracking_candidates=detections,
                    selected_event_reasons=["new_tracking_candidate"],
                )
                self.assertTrue(_wait_until_runtime_idle(runtime))

                decision_observation = tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=2001.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-race-2": _track_snapshot()},
                    tracking_candidates=detections,
                    selected_event_reasons=["new_tracking_candidate"],
                )
                self.assertIsNotNone(decision_observation)
                self.assertIn("inv-race-2", state.tracking_decision_investigations)

                self.assertEqual(
                    len(runtime_agent.call_follow_up_contexts),
                    1,
                    "No second provider call should start before MCP result is published",
                )

                state.tracking_decision_ipc_transport = _FakeResultTransport(
                    [
                        {
                            "investigation_id": "inv-race-2",
                            "request_id": "req-race-2",
                            "tool_name": "get_identity_snapshot",
                            "status": "ok",
                            "idempotency_key": "rk-race-2",
                            "result_payload": {
                                "available": True,
                                "source_generation": 3,
                                "source_version": 3,
                            },
                        }
                    ]
                )
                tracking_loop._consume_tracking_decision_results_if_enabled(
                    state=state,
                    tracking_runtime=runtime,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracking_candidates=detections,
                    now=2002.0,
                )

                self.assertTrue(_wait_until_runtime_idle(runtime))

                tracking_loop._apply_tracking_agent_runtime_cycle(
                    state=state,
                    tracking_runtime=runtime,
                    now=2003.0,
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracks={"det-race-2": _track_snapshot()},
                    tracking_candidates=detections,
                    selected_event_reasons=["new_tracking_candidate"],
                )

                self.assertEqual(len(runtime_agent.call_follow_up_contexts), 2)
                follow_up_context = runtime_agent.call_follow_up_contexts[1]
                self.assertIsInstance(follow_up_context, dict)
                assert isinstance(follow_up_context, dict)
                self.assertEqual(follow_up_context.get("investigation_id"), "inv-race-2")
                self.assertEqual(follow_up_context.get("request_id"), "req-race-2")
                self.assertEqual(follow_up_context.get("result_status"), "ok")

                investigation_state = state.tracking_decision_investigations.get("inv-race-2")
                self.assertIsInstance(investigation_state, dict)
                assert isinstance(investigation_state, dict)
                self.assertTrue(investigation_state.get("terminal", False))
                final_decision = investigation_state.get("final_decision")
                self.assertIsInstance(final_decision, dict)
                assert isinstance(final_decision, dict)
                self.assertEqual(final_decision.get("action"), "inconclusive")
        finally:
            runtime.shutdown()


if __name__ == "__main__":
    unittest.main()
