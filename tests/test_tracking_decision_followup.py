from __future__ import annotations

import copy
import types
import unittest
from unittest.mock import patch

from agents.contracts import TrackingAgentObservation, VisualAssociationUpdate
from agents.dashboard.state import SharedDashboardState
from agents.dashboard import tracking_loop
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


if __name__ == "__main__":
    unittest.main()
