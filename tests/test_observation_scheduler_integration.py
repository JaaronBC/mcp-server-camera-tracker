from __future__ import annotations

import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport
from agents.orchestration.tools import ops


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append((tool_name, dict(arguments)))
        if tool_name == "get_track_evidence_history":
            payload = {
                "available": True,
                "state": "recorded_observation",
                "messages": [
                    {
                        "message_id": "msg-observation",
                        "bounded_history": [{"observation_id": "obs-2"}],
                    }
                ],
            }
            return SimpleNamespace(content=[SimpleNamespace(json=payload)])
        if tool_name == "get_identity_snapshot":
            payload = {
                "available": True,
                "state": "snapshot_available",
                "app_identity_id": arguments.get("app_identity_id"),
            }
            return SimpleNamespace(content=[SimpleNamespace(json=payload)])
        return SimpleNamespace(content=[SimpleNamespace(json={"available": False, "error": "unexpected_tool"})])


class TestObservationSchedulerIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_integrates_observation_ipc_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            requests_path = root / "tracking_decision_tool_requests.jsonl"
            results_path = root / "tracking_decision_tool_results.jsonl"
            request_index_path = root / "latest_tracking_decision_request_index.json"
            result_index_path = root / "latest_tracking_decision_result_index.json"

            environment = {
                "OBSERVATION_TOOL_REQUESTS_ENABLED": "1",
                "PHASE2D_TRACKING_DECISION_IPC_ENABLED": "1",
                "PHASE2D_TRACKING_DECISION_TOOL_REQUESTS_PATH": str(requests_path),
                "PHASE2D_TRACKING_DECISION_TOOL_RESULTS_PATH": str(results_path),
                "PHASE2D_TRACKING_DECISION_REQUEST_INDEX_PATH": str(request_index_path),
                "PHASE2D_TRACKING_DECISION_RESULT_INDEX_PATH": str(result_index_path),
                "PHASE2D_TRACKING_DECISION_IPC_MAX_REQUESTS_PER_CYCLE": "8",
                "PHASE2D_TRACKING_DECISION_IPC_TOOL_TIMEOUT_SECONDS": "0.5",
            }

            fake_session = _FakeSession()
            ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS.clear()

            tracking_iterations = 0

            async def fake_run_tracking_cycle(_session: object, _address: list[object]) -> None:
                nonlocal tracking_iterations
                tracking_iterations += 1
                if tracking_iterations >= 6:
                    raise KeyboardInterrupt()

            def fake_make_monitor_movement(*, run_agent_ops_pipeline_fn, analysis_interval_seconds: float):
                async def _impl(session, decision_ipc_fixture_cycle_fn=None, **kwargs):
                    await runtime_flow.monitor_movement(
                        session=session,
                        tracking_address=("127.0.0.1", 5005),
                        analysis_interval_seconds=analysis_interval_seconds,
                        enable_agent_ops_heartbeat=kwargs["enable_agent_ops_heartbeat"],
                        agent_ops_interval_seconds=kwargs["agent_ops_interval_seconds"],
                        agent_ops_max_tracks=kwargs["agent_ops_max_tracks"],
                        agent_ops_summary_every_cycles=kwargs["agent_ops_summary_every_cycles"],
                        agent_ops_summary_window_size=kwargs["agent_ops_summary_window_size"],
                        handoff_degraded_warn_rate=kwargs["handoff_degraded_warn_rate"],
                        handoff_degraded_critical_rate=kwargs["handoff_degraded_critical_rate"],
                        depth_coverage_summary_enabled=kwargs["depth_coverage_summary_enabled"],
                        depth_coverage_summary_every_cycles=kwargs["depth_coverage_summary_every_cycles"],
                        depth_coverage_sample_limit=kwargs["depth_coverage_sample_limit"],
                        run_tracking_cycle_fn=fake_run_tracking_cycle,
                        run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
                        emit=lambda _message: None,
                        verification_manager=kwargs.get("verification_manager"),
                        verification_request_source_fn=kwargs.get("verification_request_source_fn"),
                        verification_feedback_publish_fn=kwargs.get("verification_feedback_publish_fn"),
                        verification_results_persist_fn=kwargs.get("verification_results_persist_fn"),
                        verification_submissions_persist_fn=kwargs.get("verification_submissions_persist_fn"),
                        decision_ipc_fixture_cycle_fn=decision_ipc_fixture_cycle_fn,
                    )

                return _impl

            def base_assessment_result() -> dict:
                return {
                    "available": True,
                    "track_id": "track-1",
                    "summary": {
                        "track_id": "track-1",
                        "created_at": 1000.0,
                        "sample_count": 3,
                        "velocity_x": 1.1,
                        "velocity_z": 0.2,
                        "speed": 1.2,
                        "confidence": 0.8,
                        "supporting_observation_ids": ["obs-1"],
                        "uncertainties": ["insufficient_history"],
                        "needs_more_observations": True,
                    },
                    "assessment": {
                        "assessment_label": "stable_motion",
                        "summary": "stable",
                        "uncertainties": [],
                    },
                    "agents": {
                        "observation": {
                            "status": "ok",
                            "model": "mock-observation",
                            "latency_ms": 1.0,
                            "error_code": None,
                            "attempts": 1,
                            "retry_count": 0,
                        },
                        "reasoning": {
                            "status": "ok",
                            "model": "mock-reasoning",
                            "latency_ms": 1.0,
                            "error_code": None,
                            "attempts": 1,
                            "retry_count": 0,
                        },
                    },
                    "handoff_validation": {"status": "ok", "error_code": None, "error": None},
                    "evidence_request": None,
                }

            def make_result(*_args, **_kwargs):
                return copy.deepcopy(base_assessment_result())

            with patch.dict(os.environ, environment, clear=False):
                with patch("main.runtime_agent.make_monitor_movement", side_effect=fake_make_monitor_movement):
                    with patch(
                        "agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment",
                        side_effect=make_result,
                    ):
                        first_eval = ops._evaluate_track_for_cycle(
                            "track-1",
                            request_evidence_on_uncertainty=True,
                            correlation_id="corr-cycle-1",
                            cycle_id="cycle-1",
                        )

                first_request = first_eval["result"].get("observation_tool_request")
                self.assertIsInstance(first_request, dict)
                self.assertTrue(first_request.get("submitted"))
                self.assertEqual(
                    first_eval["result"]["observation_decision"].get("phase"),
                    "tool_request_submitted",
                )

                dashboard_transport = TrackingDecisionIpcTransport(
                    requests_path=requests_path,
                    results_path=results_path,
                    request_index_path=request_index_path,
                    result_index_path=result_index_path,
                    writer_role="dashboard",
                )
                tracking_submitted = dashboard_transport.dashboard_submit_decision_request(
                    decision={
                        "action": "request_identity_snapshot",
                        "investigation_id": "inv-tracking-1",
                        "iteration_index": 1,
                        "tool_requests": [
                            {
                                "request_id": "req-tracking-1",
                                "tool_name": "get_identity_snapshot",
                                "arguments": {"app_identity_id": "app-1"},
                            }
                        ],
                    },
                    llm_correlation_id="corr-tracking-1",
                )
                self.assertTrue(tracking_submitted["accepted"])

                with patch("main.runtime_agent.make_monitor_movement", side_effect=fake_make_monitor_movement):
                    with self.assertRaises(KeyboardInterrupt):
                        await main.monitor_movement(
                            fake_session,
                            enable_agent_ops_heartbeat=False,
                            agent_ops_interval_seconds=999.0,
                            agent_ops_max_tracks=2,
                            agent_ops_summary_every_cycles=10,
                            agent_ops_summary_window_size=10,
                            handoff_degraded_warn_rate=0.1,
                            handoff_degraded_critical_rate=0.3,
                            depth_coverage_summary_enabled=False,
                            depth_coverage_summary_every_cycles=10,
                            depth_coverage_sample_limit=10,
                            enable_visual_verification_queue=False,
                        )

                observation_calls = [
                    arguments
                    for tool_name, arguments in fake_session.calls
                    if tool_name == "get_track_evidence_history"
                ]
                tracking_calls = [
                    arguments
                    for tool_name, arguments in fake_session.calls
                    if tool_name == "get_identity_snapshot"
                ]
                self.assertEqual(len(observation_calls), 1)
                self.assertEqual(len(tracking_calls), 1)

                captured_followup_payload: dict[str, object] = {}

                def fake_followup(payload: dict, **_kwargs):
                    captured_followup_payload["payload"] = payload
                    return (
                        {
                            "summary": "follow-up assessment",
                            "measured_facts": ["deterministic_values_preserved"],
                            "interpretations": ["evidence_correlated"],
                            "hypotheses": [],
                            "supporting_observation_ids": ["obs-1", "obs-2"],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {
                            "status": "ok",
                            "model": "mock-observation",
                            "latency_ms": 1.0,
                            "error_code": None,
                            "attempts": 1,
                            "retry_count": 0,
                        },
                    )

                with patch(
                    "agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment",
                    side_effect=make_result,
                ):
                    with patch(
                        "agents.orchestration.tools.ops.dashboard_reasoning._call_observation_agent_with_retry",
                        side_effect=fake_followup,
                    ):
                        second_eval = ops._evaluate_track_for_cycle(
                            "track-1",
                            request_evidence_on_uncertainty=True,
                            correlation_id="corr-cycle-1",
                            cycle_id="cycle-2",
                        )

                self.assertEqual(
                    second_eval["result"]["observation_decision"].get("phase"),
                    "follow_up_terminal",
                )
                self.assertEqual(
                    second_eval["result"].get("observation_tool_result", {}).get("status"),
                    "ok",
                )
                self.assertIsInstance(captured_followup_payload.get("payload"), dict)
                evidence_payload = (
                    captured_followup_payload["payload"]
                    .get("evidence_tool_result", {})
                    .get("payload", {})
                )
                self.assertIsInstance(evidence_payload, dict)
                self.assertEqual(len(evidence_payload.get("messages", [])), 1)

                observation_lookup = dashboard_transport.lookup_results_for_request(
                    investigation_id=first_request["investigation_id"],
                    request_id=first_request["request_id"],
                    tool_name="get_track_evidence_history",
                )
                tracking_lookup = dashboard_transport.lookup_results_for_request(
                    investigation_id="inv-tracking-1",
                    request_id="req-tracking-1",
                    tool_name="get_identity_snapshot",
                )
                wrong_lookup = dashboard_transport.lookup_results_for_request(
                    investigation_id=first_request["investigation_id"],
                    request_id=first_request["request_id"],
                    tool_name="get_identity_snapshot",
                )
                self.assertEqual(observation_lookup["result_count"], 1)
                self.assertEqual(tracking_lookup["result_count"], 1)
                self.assertEqual(wrong_lookup["result_count"], 0)

                with patch(
                    "agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment",
                    side_effect=make_result,
                ):
                    new_first_eval = ops._evaluate_track_for_cycle(
                        "track-1",
                        request_evidence_on_uncertainty=True,
                        correlation_id="corr-cycle-2",
                        cycle_id="cycle-3",
                    )
                    new_second_eval = ops._evaluate_track_for_cycle(
                        "track-1",
                        request_evidence_on_uncertainty=True,
                        correlation_id="corr-cycle-2",
                        cycle_id="cycle-4",
                    )

                self.assertEqual(
                    new_first_eval["result"]["observation_decision"].get("phase"),
                    "tool_request_submitted",
                )
                self.assertEqual(
                    new_second_eval["result"]["observation_decision"].get("phase"),
                    "awaiting_tool_result",
                )

                ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS["track-timeout"] = {
                    "investigation_id": "inv-timeout",
                    "request_id": "req-timeout",
                    "tool_name": "get_track_evidence_history",
                    "created_at": 0.0,
                }
                timeout_base = copy.deepcopy(base_assessment_result())
                timeout_base["track_id"] = "track-timeout"
                timeout_base["summary"]["track_id"] = "track-timeout"
                with patch(
                    "agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment",
                    return_value=timeout_base,
                ):
                    with patch("agents.orchestration.tools.ops.OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS", 1.5):
                        with patch("agents.orchestration.tools.ops.time.time", return_value=10.0):
                            timeout_eval = ops._evaluate_track_for_cycle(
                                "track-timeout",
                                request_evidence_on_uncertainty=True,
                                correlation_id="corr-timeout",
                                cycle_id="cycle-timeout",
                            )

                self.assertEqual(
                    timeout_eval["result"]["observation_decision"].get("phase"),
                    "tool_result_timeout",
                )
                self.assertNotIn("track-timeout", ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS)

                self.assertIn("observation_tool_result", second_eval["result"])
                self.assertIn("summary", second_eval["result"])
                self.assertIn("assessment", second_eval["result"])
                self.assertIsInstance(second_eval["result"].get("agents"), dict)
                self.assertIsInstance(second_eval["result"]["agents"].get("reasoning"), dict)


if __name__ == "__main__":
    unittest.main()