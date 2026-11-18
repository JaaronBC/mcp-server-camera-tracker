from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.observation import agent_observation
from agents.orchestration.tools import ops


class _FakeTransport:
    def __init__(self, *, submit_response: dict | None = None, latest_result: dict | None = None, results: list[dict] | None = None) -> None:
        self.submit_response = submit_response if submit_response is not None else {"accepted": True, "rejected": False, "duplicate": False}
        self.latest_result = latest_result
        self.results = list(results) if isinstance(results, list) else None
        self.submitted: list[dict] = []
        self.registered_handoffs: list[dict] = []

    def dashboard_submit_decision_request(self, *, decision: dict, llm_correlation_id: str | None = None, now_timestamp: float | None = None) -> dict:
        self.submitted.append({"decision": decision, "llm_correlation_id": llm_correlation_id})
        return dict(self.submit_response)

    def lookup_results_for_request(self, *, investigation_id: str, request_id: str, tool_name: str, now_timestamp: float | None = None) -> dict:
        if isinstance(self.results, list):
            latest = self.results[0] if self.results else None
            return {
                "available": True,
                "result_count": len(self.results),
                "latest": (dict(latest) if isinstance(latest, dict) else None),
                "results": [dict(item) for item in self.results if isinstance(item, dict)],
            }
        return {
            "available": True,
            "result_count": (1 if isinstance(self.latest_result, dict) else 0),
            "latest": (dict(self.latest_result) if isinstance(self.latest_result, dict) else None),
            "results": ([dict(self.latest_result)] if isinstance(self.latest_result, dict) else []),
        }

    def dashboard_register_observation_follow_up_assessment(self, **kwargs) -> dict:
        self.registered_handoffs.append(dict(kwargs))
        return {
            "published": True,
            "duplicate": False,
            "rejected": False,
            "reason": None,
        }


class TestObservationAutonomyCycle(unittest.TestCase):
    def setUp(self) -> None:
        ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS.clear()

    def tearDown(self) -> None:
        ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS.clear()

    def _base_result(self, *, needs_more_observations: bool, speed: float | None = 1.0) -> dict:
        return {
            "available": True,
            "track_id": "track-1",
            "summary": {
                "track_id": "track-1",
                "created_at": 1000.0,
                "sample_count": (3 if speed is not None else 1),
                "velocity_x": 1.25,
                "velocity_z": 0.5,
                "speed": speed,
                "confidence": 0.8,
                "supporting_observation_ids": ["obs-1", "obs-2"],
                "uncertainties": (["insufficient_history"] if needs_more_observations else []),
                "needs_more_observations": needs_more_observations,
            },
            "assessment": {
                "assessment_label": "stable_motion",
                "summary": "Stable.",
                "uncertainties": [],
            },
            "agents": {
                "observation": {
                    "status": "ok",
                    "model": "gpt-oss-120b",
                    "latency_ms": 10.0,
                    "error_code": None,
                    "attempts": 1,
                    "retry_count": 0,
                },
                "reasoning": {
                    "status": "ok",
                    "model": "gpt-oss-120b",
                    "latency_ms": 9.0,
                    "error_code": None,
                    "attempts": 1,
                    "retry_count": 0,
                },
            },
            "handoff_validation": {"status": "ok", "error_code": None, "error": None},
            "evidence_request": None,
        }

    def _attach_explicit_decision(self, result: dict, decision: dict) -> dict:
        payload = dict(result)
        summary = dict(payload.get("summary", {}))
        summary["observation_decision_present"] = True
        summary["observation_decision"] = dict(decision)
        payload["summary"] = summary
        return payload

    def test_direct_assessment_without_tool_request(self) -> None:
        fake_result = self._base_result(needs_more_observations=False, speed=1.1)
        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=fake_result):
            evaluated = ops._evaluate_track_for_cycle(
                "track-1",
                request_evidence_on_uncertainty=True,
                correlation_id="corr-1",
                cycle_id="cycle-1",
            )

        decision = evaluated["result"].get("observation_decision")
        self.assertIsInstance(decision, dict)
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_ASSESS_NOW)
        self.assertEqual(decision.get("decision_source"), "llm_summary_inference")
        self.assertNotIn("observation_tool_request", evaluated["result"])

    def test_valid_explicit_assess_now_is_preferred(self) -> None:
        base = self._base_result(needs_more_observations=True, speed=1.0)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_ASSESS_NOW,
                "reason": "operator_review_not_required",
                "uncertainty": [],
                "tool_requests": [],
            },
        )
        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
            evaluated = ops._evaluate_track_for_cycle(
                "track-1",
                request_evidence_on_uncertainty=True,
                correlation_id="corr-explicit-assess",
                cycle_id="cycle-explicit-assess",
            )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_ASSESS_NOW)
        self.assertEqual(decision.get("decision_source"), "llm_explicit")
        self.assertNotIn("observation_tool_request", evaluated["result"])

    def test_valid_explicit_evidence_request_is_preferred(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.2)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                "reason": "need_bounded_history",
                "uncertainty": ["history_gap"],
                "tool_requests": [
                    {
                        "request_id": "req-explicit-1",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-1",
                            "max_messages": 5,
                            "max_history_records": 12,
                            "max_scan_records": 120,
                        },
                    }
                ],
            },
        )
        fake_transport = _FakeTransport(
            submit_response={"accepted": True, "rejected": False, "duplicate": False, "reason": None}
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
                evaluated = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-explicit-request",
                    cycle_id="cycle-explicit-request",
                )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE)
        self.assertEqual(decision.get("decision_source"), "llm_explicit")
        self.assertEqual(decision.get("phase"), "tool_request_submitted")

    def test_explicit_evidence_request_empty_arguments_uses_context_track_id(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.2)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                "reason": "need_bounded_history",
                "uncertainty": ["history_gap"],
                "tool_requests": [
                    {
                        "request_id": "req-explicit-empty-args",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {},
                    }
                ],
            },
        )
        fake_transport = _FakeTransport(
            submit_response={"accepted": True, "rejected": False, "duplicate": False, "reason": None}
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
                evaluated = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-explicit-empty-args",
                    cycle_id="cycle-explicit-empty-args",
                )

        tool_request = evaluated["result"].get("observation_tool_request", {})
        self.assertEqual(tool_request.get("arguments", {}).get("track_id"), "track-1")

    def test_explicit_evidence_request_cannot_override_context_track_id(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.2)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                "reason": "need_bounded_history",
                "uncertainty": ["history_gap"],
                "tool_requests": [
                    {
                        "request_id": "req-explicit-override-track-id",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-llm-override",
                            "max_messages": 7,
                            "max_history_records": 22,
                        },
                    }
                ],
            },
        )
        fake_transport = _FakeTransport(
            submit_response={"accepted": True, "rejected": False, "duplicate": False, "reason": None}
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
                evaluated = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-explicit-track-override",
                    cycle_id="cycle-explicit-track-override",
                )

        tool_request = evaluated["result"].get("observation_tool_request", {})
        arguments = tool_request.get("arguments", {})
        self.assertEqual(arguments.get("track_id"), "track-1")
        self.assertEqual(arguments.get("max_messages"), 7)
        self.assertEqual(arguments.get("max_history_records"), 22)

    def test_explicit_evidence_request_with_unknown_argument_rejected_pre_dispatch(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.2)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                "reason": "need_bounded_history",
                "uncertainty": ["history_gap"],
                "tool_requests": [
                    {
                        "request_id": "req-explicit-unknown-argument",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-llm-override",
                            "since_timestamp": 123.4,
                        },
                    }
                ],
            },
        )
        fake_transport = _FakeTransport(
            submit_response={"accepted": True, "rejected": False, "duplicate": False, "reason": None}
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
                evaluated = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-explicit-unknown-argument",
                    cycle_id="cycle-explicit-unknown-argument",
                )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("phase"), "tool_request_pre_dispatch_rejected")
        self.assertEqual(decision.get("error_code"), "invalid_tool_arguments")
        self.assertIn("since_timestamp", decision.get("unknown_arguments", []))
        tool_request = evaluated["result"].get("observation_tool_request", {})
        self.assertFalse(bool(tool_request.get("submitted", True)))
        self.assertTrue(bool(tool_request.get("rejected", False)))
        self.assertEqual(tool_request.get("error_code"), "invalid_tool_arguments")
        self.assertIn("since_timestamp", tool_request.get("unknown_arguments", []))
        self.assertEqual(len(fake_transport.submitted), 0)

    def test_explicit_evidence_request_with_invalid_argument_type_rejected_pre_dispatch(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.2)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                "reason": "need_bounded_history",
                "uncertainty": ["history_gap"],
                "tool_requests": [
                    {
                        "request_id": "req-explicit-invalid-type",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-llm-override",
                            "max_messages": "5",
                        },
                    }
                ],
            },
        )
        fake_transport = _FakeTransport(
            submit_response={"accepted": True, "rejected": False, "duplicate": False, "reason": None}
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
                evaluated = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-explicit-invalid-type",
                    cycle_id="cycle-explicit-invalid-type",
                )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("phase"), "tool_request_pre_dispatch_rejected")
        self.assertEqual(decision.get("error_code"), "invalid_tool_arguments")
        self.assertIn(
            "track_evidence_history_max_messages_invalid",
            decision.get("validation_errors", []),
        )
        self.assertEqual(len(fake_transport.submitted), 0)

    def test_valid_explicit_predict_bounded_is_preferred(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.1)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_PREDICT_BOUNDED,
                "reason": "bounded_projection_ok",
                "uncertainty": ["minor_occlusion"],
                "tool_requests": [],
            },
        )
        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
            evaluated = ops._evaluate_track_for_cycle(
                "track-1",
                request_evidence_on_uncertainty=True,
                correlation_id="corr-explicit-predict",
                cycle_id="cycle-explicit-predict",
            )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_PREDICT_BOUNDED)
        self.assertEqual(decision.get("decision_source"), "llm_explicit")

    def test_valid_explicit_inconclusive_is_preferred(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.1)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_INCONCLUSIVE,
                "reason": "insufficient_consistency",
                "uncertainty": ["identity_ambiguity"],
                "tool_requests": [],
            },
        )
        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
            evaluated = ops._evaluate_track_for_cycle(
                "track-1",
                request_evidence_on_uncertainty=True,
                correlation_id="corr-explicit-inconclusive",
                cycle_id="cycle-explicit-inconclusive",
            )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_INCONCLUSIVE)
        self.assertEqual(decision.get("decision_source"), "llm_explicit")

    def test_missing_explicit_decision_uses_legacy_inference(self) -> None:
        base = self._base_result(needs_more_observations=True, speed=1.0)
        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=base):
            evaluated = ops._evaluate_track_for_cycle(
                "track-1",
                request_evidence_on_uncertainty=True,
                correlation_id="corr-missing-explicit",
                cycle_id="cycle-missing-explicit",
            )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("decision_source"), "llm_summary_inference")

    def test_invalid_explicit_tool_request_is_terminal_conservative(self) -> None:
        base = self._base_result(needs_more_observations=True, speed=1.0)
        payload = dict(base)
        summary = dict(payload.get("summary", {}))
        summary["observation_decision_present"] = True
        summary["observation_decision_validation_error"] = "decision_tool_name_not_allowed"
        payload["summary"] = summary

        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=payload):
            evaluated = ops._evaluate_track_for_cycle(
                "track-1",
                request_evidence_on_uncertainty=True,
                correlation_id="corr-invalid-explicit",
                cycle_id="cycle-invalid-explicit",
            )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_INCONCLUSIVE)
        self.assertEqual(decision.get("decision_source"), "invalid_explicit_decision")
        self.assertEqual(decision.get("phase"), "terminal_invalid_explicit_decision")
        self.assertNotIn("observation_tool_request", evaluated["result"])

    def test_explicit_request_with_budget_exhaustion_rejects_safely(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.3)
        explicit = self._attach_explicit_decision(
            base,
            {
                "action": ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                "reason": "need_evidence",
                "uncertainty": ["history_gap"],
                "tool_requests": [
                    {
                        "request_id": "req-explicit-budget",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-1",
                        },
                    }
                ],
            },
        )
        fake_transport = _FakeTransport(
            submit_response={"accepted": False, "rejected": True, "duplicate": False, "reason": "tool_budget_exhausted"}
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=explicit):
                evaluated = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-budget-exhausted",
                    cycle_id="cycle-budget-exhausted",
                )

        decision = evaluated["result"].get("observation_decision", {})
        self.assertEqual(decision.get("phase"), "tool_request_rejected")
        self.assertIn(decision.get("action"), {ops.OBSERVATION_ACTION_PREDICT_BOUNDED, ops.OBSERVATION_ACTION_INCONCLUSIVE})

    def test_valid_request_then_followup_result(self) -> None:
        first_result = self._base_result(needs_more_observations=True, speed=1.0)
        followup_result_event = {
            "status": "ok",
            "error_code": None,
            "error": None,
            "idempotency_key": "inv:req:get_track_evidence_history",
            "result_payload": {
                "available": True,
                "state": "recorded_observation",
                "messages": [
                    {
                        "message_id": "msg-1",
                        "bounded_history": [
                            {"observation_id": "obs-3"},
                        ],
                    }
                ],
            },
        }
        fake_transport = _FakeTransport(
            submit_response={"accepted": True, "rejected": False, "duplicate": False, "reason": None},
            latest_result=None,
        )

        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=first_result):
                first_eval = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-2",
                    cycle_id="cycle-2",
                )

        self.assertIn("observation_tool_request", first_eval["result"])
        self.assertEqual(
            first_eval["result"]["observation_decision"].get("phase"),
            "tool_request_submitted",
        )

        fake_transport.latest_result = followup_result_event
        follow_up_payload = {
            "summary": "Evidence-grounded update.",
            "measured_facts": ["deterministic_speed_preserved"],
            "interpretations": ["additional_history_supports_assessment"],
            "hypotheses": [],
            "supporting_observation_ids": ["obs-1", "obs-2", "obs-3"],
            "conflicting_evidence": [],
            "uncertainties": [],
            "needs_more_observations": False,
        }
        follow_up_meta = {
            "status": "ok",
            "model": "gpt-oss-120b",
            "latency_ms": 8.0,
            "error_code": None,
            "attempts": 1,
            "retry_count": 0,
        }

        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=first_result):
                with patch(
                    "agents.orchestration.tools.ops.dashboard_reasoning._call_observation_agent_with_retry",
                    return_value=(follow_up_payload, follow_up_meta),
                ):
                    second_eval = ops._evaluate_track_for_cycle(
                        "track-1",
                        request_evidence_on_uncertainty=True,
                        correlation_id="corr-2",
                        cycle_id="cycle-3",
                    )

        self.assertIn("observation_tool_result", second_eval["result"])
        self.assertEqual(
            second_eval["result"]["observation_decision"].get("phase"),
            "follow_up_terminal",
        )
        self.assertEqual(len(fake_transport.registered_handoffs), 1)
        handoff = fake_transport.registered_handoffs[0]
        self.assertEqual(handoff.get("track_id"), "track-1")
        self.assertEqual(handoff.get("idempotency_key"), "inv:req:get_track_evidence_history")
        assessment = handoff.get("assessment")
        self.assertIsInstance(assessment, dict)
        assert isinstance(assessment, dict)
        self.assertEqual(assessment.get("state"), "observation_follow_up_completed")
        self.assertIn("obs-3", assessment.get("supporting_observation_ids", []))
        self.assertFalse(assessment.get("provenance", {}).get("identity_mutation_authorized", True))

    def test_followup_explicit_request_is_terminal_and_not_reissued(self) -> None:
        first_result = self._base_result(needs_more_observations=True, speed=1.0)
        fake_transport = _FakeTransport(
            submit_response={"accepted": True, "rejected": False, "duplicate": False, "reason": None},
            latest_result=None,
        )

        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=first_result):
                first_eval = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-followup-terminal",
                    cycle_id="cycle-followup-terminal-1",
                )

        self.assertEqual(first_eval["result"]["observation_decision"].get("phase"), "tool_request_submitted")

        fake_transport.latest_result = {
            "status": "ok",
            "error_code": None,
            "error": None,
            "idempotency_key": "inv:req:get_track_evidence_history",
            "result_payload": {"available": True, "messages": []},
        }
        follow_up_payload = {
            "summary": "Still uncertain, asks for more.",
            "measured_facts": [],
            "interpretations": [],
            "hypotheses": [],
            "supporting_observation_ids": ["obs-1", "obs-2"],
            "conflicting_evidence": [],
            "uncertainties": ["more_data_needed"],
            "needs_more_observations": True,
            "decision": {
                "action": ops.OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
                "reason": "request_more_again",
                "uncertainty": ["more_data_needed"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {"track_id": "track-1"},
                    }
                ],
            },
        }
        follow_up_meta = {
            "status": "ok",
            "model": "gpt-oss-120b",
            "latency_ms": 8.0,
            "error_code": None,
            "attempts": 1,
            "retry_count": 0,
        }

        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=first_result):
                with patch(
                    "agents.orchestration.tools.ops.dashboard_reasoning._call_observation_agent_with_retry",
                    return_value=(follow_up_payload, follow_up_meta),
                ):
                    second_eval = ops._evaluate_track_for_cycle(
                        "track-1",
                        request_evidence_on_uncertainty=True,
                        correlation_id="corr-followup-terminal",
                        cycle_id="cycle-followup-terminal-2",
                    )

        decision = second_eval["result"].get("observation_decision", {})
        self.assertEqual(decision.get("decision_source"), "invalid_explicit_decision")
        self.assertEqual(decision.get("phase"), "terminal_invalid_explicit_decision")
        self.assertEqual(decision.get("validation_error"), "decision_request_not_allowed_in_follow_up")
        self.assertEqual(len(fake_transport.submitted), 1)

    def test_rejected_tool_request_uses_bounded_prediction_when_supported(self) -> None:
        base = self._base_result(needs_more_observations=True, speed=1.5)
        fake_transport = _FakeTransport(
            submit_response={"accepted": False, "rejected": True, "duplicate": False, "reason": "invalid_tool_arguments"}
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=base):
                evaluated = ops._evaluate_track_for_cycle(
                    "track-1",
                    request_evidence_on_uncertainty=True,
                    correlation_id="corr-4",
                    cycle_id="cycle-4",
                )
        decision = evaluated["result"].get("observation_decision")
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_PREDICT_BOUNDED)

    def test_timeout_or_unavailable_evidence_can_end_inconclusive(self) -> None:
        base = self._base_result(needs_more_observations=True, speed=None)
        ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS["track-1"] = {
            "investigation_id": "inv-1",
            "request_id": "req-1",
            "tool_name": "get_track_evidence_history",
            "created_at": 1000.0,
        }
        fake_transport = _FakeTransport(
            latest_result={
                "status": "timeout",
                "error_code": "tool_timeout",
                "error": "tool_timeout",
                "result_payload": {"available": False},
            }
        )

        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=base):
                with patch(
                    "agents.orchestration.tools.ops.dashboard_reasoning._call_observation_agent_with_retry",
                    return_value=(None, {"status": "error", "error_code": "timeout"}),
                ):
                    evaluated = ops._evaluate_track_for_cycle(
                        "track-1",
                        request_evidence_on_uncertainty=True,
                        correlation_id="corr-5",
                        cycle_id="cycle-5",
                    )

        decision = evaluated["result"].get("observation_decision")
        self.assertIn(decision.get("action"), {ops.OBSERVATION_ACTION_INCONCLUSIVE, ops.OBSERVATION_ACTION_PREDICT_BOUNDED})

    def test_inconclusive_when_prediction_not_supported_and_needs_more_observations(self) -> None:
        decision = ops._infer_observation_decision_from_summary(
            summary_payload={
                "track_id": "track-1",
                "needs_more_observations": True,
                "sample_count": 1,
                "speed": None,
                "uncertainties": ["insufficient_history"],
            },
            request_evidence_on_uncertainty=False,
        )
        self.assertEqual(decision.get("action"), ops.OBSERVATION_ACTION_INCONCLUSIVE)

    def test_numeric_fabrication_attempt_does_not_override_deterministic_values(self) -> None:
        fallback_summary = {
            "track_id": "track-1",
            "created_at": 1000.0,
            "sample_count": 5,
            "velocity_x": 1.25,
            "velocity_z": 0.5,
            "speed": 1.35,
            "confidence": 0.9,
            "supporting_observation_ids": ["obs-1"],
            "uncertainties": [],
            "needs_more_observations": False,
        }
        fabricated_payload = {
            "summary": "Fake numeric overrides.",
            "measured_facts": ["speed=999"],
            "interpretations": ["fabricated"],
            "hypotheses": [],
            "supporting_observation_ids": ["obs-1"],
            "conflicting_evidence": [],
            "uncertainties": [],
            "needs_more_observations": False,
            "speed": 999.0,
            "velocity_x": 999.0,
            "velocity_z": 999.0,
        }

        normalized = agent_observation.validate_observation_llm_payload(
            fabricated_payload,
            fallback_summary=fallback_summary,
            valid_observation_ids={"obs-1"},
        )
        self.assertEqual(normalized.get("speed"), 1.35)
        self.assertEqual(normalized.get("velocity_x"), 1.25)
        self.assertEqual(normalized.get("velocity_z"), 0.5)

    def test_validate_preserves_valid_explicit_decision(self) -> None:
        fallback_summary = {
            "track_id": "track-1",
            "created_at": 1000.0,
            "sample_count": 5,
            "velocity_x": 1.25,
            "velocity_z": 0.5,
            "speed": 1.35,
            "confidence": 0.9,
            "supporting_observation_ids": ["obs-1"],
            "uncertainties": [],
            "needs_more_observations": False,
        }
        payload = {
            "summary": "Explicit decision payload.",
            "measured_facts": [],
            "interpretations": [],
            "hypotheses": [],
            "supporting_observation_ids": ["obs-1"],
            "conflicting_evidence": [],
            "uncertainties": [],
            "needs_more_observations": False,
            "decision": {
                "action": "assess_now",
                "reason": "enough_evidence",
                "uncertainty": [],
                "tool_requests": [],
            },
        }

        normalized = agent_observation.validate_observation_llm_payload(
            payload,
            fallback_summary=fallback_summary,
            valid_observation_ids={"obs-1"},
        )
        decision = normalized.get("observation_decision")
        self.assertIsInstance(decision, dict)
        self.assertEqual(decision.get("action"), "assess_now")
        self.assertIsNone(normalized.get("observation_decision_validation_error"))

    def test_validate_rejects_invalid_explicit_tool_request(self) -> None:
        fallback_summary = {
            "track_id": "track-1",
            "created_at": 1000.0,
            "sample_count": 5,
            "velocity_x": 1.25,
            "velocity_z": 0.5,
            "speed": 1.35,
            "confidence": 0.9,
            "supporting_observation_ids": ["obs-1"],
            "uncertainties": [],
            "needs_more_observations": True,
        }
        payload = {
            "summary": "Invalid tool decision payload.",
            "measured_facts": [],
            "interpretations": [],
            "hypotheses": [],
            "supporting_observation_ids": ["obs-1"],
            "conflicting_evidence": [],
            "uncertainties": ["history_gap"],
            "needs_more_observations": True,
            "decision": {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more",
                "uncertainty": ["history_gap"],
                "tool_requests": [
                    {
                        "tool_name": "unauthorized_tool",
                        "arguments": {"track_id": "track-1"},
                    }
                ],
            },
        }

        normalized = agent_observation.validate_observation_llm_payload(
            payload,
            fallback_summary=fallback_summary,
            valid_observation_ids={"obs-1"},
        )
        self.assertEqual(
            normalized.get("observation_decision_validation_error"),
            "decision_tool_request_required",
        )
        self.assertNotIn("observation_decision", normalized)

    def test_validate_explicit_decision_with_blank_reason_uses_reason_missing_fallback(self) -> None:
        fallback_summary = {
            "track_id": "track-1",
            "created_at": 1000.0,
            "sample_count": 5,
            "velocity_x": 1.25,
            "velocity_z": 0.5,
            "speed": 1.35,
            "confidence": 0.9,
            "supporting_observation_ids": ["obs-1"],
            "uncertainties": [],
            "needs_more_observations": False,
        }
        payload = {
            "summary": "Explicit decision payload with blank reason.",
            "measured_facts": [],
            "interpretations": [],
            "hypotheses": [],
            "supporting_observation_ids": ["obs-1"],
            "conflicting_evidence": [],
            "uncertainties": [],
            "needs_more_observations": False,
            "decision": {
                "action": "assess_now",
                "reason": "   ",
                "uncertainty": [],
                "tool_requests": [],
            },
        }

        normalized = agent_observation.validate_observation_llm_payload(
            payload,
            fallback_summary=fallback_summary,
            valid_observation_ids={"obs-1"},
        )
        decision = normalized.get("observation_decision")
        self.assertIsInstance(decision, dict)
        self.assertEqual(decision.get("action"), "assess_now")
        self.assertEqual(decision.get("reason"), "decision_reason_missing")
        self.assertIsNone(normalized.get("observation_decision_validation_error"))

    def test_malformed_followup_llm_response_keeps_contract(self) -> None:
        base = self._base_result(needs_more_observations=True, speed=1.2)
        ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS["track-1"] = {
            "investigation_id": "inv-1",
            "request_id": "req-1",
            "tool_name": "get_track_evidence_history",
            "created_at": 1000.0,
        }
        fake_transport = _FakeTransport(
            latest_result={
                "status": "ok",
                "error_code": None,
                "error": None,
                "result_payload": {"available": True, "messages": []},
            }
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=base):
                with patch(
                    "agents.orchestration.tools.ops.dashboard_reasoning._call_observation_agent_with_retry",
                    return_value=(None, {"status": "error", "error_code": "llm_invalid_json_response"}),
                ):
                    evaluated = ops._evaluate_track_for_cycle(
                        "track-1",
                        request_evidence_on_uncertainty=True,
                        correlation_id="corr-6",
                        cycle_id="cycle-6",
                    )

        self.assertIn("summary", evaluated["result"])
        self.assertIn("assessment", evaluated["result"])

    def test_observation_to_reasoning_handoff_contract_remains_present(self) -> None:
        base = self._base_result(needs_more_observations=False, speed=1.0)
        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=base):
            evaluated = ops._evaluate_track_for_cycle(
                "track-1",
                request_evidence_on_uncertainty=True,
                correlation_id="corr-7",
                cycle_id="cycle-7",
            )

        result = evaluated["result"]
        self.assertIsInstance(result.get("summary"), dict)
        self.assertIsInstance(result.get("assessment"), dict)
        self.assertIsInstance(result.get("agents"), dict)

    def test_duplicate_latest_result_is_ignored_when_nonduplicate_exists(self) -> None:
        ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS["track-1"] = {
            "investigation_id": "inv-dup",
            "request_id": "req-dup",
            "tool_name": "get_track_evidence_history",
            "created_at": 1000.0,
        }
        fake_transport = _FakeTransport(
            results=[
                {
                    "status": "duplicate",
                    "finished_at": 1000.3,
                    "result_payload": {"available": False},
                },
                {
                    "status": "ok",
                    "finished_at": 1000.2,
                    "result_payload": {"available": True},
                },
            ]
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            result = ops._extract_observation_tool_result_for_pending("track-1")

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "ok")

    def test_result_older_than_pending_created_at_is_ignored(self) -> None:
        ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS["track-1"] = {
            "investigation_id": "inv-stale",
            "request_id": "req-stale",
            "tool_name": "get_track_evidence_history",
            "created_at": 2000.0,
        }
        fake_transport = _FakeTransport(
            results=[
                {
                    "status": "stale",
                    "finished_at": 1000.1,
                    "result_payload": {"available": False},
                }
            ]
        )
        with patch("agents.orchestration.tools.ops._observation_ipc_transport", return_value=fake_transport):
            result = ops._extract_observation_tool_result_for_pending("track-1")

        self.assertIsNone(result)

    def test_pending_followup_timeout_falls_back_and_clears_pending(self) -> None:
        base = self._base_result(needs_more_observations=True, speed=1.3)
        ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS["track-1"] = {
            "investigation_id": "inv-timeout",
            "request_id": "req-timeout",
            "tool_name": "get_track_evidence_history",
            "created_at": 0.0,
        }

        with patch("agents.orchestration.tools.ops.dashboard_reasoning.simulate_reasoning_assessment", return_value=base):
            with patch("agents.orchestration.tools.ops.OBSERVATION_TOOL_REQUEST_TIMEOUT_SECONDS", 1.5):
                with patch("agents.orchestration.tools.ops.time.time", return_value=10.0):
                    evaluated = ops._evaluate_track_for_cycle(
                        "track-1",
                        request_evidence_on_uncertainty=True,
                        correlation_id="corr-timeout",
                        cycle_id="cycle-timeout",
                    )

        decision = evaluated["result"].get("observation_decision")
        self.assertIsInstance(decision, dict)
        self.assertEqual(decision.get("phase"), "tool_result_timeout")
        self.assertIn(decision.get("action"), {ops.OBSERVATION_ACTION_PREDICT_BOUNDED, ops.OBSERVATION_ACTION_INCONCLUSIVE})
        self.assertNotIn("track-1", ops._PENDING_OBSERVATION_TOOL_FOLLOWUPS)


if __name__ == "__main__":
    unittest.main()
