from __future__ import annotations

import json
import os
import tempfile
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch

import tools.observation_autonomy_provider_harness as harness
from agents.observation.tools.track_evidence_history import get_track_evidence_history
from agents.observation.tools.track_evidence_history_contract import (
    TRACK_EVIDENCE_HISTORY_MODEL_SELECTABLE_ARGUMENTS,
    TRACK_EVIDENCE_HISTORY_SUPPORTED_ARGUMENTS,
    TRACK_EVIDENCE_HISTORY_TRUSTED_METADATA_ARGUMENTS,
    validate_track_evidence_history_arguments,
)
from tools.observation_autonomy_provider_harness import run_single_investigation


class ObservationProviderHarnessPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.fixture_path = (
            self.project_root
            / "tests"
            / "fixtures"
            / "observation_autonomy"
            / "single_investigation_evidence_gap.json"
        )

    def _fixture_payload(self) -> dict:
        return json.loads(self.fixture_path.read_text(encoding="utf-8"))

    def _write_fixture(self, run_dir: Path, payload: dict) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        fixture = run_dir / "fixture.json"
        fixture.write_text(json.dumps(payload), encoding="utf-8")
        return fixture

    def _run_concise_preflight_with_first_decision(self, decision_payload: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["preflight_observation_responses"][0]["decision"] = decision_payload
            fixture = self._write_fixture(run_dir, payload)
            return run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
                concise_prompts=True,
            )

    def test_preflight_harness_report_and_budgets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            report = run_single_investigation(
                fixture_path=self.fixture_path,
                run_dir=run_dir,
                preflight_only=True,
            )

            self.assertEqual(report.get("status"), "ok")
            investigation = report.get("investigation", {})

            self.assertEqual(investigation.get("observation_llm_call_count"), 2)
            self.assertEqual(investigation.get("mcp_evidence_call_count"), 1)
            self.assertEqual(investigation.get("provider_call_attempted_total"), 0)

            by_role = investigation.get("provider_call_attempted_by_role", {})
            self.assertEqual(by_role.get("observation_text", 0), 0)
            self.assertEqual(by_role.get("tracking_vlm", 0), 0)
            self.assertEqual(by_role.get("reasoning_text", 0), 0)

            validation = investigation.get("validation", {})
            self.assertTrue(validation.get("deterministic_measurements_protected"))
            self.assertTrue(validation.get("follow_up_received_correlated_evidence"))
            self.assertEqual(investigation.get("first_pass_decision_source"), "llm_summary_inference")
            self.assertEqual(investigation.get("explicit_autonomy_demonstrated"), False)
            self.assertEqual(investigation.get("successful_explicit_observation_autonomy"), False)
            self.assertEqual(investigation.get("evidence_retrieval_status"), "retrieved")
            self.assertEqual(investigation.get("outcome"), "successful_evidence_based_assessment")

            self.assertTrue((run_dir / "observation_autonomy_harness_report.json").is_file())
            self.assertTrue((run_dir / "tracking_decision_tool_requests.jsonl").is_file())
            self.assertTrue((run_dir / "tracking_decision_tool_results.jsonl").is_file())

    def test_preflight_scheduler_non_blocking_iterations(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            report = run_single_investigation(
                fixture_path=self.fixture_path,
                run_dir=run_dir,
                preflight_only=True,
            )
            self.assertEqual(report.get("status"), "ok")
            scheduler = report.get("investigation", {}).get("scheduler", {})
            self.assertGreaterEqual(int(scheduler.get("tracking_iterations", 0)), 2)
            self.assertTrue(bool(scheduler.get("stopped_by_keyboard_interrupt", False)))

    def test_preflight_report_is_persisted_and_parseable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            report = run_single_investigation(
                fixture_path=self.fixture_path,
                run_dir=run_dir,
                preflight_only=True,
            )
            self.assertEqual(report.get("status"), "ok")

            persisted = run_dir / "observation_autonomy_harness_report.json"
            self.assertTrue(persisted.is_file())
            payload = json.loads(persisted.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("status"), "ok")
            self.assertEqual(payload.get("mode"), "preflight")
            self.assertEqual(payload.get("live_mode_enabled"), False)
            safety = payload.get("safety", {})
            self.assertEqual(safety.get("camera_interaction"), "disabled")
            self.assertEqual(safety.get("tracking_identity_mutation"), "disabled")

    def test_live_mode_uses_observation_provider_path_not_preflight_invoke_fn(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            captured_kwargs: list[dict] = []

            def _fake_call(user_payload: dict, **kwargs: dict):
                captured_kwargs.append(dict(kwargs))
                return (
                    {
                        "summary": "live-mode test response",
                        "measured_facts": [],
                        "interpretations": [],
                        "hypotheses": [],
                        "supporting_observation_ids": ["obs-100"],
                        "conflicting_evidence": [],
                        "uncertainties": [],
                        "needs_more_observations": False,
                    },
                    {
                        "status": "ok",
                        "model": "test-model",
                        "latency_ms": 0.1,
                        "error": None,
                        "error_code": None,
                        "attempts": 1,
                        "retry_count": 0,
                    },
                )

            with patch("tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry", side_effect=_fake_call):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

            self.assertEqual(report.get("status"), "ok")
            self.assertEqual(report.get("mode"), "live_provider")
            self.assertEqual(report.get("live_mode_enabled"), True)
            self.assertGreaterEqual(len(captured_kwargs), 1)
            self.assertNotIn("invoke_fn", captured_kwargs[0])

    def test_live_mode_rejects_nonempty_run_dir_before_any_provider_call(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "existing_artifact.txt").write_text("stale", encoding="utf-8")

            with patch("tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry") as call_mock:
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

            self.assertEqual(report.get("status"), "error")
            self.assertEqual(report.get("error"), "RuntimeError:live_run_dir_not_empty")
            self.assertEqual(report.get("mode"), "live_provider")
            call_mock.assert_not_called()

    def test_live_mode_allows_fresh_run_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "fresh-run"
            captured_calls = 0

            def _fake_call(user_payload: dict, **kwargs: dict):
                nonlocal captured_calls
                captured_calls += 1
                return (
                    {
                        "summary": "live-mode fresh dir response",
                        "measured_facts": [],
                        "interpretations": [],
                        "hypotheses": [],
                        "supporting_observation_ids": ["obs-100"],
                        "conflicting_evidence": [],
                        "uncertainties": [],
                        "needs_more_observations": False,
                        "decision": {
                            "action": "assess_now",
                            "reason": "enough_evidence",
                            "uncertainty": [],
                            "tool_requests": [],
                        },
                    },
                    {
                        "status": "ok",
                        "model": "test-model",
                        "latency_ms": 0.1,
                        "error": None,
                        "error_code": None,
                        "attempts": 1,
                        "retry_count": 0,
                    },
                )

            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                    concise_prompts=True,
                )

            self.assertEqual(report.get("status"), "ok")
            self.assertEqual(report.get("mode"), "live_provider")
            self.assertGreaterEqual(captured_calls, 1)

    def test_explicit_request_decision_demonstrates_tool_request_autonomy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["preflight_observation_responses"][0]["decision"] = {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": payload["track_id"],
                            "max_messages": 5,
                            "max_history_records": 12,
                            "max_scan_records": 120,
                        },
                    }
                ],
            }
            payload["preflight_observation_responses"][1]["decision"] = {
                "action": "inconclusive",
                "reason": "terminal_after_followup",
                "uncertainty": ["residual_risk"],
                "tool_requests": [],
            }
            fixture = self._write_fixture(run_dir, payload)
            report = run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
            )

            self.assertEqual(report.get("status"), "ok")
            investigation = report.get("investigation", {})
            self.assertEqual(investigation.get("first_pass_decision_source"), "llm_explicit")
            self.assertEqual(investigation.get("explicit_autonomy_demonstrated"), True)
            self.assertEqual(investigation.get("tool_request_autonomy_demonstrated"), True)
            self.assertEqual(investigation.get("successful_explicit_observation_autonomy"), True)
            self.assertEqual(investigation.get("final_decision", {}).get("phase"), "follow_up_terminal")
            classification = investigation.get("classification", {})
            self.assertTrue(classification.get("explicit_tool_request_validated"))
            self.assertTrue(classification.get("explicit_tool_request_dispatched"))
            self.assertTrue(classification.get("explicit_tool_request_followed_by_terminal_second_pass"))

    def test_explicit_non_request_decision_does_not_claim_tool_request_autonomy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["preflight_observation_responses"][0]["needs_more_observations"] = False
            payload["preflight_observation_responses"][0]["decision"] = {
                "action": "assess_now",
                "reason": "enough_evidence",
                "uncertainty": [],
                "tool_requests": [],
            }
            fixture = self._write_fixture(run_dir, payload)
            report = run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
            )

            self.assertEqual(report.get("status"), "ok")
            investigation = report.get("investigation", {})
            self.assertEqual(investigation.get("first_pass_decision_source"), "llm_explicit")
            self.assertEqual(investigation.get("explicit_autonomy_demonstrated"), True)
            self.assertEqual(investigation.get("tool_request_autonomy_demonstrated"), False)
            classification = investigation.get("classification", {})
            self.assertTrue(classification.get("explicit_non_request_decision"))

    def test_explicit_predict_bounded_without_support_is_downgraded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["summary_measurements"]["sample_count"] = 1
            payload["summary_measurements"]["speed"] = None
            payload["preflight_observation_responses"][0]["needs_more_observations"] = True
            payload["preflight_observation_responses"][0]["decision"] = {
                "action": "predict_bounded",
                "reason": "attempt_prediction",
                "uncertainty": ["limited_history"],
                "tool_requests": [],
            }
            fixture = self._write_fixture(run_dir, payload)
            report = run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
            )

            self.assertEqual(report.get("status"), "ok")
            investigation = report.get("investigation", {})
            first_decision = investigation.get("first_decision", {})
            self.assertEqual(first_decision.get("action"), "inconclusive")
            self.assertEqual(first_decision.get("reason"), "predict_bounded_not_supported_by_deterministic_evidence")
            self.assertTrue(bool(first_decision.get("predict_bounded_downgraded")))
            classification = investigation.get("classification", {})
            self.assertTrue(classification.get("predict_bounded_downgraded"))

    def test_harness_applies_output_token_override_during_run_and_restores_after(self) -> None:
        previous = os.environ.get("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS")
        os.environ["OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"] = "321"
        observed_values: list[str | None] = []
        try:
            with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
                run_dir = Path(temp_dir) / "run"

                def _fake_call(user_payload: dict, **kwargs: dict):
                    observed_values.append(os.environ.get("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"))
                    return (
                        {
                            "summary": "override check",
                            "measured_facts": [],
                            "interpretations": [],
                            "hypotheses": [],
                            "supporting_observation_ids": ["obs-100"],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {
                            "status": "ok",
                            "model": "test-model",
                            "latency_ms": 0.1,
                            "error": None,
                            "error_code": None,
                            "attempts": 1,
                            "retry_count": 0,
                        },
                    )

                with patch(
                    "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                    side_effect=_fake_call,
                ):
                    report = run_single_investigation(
                        fixture_path=self.fixture_path,
                        run_dir=run_dir,
                        preflight_only=False,
                    )

                self.assertEqual(report.get("status"), "ok")
                self.assertGreaterEqual(len(observed_values), 1)
                self.assertTrue(all(value == "1024" for value in observed_values))
                persisted = json.loads((run_dir / "observation_autonomy_harness_report.json").read_text(encoding="utf-8"))
                pass_attempts = persisted.get("investigation", {}).get("pass_attempts", {})
                self.assertIsInstance(pass_attempts, dict)
                self.assertTrue(bool(pass_attempts.get("pass1", {}).get("attempted", False)))
        finally:
            self.assertEqual(os.environ.get("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"), "321")
            if previous is None:
                os.environ.pop("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS", None)
            else:
                os.environ["OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"] = previous

    def test_harness_restores_environment_after_exception(self) -> None:
        previous = os.environ.get("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS")
        os.environ["OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"] = "654"
        observed_values: list[str | None] = []
        try:
            with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
                run_dir = Path(temp_dir) / "run"

                def _failing_call(user_payload: dict, **kwargs: dict):
                    observed_values.append(os.environ.get("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"))
                    raise RuntimeError("forced_failure")

                with patch(
                    "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                    side_effect=_failing_call,
                ):
                    report = run_single_investigation(
                        fixture_path=self.fixture_path,
                        run_dir=run_dir,
                        preflight_only=False,
                    )

                self.assertEqual(report.get("status"), "error")
                self.assertGreaterEqual(len(observed_values), 1)
                self.assertTrue(all(value == "1024" for value in observed_values))
        finally:
            self.assertEqual(os.environ.get("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"), "654")
            if previous is None:
                os.environ.pop("OBSERVATION_AGENT_MAX_OUTPUT_TOKENS", None)
            else:
                os.environ["OBSERVATION_AGENT_MAX_OUTPUT_TOKENS"] = previous

    def test_harness_enforces_single_attempt_per_pass_retry_limit_zero(self) -> None:
        captured_kwargs: list[dict] = []

        def _fake_call(user_payload: dict, **kwargs: dict):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                return (
                    {
                        "summary": "Need follow-up evidence.",
                        "measured_facts": [],
                        "interpretations": [],
                        "hypotheses": [],
                        "supporting_observation_ids": ["obs-100", "obs-101"],
                        "conflicting_evidence": [],
                        "uncertainties": ["occlusion"],
                        "needs_more_observations": True,
                        "decision": {
                            "action": "request_additional_tracking_evidence",
                            "reason": "need_history",
                            "uncertainty": ["occlusion"],
                            "tool_requests": [
                                {
                                    "tool_name": "get_track_evidence_history",
                                    "arguments": {
                                        "track_id": "fixture-track-001",
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "status": "ok",
                        "model": "test-model",
                        "latency_ms": 0.1,
                        "error": None,
                        "error_code": None,
                        "attempts": 1,
                        "retry_count": 0,
                    },
                )
            return (
                {
                    "summary": "Terminal follow-up assessment.",
                    "measured_facts": [],
                    "interpretations": [],
                    "hypotheses": [],
                    "supporting_observation_ids": ["obs-100", "obs-101", "obs-102"],
                    "conflicting_evidence": [],
                    "uncertainties": [],
                    "needs_more_observations": False,
                    "decision": {
                        "action": "assess_now",
                        "reason": "enough_evidence",
                        "uncertainty": [],
                        "tool_requests": [],
                    },
                },
                {
                    "status": "ok",
                    "model": "test-model",
                    "latency_ms": 0.1,
                    "error": None,
                    "error_code": None,
                    "attempts": 1,
                    "retry_count": 0,
                },
            )

        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

        self.assertEqual(report.get("status"), "ok")
        self.assertEqual(len(captured_kwargs), 2)
        self.assertEqual(captured_kwargs[0].get("retry_limit"), 0)
        self.assertEqual(captured_kwargs[1].get("retry_limit"), 0)

    def test_first_pass_output_truncated_skips_second_pass_with_phase_diagnostics(self) -> None:
        captured_kwargs: list[dict] = []

        def _fake_call(user_payload: dict, **kwargs: dict):
            captured_kwargs.append(dict(kwargs))
            return (
                None,
                {
                    "status": "error",
                    "model": "test-model",
                    "latency_ms": 0.2,
                    "error": "RuntimeError:llm_output_truncated",
                    "error_code": "output_truncated",
                    "attempts": 1,
                    "retry_count": 0,
                },
            )

        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

        self.assertEqual(report.get("status"), "error")
        self.assertEqual(len(captured_kwargs), 1)
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("phase_failure", {}).get("code"), "first_pass_output_truncated")
        self.assertEqual(investigation.get("phase_failure", {}).get("phase"), "pass1")
        second_pass = investigation.get("second_pass", {})
        self.assertFalse(bool(second_pass.get("attempted", True)))
        self.assertTrue(bool(second_pass.get("skipped", False)))
        self.assertEqual(second_pass.get("skip_reason"), "first_pass_output_truncated")
        pass_attempts = investigation.get("pass_attempts", {})
        pass1 = pass_attempts.get("pass1", {})
        self.assertEqual(pass1.get("llm_attempts"), 1)
        self.assertEqual(pass1.get("retry_count"), 0)
        self.assertTrue(bool(pass1.get("truncated", False)))

    def test_first_pass_provider_failure_does_not_retry_and_skips_second_pass(self) -> None:
        captured_kwargs: list[dict] = []

        def _fake_call(user_payload: dict, **kwargs: dict):
            captured_kwargs.append(dict(kwargs))
            return (
                None,
                {
                    "status": "error",
                    "model": "test-model",
                    "latency_ms": 0.2,
                    "error": "RuntimeError:llm_invalid_json_response",
                    "error_code": "invalid_response_format",
                    "attempts": 1,
                    "retry_count": 0,
                },
            )

        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

        self.assertEqual(report.get("status"), "error")
        self.assertEqual(len(captured_kwargs), 1)
        self.assertEqual(captured_kwargs[0].get("retry_limit"), 0)
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("phase_failure", {}).get("code"), "first_pass_provider_error")
        self.assertTrue(bool(investigation.get("second_pass", {}).get("skipped", False)))

    def test_second_pass_budget_exhaustion_has_phase_specific_classification(self) -> None:
        call_count = 0

        def _fake_call(user_payload: dict, **kwargs: dict):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    {
                        "summary": "Need follow-up evidence.",
                        "measured_facts": [],
                        "interpretations": [],
                        "hypotheses": [],
                        "supporting_observation_ids": ["obs-100", "obs-101"],
                        "conflicting_evidence": [],
                        "uncertainties": ["occlusion"],
                        "needs_more_observations": True,
                        "decision": {
                            "action": "request_additional_tracking_evidence",
                            "reason": "need_history",
                            "uncertainty": ["occlusion"],
                            "tool_requests": [
                                {
                                    "tool_name": "get_track_evidence_history",
                                    "arguments": {
                                        "track_id": "fixture-track-001",
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "status": "ok",
                        "model": "test-model",
                        "latency_ms": 0.1,
                        "error": None,
                        "error_code": None,
                        "attempts": 1,
                        "retry_count": 0,
                    },
                )
            return (
                None,
                {
                    "status": "error",
                    "model": "test-model",
                    "latency_ms": 0.1,
                    "error": "ProviderCallBudgetExceeded:provider_call_budget_exhausted:observation_text:global_limit_exhausted",
                    "error_code": "budget_exhausted",
                    "attempts": 1,
                    "retry_count": 0,
                },
            )

        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

        self.assertEqual(report.get("status"), "error")
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("phase_failure", {}).get("code"), "second_pass_budget_exhausted")
        self.assertEqual(investigation.get("phase_failure", {}).get("phase"), "pass2")
        pass2 = investigation.get("pass_attempts", {}).get("pass2", {})
        self.assertEqual(pass2.get("llm_attempts"), 1)
        self.assertEqual(pass2.get("retry_count"), 0)

    def test_request_builder_enforces_authoritative_track_id_when_llm_args_empty(self) -> None:
        decision = {
            "tool_requests": [
                {
                    "tool_name": "get_track_evidence_history",
                    "arguments": {},
                }
            ]
        }
        built = harness._build_request_decision_from_observation_decision(
            investigation_id="inv-1",
            request_id="req-1",
            track_id="track-authoritative",
            correlation_id="corr-1",
            observation_decision=decision,
        )
        arguments = built["tool_requests"][0]["arguments"]
        self.assertEqual(arguments.get("track_id"), "track-authoritative")
        self.assertEqual(arguments.get("max_messages"), 5)
        self.assertEqual(arguments.get("max_history_records"), 12)
        self.assertEqual(arguments.get("max_scan_records"), 120)

    def test_request_builder_prevents_llm_track_id_override_and_preserves_optional_args(self) -> None:
        decision = {
            "tool_requests": [
                {
                    "tool_name": "get_track_evidence_history",
                    "arguments": {
                        "track_id": "track-llm-override",
                        "max_messages": 7,
                        "max_history_records": 20,
                    },
                }
            ]
        }
        built = harness._build_request_decision_from_observation_decision(
            investigation_id="inv-2",
            request_id="req-2",
            track_id="track-authoritative",
            correlation_id="corr-2",
            observation_decision=decision,
        )
        arguments = built["tool_requests"][0]["arguments"]
        self.assertEqual(arguments.get("track_id"), "track-authoritative")
        self.assertEqual(arguments.get("max_messages"), 7)
        self.assertEqual(arguments.get("max_history_records"), 20)

    def test_unsupported_observation_tool_arguments_are_rejected_by_ipc_validation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["preflight_observation_responses"][0]["decision"] = {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-llm-override",
                            "max_messages": 5,
                            "unsupported_flag": "not_allowed",
                        },
                    }
                ],
            }
            payload["preflight_observation_responses"][1]["decision"] = {
                "action": "inconclusive",
                "reason": "terminal_after_followup",
                "uncertainty": ["residual_risk"],
                "tool_requests": [],
            }
            fixture = self._write_fixture(run_dir, payload)
            report = run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
            )

            self.assertEqual(report.get("status"), "ok")
            investigation = report.get("investigation", {})
            self.assertFalse(bool(investigation.get("request_submitted", False)))
            correlated = investigation.get("correlated_result", {})
            errors = correlated.get("result_payload", {}).get("errors", [])
            self.assertEqual(correlated.get("error_code"), "invalid_tool_arguments")
            self.assertIn("track_evidence_history_unknown_arguments", errors)
            self.assertEqual(investigation.get("mcp_evidence_call_count"), 0)
            self.assertEqual(investigation.get("second_pass", {}).get("attempted"), False)
            self.assertEqual(investigation.get("outcome"), "justified_inconclusive")

    def test_invalid_type_observation_tool_arguments_rejected_pre_dispatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["preflight_observation_responses"][0]["decision"] = {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-llm-override",
                            "max_messages": "5",
                        },
                    }
                ],
            }
            fixture = self._write_fixture(run_dir, payload)
            report = run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
            )

            self.assertEqual(report.get("status"), "ok")
            investigation = report.get("investigation", {})
            self.assertFalse(bool(investigation.get("request_submitted", False)))
            self.assertEqual(investigation.get("mcp_evidence_call_count"), 0)
            self.assertEqual(investigation.get("second_pass", {}).get("attempted"), False)
            correlated = investigation.get("correlated_result", {})
            self.assertEqual(correlated.get("error_code"), "invalid_tool_arguments")
            self.assertIn(
                "track_evidence_history_max_messages_invalid",
                correlated.get("result_payload", {}).get("errors", []),
            )

    def test_track_evidence_history_contract_parity_with_runtime_validator_and_tool_signature(self) -> None:
        runtime_allowed_keys = {
            "track_id",
            "max_messages",
            "max_history_records",
            "max_scan_records",
            "now_timestamp",
            "investigation_id",
            "correlation_id",
        }
        self.assertEqual(TRACK_EVIDENCE_HISTORY_SUPPORTED_ARGUMENTS, runtime_allowed_keys)
        self.assertEqual(
            set(TRACK_EVIDENCE_HISTORY_MODEL_SELECTABLE_ARGUMENTS.keys()),
            {"max_messages", "max_history_records", "max_scan_records"},
        )
        self.assertEqual(
            set(TRACK_EVIDENCE_HISTORY_TRUSTED_METADATA_ARGUMENTS.keys()),
            {"track_id", "now_timestamp", "investigation_id", "correlation_id"},
        )

        signature_parameters = set(inspect.signature(get_track_evidence_history).parameters.keys())
        self.assertTrue(runtime_allowed_keys.issubset(signature_parameters))

    def test_track_evidence_history_contract_accepts_supported_arguments(self) -> None:
        validation = validate_track_evidence_history_arguments(
            {
                "track_id": "fixture-track-001",
                "max_messages": 5,
                "max_history_records": 12,
                "max_scan_records": 120,
                "now_timestamp": 1734023400.5,
                "investigation_id": "inv-1",
                "correlation_id": "corr-1",
            }
        )
        self.assertTrue(bool(validation.get("valid", False)))
        self.assertEqual(validation.get("errors"), [])

    def test_track_evidence_history_contract_rejects_since_timestamp(self) -> None:
        validation = validate_track_evidence_history_arguments(
            {
                "track_id": "fixture-track-001",
                "since_timestamp": 1734023390.0,
            }
        )
        self.assertFalse(bool(validation.get("valid", True)))
        self.assertIn("track_evidence_history_unknown_arguments", validation.get("errors", []))
        self.assertIn("since_timestamp", validation.get("unknown_arguments", []))

    def test_classifies_timeout_missing_and_tool_failure_before_justified_inconclusive(self) -> None:
        final_decision = {"action": "inconclusive"}
        validation = {"allows_inconclusive": True}
        self.assertEqual(
            harness._classify_investigation_outcome(
                evidence_retrieval_status="timeout",
                final_decision=final_decision,
                validation=validation,
            ),
            "tool_timeout_inconclusive",
        )
        self.assertEqual(
            harness._classify_investigation_outcome(
                evidence_retrieval_status="missing",
                final_decision=final_decision,
                validation=validation,
            ),
            "missing_evidence_inconclusive",
        )
        self.assertEqual(
            harness._classify_investigation_outcome(
                evidence_retrieval_status="failed",
                final_decision=final_decision,
                validation=validation,
            ),
            "tool_failure_inconclusive",
        )

    def test_successful_retrieval_inconclusive_stays_justified(self) -> None:
        outcome = harness._classify_investigation_outcome(
            evidence_retrieval_status="retrieved",
            final_decision={"action": "inconclusive"},
            validation={"allows_inconclusive": True},
        )
        self.assertEqual(outcome, "justified_inconclusive")

    def test_successful_retrieval_assessment_classification(self) -> None:
        outcome = harness._classify_investigation_outcome(
            evidence_retrieval_status="retrieved",
            final_decision={"action": "assess_now"},
            validation={"allows_inconclusive": True},
        )
        self.assertEqual(outcome, "successful_evidence_based_assessment")

    def test_terminal_provenance_explicit_with_evidence_support(self) -> None:
        provenance = harness._classify_terminal_decision_provenance(
            first_pass_decision_source="llm_explicit",
            evidence_retrieval_status="retrieved",
            final_decision={
                "action": "inconclusive",
                "reason": "residual_occlusion",
                "decision_source": "llm_explicit",
                "phase": "follow_up_terminal",
            },
            final_summary={"supporting_observation_ids": ["obs-102"]},
            correlated_result={
                "result_payload": {
                    "messages": [
                        {
                            "bounded_history": [
                                {"observation_id": "obs-102"},
                            ]
                        }
                    ]
                }
            },
        )
        self.assertEqual(provenance, "explicit_terminal_with_evidence_support")

    def test_terminal_provenance_explicit_missing_reason(self) -> None:
        provenance = harness._classify_terminal_decision_provenance(
            first_pass_decision_source="llm_explicit",
            evidence_retrieval_status="retrieved",
            final_decision={
                "action": "inconclusive",
                "reason": "decision_reason_missing",
                "decision_source": "llm_explicit",
                "phase": "follow_up_terminal",
            },
            final_summary={"supporting_observation_ids": ["obs-102"]},
            correlated_result={
                "result_payload": {
                    "messages": [
                        {
                            "bounded_history": [
                                {"observation_id": "obs-102"},
                            ]
                        }
                    ]
                }
            },
        )
        self.assertEqual(provenance, "explicit_terminal_missing_reason")

    def test_terminal_provenance_invalid_explicit_decision(self) -> None:
        provenance = harness._classify_terminal_decision_provenance(
            first_pass_decision_source="llm_explicit",
            evidence_retrieval_status="retrieved",
            final_decision={
                "action": "inconclusive",
                "reason": "invalid_explicit_decision",
                "decision_source": "invalid_explicit_decision",
                "phase": "terminal_invalid_explicit_decision",
            },
            final_summary={"supporting_observation_ids": ["obs-102"]},
            correlated_result={
                "result_payload": {
                    "messages": [
                        {
                            "bounded_history": [
                                {"observation_id": "obs-102"},
                            ]
                        }
                    ]
                }
            },
        )
        self.assertEqual(provenance, "invalid_explicit_decision")

    def test_terminal_provenance_orchestration_fallback(self) -> None:
        provenance = harness._classify_terminal_decision_provenance(
            first_pass_decision_source="llm_summary_inference",
            evidence_retrieval_status="retrieved",
            final_decision={
                "action": "inconclusive",
                "reason": "residual_occlusion",
                "decision_source": "llm_summary_inference",
                "phase": "follow_up_terminal",
            },
            final_summary={"supporting_observation_ids": ["obs-102"]},
            correlated_result={
                "result_payload": {
                    "messages": [
                        {
                            "bounded_history": [
                                {"observation_id": "obs-102"},
                            ]
                        }
                    ]
                }
            },
        )
        self.assertEqual(provenance, "orchestration_fallback")

    def test_terminal_provenance_explicit_insufficient_evidence(self) -> None:
        provenance = harness._classify_terminal_decision_provenance(
            first_pass_decision_source="llm_explicit",
            evidence_retrieval_status="retrieved",
            final_decision={
                "action": "inconclusive",
                "reason": "residual_occlusion",
                "decision_source": "llm_explicit",
                "phase": "follow_up_terminal",
            },
            final_summary={"supporting_observation_ids": ["obs-999"]},
            correlated_result={
                "result_payload": {
                    "messages": [
                        {
                            "bounded_history": [
                                {"observation_id": "obs-102"},
                            ]
                        }
                    ]
                }
            },
        )
        self.assertEqual(provenance, "explicit_terminal_insufficient_evidence")

    def test_second_pass_uses_terminal_prompt_that_prohibits_tool_requests(self) -> None:
        captured_kwargs: list[dict] = []

        def _fake_call(user_payload: dict, **kwargs: dict):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                return (
                    {
                        "summary": "Need more evidence due to uncertainty.",
                        "measured_facts": [],
                        "interpretations": [],
                        "hypotheses": [],
                        "supporting_observation_ids": ["obs-100", "obs-101"],
                        "conflicting_evidence": [],
                        "uncertainties": ["occlusion"],
                        "needs_more_observations": True,
                        "decision": {
                            "action": "request_additional_tracking_evidence",
                            "reason": "need_history",
                            "uncertainty": ["occlusion"],
                            "tool_requests": [
                                {
                                    "tool_name": "get_track_evidence_history",
                                    "arguments": {
                                        "track_id": "fixture-track-001",
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "status": "ok",
                        "model": "test-model",
                        "latency_ms": 0.1,
                        "error": None,
                        "error_code": None,
                        "attempts": 1,
                        "retry_count": 0,
                    },
                )
            return (
                {
                    "summary": "Terminal follow-up assessment.",
                    "measured_facts": [],
                    "interpretations": [],
                    "hypotheses": [],
                    "supporting_observation_ids": ["obs-100", "obs-101", "obs-102"],
                    "conflicting_evidence": [],
                    "uncertainties": [],
                    "needs_more_observations": False,
                    "decision": {
                        "action": "assess_now",
                        "reason": "enough_evidence",
                        "uncertainty": [],
                        "tool_requests": [],
                    },
                },
                {
                    "status": "ok",
                    "model": "test-model",
                    "latency_ms": 0.1,
                    "error": None,
                    "error_code": None,
                    "attempts": 1,
                    "retry_count": 0,
                },
            )

        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

        self.assertEqual(report.get("status"), "ok")
        self.assertGreaterEqual(len(captured_kwargs), 2)
        first_prompt = captured_kwargs[0].get("system_prompt")
        second_prompt = captured_kwargs[1].get("system_prompt")
        self.assertIsNone(first_prompt)
        self.assertEqual(second_prompt, harness.HARNESS_SECOND_PASS_TERMINAL_SYSTEM_PROMPT)
        self.assertIn("no additional tool requests are permitted", second_prompt)
        self.assertIn("assess_now, predict_bounded, inconclusive", second_prompt)
        self.assertIn("decision.reason must be meaningful and evidence-specific", second_prompt)

    def test_concise_prompt_mode_is_opt_in_and_applies_both_pass_prompts(self) -> None:
        captured_kwargs: list[dict] = []

        def _fake_call(user_payload: dict, **kwargs: dict):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                return (
                    {
                        "summary": "Need more evidence due to uncertainty.",
                        "measured_facts": [],
                        "interpretations": [],
                        "hypotheses": [],
                        "supporting_observation_ids": ["obs-100", "obs-101"],
                        "conflicting_evidence": [],
                        "uncertainties": ["occlusion"],
                        "needs_more_observations": True,
                        "decision": {
                            "action": "request_additional_tracking_evidence",
                            "reason": "need_history",
                            "uncertainty": ["occlusion"],
                            "tool_requests": [
                                {
                                    "tool_name": "get_track_evidence_history",
                                    "arguments": {
                                        "track_id": "fixture-track-001",
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "status": "ok",
                        "model": "test-model",
                        "latency_ms": 0.1,
                        "error": None,
                        "error_code": None,
                        "attempts": 1,
                        "retry_count": 0,
                    },
                )
            return (
                {
                    "summary": "Terminal follow-up assessment.",
                    "measured_facts": [],
                    "interpretations": [],
                    "hypotheses": [],
                    "supporting_observation_ids": ["obs-100", "obs-101", "obs-102"],
                    "conflicting_evidence": [],
                    "uncertainties": [],
                    "needs_more_observations": False,
                    "decision": {
                        "action": "assess_now",
                        "reason": "enough_evidence",
                        "uncertainty": [],
                        "tool_requests": [],
                    },
                },
                {
                    "status": "ok",
                    "model": "test-model",
                    "latency_ms": 0.1,
                    "error": None,
                    "error_code": None,
                    "attempts": 1,
                    "retry_count": 0,
                },
            )

        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                    concise_prompts=True,
                )

        self.assertEqual(report.get("status"), "ok")
        self.assertEqual(report.get("prompt_mode"), "concise_harness")
        self.assertGreaterEqual(len(captured_kwargs), 2)
        self.assertEqual(captured_kwargs[0].get("system_prompt"), harness.HARNESS_FIRST_PASS_CONCISE_SYSTEM_PROMPT)
        self.assertEqual(
            captured_kwargs[1].get("system_prompt"),
            harness.HARNESS_SECOND_PASS_TERMINAL_CONCISE_SYSTEM_PROMPT,
        )

    def test_concise_mode_valid_explicit_request_runs_mcp_and_terminal_second_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["preflight_observation_responses"][0]["decision"] = {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": payload["track_id"],
                            "max_messages": 5,
                            "max_history_records": 12,
                            "max_scan_records": 120,
                        },
                    }
                ],
            }
            payload["preflight_observation_responses"][1]["decision"] = {
                "action": "inconclusive",
                "reason": "residual_occlusion_after_followup",
                "uncertainty": ["residual_risk"],
                "tool_requests": [],
            }
            payload["preflight_observation_responses"][1]["supporting_observation_ids"] = [
                "obs-100",
                "obs-101",
                "obs-102",
            ]

            fixture = self._write_fixture(run_dir, payload)
            report = run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
                concise_prompts=True,
            )

        self.assertEqual(report.get("status"), "ok")
        self.assertEqual(report.get("prompt_mode"), "concise_harness")
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("first_pass_decision_source"), "llm_explicit")
        self.assertTrue(bool(investigation.get("request_submitted", False)))
        self.assertEqual(investigation.get("mcp_evidence_call_count"), 1)
        self.assertEqual(investigation.get("evidence_retrieval_status"), "retrieved")
        self.assertTrue(bool(investigation.get("second_pass", {}).get("attempted", False)))
        self.assertEqual(investigation.get("second_pass", {}).get("status"), "ok")
        self.assertEqual(investigation.get("final_decision", {}).get("decision_source"), "llm_explicit")
        self.assertEqual(investigation.get("final_decision", {}).get("phase"), "follow_up_terminal")
        self.assertEqual(
            investigation.get("classification", {}).get("terminal_decision_provenance"),
            "explicit_terminal_with_evidence_support",
        )
        self.assertEqual(investigation.get("evidence_references", {}).get("message_ids"), ["msg-1"])

    def test_concise_mode_request_action_missing_tool_requests_is_rejected(self) -> None:
        report = self._run_concise_preflight_with_first_decision(
            {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
            }
        )

        self.assertEqual(report.get("status"), "ok")
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("first_decision", {}).get("decision_source"), "invalid_explicit_decision")
        self.assertEqual(investigation.get("first_decision", {}).get("validation_error"), "decision_tool_request_required")
        self.assertFalse(bool(investigation.get("request_submitted", False)))
        self.assertEqual(investigation.get("mcp_evidence_call_count"), 0)
        self.assertFalse(bool(investigation.get("second_pass", {}).get("attempted", False)))

    def test_concise_mode_request_action_with_empty_tool_requests_is_rejected(self) -> None:
        report = self._run_concise_preflight_with_first_decision(
            {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
                "tool_requests": [],
            }
        )

        self.assertEqual(report.get("status"), "ok")
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("first_decision", {}).get("decision_source"), "invalid_explicit_decision")
        self.assertEqual(investigation.get("first_decision", {}).get("validation_error"), "decision_tool_request_required")
        self.assertFalse(bool(investigation.get("request_submitted", False)))
        self.assertEqual(investigation.get("mcp_evidence_call_count"), 0)
        self.assertFalse(bool(investigation.get("second_pass", {}).get("attempted", False)))

    def test_concise_mode_request_action_with_malformed_arguments_is_rejected(self) -> None:
        report = self._run_concise_preflight_with_first_decision(
            {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "since_timestamp": 1734023400.1,
                        },
                    }
                ],
            }
        )

        self.assertEqual(report.get("status"), "ok")
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("first_decision", {}).get("decision_source"), "llm_explicit")
        request_submission = investigation.get("request_submission", {})
        self.assertTrue(bool(request_submission.get("rejected", False)))
        self.assertEqual(request_submission.get("reason"), "invalid_tool_arguments_pre_dispatch")
        self.assertFalse(bool(investigation.get("request_submitted", False)))
        self.assertEqual(investigation.get("mcp_evidence_call_count"), 0)
        self.assertFalse(bool(investigation.get("second_pass", {}).get("attempted", False)))

    def test_compact_correlated_evidence_preserves_required_fields_and_drops_transport_metadata(self) -> None:
        correlated_result = {
            "status": "ok",
            "error_code": None,
            "error": None,
            "result_payload": {
                "available": True,
                "track_id": "fixture-track-001",
                "request_id": "req-1",
                "tool_name": "get_track_evidence_history",
                "started_at": 1.0,
                "finished_at": 2.0,
                "execution_mode": "bounded_mcp_dispatch",
                "messages": [
                    {
                        "message_id": "msg-1",
                        "source": "tracking-evidence-history",
                        "bounded_history": [
                            {
                                "observation_id": "obs-102",
                                "timestamp": 100.0,
                                "x": 0.28,
                                "z": -0.05,
                                "confidence": 0.54,
                                "quality_flags": ["depth_missing"],
                            }
                        ],
                    }
                ],
                "history_records": [
                    {
                        "observation_id": "obs-102",
                        "timestamp": 100.0,
                        "confidence": 0.54,
                    }
                ],
            },
        }

        compact = harness._compact_correlated_evidence_tool_result(
            correlated_result,
            authoritative_track_id="fixture-track-001",
        )

        payload = compact.get("payload", {})
        self.assertEqual(compact.get("status"), "ok")
        self.assertEqual(payload.get("track_id"), "fixture-track-001")
        self.assertEqual(payload.get("supporting_observation_ids"), ["obs-102"])
        self.assertIsInstance(payload.get("messages"), list)
        self.assertNotIn("request_id", payload)
        self.assertNotIn("tool_name", payload)
        self.assertNotIn("execution_mode", payload)

    def test_second_pass_failure_preserves_partial_investigation_telemetry(self) -> None:
        call_count = 0

        def _fake_call(user_payload: dict, **kwargs: dict):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    {
                        "summary": "Need follow-up evidence.",
                        "measured_facts": [],
                        "interpretations": [],
                        "hypotheses": [],
                        "supporting_observation_ids": ["obs-100", "obs-101"],
                        "conflicting_evidence": [],
                        "uncertainties": ["occlusion"],
                        "needs_more_observations": True,
                        "decision": {
                            "action": "request_additional_tracking_evidence",
                            "reason": "need_history",
                            "uncertainty": ["occlusion"],
                            "tool_requests": [
                                {
                                    "tool_name": "get_track_evidence_history",
                                    "arguments": {
                                        "track_id": "fixture-track-001",
                                    },
                                }
                            ],
                        },
                    },
                    {
                        "status": "ok",
                        "model": "test-model",
                        "latency_ms": 0.1,
                        "error": None,
                        "error_code": None,
                        "attempts": 1,
                        "retry_count": 0,
                    },
                )
            return (
                None,
                {
                    "status": "error",
                    "model": "test-model",
                    "latency_ms": 0.2,
                    "error": "RuntimeError:llm_output_truncated",
                    "error_code": "output_truncated",
                    "attempts": 1,
                    "retry_count": 0,
                },
            )

        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            with patch(
                "tools.observation_autonomy_provider_harness.agent_observation.call_observation_agent_with_retry",
                side_effect=_fake_call,
            ):
                report = run_single_investigation(
                    fixture_path=self.fixture_path,
                    run_dir=run_dir,
                    preflight_only=False,
                )

        self.assertEqual(report.get("status"), "error")
        investigation = report.get("investigation", {})
        self.assertEqual(investigation.get("first_decision", {}).get("decision_source"), "llm_explicit")
        self.assertTrue(bool(investigation.get("request_submitted", False)))
        request_decision = investigation.get("request_decision", {})
        request_arguments = request_decision.get("tool_requests", [{}])[0].get("arguments", {})
        self.assertEqual(request_arguments.get("track_id"), "fixture-track-001")
        self.assertEqual(investigation.get("correlated_result", {}).get("status"), "ok")
        self.assertIsNone(investigation.get("correlated_result", {}).get("error_code"))
        self.assertEqual(investigation.get("evidence_retrieval_status"), "retrieved")
        second_pass = investigation.get("second_pass", {})
        self.assertTrue(bool(second_pass.get("attempted", False)))
        self.assertEqual(second_pass.get("status"), "error")
        self.assertEqual(second_pass.get("error_code"), "output_truncated")
        self.assertEqual(investigation.get("phase_failure", {}).get("code"), "second_pass_output_truncated")
        self.assertEqual(investigation.get("phase_failure", {}).get("phase"), "pass2")
        pass_attempts = investigation.get("pass_attempts", {})
        self.assertTrue(bool(pass_attempts.get("pass1", {}).get("attempted", False)))
        self.assertTrue(bool(pass_attempts.get("pass2", {}).get("attempted", False)))
        self.assertEqual(pass_attempts.get("pass1", {}).get("llm_attempts"), 1)
        self.assertEqual(pass_attempts.get("pass1", {}).get("retry_count"), 0)
        self.assertEqual(pass_attempts.get("pass2", {}).get("llm_attempts"), 1)
        self.assertEqual(pass_attempts.get("pass2", {}).get("retry_count"), 0)

    def test_follow_up_remains_terminal_if_second_pass_requests_another_tool(self) -> None:
        with tempfile.TemporaryDirectory(prefix="obs-provider-harness-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            payload = self._fixture_payload()
            payload["preflight_observation_responses"][0]["decision"] = {
                "action": "request_additional_tracking_evidence",
                "reason": "need_more_tracking_history",
                "uncertainty": ["occlusion"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": payload["track_id"],
                        },
                    }
                ],
            }
            payload["preflight_observation_responses"][1]["decision"] = {
                "action": "request_additional_tracking_evidence",
                "reason": "ask_again",
                "uncertainty": ["residual_risk"],
                "tool_requests": [
                    {
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": payload["track_id"],
                        },
                    }
                ],
            }
            fixture = self._write_fixture(run_dir, payload)
            report = run_single_investigation(
                fixture_path=fixture,
                run_dir=run_dir,
                preflight_only=True,
            )

            self.assertEqual(report.get("status"), "ok")
            investigation = report.get("investigation", {})
            self.assertEqual(investigation.get("mcp_evidence_call_count"), 1)
            final_decision = investigation.get("final_decision", {})
            self.assertEqual(final_decision.get("decision_source"), "invalid_explicit_decision")
            self.assertEqual(final_decision.get("phase"), "terminal_invalid_explicit_decision")


if __name__ == "__main__":
    unittest.main()
