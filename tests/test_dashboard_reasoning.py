# Handles test dashboard reasoning behavior.
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from tools import dashboardReasoning


class TestDashboardReasoning(unittest.TestCase):
    def tearDown(self) -> None:
        # Performs tearDown logic.
        dashboardReasoning._reset_reasoning_state()

    def test_reasoning_returns_unavailable_when_no_radar_state(self) -> None:
        # Performs test reasoning returns unavailable when no radar state logic.
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_path):
                result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_observation_summary_returns_unavailable_when_no_radar_state(self) -> None:
        # Performs test observation summary returns unavailable when no radar state logic.
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_path):
                result = dashboardReasoning.simulate_observation_summary("track-1")

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_reasoning_returns_unavailable_when_radar_state_stale(self) -> None:
        # Performs test reasoning returns unavailable when radar state stale logic.
        payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            os.utime(radar_path, (1000.0, 1000.0))

            with patch.dict(
                os.environ,
                {"DASHBOARD_RADAR_STATE_MAX_AGE_SECONDS": "2"},
                clear=False,
            ):
                with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                    with patch("tools.dashboardReasoning.time.time", return_value=1005.0):
                        result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_stale")
        self.assertEqual(result["radar_state_max_age_seconds"], 2.0)
        self.assertEqual(result["radar_state_age_seconds"], 5.0)

    def test_evidence_request_returns_unavailable_when_radar_state_stale(self) -> None:
        # Performs test evidence request returns unavailable when radar state stale logic.
        payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            os.utime(radar_path, (2000.0, 2000.0))

            with patch.dict(
                os.environ,
                {"DASHBOARD_RADAR_STATE_MAX_AGE_SECONDS": "1"},
                clear=False,
            ):
                with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                    with patch("tools.dashboardReasoning.time.time", return_value=2003.0):
                        result = dashboardReasoning.simulate_evidence_request("track-1", "need_more")

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_stale")
        self.assertEqual(result["radar_state_max_age_seconds"], 1.0)
        self.assertEqual(result["radar_state_age_seconds"], 3.0)

    def test_reasoning_requests_evidence_for_insufficient_history(self) -> None:
        # Performs test reasoning requests evidence for insufficient history logic.
        payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Insufficient history for confident trajectory interpretation.",
                            "measured_facts": ["single observation"],
                            "interpretations": ["history too short"],
                            "hypotheses": ["possible loitering not confirmable"],
                            "supporting_observation_ids": ["track-1:1000.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": ["insufficient_history"],
                            "needs_more_observations": True,
                        },
                        {
                            "status": "ok",
                            "model": "gpt-oss-120b",
                            "latency_ms": 12.5,
                            "error": None,
                            "attempts": 1,
                        },
                    ),
                ):
                    with patch(
                        "tools.dashboardReasoning._call_reasoning_agent_with_retry",
                        return_value=(
                            {
                                "assessment_label": "requires_more_observation",
                                "summary": "Need additional observations before operator action.",
                                "rationale": "Single sample cannot establish trajectory pattern.",
                                "recommendation": "collect_more_observations",
                                "supporting_observation_ids": ["track-1:1000.000000"],
                                "uncertainties": ["insufficient_history"],
                                "limitations": ["single_frame"],
                                "alternative_interpretations": ["normal transit"],
                                "missing_evidence": ["more_timestamps"],
                                "contradictory_evidence": [],
                                "request_more_observations": True,
                            },
                            {
                                "status": "ok",
                                "model": "gpt-oss-120b",
                                "latency_ms": 18.0,
                                "error": None,
                                "attempts": 1,
                            },
                        ),
                    ):
                        result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertTrue(result["available"])
        self.assertEqual(result["assessment"]["assessment_label"], "requires_more_observation")
        self.assertIsNotNone(result["evidence_request"])
        self.assertEqual(result["agents"]["observation"]["status"], "ok")
        self.assertEqual(result["agents"]["reasoning"]["status"], "ok")

    def test_reasoning_detects_fast_motion_after_two_updates(self) -> None:
        # Performs test reasoning detects fast motion after two updates logic.
        first_payload = {
            "timestamp": 1000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1000.0,
                    "ground_position": {"x": 0.0, "z": 0.0},
                    "confidence": 0.9,
                }
            ],
        }
        second_payload = {
            "timestamp": 1001.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 1001.0,
                    "ground_position": {"x": 3.0, "z": 0.0},
                    "confidence": 0.95,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                radar_path.write_text(json.dumps(first_payload), encoding="utf-8")
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Initial observation only.",
                            "measured_facts": ["one point"],
                            "interpretations": ["insufficient pattern"],
                            "hypotheses": [],
                            "supporting_observation_ids": ["track-1:1000.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": ["insufficient_history"],
                            "needs_more_observations": True,
                        },
                        {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 9.0, "error": None, "attempts": 1},
                    ),
                ):
                    with patch(
                        "tools.dashboardReasoning._call_reasoning_agent_with_retry",
                        return_value=(
                            {
                                "assessment_label": "requires_more_observation",
                                "summary": "Need more history.",
                                "rationale": "Single timestamp.",
                                "recommendation": "collect_more_observations",
                                "supporting_observation_ids": ["track-1:1000.000000"],
                                "uncertainties": ["insufficient_history"],
                                "limitations": [],
                                "alternative_interpretations": [],
                                "missing_evidence": ["more_samples"],
                                "contradictory_evidence": [],
                                "request_more_observations": True,
                            },
                            {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 10.0, "error": None, "attempts": 1},
                        ),
                    ):
                        dashboardReasoning.simulate_reasoning_assessment("track-1")

                radar_path.write_text(json.dumps(second_payload), encoding="utf-8")
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Track shows sustained high-speed departure.",
                            "measured_facts": ["speed > 2.0"],
                            "interpretations": ["rapid movement"],
                            "hypotheses": ["normal transit"],
                            "supporting_observation_ids": ["track-1:1000.000000", "track-1:1001.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 11.0, "error": None, "attempts": 1},
                    ),
                ):
                    with patch(
                        "tools.dashboardReasoning._call_reasoning_agent_with_retry",
                        return_value=(
                            {
                                "assessment_label": "requires_review",
                                "summary": "Movement merits operator review due to rapid trajectory change.",
                                "rationale": "Two-point velocity indicates rapid displacement.",
                                "recommendation": "operator_review",
                                "supporting_observation_ids": ["track-1:1000.000000", "track-1:1001.000000"],
                                "uncertainties": [],
                                "limitations": [],
                                "alternative_interpretations": ["running_person"],
                                "missing_evidence": [],
                                "contradictory_evidence": [],
                                "request_more_observations": False,
                            },
                            {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 13.0, "error": None, "attempts": 1},
                        ),
                    ):
                        result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertTrue(result["available"])
        self.assertEqual(result["assessment"]["assessment_label"], "requires_review")
        self.assertIsNotNone(result["summary"]["speed"])

    def test_observation_summary_reports_velocity_after_two_updates(self) -> None:
        # Performs test observation summary reports velocity after two updates logic.
        first_payload = {
            "timestamp": 2000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 2000.0,
                    "ground_position": {"x": 1.0, "z": 1.0},
                    "confidence": 0.8,
                }
            ],
        }
        second_payload = {
            "timestamp": 2001.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 2001.0,
                    "ground_position": {"x": 2.0, "z": 3.0},
                    "confidence": 0.85,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                radar_path.write_text(json.dumps(first_payload), encoding="utf-8")
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Initial observation.",
                            "measured_facts": ["single sample"],
                            "interpretations": [],
                            "hypotheses": [],
                            "supporting_observation_ids": ["track-1:2000.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": ["insufficient_history"],
                            "needs_more_observations": True,
                        },
                        {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 8.0, "error": None, "attempts": 1},
                    ),
                ):
                    dashboardReasoning.simulate_observation_summary("track-1")

                radar_path.write_text(json.dumps(second_payload), encoding="utf-8")
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Track moved diagonally with sustained motion.",
                            "measured_facts": ["velocity_x positive", "velocity_z positive"],
                            "interpretations": ["sustained movement"],
                            "hypotheses": ["normal transit"],
                            "supporting_observation_ids": ["track-1:2000.000000", "track-1:2001.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 9.0, "error": None, "attempts": 1},
                    ),
                ):
                    result = dashboardReasoning.simulate_observation_summary("track-1")

        self.assertTrue(result["available"])
        summary = result["summary"]
        self.assertIsNotNone(summary["velocity_x"])
        self.assertIsNotNone(summary["velocity_z"])
        self.assertIsNotNone(summary["speed"])
        self.assertTrue(summary["llm_available"])

    def test_observation_summary_falls_back_when_llm_unavailable(self) -> None:
        # Performs test observation summary falls back when llm unavailable logic.
        payload = {
            "timestamp": 4000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 4000.0,
                    "ground_position": {"x": 1.0, "z": 1.0},
                    "confidence": 0.9,
                },
                {
                    "track_id": "track-1",
                    "timestamp": 4001.0,
                    "ground_position": {"x": 2.0, "z": 1.5},
                    "confidence": 0.9,
                },
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        None,
                        {"status": "error", "model": "gpt-oss-120b", "latency_ms": 15.0, "error": "timeout", "attempts": 2},
                    ),
                ):
                    result = dashboardReasoning.simulate_observation_summary("track-1")

        self.assertTrue(result["available"])
        self.assertFalse(result["summary"]["llm_available"])
        self.assertIn("Fallback observation summary", result["summary"]["summary"])

    def test_reasoning_falls_back_when_reasoning_llm_unavailable(self) -> None:
        # Performs test reasoning falls back when reasoning llm unavailable logic.
        payload = {
            "timestamp": 5000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 5000.0,
                    "ground_position": {"x": 1.0, "z": 1.0},
                    "confidence": 0.9,
                },
                {
                    "track_id": "track-1",
                    "timestamp": 5001.0,
                    "ground_position": {"x": 5.0, "z": 1.0},
                    "confidence": 0.9,
                },
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Rapid movement observed.",
                            "measured_facts": ["high speed"],
                            "interpretations": ["trajectory change"],
                            "hypotheses": ["rapid transit"],
                            "supporting_observation_ids": ["track-1:5000.000000", "track-1:5001.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 7.0, "error": None, "attempts": 1},
                    ),
                ):
                    with patch(
                        "tools.dashboardReasoning._call_reasoning_agent_with_retry",
                        return_value=(
                            None,
                            {"status": "error", "model": "gpt-oss-120b", "latency_ms": 16.0, "error": "timeout", "attempts": 2},
                        ),
                    ):
                        result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertTrue(result["available"])
        self.assertFalse(result["assessment"]["llm_available"])
        self.assertIn(result["assessment"]["assessment_label"], {"requires_review", "requires_more_observation", "stable_motion"})

    def test_evidence_request_returns_unavailable_when_no_radar_state(self) -> None:
        # Performs test evidence request returns unavailable when no radar state logic.
        with TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "latest_radar_state.json"
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", missing_path):
                result = dashboardReasoning.simulate_evidence_request(
                    "track-1",
                    "need_more_observations",
                )

        self.assertFalse(result["available"])
        self.assertEqual(result["error"], "radar_state_unavailable")

    def test_evidence_request_is_bounded_over_multiple_calls(self) -> None:
        # Performs test evidence request is bounded over multiple calls logic.
        payload = {
            "timestamp": 3000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 3000.0,
                    "ground_position": {"x": 0.0, "z": 0.0},
                    "confidence": 0.9,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                requests = []
                for index in range(4):
                    response = dashboardReasoning.simulate_evidence_request(
                        "track-1",
                        f"reason-{index}",
                    )
                    requests.append(response)

        self.assertTrue(requests[0]["available"])
        self.assertIsNotNone(requests[0]["request"])
        self.assertIsNotNone(requests[1]["request"])
        self.assertIsNotNone(requests[2]["request"])
        self.assertIsNone(requests[3]["request"])
        self.assertTrue(requests[3]["bounded_out"])

    def test_observation_agent_uses_distinct_prompt_and_model(self) -> None:
        # Performs test observation agent uses distinct prompt and model logic.
        expected_payload = {
            "summary": "Observed sustained movement.",
            "measured_facts": ["speed high"],
            "interpretations": ["departing"],
            "hypotheses": [],
            "supporting_observation_ids": [],
            "conflicting_evidence": [],
            "uncertainties": [],
            "needs_more_observations": False,
        }

        with patch.dict(
            os.environ,
            {
                "OBSERVATION_AGENT_MODEL": "gpt-oss-120b",
                "DASHBOARD_AGENT_LLM_TIMEOUT_SECONDS": "6.5",
                "DASHBOARD_AGENT_LLM_RETRY_LIMIT": "0",
            },
            clear=False,
        ):
            with patch(
                "tools.dashboardReasoning._invoke_llm_json_response",
                new=AsyncMock(return_value=expected_payload),
            ) as invoke_mock:
                payload, meta = dashboardReasoning._call_observation_agent_with_retry(
                    {"track_id": "track-1"}
                )

        self.assertEqual(payload, expected_payload)
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["model"], "gpt-oss-120b")
        self.assertEqual(meta["attempts"], 1)
        self.assertEqual(invoke_mock.await_count, 1)
        call_kwargs = invoke_mock.await_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-oss-120b")
        self.assertIn("Observation/Behavior Agent", call_kwargs["system_prompt"])
        self.assertEqual(call_kwargs["timeout_seconds"], 6.5)

    def test_reasoning_agent_retries_and_uses_distinct_prompt(self) -> None:
        # Performs test reasoning agent retries and uses distinct prompt logic.
        expected_payload = {
            "assessment_label": "requires_review",
            "summary": "Review warranted.",
            "rationale": "Evidence indicates trajectory shift.",
            "recommendation": "operator_review",
            "supporting_observation_ids": [],
            "uncertainties": [],
            "limitations": [],
            "alternative_interpretations": [],
            "missing_evidence": [],
            "contradictory_evidence": [],
            "request_more_observations": False,
        }

        with patch.dict(
            os.environ,
            {
                "REASONING_AGENT_MODEL": "gpt-oss-120b",
                "DASHBOARD_AGENT_LLM_TIMEOUT_SECONDS": "5",
                "DASHBOARD_AGENT_LLM_RETRY_LIMIT": "1",
            },
            clear=False,
        ):
            with patch(
                "tools.dashboardReasoning._invoke_llm_json_response",
                new=AsyncMock(side_effect=[RuntimeError("transient"), expected_payload]),
            ) as invoke_mock:
                payload, meta = dashboardReasoning._call_reasoning_agent_with_retry(
                    {"track_id": "track-1"}
                )

        self.assertEqual(payload, expected_payload)
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["attempts"], 2)
        self.assertEqual(invoke_mock.await_count, 2)
        final_call_kwargs = invoke_mock.await_args.kwargs
        self.assertEqual(final_call_kwargs["model"], "gpt-oss-120b")
        self.assertIn("Reasoning/Decision-Support Agent", final_call_kwargs["system_prompt"])
        self.assertEqual(final_call_kwargs["timeout_seconds"], 5.0)

    def test_reasoning_payload_validation_filters_invalid_ids_and_label(self) -> None:
        # Performs test reasoning payload validation filters invalid ids and label logic.
        payload = {
            "assessment_label": "unsupported_label",
            "summary": "Assessment summary.",
            "rationale": "Rationale text.",
            "recommendation": "collect_more_observations",
            "supporting_observation_ids": ["valid-id", "bad-id"],
            "uncertainties": ["short_history"],
            "limitations": ["occlusion"],
            "alternative_interpretations": ["normal transit"],
            "missing_evidence": ["additional samples"],
            "contradictory_evidence": [],
            "request_more_observations": True,
        }

        assessment, request_more = dashboardReasoning._validate_reasoning_llm_payload(
            payload,
            track_id="track-1",
            created_at=1.23,
            valid_observation_ids={"valid-id"},
        )

        self.assertEqual(assessment["assessment_label"], "requires_more_observation")
        self.assertEqual(assessment["supporting_observation_ids"], ["valid-id"])
        self.assertTrue(request_more)

    def test_observation_agent_error_code_and_backoff_on_retries(self) -> None:
        # Performs test observation agent error code and backoff on retries logic.
        with patch.dict(
            os.environ,
            {
                "DASHBOARD_AGENT_LLM_RETRY_LIMIT": "2",
                "DASHBOARD_AGENT_LLM_RETRY_BACKOFF_SECONDS": "0.01",
            },
            clear=False,
        ):
            with patch(
                "tools.dashboardReasoning._invoke_llm_json_response",
                new=AsyncMock(side_effect=[RuntimeError("llm_invalid_json_response")] * 3),
            ) as invoke_mock:
                with patch("tools.dashboardReasoning.time.sleep") as sleep_mock:
                    payload, meta = dashboardReasoning._call_observation_agent_with_retry(
                        {"track_id": "track-1"}
                    )

        self.assertIsNone(payload)
        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["attempts"], 3)
        self.assertEqual(meta["retry_count"], 2)
        self.assertEqual(meta["error_code"], "invalid_response_format")
        self.assertEqual(invoke_mock.await_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)
        self.assertEqual(sleep_mock.call_args_list[0].args[0], 0.01)
        self.assertEqual(sleep_mock.call_args_list[1].args[0], 0.02)

    def test_reasoning_agent_timeout_error_code(self) -> None:
        # Performs test reasoning agent timeout error code logic.
        with patch.dict(
            os.environ,
            {
                "DASHBOARD_AGENT_LLM_RETRY_LIMIT": "0",
            },
            clear=False,
        ):
            with patch(
                "tools.dashboardReasoning._invoke_llm_json_response",
                new=AsyncMock(side_effect=TimeoutError("request timeout")),
            ):
                payload, meta = dashboardReasoning._call_reasoning_agent_with_retry(
                    {"track_id": "track-1"}
                )

        self.assertIsNone(payload)
        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["error_code"], "timeout")
        self.assertEqual(meta["attempts"], 1)
        self.assertEqual(meta["retry_count"], 0)

    def test_reasoning_handoff_validation_skips_reasoning_call(self) -> None:
        # Performs test reasoning handoff validation skips reasoning call logic.
        payload = {
            "timestamp": 6000.0,
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 6000.0,
                    "ground_position": {"x": 0.0, "z": 0.0},
                    "confidence": 0.9,
                },
                {
                    "track_id": "track-1",
                    "timestamp": 6001.0,
                    "ground_position": {"x": 0.5, "z": 0.25},
                    "confidence": 0.9,
                },
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Observation summary with malformed supporting ids.",
                            "measured_facts": ["movement observed"],
                            "interpretations": ["slow transit"],
                            "hypotheses": [],
                            "supporting_observation_ids": ["track-1:6000.000000", "track-1:6001.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {"status": "ok", "model": "gpt-oss-120b", "latency_ms": 5.0, "error": None, "attempts": 1},
                    ),
                ):
                    with patch(
                        "tools.dashboardReasoning._validate_observation_llm_payload",
                        return_value={
                            "summary": "broken",
                            "supporting_observation_ids": "not-a-list",
                            "needs_more_observations": False,
                        },
                    ):
                        with patch(
                            "tools.dashboardReasoning._call_reasoning_agent_with_retry"
                        ) as reasoning_mock:
                            result = dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertTrue(result["available"])
        self.assertFalse(result["assessment"]["llm_available"])
        self.assertEqual(result["agents"]["reasoning"]["status"], "skipped")
        self.assertEqual(result["agents"]["reasoning"]["error_code"], "invalid_handoff_payload")
        self.assertEqual(result["handoff_validation"]["status"], "error")
        self.assertEqual(result["handoff_validation"]["error_code"], "invalid_handoff_payload")
        reasoning_mock.assert_not_called()

    def test_build_reasoning_handoff_payload_filters_to_history_ids(self) -> None:
        # Performs test build reasoning handoff payload filters to history ids logic.
        summary = {
            "summary": "ok",
            "supporting_observation_ids": ["obs-1", "unknown"],
        }
        history_records = [
            {"observation_id": "obs-1"},
            {"observation_id": "obs-2"},
        ]

        payload, meta = dashboardReasoning._build_reasoning_handoff_payload(
            track_id="track-1",
            created_at=123.0,
            summary=summary,
            history_records=history_records,
        )

        self.assertEqual(meta["status"], "ok")
        self.assertEqual(payload["observation_summary"]["supporting_observation_ids"], ["obs-1"])

    def test_observation_llm_input_includes_tracking_and_coordinate_evidence(self) -> None:
        # Performs test observation llm input includes tracking and coordinate evidence logic.
        radar_state = {
            "coordinate_system": "calibrated_ground_plane",
            "calibration_status": "loaded",
            "camera_status": "connected",
            "detector_status": "running",
            "last_observation_timestamp": 7100.0,
            "tracking_agent_observation": {
                "timestamp": 7100.0,
                "source_agent": "visual_tracking",
                "vlm_model": "qwen-3.8-27b",
                "proposal_latency_ms": 12.3,
                "uncertainties": ["occlusion_risk"],
                "updates": [
                    {
                        "detection_id": "track-1",
                        "resolved_track_id": "track-1",
                        "accepted": True,
                        "rejection_reasons": [],
                        "uncertainties": [],
                        "supporting_evidence": ["visual_continuity"],
                    }
                ],
            },
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 7100.0,
                    "pixel_position": {"x": 100.0, "y": 200.0},
                    "ground_position": {"x": 1.5, "z": 4.0},
                    "ground_coordinate_unit": "meters",
                    "position_kind": "calibrated_ground_plane",
                    "relative_depth": 0.42,
                    "confidence": 0.91,
                    "quality_flags": ["ground_plane_homography"],
                },
                {
                    "track_id": "track-2",
                    "timestamp": 7100.0,
                    "ground_position": {"x": 4.0, "z": 2.0},
                },
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(radar_state), encoding="utf-8")

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                captured_payloads: list[dict] = []

                def _capture_observation_payload(user_payload: dict):
                    # Performs capture observation payload logic.
                    captured_payloads.append(user_payload)
                    return (
                        {
                            "summary": "Connected evidence payload received.",
                            "measured_facts": [],
                            "interpretations": [],
                            "hypotheses": [],
                            "supporting_observation_ids": user_payload.get("history_records", [])[0:1],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {
                            "status": "ok",
                            "model": "gpt-oss-120b",
                            "latency_ms": 1.0,
                            "error": None,
                            "attempts": 1,
                            "retry_count": 0,
                        },
                    )

                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    side_effect=_capture_observation_payload,
                ):
                    dashboardReasoning.simulate_observation_summary("track-1")

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]

        self.assertIn("latest_track_snapshot", payload)
        self.assertIn("tracking_agent_context", payload)
        self.assertIn("coordinate_context", payload)

        latest_track_snapshot = payload["latest_track_snapshot"]
        self.assertTrue(latest_track_snapshot["available"])
        self.assertEqual(latest_track_snapshot["track_id"], "track-1")
        self.assertEqual(latest_track_snapshot["ground_coordinate_unit"], "meters")
        self.assertEqual(latest_track_snapshot["position_kind"], "calibrated_ground_plane")
        self.assertEqual(latest_track_snapshot["relative_depth"], 0.42)

        tracking_agent_context = payload["tracking_agent_context"]
        self.assertTrue(tracking_agent_context["available"])
        self.assertEqual(tracking_agent_context["vlm_model"], "qwen-3.8-27b")
        self.assertEqual(len(tracking_agent_context["selected_updates"]), 1)
        self.assertEqual(
            tracking_agent_context["selected_updates"][0]["detection_id"],
            "track-1",
        )

        coordinate_context = payload["coordinate_context"]
        self.assertTrue(coordinate_context["available"])
        self.assertEqual(coordinate_context["coordinate_system"], "calibrated_ground_plane")
        self.assertEqual(coordinate_context["calibration_status"], "loaded")

    def test_reasoning_handoff_preserves_tracking_and_coordinate_evidence(self) -> None:
        # Performs test reasoning handoff preserves tracking and coordinate evidence logic.
        radar_state = {
            "coordinate_system": "calibrated_ground_plane",
            "calibration_status": "loaded",
            "camera_status": "connected",
            "detector_status": "running",
            "last_observation_timestamp": 7200.0,
            "tracking_agent_observation": {
                "timestamp": 7200.0,
                "source_agent": "visual_tracking",
                "vlm_model": "qwen-3.8-27b",
                "proposal_latency_ms": 15.2,
                "uncertainties": ["brief_occlusion"],
                "updates": [
                    {
                        "detection_id": "track-1",
                        "resolved_track_id": "track-1",
                        "accepted": True,
                        "rejection_reasons": [],
                        "uncertainties": [],
                        "supporting_evidence": ["depth_consistency"],
                    }
                ],
            },
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 7200.0,
                    "pixel_position": {"x": 120.0, "y": 240.0},
                    "ground_position": {"x": 2.0, "z": 5.0},
                    "ground_coordinate_unit": "meters",
                    "position_kind": "calibrated_ground_plane",
                    "relative_depth": 0.31,
                    "confidence": 0.89,
                    "quality_flags": ["ground_plane_homography"],
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_path.write_text(json.dumps(radar_state), encoding="utf-8")

            with patch("tools.dashboardReasoning.RADAR_STATE_PATH", radar_path):
                captured_reasoning_inputs: list[dict] = []

                with patch(
                    "tools.dashboardReasoning._call_observation_agent_with_retry",
                    return_value=(
                        {
                            "summary": "Observation includes rich track context.",
                            "measured_facts": ["single validated point"],
                            "interpretations": ["transit likely"],
                            "hypotheses": [],
                            "supporting_observation_ids": ["track-1:7200.000000"],
                            "conflicting_evidence": [],
                            "uncertainties": [],
                            "needs_more_observations": False,
                        },
                        {
                            "status": "ok",
                            "model": "gpt-oss-120b",
                            "latency_ms": 1.0,
                            "error": None,
                            "attempts": 1,
                            "retry_count": 0,
                        },
                    ),
                ):

                    def _capture_reasoning_payload(user_payload: dict):
                        # Performs capture reasoning payload logic.
                        captured_reasoning_inputs.append(user_payload)
                        return (
                            {
                                "assessment_label": "requires_review",
                                "summary": "Review due to movement pattern.",
                                "rationale": "Trajectory and context warrant attention.",
                                "recommendation": "operator_review",
                                "supporting_observation_ids": ["track-1:7200.000000"],
                                "uncertainties": [],
                                "limitations": [],
                                "alternative_interpretations": [],
                                "missing_evidence": [],
                                "contradictory_evidence": [],
                                "request_more_observations": False,
                            },
                            {
                                "status": "ok",
                                "model": "gpt-oss-120b",
                                "latency_ms": 1.0,
                                "error": None,
                                "attempts": 1,
                                "retry_count": 0,
                            },
                        )

                    with patch(
                        "tools.dashboardReasoning._call_reasoning_agent_with_retry",
                        side_effect=_capture_reasoning_payload,
                    ):
                        dashboardReasoning.simulate_reasoning_assessment("track-1")

        self.assertEqual(len(captured_reasoning_inputs), 1)
        reasoning_input = captured_reasoning_inputs[0]
        observation_summary = reasoning_input["observation_summary"]

        self.assertIn("latest_track_snapshot", observation_summary)
        self.assertIn("tracking_agent_context", observation_summary)
        self.assertIn("coordinate_context", observation_summary)

        latest_track_snapshot = observation_summary["latest_track_snapshot"]
        self.assertTrue(latest_track_snapshot["available"])
        self.assertEqual(latest_track_snapshot["ground_coordinate_unit"], "meters")
        self.assertEqual(latest_track_snapshot["position_kind"], "calibrated_ground_plane")

        tracking_agent_context = observation_summary["tracking_agent_context"]
        self.assertTrue(tracking_agent_context["available"])
        self.assertEqual(tracking_agent_context["vlm_model"], "qwen-3.8-27b")
        self.assertEqual(
            tracking_agent_context["selected_updates"][0]["detection_id"],
            "track-1",
        )

        coordinate_context = observation_summary["coordinate_context"]
        self.assertTrue(coordinate_context["available"])
        self.assertEqual(coordinate_context["coordinate_system"], "calibrated_ground_plane")
        self.assertEqual(coordinate_context["calibration_status"], "loaded")

    def test_extract_latest_track_snapshot_normalizes_relative_units(self) -> None:
        # Performs test extract latest track snapshot normalizes relative units logic.
        radar_state = {
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 8000.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "ground_coordinate_unit": "meters",
                    "position_kind": "relative_ground_estimate",
                    "quality_flags": ["estimated_ground_from_image"],
                }
            ]
        }

        snapshot = dashboardReasoning._extract_latest_track_snapshot(radar_state, "track-1")

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["position_kind"], "relative_ground_estimate")
        self.assertEqual(snapshot["ground_coordinate_unit"], "relative_units")
        self.assertIn("ground_unit_normalized:relative_units", snapshot["quality_flags"])

    def test_extract_latest_track_snapshot_downgrades_invalid_calibrated_unit(self) -> None:
        # Performs test extract latest track snapshot downgrades invalid calibrated unit logic.
        radar_state = {
            "tracks": [
                {
                    "track_id": "track-1",
                    "timestamp": 8001.0,
                    "ground_position": {"x": 1.0, "z": 2.0},
                    "ground_coordinate_unit": "relative_units",
                    "position_kind": "calibrated_ground_plane",
                    "quality_flags": ["ground_plane_homography"],
                }
            ]
        }

        snapshot = dashboardReasoning._extract_latest_track_snapshot(radar_state, "track-1")

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["position_kind"], "relative_ground_estimate")
        self.assertEqual(snapshot["ground_coordinate_unit"], "relative_units")
        self.assertIn(
            "position_kind_normalized:invalid_calibrated_unit",
            snapshot["quality_flags"],
        )


if __name__ == "__main__":
    unittest.main()
