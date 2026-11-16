from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
from uuid import uuid4

HARNESS_PATH = Path("/tmp/mcp_camera_tracker_runs/run_tracking_only_harness.py")


def _load_harness(env_overrides: dict[str, str] | None = None) -> ModuleType:
    if not HARNESS_PATH.is_file():
        raise unittest.SkipTest(f"Harness script not found: {HARNESS_PATH}")

    module_name = f"tracking_only_harness_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(HARNESS_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load harness module spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with patch.dict(os.environ, env_overrides or {}, clear=False):
        spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


class TestTrackingOnlyHarnessControls(unittest.TestCase):
    def test_video_override_preserves_default_when_unset(self) -> None:
        module = _load_harness({"TRACKING_ONLY_VIDEO_PATH": "", "TRACKING_ONLY_USE_SHORT_CLIP": "0"})
        self.assertEqual(module.VIDEO_PATH, module.DEFAULT_VIDEO_PATH)

    def test_short_clip_selection_used_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected_clip = (root / "occlusion_short_clip.mp4").resolve()
            selected_clip.write_bytes(b"clip")

            module = _load_harness(
                {
                    "TRACKING_ONLY_VIDEO_PATH": "",
                    "TRACKING_ONLY_USE_SHORT_CLIP": "1",
                    "TRACKING_ONLY_SHORT_CLIP_PATH": str(selected_clip),
                }
            )
            self.assertEqual(module.VIDEO_PATH, selected_clip)

    def test_video_override_requires_exact_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = (root / "selected_occlusion.mp4").resolve()
            selected.write_bytes(b"video")

            module = _load_harness({"TRACKING_ONLY_VIDEO_PATH": str(selected)})
            self.assertEqual(module.VIDEO_PATH, selected)
            self.assertFalse(any(err.startswith("missing_video_file:") for err in module._preflight_errors()))

            missing = (root / "missing_occlusion.mp4").resolve()
            module_missing = _load_harness({"TRACKING_ONLY_VIDEO_PATH": str(missing)})
            errors = module_missing._preflight_errors()
            self.assertIn(f"missing_video_file:{missing}", errors)

    def test_budget_and_phase2d_paths_propagate_and_validate(self) -> None:
        module = _load_harness(
            {
                "TRACKING_ONLY_BUDGET_TOTAL": "2",
                "TRACKING_ONLY_BUDGET_TRACKING": "2",
                "TRACKING_ONLY_BUDGET_OBS_TEXT": "0",
                "TRACKING_ONLY_BUDGET_REASONING_TEXT": "0",
                "TRACKING_ONLY_BUDGET_OBS_VISUAL": "0",
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            env = module._build_env(paths, module.ORIGINAL_REPO_ROOT)

            self.assertEqual(env["PROVIDER_CALL_BUDGET_TOTAL_LIMIT"], "2")
            self.assertEqual(env["PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT"], "2")
            self.assertEqual(env["PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT"], "0")
            self.assertEqual(env["PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT"], "0")
            self.assertEqual(env["PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT"], "0")

            self.assertEqual(env["PHASE2D_TRACKING_DECISION_TOOL_REQUESTS_PATH"], str(paths.tracking_decision_tool_requests))
            self.assertEqual(env["PHASE2D_TRACKING_DECISION_TOOL_RESULTS_PATH"], str(paths.tracking_decision_tool_results))
            self.assertEqual(env["PHASE2D_TRACKING_DECISION_REQUEST_INDEX_PATH"], str(paths.tracking_decision_request_index))
            self.assertEqual(env["PHASE2D_TRACKING_DECISION_RESULT_INDEX_PATH"], str(paths.tracking_decision_result_index))
            self.assertEqual(env["PHASE2D_TRACKING_DECISION_IPC_ENABLED"], "1")

            self.assertEqual(module._validate_controlled_isolation(paths, env), [])

    def test_isolation_validation_rejects_non_tracking_budgets(self) -> None:
        module = _load_harness({"TRACKING_ONLY_BUDGET_OBS_TEXT": "1"})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            env = module._build_env(paths, module.ORIGINAL_REPO_ROOT)
            errors = module._validate_controlled_isolation(paths, env)
            self.assertTrue(any(err.startswith("non_tracking_budget_must_be_zero:") for err in errors))

    def test_outcome_classification_handles_legitimate_inconclusive_and_accepted(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))

            empty_artifacts = module._collect_decision_artifacts(paths)
            empty_report = {
                "error": None,
                "stopped_by": None,
                "milestones": {"provider_response_parsed_successfully": {"status": "unknown"}},
            }
            outcome = module._classify_investigation_outcome(empty_report, empty_artifacts)
            self.assertEqual(outcome["status"], "inconclusive")
            self.assertEqual(outcome["reason"], "missing_tool_request")

            request_rows = [
                {
                    "investigation_id": "inv-1",
                    "decision": {
                        "action": "request_additional_tracking_evidence",
                        "investigation_id": "inv-1",
                        "tool_requests": [
                            {
                                "request_id": "req-1",
                                "tool_name": "get_track_evidence_history",
                                "arguments": {"track_id": "track-1"},
                            }
                        ],
                    },
                },
                {
                    "investigation_id": "inv-1",
                    "decision": {
                        "action": "finalize_investigation",
                        "investigation_id": "inv-1",
                        "tool_requests": [],
                    },
                },
            ]
            result_rows = [
                {
                    "investigation_id": "inv-1",
                    "request_id": "req-1",
                    "status": "ok",
                    "result": {"evidence": []},
                }
            ]

            _write_jsonl(paths.tracking_decision_tool_requests, request_rows)
            _write_jsonl(paths.tracking_decision_tool_results, result_rows)

            decision_artifacts = module._collect_decision_artifacts(paths)
            accepted_report = {
                "error": None,
                "stopped_by": None,
                "milestones": {"provider_response_parsed_successfully": {"status": "hit"}},
            }
            accepted = module._classify_investigation_outcome(accepted_report, decision_artifacts)
            self.assertEqual(accepted["status"], "accepted")

            invalid_llm_report = {
                "error": None,
                "stopped_by": None,
                "milestones": {"provider_response_parsed_successfully": {"status": "miss"}},
            }
            inconclusive = module._classify_investigation_outcome(invalid_llm_report, decision_artifacts)
            self.assertEqual(inconclusive["status"], "inconclusive")
            self.assertEqual(inconclusive["reason"], "invalid_llm_response")

    def test_outcome_classification_accepts_legitimate_no_tool_inconclusive_initial_decision(self) -> None:
        module = _load_harness({})
        report = {
            "error": None,
            "stopped_by": "post_success_observation_published",
            "milestones": {
                "provider_response_parsed_successfully": {
                    "status": "hit",
                }
            },
        }
        decision_artifacts = {
            "initial_decision": {
                "action": "inconclusive",
                "reason": "insufficient_evidence",
                "investigation_id": "inv-1",
                "tool_requests": [],
            },
            "follow_up_decision": None,
            "tool_request_count": 0,
            "result_event_count": 0,
            "first_tool_request": None,
            "first_tool_result": None,
        }
        outcome = module._classify_investigation_outcome(report, decision_artifacts)
        self.assertEqual(outcome["status"], "inconclusive")
        self.assertEqual(outcome["reason"], "initial_inconclusive_decision_without_tool_request")

    def test_first_investigation_gate_selects_once_and_rejects_second_candidate(self) -> None:
        module = _load_harness({})

        selected, allowed, reason = module._gate_select_first_investigation(None, "inv-a")
        self.assertTrue(allowed)
        self.assertEqual(reason, "selected_first_investigation")
        self.assertEqual(selected, "inv-a")

        selected, allowed, reason = module._gate_select_first_investigation(selected, "inv-b")
        self.assertFalse(allowed)
        self.assertEqual(reason, "non_selected_investigation_rejected")
        self.assertEqual(selected, "inv-a")

    def test_pre_llm_gate_allows_only_selected_investigation_follow_up(self) -> None:
        module = _load_harness({})

        allowed, reason = module._gate_allows_tracking_submission(
            "inv-a",
            {"investigation_id": "inv-a", "request_id": "req-1"},
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "selected_follow_up_allowed")

        allowed, reason = module._gate_allows_tracking_submission(
            "inv-a",
            {"investigation_id": "inv-b", "request_id": "req-2"},
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "blocked_pre_llm_non_selected_follow_up")

        allowed, reason = module._gate_allows_tracking_submission("inv-a", None)
        self.assertFalse(allowed)
        self.assertEqual(reason, "blocked_pre_llm_new_investigation_candidate")

    def test_selected_investigation_allows_initial_then_follow_up(self) -> None:
        module = _load_harness({})

        allowed, reason = module._gate_allows_tracking_submission(None, None)
        self.assertTrue(allowed)
        self.assertEqual(reason, "no_selected_investigation")

        selected, selected_allowed, selected_reason = module._gate_select_first_investigation(None, "inv-a")
        self.assertTrue(selected_allowed)
        self.assertEqual(selected_reason, "selected_first_investigation")
        self.assertEqual(selected, "inv-a")

        allowed, reason = module._gate_allows_tracking_submission(
            selected,
            {"investigation_id": "inv-a", "request_id": "req-follow-up"},
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "selected_follow_up_allowed")

    def test_process_probe_confirms_shim_scope_and_pre_llm_wrapping(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            env = module._build_env(paths)

            with_env = module._probe_child_gate_installation(
                module.PYTHON_PATH,
                module.ORIGINAL_REPO_ROOT,
                env,
                expect_wrapped=True,
                shim_dir=paths.first_investigation_gate_shim_dir,
            )
            without_env = module._probe_child_gate_installation(
                module.PYTHON_PATH,
                module.ORIGINAL_REPO_ROOT,
                dict(os.environ),
                expect_wrapped=False,
                shim_dir=paths.first_investigation_gate_shim_dir,
            )

            self.assertTrue(with_env["passed"], with_env)
            self.assertTrue(with_env["details"]["runtime_submit_wrapped"])
            self.assertTrue(with_env["details"]["tracking_loop_alias_wrapped"])
            self.assertTrue(with_env["details"]["decision_submit_wrapped"])
            self.assertTrue(without_env["passed"], without_env)

    def test_process_probe_confirms_child_budget_env_enforced(self) -> None:
        module = _load_harness(
            {
                "TRACKING_ONLY_BUDGET_TOTAL": "2",
                "TRACKING_ONLY_BUDGET_TRACKING": "2",
                "TRACKING_ONLY_BUDGET_OBS_TEXT": "0",
                "TRACKING_ONLY_BUDGET_REASONING_TEXT": "0",
                "TRACKING_ONLY_BUDGET_OBS_VISUAL": "0",
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            env = module._build_env(paths)
            probe = module._probe_child_budget_env(module.PYTHON_PATH, module.ORIGINAL_REPO_ROOT, env)
            self.assertTrue(probe["passed"], probe)
            self.assertEqual(probe["details"]["PROVIDER_CALL_BUDGET_TOTAL_LIMIT"], "2")
            self.assertEqual(probe["details"]["PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT"], "2")

    def test_process_probe_confirms_child_tracking_agent_config(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            runtime_env = module._build_env(paths)
            probe_env, _ = module._build_prelaunch_probe_env(runtime_env, Path(temp_dir))
            probe = module._probe_child_tracking_agent_config(module.PYTHON_PATH, module.ORIGINAL_REPO_ROOT, probe_env)
            self.assertTrue(probe["passed"], probe)
            self.assertEqual(probe["details"]["phase2d_tracking_decision_ipc_enabled_env"], "1")
            self.assertTrue(probe["details"]["dashboard_tracking_decision_ipc_enabled"])
            self.assertTrue(probe["details"]["dashboard_state_tracking_decision_ipc_enabled"])
            self.assertEqual(probe["details"]["tracking_agent_compact_decision_mode"], "1")
            self.assertEqual(
                probe["details"]["tracking_agent_request_timeout_seconds_env"],
                str(module.TIMEOUT_SECONDS),
            )
            self.assertEqual(probe["details"]["request_kwargs_max_tokens"], 768)
            self.assertEqual(probe["details"]["fake_completion_call_count"], 1)

            self.assertTrue(probe["details"]["semantic_probe_input_omits_tool_requests"])
            self.assertTrue(probe["details"]["semantic_probe_input_omits_tool_name"])
            self.assertTrue(probe["details"]["semantic_probe_input_omits_provenance"])
            self.assertTrue(probe["details"]["semantic_probe_input_omits_investigation_id"])
            self.assertTrue(probe["details"]["semantic_probe_input_omits_request_id"])
            self.assertTrue(probe["details"]["semantic_probe_input_omits_correlation_id"])

            self.assertEqual(
                probe["details"]["semantic_probe_action"],
                "request_additional_tracking_evidence",
            )
            self.assertEqual(
                probe["details"]["semantic_probe_tool_name"],
                "get_track_evidence_history",
            )
            self.assertEqual(
                probe["details"]["semantic_probe_investigation_id"],
                "inv-semantic-probe-1",
            )
            self.assertEqual(
                probe["details"]["semantic_probe_request_id"],
                "req-semantic-probe-1",
            )
            self.assertEqual(
                probe["details"]["semantic_probe_correlation_id"],
                "corr-semantic-probe-1",
            )
            self.assertEqual(probe["details"]["semantic_probe_tool_request_count"], 1)
            self.assertTrue(probe["details"]["semantic_probe_canonical_tool_request_shape"])

            semantic_arguments = probe["details"]["semantic_probe_arguments"]
            self.assertIsInstance(semantic_arguments, dict)
            self.assertEqual(semantic_arguments.get("track_id"), "track-semantic-probe-1")
            self.assertEqual(semantic_arguments.get("max_messages"), 3)
            self.assertEqual(semantic_arguments.get("max_history_records"), 2)

            self.assertEqual(probe["details"]["semantic_probe_validation_errors"], [])
            self.assertEqual(probe["details"]["semantic_probe_validation_error_count"], 0)
            self.assertTrue(probe["details"]["semantic_probe_validator_passed"])
            self.assertTrue(probe["details"]["semantic_probe_negative_rejected"])
            self.assertIn(
                "decision_action_invalid",
                probe["details"]["semantic_probe_negative_validation_errors"],
            )

    def test_process_probe_rejects_negative_semantic_action(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            runtime_env = module._build_env(paths)
            probe_env, _ = module._build_prelaunch_probe_env(runtime_env, Path(temp_dir))
            probe = module._probe_child_tracking_agent_config(
                module.PYTHON_PATH,
                module.ORIGINAL_REPO_ROOT,
                probe_env,
            )
            self.assertTrue(probe["passed"], probe)
            negative_errors = probe["details"].get("semantic_probe_negative_validation_errors")
            self.assertIsInstance(negative_errors, list)
            self.assertGreater(len(negative_errors), 0)
            self.assertIn("decision_action_invalid", negative_errors)

    def test_probe_artifact_paths_are_disjoint_from_runtime_paths(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            paths = module._build_paths(run_dir)
            runtime_env = module._build_env(paths, module.ORIGINAL_REPO_ROOT)
            probe_env, _ = module._build_prelaunch_probe_env(runtime_env, run_dir)
            isolation = module._probe_path_isolation_summary(runtime_env, probe_env)

            self.assertTrue(isolation["all_disjoint"], isolation)
            for key in module._PROBE_ISOLATED_PATH_ENV_KEYS:
                self.assertTrue(isolation["keys"][key]["disjoint"], key)

    def test_fake_probe_success_does_not_create_runtime_provider_attempt(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            paths = module._build_paths(run_dir)
            runtime_env = module._build_env(paths)
            probe_env, _ = module._build_prelaunch_probe_env(runtime_env, run_dir)

            probe_log = Path(probe_env["PROVIDER_CALL_BUDGET_LOG_PATH"]).resolve()
            probe_event = {
                "event_type": "provider_call_attempt",
                "role": "tracking_vlm",
                "outcome": "ok",
                "attempt_id": "probe-attempt-1",
                "timestamp": 100.0,
                "latency_ms": 0.2,
            }
            _write_jsonl(probe_log, [probe_event])

            self.assertFalse(paths.budget_log.exists())
            self.assertEqual(module._extract_tracking_events(paths.budget_log), [])
            summary = module._summarize_provider_events(paths.budget_log)
            self.assertEqual(summary["attempts"], 0)

    def test_fake_probe_success_does_not_trigger_post_success_grace_expiry(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            paths = module._build_paths(run_dir)
            runtime_env = module._build_env(paths)
            probe_env, _ = module._build_prelaunch_probe_env(runtime_env, run_dir)

            probe_log = Path(probe_env["PROVIDER_CALL_BUDGET_LOG_PATH"]).resolve()
            _write_jsonl(
                probe_log,
                [
                    {
                        "event_type": "provider_call_attempt",
                        "role": "tracking_vlm",
                        "outcome": "ok",
                        "attempt_id": "probe-attempt-2",
                        "timestamp": 123.0,
                    }
                ],
            )

            runtime_events = module._extract_tracking_events(paths.budget_log)
            first_success = next(
                (event for event in runtime_events if module._is_successful_provider_attempt(event)),
                None,
            )
            self.assertIsNone(first_success)

    def test_runtime_budget_remains_zero_after_prelaunch_probe_writes(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            paths = module._build_paths(run_dir)
            runtime_env = module._build_env(paths)
            probe_env, _ = module._build_prelaunch_probe_env(runtime_env, run_dir)

            probe_state = Path(probe_env["PROVIDER_CALL_BUDGET_STATE_PATH"]).resolve()
            probe_state.parent.mkdir(parents=True, exist_ok=True)
            probe_state.write_text(
                json.dumps({"total_count": 1, "role_counts": {"tracking_vlm": 1}}),
                encoding="utf-8",
            )

            self.assertFalse(paths.budget_state.exists())
            runtime_summary = module._summarize_provider_events(paths.budget_log)
            self.assertEqual(runtime_summary["attempts"], 0)

    def test_genuine_runtime_event_is_still_counted(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            _write_jsonl(
                paths.budget_log,
                [
                    {
                        "event_type": "provider_call_attempt",
                        "role": "tracking_vlm",
                        "outcome": "ok",
                        "attempt_id": "runtime-attempt-1",
                        "timestamp": 200.0,
                        "latency_ms": 111.0,
                    }
                ],
            )

            events = module._extract_tracking_events(paths.budget_log)
            self.assertEqual(len(events), 1)
            self.assertTrue(module._is_successful_provider_attempt(events[0]))
            summary = module._summarize_provider_events(paths.budget_log)
            self.assertEqual(summary["attempts"], 1)

    def test_missing_runtime_observation_not_interpreted_as_successful_decision(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            milestones = module._milestone_state()
            artifact_timing = {"observation_records": []}
            module._scan_radar_for_milestones(
                Path(temp_dir) / "missing_latest_radar_state.json",
                10.0,
                20.0,
                milestones,
                set(),
                artifact_timing,
            )
            self.assertEqual(milestones["tracking_agent_returned_valid_observation"]["status"], "unknown")

            decision_artifacts = module._collect_decision_artifacts(paths)
            report = {
                "error": None,
                "stopped_by": "post_success_observation_grace_expired",
                "milestones": milestones,
            }
            outcome = module._classify_investigation_outcome(report, decision_artifacts)
            self.assertNotEqual(outcome["status"], "accepted")

    def test_fallback_observation_without_decision_does_not_satisfy_post_success_completion(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_payload = {
                "tracking_agent_observation": {
                    "timestamp": 15.0,
                    "source_agent": "visual_tracking",
                    "vlm_model": "qwen-3.8-27b",
                    "observation_id": "obs-fallback",
                    "uncertainties": ["vlm_unavailable"],
                    "decision": None,
                    "updates": [
                        {
                            "detection_id": "d1",
                            "uncertainties": ["no_vlm_proposal_used"],
                        }
                    ],
                }
            }
            radar_path.write_text(json.dumps(radar_payload), encoding="utf-8")

            milestones = module._milestone_state()
            artifact_timing = {"observation_records": []}
            module._scan_radar_for_milestones(
                radar_path,
                10.0,
                20.0,
                milestones,
                set(),
                artifact_timing,
            )
            self.assertEqual(milestones["fallback_observation_published"]["status"], "hit")

            records = artifact_timing["observation_records"]
            self.assertEqual(len(records), 1)
            record = records[0]
            decision_cycle = {
                "initial_decision_completed": False,
            }
            self.assertFalse(module._observation_record_satisfies_post_success_completion(record, decision_cycle))

    def test_invalid_decision_dictionary_observation_is_not_counted_as_completed_or_dispatched(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            radar_payload = {
                "tracking_agent_observation": {
                    "timestamp": 15.0,
                    "source_agent": "visual_tracking",
                    "vlm_model": "qwen-3.8-27b",
                    "observation_id": "obs-invalid-decision",
                    "uncertainties": ["vlm_invalid_decision_contract"],
                    "decision": {
                        "action": "request_additional_tracking_evidence",
                    },
                    "updates": [
                        {
                            "detection_id": "d1",
                            "uncertainties": ["no_vlm_proposal_used"],
                        }
                    ],
                }
            }
            radar_path.write_text(json.dumps(radar_payload), encoding="utf-8")

            milestones = module._milestone_state()
            artifact_timing = {"observation_records": []}
            module._scan_radar_for_milestones(
                radar_path,
                10.0,
                20.0,
                milestones,
                set(),
                artifact_timing,
            )

            records = artifact_timing["observation_records"]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertTrue(record["decision_present"])

            decision_cycle = {
                "initial_decision_completed": False,
                "mcp_request_submitted": False,
                "mcp_result_processed": False,
            }
            self.assertFalse(module._observation_record_satisfies_post_success_completion(record, decision_cycle))
            self.assertFalse(decision_cycle["mcp_request_submitted"])
            self.assertFalse(decision_cycle["mcp_result_processed"])

            outcome = module._classify_investigation_outcome(
                {
                    "error": None,
                    "stopped_by": "post_success_observation_grace_expired",
                    "milestones": {"provider_response_parsed_successfully": {"status": "miss"}},
                },
                {
                    "tool_request_count": 0,
                    "result_event_count": 0,
                    "initial_decision": None,
                    "follow_up_decision": None,
                },
            )
            self.assertNotEqual(outcome["status"], "accepted")

    def test_valid_inconclusive_decision_counts_as_completed_initial_decision(self) -> None:
        module = _load_harness({})
        decision_artifacts = {
            "initial_decision": {
                "action": "inconclusive",
                "reason": "insufficient_evidence",
                "investigation_id": "inv-1",
                "tool_requests": [],
            },
            "follow_up_decision": None,
            "tool_request_count": 0,
            "result_event_count": 0,
            "first_tool_request": None,
            "first_tool_result": None,
        }
        cycle = module._decision_cycle_state(decision_artifacts)
        self.assertTrue(cycle["initial_decision_valid"])
        self.assertEqual(cycle["initial_decision_action"], "inconclusive")
        self.assertTrue(cycle["initial_decision_completed"])
        self.assertFalse(cycle["mcp_request_submitted"])
        self.assertFalse(cycle["mcp_result_processed"])

    def test_valid_mcp_evidence_request_without_result_not_completed(self) -> None:
        module = _load_harness({})
        decision_artifacts = {
            "initial_decision": {
                "action": "request_additional_tracking_evidence",
                "investigation_id": "inv-1",
                "tool_requests": [
                    {
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {"track_id": "track-1"},
                    }
                ],
            },
            "follow_up_decision": None,
            "tool_request_count": 1,
            "result_event_count": 0,
            "first_tool_request": {
                "request_id": "req-1",
                "tool_name": "get_track_evidence_history",
            },
            "first_tool_result": None,
        }
        cycle = module._decision_cycle_state(decision_artifacts)
        self.assertTrue(cycle["initial_decision_valid"])
        self.assertTrue(cycle["mcp_request_submitted"])
        self.assertFalse(cycle["mcp_result_processed"])
        self.assertFalse(cycle["follow_up_decision_valid"])
        self.assertFalse(cycle["initial_decision_completed"])
        self.assertFalse(cycle["evidence_cycle_completed"])

    def test_completed_mcp_cycle_requires_result_and_follow_up_decision(self) -> None:
        module = _load_harness({})
        decision_artifacts = {
            "initial_decision": {
                "action": "request_additional_tracking_evidence",
                "investigation_id": "inv-1",
                "tool_requests": [
                    {
                        "request_id": "req-1",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {"track_id": "track-1"},
                    }
                ],
            },
            "follow_up_decision": {
                "action": "inconclusive",
                "investigation_id": "inv-1",
                "reason": "evidence_not_sufficient",
                "tool_requests": [],
            },
            "tool_request_count": 1,
            "result_event_count": 1,
            "first_tool_request": {
                "request_id": "req-1",
                "tool_name": "get_track_evidence_history",
            },
            "first_tool_result": {
                "status": "ok",
            },
        }
        cycle = module._decision_cycle_state(decision_artifacts)
        self.assertTrue(cycle["mcp_request_submitted"])
        self.assertTrue(cycle["mcp_result_processed"])
        self.assertTrue(cycle["follow_up_decision_valid"])
        self.assertTrue(cycle["initial_decision_completed"])
        self.assertTrue(cycle["evidence_cycle_completed"])

        record = {
            "scope": "in_window",
            "schema_valid": True,
            "fallback_only": False,
            "decision_present": True,
        }
        self.assertTrue(module._observation_record_satisfies_post_success_completion(record, cycle))

    def test_run_harness_fails_closed_when_gate_probe_fails(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = (root / "run").resolve()
            isolated_repo = (root / "isolated_repo").resolve()
            captures_dir = (root / "captures").resolve()
            captures_dir.mkdir(parents=True, exist_ok=True)
            video_path = (root / "video.mp4").resolve()
            video_path.write_bytes(b"video")

            def _fake_workspace_copy(src_repo_root: Path, dst_repo_root: Path) -> dict:
                dst_repo_root.mkdir(parents=True, exist_ok=True)
                (dst_repo_root / "captures").mkdir(parents=True, exist_ok=True)
                (dst_repo_root / "main.py").write_text("print('noop')\n", encoding="utf-8")
                return {
                    "source_repo_root": str(src_repo_root),
                    "isolated_repo_root": str(dst_repo_root),
                    "isolated_captures_dir": str((dst_repo_root / "captures").resolve()),
                    "copy_duration_seconds": 0.0,
                }

            with patch.object(module, "RUN_DIR", run_dir), \
                patch.object(module, "ISOLATED_REPO_ROOT", isolated_repo), \
                patch.object(module, "REPORT_PATH", (run_dir / "tracking_only_report.json").resolve()), \
                patch.object(module, "VIDEO_PATH", video_path), \
                patch.object(module, "ORIGINAL_CAPTURES_DIR", captures_dir), \
                patch.object(module, "_preflight_errors", return_value=[]), \
                patch.object(module, "_probe_child_interpreter", return_value={"passed": True}), \
                patch.object(module, "_build_isolated_workspace", side_effect=_fake_workspace_copy), \
                patch.object(module, "_server_process_snapshot", return_value=[]), \
                patch.object(module, "_dashboard_process_snapshot", return_value=[]), \
                patch.object(module, "_probe_child_gate_installation", side_effect=[
                    {"passed": False, "details": {"runtime_submit_wrapped": False}},
                    {"passed": True, "details": {}},
                ]), \
                patch.object(module, "_probe_child_budget_env", return_value={"passed": True, "details": {}}), \
                patch.object(module, "_probe_child_tracking_agent_config", return_value={"passed": True, "details": {}}):
                exit_code = module.run_harness()

            self.assertEqual(exit_code, 1)

            report = json.loads((run_dir / "artifacts" / "tracking_only_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("error"), "prelaunch_probe_failed")
            self.assertIn("child_gate_probe_with_harness_env_failed", report["preflight"]["errors"])

    def test_identity_registry_classification_distinguishes_verified_vs_missing(self) -> None:
        module = _load_harness({})

        verified = module._classify_identity_registry_evidence(
            {
                "tracking_identity_snapshot": {"exists": True, "sha256": "abc"},
            },
            {
                "tracking_identity_snapshot": {"exists": True, "sha256": "abc"},
            },
        )
        self.assertEqual(verified["status"], "verified_no_mutation")

        missing = module._classify_identity_registry_evidence(
            {
                "tracking_identity_snapshot": {"exists": False},
            },
            {
                "tracking_identity_snapshot": {"exists": True, "sha256": "def"},
            },
        )
        self.assertEqual(missing["status"], "missing_or_unavailable_registry_evidence")

    def test_isolation_validation_rejects_disabled_tracking_decision_ipc(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module._build_paths(Path(temp_dir))
            env = module._build_env(paths)
            env["PHASE2D_TRACKING_DECISION_IPC_ENABLED"] = "0"
            errors = module._validate_controlled_isolation(paths, env)
            self.assertTrue(
                any(error.startswith("tracking_decision_ipc_must_be_enabled:") for error in errors),
                errors,
            )

    def test_run_harness_fails_closed_when_child_ipc_probe_reports_disabled(self) -> None:
        module = _load_harness({})
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = (root / "run").resolve()
            isolated_repo = (root / "isolated_repo").resolve()
            captures_dir = (root / "captures").resolve()
            captures_dir.mkdir(parents=True, exist_ok=True)
            video_path = (root / "video.mp4").resolve()
            video_path.write_bytes(b"video")

            def _fake_workspace_copy(src_repo_root: Path, dst_repo_root: Path) -> dict:
                dst_repo_root.mkdir(parents=True, exist_ok=True)
                (dst_repo_root / "captures").mkdir(parents=True, exist_ok=True)
                (dst_repo_root / "main.py").write_text("print('noop')\n", encoding="utf-8")
                return {
                    "source_repo_root": str(src_repo_root),
                    "isolated_repo_root": str(dst_repo_root),
                    "isolated_captures_dir": str((dst_repo_root / "captures").resolve()),
                    "copy_duration_seconds": 0.0,
                }

            with patch.object(module, "RUN_DIR", run_dir), \
                patch.object(module, "ISOLATED_REPO_ROOT", isolated_repo), \
                patch.object(module, "REPORT_PATH", (run_dir / "tracking_only_report.json").resolve()), \
                patch.object(module, "VIDEO_PATH", video_path), \
                patch.object(module, "ORIGINAL_CAPTURES_DIR", captures_dir), \
                patch.object(module, "_preflight_errors", return_value=[]), \
                patch.object(module, "_probe_child_interpreter", return_value={"passed": True}), \
                patch.object(module, "_build_isolated_workspace", side_effect=_fake_workspace_copy), \
                patch.object(module, "_server_process_snapshot", return_value=[]), \
                patch.object(module, "_dashboard_process_snapshot", return_value=[]), \
                patch.object(module, "_probe_child_gate_installation", side_effect=[
                    {"passed": True, "details": {"runtime_submit_wrapped": True}},
                    {"passed": True, "details": {}},
                ]), \
                patch.object(module, "_probe_child_budget_env", return_value={"passed": True, "details": {}}), \
                patch.object(
                    module,
                    "_probe_child_tracking_agent_config",
                    return_value={
                        "passed": False,
                        "details": {
                            "phase2d_tracking_decision_ipc_enabled_env": "0",
                            "dashboard_tracking_decision_ipc_enabled": False,
                            "dashboard_state_tracking_decision_ipc_enabled": False,
                        },
                        "mismatches": [
                            "PHASE2D_TRACKING_DECISION_IPC_ENABLED_ENV:expected=1:actual=0",
                        ],
                    },
                ):
                exit_code = module.run_harness()

            self.assertEqual(exit_code, 1)

            report = json.loads((run_dir / "artifacts" / "tracking_only_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("error"), "prelaunch_probe_failed")
            self.assertIn("child_tracking_agent_config_probe_failed", report["preflight"]["errors"])


if __name__ == "__main__":
    unittest.main()
