from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

mcp_stub = sys.modules.get("mcp")
if mcp_stub is None:
    mcp_stub = types.ModuleType("mcp")
    sys.modules["mcp"] = mcp_stub

if not hasattr(mcp_stub, "ClientSession"):
    class _StubClientSession:
        pass

    mcp_stub.ClientSession = _StubClientSession

if not hasattr(mcp_stub, "StdioServerParameters"):
    class _StubStdioServerParameters:
        def __init__(self, *, command: str, args: list[str], env: dict[str, str], cwd: str):
            self.command = command
            self.args = args
            self.env = env
            self.cwd = cwd

    mcp_stub.StdioServerParameters = _StubStdioServerParameters

from agents.orchestration.runtime import flow
from agents.orchestration import agent_orchestration


class TestPhase16ESubmissionDecisionLogging(unittest.TestCase):
    def test_resolve_submission_decisions_path_prefers_explicit_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_decision_path_explicit_") as tmp:
            explicit_path = Path(tmp) / "custom.jsonl"
            with patch.dict(
                os.environ,
                {
                    "PHASE16E_VERIFICATION_SUBMISSION_DECISIONS_PATH": str(explicit_path),
                    "PHASE16E_VERIFICATION_RESULTS_PATH": str(Path(tmp) / "results.json"),
                },
                clear=False,
            ):
                resolved = flow._resolve_submission_decisions_path()
            self.assertEqual(resolved, explicit_path.resolve())

    def test_resolve_submission_decisions_path_falls_back_to_results_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_decision_path_results_") as tmp:
            results_path = Path(tmp) / "run" / "latest_verification_queue_results.json"
            with patch.dict(
                os.environ,
                {
                    "PHASE16E_VERIFICATION_SUBMISSION_DECISIONS_PATH": "",
                    "PHASE16E_VERIFICATION_RESULTS_PATH": str(results_path),
                },
                clear=False,
            ):
                resolved = flow._resolve_submission_decisions_path()
            self.assertEqual(resolved, results_path.resolve().parent / "verification_submission_decisions.jsonl")

    def test_append_submission_decision_writes_safe_jsonl(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_decision_append_") as tmp:
            decisions_path = Path(tmp) / "run" / "verification_submission_decisions.jsonl"
            payload = {
                "run_id": "run-1",
                "event_origin": "production_monitor",
                "trigger_event_id": "evt-1",
                "track_id": "track-1",
                "evidence_requested": True,
                "evidence_message_id": None,
                "queue_request_id": "req-1",
                "submission_eligibility": True,
                "skip_reason": None,
                "submission_attempted": True,
                "submission_accepted": False,
            }
            with patch.dict(
                os.environ,
                {
                    "PHASE16E_VERIFICATION_SUBMISSION_DECISIONS_PATH": str(decisions_path),
                },
                clear=False,
            ):
                flow._append_submission_decision(payload)

            lines = decisions_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["run_id"], "run-1")
            self.assertEqual(record["event_origin"], "production_monitor")
            self.assertEqual(record["queue_request_id"], "req-1")
            self.assertIn("logged_at", record)
            self.assertNotIn("OPENROUTER_API_KEY", record)

    def test_startup_activation_skip_logging(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_startup_skip_log_") as tmp:
            decisions_path = Path(tmp) / "run" / "verification_submission_decisions.jsonl"
            events = [
                {
                    "task_id": "cycle-1:track-1",
                    "track_id": "sim-1",
                    "evidence_requested": True,
                }
            ]

            with patch.dict(
                os.environ,
                {
                    "PHASE16E_VERIFICATION_SUBMISSION_DECISIONS_PATH": str(decisions_path),
                    "PHASE16E_RUN_ID": "run-startup",
                },
                clear=False,
            ):
                agent_orchestration._record_startup_activation_submission_skips(events)

            lines = decisions_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["run_id"], "run-startup")
            self.assertEqual(record["event_origin"], "startup_activation")
            self.assertEqual(record["track_id"], "sim-1")
            self.assertEqual(record["skip_reason"], "startup_activation_cycle_no_submission_path")
            self.assertFalse(record["submission_attempted"])
            self.assertFalse(record["submission_accepted"])


if __name__ == "__main__":
    unittest.main()
