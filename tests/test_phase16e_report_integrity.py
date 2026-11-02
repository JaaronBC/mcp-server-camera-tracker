from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from tools.phase16e_report import generate_report


class TestPhase16EReportIntegrity(unittest.TestCase):
    def _write_base(self, run_dir: Path, run_id: str = "run-1") -> float:
        start_ts = time.time()
        end_ts = start_ts + 1.0
        (run_dir / "start_ts.txt").write_text(f"{start_ts}\n", encoding="utf-8")
        (run_dir / "end_ts.txt").write_text(f"{end_ts}\n", encoding="utf-8")
        (run_dir / "main_exit_code.txt").write_text("0\n", encoding="utf-8")
        (run_dir / "run_interrupted.txt").write_text("0\n", encoding="utf-8")
        (run_dir / "run_id.txt").write_text(f"{run_id}\n", encoding="utf-8")
        (run_dir / "post_shutdown_processes.txt").write_text("", encoding="utf-8")
        (run_dir / "provider_call_budget_events.jsonl").write_text("", encoding="utf-8")
        (run_dir / "provider_call_budget_state.json").write_text(
            json.dumps(
                {
                    "totals": {
                        "per_role_attempted": {
                            "tracking_vlm": 0,
                            "observation_text": 0,
                            "reasoning_text": 0,
                            "observation_visual": 1,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return start_ts

    def test_result_without_submission_fails_queue_submission(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_integrity_no_submission_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            start_ts = self._write_base(run_dir)

            (run_dir / "latest_verification_queue_results.json").write_text(
                json.dumps([
                    {"run_id": "run-1", "track_id": "track-1", "request_id": "req-1", "validation_outcome": "accepted"}
                ]),
                encoding="utf-8",
            )
            (run_dir / "latest_verification_queue_submissions.json").write_text("[]", encoding="utf-8")
            (run_dir / "agent_events.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "event_type": "reasoning_cycle",
                        "timestamp": start_ts + 0.2,
                        "track_id": "track-1",
                        "evidence_requested": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = generate_report(run_dir)
            self.assertEqual(report["verdict"], "FAIL")
            self.assertIn("queue_submission_missing:track-1", report["failure_reasons"])

    def test_track_match_with_different_request_id_does_not_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_integrity_reqid_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            start_ts = self._write_base(run_dir)

            (run_dir / "latest_verification_queue_submissions.json").write_text(
                json.dumps([
                    {
                        "run_id": "run-1",
                        "track_id": "track-1",
                        "request_id": "req-submission",
                        "submitted_at": start_ts + 0.25,
                        "accepted": True,
                        "status": "queued",
                        "message_id": "msg-1",
                    }
                ]),
                encoding="utf-8",
            )
            (run_dir / "latest_verification_queue_results.json").write_text(
                json.dumps([
                    {"run_id": "run-1", "track_id": "track-1", "request_id": "req-different", "validation_outcome": "accepted"}
                ]),
                encoding="utf-8",
            )
            (run_dir / "agent_events.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "event_type": "reasoning_cycle",
                        "timestamp": start_ts + 0.2,
                        "track_id": "track-1",
                        "evidence_requested": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = generate_report(run_dir)
            row = report["correlated_chain_rows"][0]
            self.assertEqual(row["queue_request_id"], "req-submission")
            self.assertFalse(row["verification_success"])


if __name__ == "__main__":
    unittest.main()
