from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from tools.phase16e_report import generate_report


class TestPhase16EReportCorrelation(unittest.TestCase):
    def test_correlated_rows_include_only_triggering_reasoning_events(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_report_corr_") as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            captures_dir = root / "captures"
            run_dir.mkdir(parents=True, exist_ok=True)
            captures_dir.mkdir(parents=True, exist_ok=True)

            start_ts = time.time()
            end_ts = start_ts + 1.0

            (run_dir / "start_ts.txt").write_text(f"{start_ts}\n", encoding="utf-8")
            (run_dir / "run_id.txt").write_text("run-1\n", encoding="utf-8")
            (run_dir / "end_ts.txt").write_text(f"{end_ts}\n", encoding="utf-8")
            (run_dir / "main_exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "run_interrupted.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "post_shutdown_processes.txt").write_text("", encoding="utf-8")
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
            (run_dir / "provider_call_budget_events.jsonl").write_text("", encoding="utf-8")
            (run_dir / "latest_verification_queue_results.json").write_text(
                json.dumps([
                    {
                        "track_id": "track-triggered",
                        "request_id": "verifyreq-1",
                        "run_id": "run-1",
                        "evidence_message_id": "trkmsg-1",
                        "validation_outcome": "accepted",
                    },
                    {
                        "track_id": "track-triggered",
                        "request_id": "verifyreq-old",
                        "run_id": "other-run",
                        "evidence_message_id": "trkmsg-old",
                        "validation_outcome": "accepted",
                    }
                ]),
                encoding="utf-8",
            )
            (run_dir / "latest_verification_queue_submissions.json").write_text(
                json.dumps([
                    {
                        "request_id": "verifyreq-old",
                        "run_id": "other-run",
                        "source": "reasoning_cycle",
                        "track_id": "track-triggered",
                        "message_id": "trkmsg-old",
                        "submitted_at": start_ts + 0.15,
                        "accepted": True,
                        "status": "queued",
                        "reason": None,
                    },
                    {
                        "request_id": "verifyreq-1",
                        "run_id": "run-1",
                        "source": "reasoning_cycle",
                        "track_id": "track-triggered",
                        "message_id": "trkmsg-1",
                        "submitted_at": start_ts + 0.25,
                        "accepted": True,
                        "status": "queued",
                        "reason": None,
                    },
                ]),
                encoding="utf-8",
            )

            events = [
                {
                    "run_id": "run-1",
                    "event_type": "reasoning_cycle",
                    "event_origin": "production_monitor",
                    "timestamp": start_ts + 0.1,
                    "track_id": "track-nontrigger",
                    "evidence_requested": False,
                },
                {
                    "run_id": "run-1",
                    "event_type": "reasoning_cycle",
                    "event_origin": "production_monitor",
                    "timestamp": start_ts + 0.2,
                    "track_id": "track-triggered",
                    "evidence_requested": True,
                },
                {
                    "run_id": "run-1",
                    "event_type": "phase16e_trigger",
                    "event_origin": "production_monitor",
                    "timestamp": start_ts + 0.3,
                    "track_id": "track-phase16e",
                    "automatic_evidence_requested": True,
                },
            ]
            with (run_dir / "agent_events.jsonl").open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event) + "\n")

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                report = generate_report(run_dir)
            finally:
                os.chdir(previous_cwd)

            correlated_rows = report.get("correlated_chain_rows", [])
            track_ids = {row.get("track_id") for row in correlated_rows}
            self.assertEqual(track_ids, {"track-triggered", "track-phase16e"})

            triggered_row = next(row for row in correlated_rows if row.get("track_id") == "track-triggered")
            self.assertTrue(triggered_row.get("automatic_evidence_requested"))
            self.assertTrue(triggered_row.get("eligible_for_genuine_chain"))
            self.assertTrue(triggered_row.get("queue_submission"))
            self.assertEqual(triggered_row.get("queue_request_id"), "verifyreq-1")
            self.assertEqual(triggered_row.get("evidence_message_id"), "trkmsg-1")

            fallback_row = next(row for row in correlated_rows if row.get("track_id") == "track-phase16e")
            self.assertFalse(fallback_row.get("eligible_for_genuine_chain"))

    def test_missing_run_id_records_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_report_run_id_missing_") as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            captures_dir = root / "captures"
            run_dir.mkdir(parents=True, exist_ok=True)
            captures_dir.mkdir(parents=True, exist_ok=True)

            start_ts = time.time()
            end_ts = start_ts + 1.0

            (run_dir / "start_ts.txt").write_text(f"{start_ts}\n", encoding="utf-8")
            (run_dir / "end_ts.txt").write_text(f"{end_ts}\n", encoding="utf-8")
            (run_dir / "main_exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "run_interrupted.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "post_shutdown_processes.txt").write_text("", encoding="utf-8")
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
            (run_dir / "provider_call_budget_events.jsonl").write_text("", encoding="utf-8")
            (run_dir / "latest_verification_queue_results.json").write_text(
                json.dumps([
                    {
                        "track_id": "track-1",
                        "request_id": "req-1",
                        "evidence_message_id": "msg-1",
                        "validation_outcome": "accepted",
                    }
                ]),
                encoding="utf-8",
            )
            (run_dir / "latest_verification_queue_submissions.json").write_text(
                json.dumps([
                    {
                        "request_id": "req-1",
                        "track_id": "track-1",
                        "message_id": "msg-1",
                        "submitted_at": start_ts + 0.25,
                        "accepted": True,
                        "status": "queued",
                    }
                ]),
                encoding="utf-8",
            )

            with (run_dir / "agent_events.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event_type": "reasoning_cycle",
                            "timestamp": start_ts + 0.2,
                            "track_id": "track-1",
                            "evidence_requested": True,
                        }
                    )
                    + "\n"
                )

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                report = generate_report(run_dir)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(report.get("verdict"), "FAIL")
            self.assertIn("run_id_missing", report.get("failure_reasons", []))
            self.assertIn("queue_records_missing_run_id:1", report.get("failure_reasons", []))
            self.assertIn("submission_records_missing_run_id:1", report.get("failure_reasons", []))

    def test_mismatched_run_id_records_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_report_run_id_mismatch_") as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            captures_dir = root / "captures"
            run_dir.mkdir(parents=True, exist_ok=True)
            captures_dir.mkdir(parents=True, exist_ok=True)

            start_ts = time.time()
            end_ts = start_ts + 1.0

            (run_dir / "start_ts.txt").write_text(f"{start_ts}\n", encoding="utf-8")
            (run_dir / "run_id.txt").write_text("run-good\n", encoding="utf-8")
            (run_dir / "end_ts.txt").write_text(f"{end_ts}\n", encoding="utf-8")
            (run_dir / "main_exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "run_interrupted.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "post_shutdown_processes.txt").write_text("", encoding="utf-8")
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
            (run_dir / "provider_call_budget_events.jsonl").write_text("", encoding="utf-8")
            (run_dir / "latest_verification_queue_results.json").write_text(
                json.dumps([
                    {
                        "track_id": "track-1",
                        "request_id": "req-1",
                        "run_id": "run-other",
                        "evidence_message_id": "msg-1",
                        "validation_outcome": "accepted",
                    }
                ]),
                encoding="utf-8",
            )
            (run_dir / "latest_verification_queue_submissions.json").write_text(
                json.dumps([
                    {
                        "request_id": "req-1",
                        "run_id": "run-other",
                        "track_id": "track-1",
                        "message_id": "msg-1",
                        "submitted_at": start_ts + 0.25,
                        "accepted": True,
                        "status": "queued",
                    }
                ]),
                encoding="utf-8",
            )

            with (run_dir / "agent_events.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event_type": "reasoning_cycle",
                            "timestamp": start_ts + 0.2,
                            "track_id": "track-1",
                            "evidence_requested": True,
                        }
                    )
                    + "\n"
                )

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                report = generate_report(run_dir)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(report.get("verdict"), "FAIL")
            self.assertIn("queue_records_mismatched_run_id:1", report.get("failure_reasons", []))
            self.assertIn("submission_records_mismatched_run_id:1", report.get("failure_reasons", []))

    def test_missing_run_local_artifact_records_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_report_missing_artifact_") as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            captures_dir = root / "captures"
            run_dir.mkdir(parents=True, exist_ok=True)
            captures_dir.mkdir(parents=True, exist_ok=True)

            start_ts = time.time()
            end_ts = start_ts + 1.0

            (run_dir / "start_ts.txt").write_text(f"{start_ts}\n", encoding="utf-8")
            (run_dir / "run_id.txt").write_text("run-1\n", encoding="utf-8")
            (run_dir / "end_ts.txt").write_text(f"{end_ts}\n", encoding="utf-8")
            (run_dir / "main_exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "run_interrupted.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "post_shutdown_processes.txt").write_text("", encoding="utf-8")
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
            (run_dir / "provider_call_budget_events.jsonl").write_text("", encoding="utf-8")
            (run_dir / "latest_verification_queue_results.json").write_text(
                json.dumps([
                    {
                        "track_id": "track-1",
                        "request_id": "req-1",
                        "run_id": "run-1",
                        "evidence_message_id": "msg-1",
                        "validation_outcome": "accepted",
                    }
                ]),
                encoding="utf-8",
            )
            (run_dir / "latest_verification_queue_submissions.json").write_text(
                json.dumps([
                    {
                        "request_id": "req-1",
                        "run_id": "run-1",
                        "track_id": "track-1",
                        "message_id": "msg-1",
                        "submitted_at": start_ts + 0.25,
                        "accepted": True,
                        "status": "queued",
                    }
                ]),
                encoding="utf-8",
            )
            (run_dir / "run_local_artifact_status.json").write_text(
                json.dumps(
                    {
                        "expected": [
                            "provider_call_budget_state.json",
                            "provider_call_budget_events.jsonl",
                            "latest_verification_queue_results.json",
                            "latest_verification_queue_submissions.json",
                            "operator_tracking_feedback.json",
                        ],
                        "present": [
                            "provider_call_budget_state.json",
                            "provider_call_budget_events.jsonl",
                            "latest_verification_queue_results.json",
                            "latest_verification_queue_submissions.json",
                        ],
                        "missing": ["operator_tracking_feedback.json"],
                    }
                ),
                encoding="utf-8",
            )

            with (run_dir / "agent_events.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event_type": "reasoning_cycle",
                            "timestamp": start_ts + 0.2,
                            "track_id": "track-1",
                            "evidence_requested": True,
                        }
                    )
                    + "\n"
                )

            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                report = generate_report(run_dir)
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(report.get("verdict"), "FAIL")
            self.assertIn("run_local_artifact_missing:operator_tracking_feedback.json", report.get("failure_reasons", []))


if __name__ == "__main__":
    unittest.main()
