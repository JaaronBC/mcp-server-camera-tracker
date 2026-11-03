from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import main as runtime_main
from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.verification_feedback import VerificationFeedbackPublisher
from agents.orchestration.runtime.verification_queue import VerificationQueueManager
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer
from tools.phase16e_report import generate_report


class _FakeItem:
    def __init__(self, payload: dict[str, object]) -> None:
        self.json = payload


class _FakeResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self.content = [_FakeItem(payload)]


class _FakeSession:
    def __init__(self, handler) -> None:
        self._handler = handler

    async def call_tool(self, tool_name, arguments):
        return await self._handler(tool_name, arguments)


def _tracking_envelope_payload(
    *,
    message_id: str,
    track_id: str,
    timestamp: float,
    source_evidence_refs: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "message_id": message_id,
        "track_id": track_id,
        "timestamp": timestamp,
        "tracking_claims": {
            "claimed_resolved_track_id": track_id,
            "association_status": "accepted",
            "possible_id_switch": False,
            "visibility_status": "visible",
            "claim_reasons": ["camera_tracking_fixture"],
        },
        "frame_id": "frame-001",
        "detection_id": "det-001",
        "source_evidence_refs": list(source_evidence_refs),
        "uncertainty": [],
        "quality_flags": ["position_kind:calibrated_ground_plane"],
        "track_history": [],
    }


def _write_validated_tracking_envelope(
    root: Path,
    *,
    message_id: str,
    track_id: str,
    timestamp: float,
    source_evidence_refs: list[str],
) -> None:
    envelopes = root / "envelopes"
    envelopes.mkdir(parents=True, exist_ok=True)
    payload = {
        "message_id": message_id,
        "stored_at": timestamp,
        "validation": {"valid": True, "errors": [], "warnings": []},
        "envelope": _tracking_envelope_payload(
            message_id=message_id,
            track_id=track_id,
            timestamp=timestamp,
            source_evidence_refs=source_evidence_refs,
        ),
    }
    (envelopes / f"{message_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _verification_envelope_payload(
    *,
    message_id: str,
    track_id: str,
    timestamp: float,
    source_evidence_refs: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "observation_id": f"verify-{message_id}",
        "referenced_track_id": track_id,
        "referenced_message_id": message_id,
        "timestamp": timestamp,
        "source_evidence_refs": list(source_evidence_refs),
        "verification_status": "supported",
        "visibility_status": "visible",
        "identity_consistency": "consistent",
        "possible_id_switch": False,
        "supporting_evidence_refs": list(source_evidence_refs),
        "contradicting_evidence_refs": [],
        "uncertainty": [],
        "limitations": [],
        "confidence": 0.91,
        "visual_content_inspected": True,
    }


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
        (run_dir / "operator_tracking_feedback.json").write_text("[]", encoding="utf-8")
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
                        "event_origin": "production_monitor",
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
                        "event_origin": "production_monitor",
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

    def test_startup_only_trigger_yields_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_integrity_startup_only_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            start_ts = self._write_base(run_dir)

            (run_dir / "latest_verification_queue_results.json").write_text("[]", encoding="utf-8")
            (run_dir / "latest_verification_queue_submissions.json").write_text("[]", encoding="utf-8")
            (run_dir / "agent_events.jsonl").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "event_type": "reasoning_cycle",
                        "event_origin": "startup_activation",
                        "timestamp": start_ts + 0.2,
                        "track_id": "sim-1",
                        "evidence_requested": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = generate_report(run_dir)
            self.assertEqual(report["verdict"], "INCONCLUSIVE")
            self.assertTrue(report["checks"]["trigger_eligibility"]["startup_only_observed"])

    def test_missing_event_origin_does_not_pass_chain(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_integrity_missing_origin_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            start_ts = self._write_base(run_dir)

            (run_dir / "latest_verification_queue_results.json").write_text("[]", encoding="utf-8")
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
            self.assertEqual(report["verdict"], "INCONCLUSIVE")
            self.assertIsNone(report["correlated_chain_rows"][0]["event_origin"])
            self.assertFalse(report["correlated_chain_rows"][0]["eligible_for_genuine_chain"])

    def test_decision_log_skip_reason_is_reported_for_production_trigger(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_integrity_decision_skip_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            start_ts = self._write_base(run_dir)

            (run_dir / "latest_verification_queue_results.json").write_text("[]", encoding="utf-8")
            (run_dir / "latest_verification_queue_submissions.json").write_text("[]", encoding="utf-8")
            event = {
                "run_id": "run-1",
                "event_type": "reasoning_cycle",
                "event_origin": "production_monitor",
                "task_id": "cycle-1:track-1",
                "timestamp": start_ts + 0.2,
                "track_id": "track-1",
                "evidence_requested": True,
            }
            (run_dir / "agent_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
            decision = {
                "run_id": "run-1",
                "event_origin": "production_monitor",
                "trigger_event_id": "cycle-1:track-1",
                "track_id": "track-1",
                "evidence_requested": True,
                "evidence_message_id": None,
                "queue_request_id": None,
                "submission_eligibility": True,
                "skip_reason": "submit_rejected:queue_saturated",
                "submission_attempted": True,
                "submission_accepted": False,
            }
            (run_dir / "verification_submission_decisions.jsonl").write_text(
                json.dumps(decision) + "\n",
                encoding="utf-8",
            )

            report = generate_report(run_dir)
            self.assertEqual(report["verdict"], "FAIL")
            self.assertIn("queue_submission_missing:track-1", report["failure_reasons"])
            row = report["correlated_chain_rows"][0]
            self.assertEqual(row["submission_reason"], "submit_rejected:queue_saturated")


class TestPhase16EProductionMonitorIntegration(unittest.IsolatedAsyncioTestCase):
    def _write_phase16e_base(self, run_dir: Path, *, start_ts: float, end_ts: float, run_id: str) -> None:
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
        runtime_main._persist_json_artifact(run_dir / "latest_verification_queue_results.json", [])
        runtime_main._persist_json_artifact(run_dir / "latest_verification_queue_submissions.json", [])

    async def _run_provider_free_scenario(
        self,
        *,
        event_origin: str,
        track_id: str,
        message_id: str,
        evidence_source_refs: list[str],
        verification_source_refs: list[str],
        tamper_result_request_id: bool = False,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="phase16e_prod_path_") as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            evidence_root = root / "tracking_evidence"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_id = "run-prod-1"
            start_ts = time.time()
            self._write_phase16e_base(run_dir, start_ts=start_ts, end_ts=start_ts + 5.0, run_id=run_id)

            _write_validated_tracking_envelope(
                evidence_root,
                message_id=message_id,
                track_id=track_id,
                timestamp=start_ts + 0.05,
                source_evidence_refs=evidence_source_refs,
            )

            feedback_path = run_dir / "operator_tracking_feedback.json"
            decisions_path = run_dir / "verification_submission_decisions.jsonl"

            publisher = VerificationFeedbackPublisher(
                artifact_path=feedback_path,
                max_records=256,
                record_ttl_seconds=45.0,
            )
            consumer = TrackingObservationFeedbackConsumer(artifact_path=feedback_path)
            consumer_accepted_request_ids: set[str] = set()

            emitted_once = False
            tracking_calls = 0
            scenario_event = {
                "run_id": run_id,
                "event_type": "reasoning_cycle",
                "event_origin": event_origin,
                "task_id": "cycle-fixed:track-1",
                "timestamp": start_ts + 0.2,
                "track_id": track_id,
                "evidence_requested": True,
                "evidence_message_id": message_id,
            }

            async def tool_handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": track_id})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope_payload(
                                message_id=str(arguments["message_id"]),
                                track_id=track_id,
                                timestamp=start_ts + 0.35,
                                source_evidence_refs=verification_source_refs,
                            ),
                        }
                    )
                raise RuntimeError(f"unexpected tool: {tool_name}")

            async def run_tracking_cycle(_session, _address):
                nonlocal tracking_calls
                tracking_calls += 1
                if tracking_calls >= 6:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            async def run_agent_ops_pipeline_fn(_session, **_kwargs):
                nonlocal emitted_once
                cycle_events = []
                if not emitted_once:
                    cycle_events = [dict(scenario_event)]
                    emitted_once = True
                return {
                    "agent_ops_enabled": True,
                    "agent_ops_summary_enabled": False,
                    "depth_summary_enabled": False,
                    "completed_cycle_increment": 1,
                    "cycle_payload": {"events": cycle_events},
                }

            def _published_request_ids_from_artifact() -> set[str]:
                if not feedback_path.is_file():
                    return set()
                payload = json.loads(feedback_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    return set()
                records = payload.get("records")
                if not isinstance(records, list):
                    return set()
                output: set[str] = set()
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    request_id = record.get("verification_request_id")
                    if isinstance(request_id, str) and request_id.strip():
                        output.add(request_id.strip())
                return output

            def persist_results(results: list[dict[str, object]]) -> None:
                publisher.publish_from_results(results, now_timestamp=time.time())
                poll_result = consumer.poll(
                    current_track_ids={track_id},
                    temporarily_lost_track_ids=set(),
                    latest_message_id_by_track={track_id: message_id},
                    now_timestamp=time.time(),
                )
                accepted_payload = poll_result.get("accepted", [])
                if isinstance(accepted_payload, list):
                    for item in accepted_payload:
                        if not isinstance(item, dict):
                            continue
                        request_id = item.get("verification_request_id")
                        if isinstance(request_id, str) and request_id.strip():
                            consumer_accepted_request_ids.add(request_id.strip())

                published_request_ids = _published_request_ids_from_artifact()
                enriched_results: list[dict[str, object]] = []
                for item in results:
                    row = dict(item)
                    request_id = row.get("request_id")
                    normalized_request_id = request_id.strip() if isinstance(request_id, str) else ""
                    row["feedback_published"] = normalized_request_id in published_request_ids
                    row["tracking_feedback_consumer_acceptance"] = normalized_request_id in consumer_accepted_request_ids
                    enriched_results.append(row)

                runtime_main._persist_json_artifact(run_dir / "latest_verification_queue_results.json", enriched_results)

            def persist_submissions(submissions: list[dict[str, object]]) -> None:
                runtime_main._persist_json_artifact(run_dir / "latest_verification_queue_submissions.json", submissions)

            env_patch = {
                "PHASE16E_RUN_ID": run_id,
                "PHASE16E_VERIFICATION_SUBMISSION_DECISIONS_PATH": str(decisions_path),
            }

            manager = VerificationQueueManager(
                evidence_root_dir=evidence_root,
                run_id=run_id,
                request_timeout_seconds=2.0,
            )
            session = _FakeSession(tool_handler)

            with patch.dict(os.environ, env_patch, clear=False):
                await manager.start(session=session)
                try:
                    with self.assertRaises(KeyboardInterrupt):
                        await runtime_flow.monitor_movement(
                            session,
                            tracking_address=("127.0.0.1", 5005),
                            analysis_interval_seconds=0,
                            enable_agent_ops_heartbeat=True,
                            agent_ops_interval_seconds=0,
                            agent_ops_max_tracks=5,
                            agent_ops_summary_every_cycles=999,
                            agent_ops_summary_window_size=10,
                            handoff_degraded_warn_rate=0.1,
                            handoff_degraded_critical_rate=0.3,
                            depth_coverage_summary_enabled=False,
                            depth_coverage_summary_every_cycles=999,
                            depth_coverage_sample_limit=10,
                            run_tracking_cycle_fn=run_tracking_cycle,
                            run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
                            emit=lambda _msg: None,
                            verification_manager=manager,
                            verification_results_persist_fn=persist_results,
                            verification_submissions_persist_fn=persist_submissions,
                            verification_feedback_publish_fn=publisher.publish_from_results,
                        )

                    await asyncio.wait_for(manager._queue.join(), timeout=2.0)
                    persist_results(manager.get_recent_results(200))
                    persist_submissions(manager.get_recent_submissions(200))
                finally:
                    await manager.stop()

            (run_dir / "agent_events.jsonl").write_text(json.dumps(scenario_event) + "\n", encoding="utf-8")
            (run_dir / "end_ts.txt").write_text(f"{time.time()}\n", encoding="utf-8")

            if tamper_result_request_id:
                results_payload = json.loads((run_dir / "latest_verification_queue_results.json").read_text(encoding="utf-8"))
                if isinstance(results_payload, list) and results_payload:
                    if isinstance(results_payload[0], dict):
                        results_payload[0]["request_id"] = "req-mismatch"
                    runtime_main._persist_json_artifact(run_dir / "latest_verification_queue_results.json", results_payload)

            report = generate_report(run_dir)
            queue_results = json.loads((run_dir / "latest_verification_queue_results.json").read_text(encoding="utf-8"))
            submissions = json.loads((run_dir / "latest_verification_queue_submissions.json").read_text(encoding="utf-8"))
            decision_lines = []
            if decisions_path.is_file():
                decision_lines = [
                    json.loads(line)
                    for line in decisions_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

            return {
                "report": report,
                "queue_results": queue_results,
                "submissions": submissions,
                "decision_lines": decision_lines,
            }

    async def test_production_monitor_path_provider_free_passes_phase16e(self) -> None:
        scenario = await self._run_provider_free_scenario(
            event_origin="production_monitor",
            track_id="track-live-1",
            message_id="msg-live-1",
            evidence_source_refs=["evidence://camera/frame-001"],
            verification_source_refs=["evidence://camera/frame-001"],
        )

        report = scenario["report"]
        queue_results = scenario["queue_results"]
        submissions = scenario["submissions"]
        decision_lines = scenario["decision_lines"]

        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(len(queue_results), 1)
        self.assertEqual(len(submissions), 1)
        self.assertTrue(any(bool(item.get("submission_accepted", False)) for item in decision_lines))

        row = report["correlated_chain_rows"][0]
        self.assertEqual(row["event_origin"], "production_monitor")
        self.assertTrue(row["eligible_for_genuine_chain"])
        self.assertTrue(row["queue_submission"])
        self.assertTrue(row["verification_success"])
        self.assertTrue(row["coordinator_acceptance"])
        self.assertTrue(row["feedback_publication"])
        self.assertTrue(row["tracking_feedback_consumer_acceptance"])
        self.assertEqual(report["checks"]["trigger_eligibility"]["eligible_row_count"], 1)

    async def test_simulated_track_does_not_produce_pass(self) -> None:
        scenario = await self._run_provider_free_scenario(
            event_origin="production_monitor",
            track_id="sim-123",
            message_id="msg-sim-1",
            evidence_source_refs=["evidence://camera/frame-sim-1"],
            verification_source_refs=["evidence://camera/frame-sim-1"],
        )
        report = scenario["report"]
        self.assertNotEqual(report["verdict"], "PASS")
        self.assertEqual(report["verdict"], "INCONCLUSIVE")

    async def test_startup_activation_origin_does_not_produce_pass(self) -> None:
        scenario = await self._run_provider_free_scenario(
            event_origin="startup_activation",
            track_id="track-startup-1",
            message_id="msg-startup-1",
            evidence_source_refs=["evidence://camera/frame-start-1"],
            verification_source_refs=["evidence://camera/frame-start-1"],
        )
        report = scenario["report"]
        self.assertNotEqual(report["verdict"], "PASS")
        self.assertEqual(report["verdict"], "INCONCLUSIVE")

    async def test_missing_provenance_source_refs_does_not_produce_pass(self) -> None:
        scenario = await self._run_provider_free_scenario(
            event_origin="production_monitor",
            track_id="track-noprovenance-1",
            message_id="msg-noprovenance-1",
            evidence_source_refs=["evidence://camera/frame-prov-1"],
            verification_source_refs=[],
        )

        report = scenario["report"]
        queue_results = scenario["queue_results"]
        self.assertNotEqual(report["verdict"], "PASS")
        self.assertEqual(report["verdict"], "FAIL")
        self.assertTrue(any(item.get("validation_outcome") == "coordinator_validation_failed" for item in queue_results))

    async def test_mismatched_request_id_cannot_produce_pass(self) -> None:
        scenario = await self._run_provider_free_scenario(
            event_origin="production_monitor",
            track_id="track-mismatch-1",
            message_id="msg-mismatch-1",
            evidence_source_refs=["evidence://camera/frame-mm-1"],
            verification_source_refs=["evidence://camera/frame-mm-1"],
            tamper_result_request_id=True,
        )

        report = scenario["report"]
        row = report["correlated_chain_rows"][0]
        self.assertNotEqual(report["verdict"], "PASS")
        self.assertEqual(report["verdict"], "FAIL")
        self.assertFalse(row["verification_success"])
        self.assertIn("feedback_publication_missing:track-mismatch-1", report["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
