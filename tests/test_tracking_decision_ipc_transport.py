from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport


def _decision(
    *,
    investigation_id: str,
    request_id: str,
    tool_name: str = "get_track_evidence_history",
    iteration_index: int = 1,
) -> dict:
    arguments = {"track_id": "track-1"}
    if tool_name == "get_identity_snapshot":
        arguments = {"app_identity_id": "app-1"}
    return {
        "action": (
            "request_identity_snapshot"
            if tool_name == "get_identity_snapshot"
            else "request_additional_tracking_evidence"
        ),
        "investigation_id": investigation_id,
        "iteration_index": iteration_index,
        "tool_requests": [
            {
                "request_id": request_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        ],
    }


def _new_transport(root: Path, **overrides) -> TrackingDecisionIpcTransport:
    return TrackingDecisionIpcTransport(
        requests_path=root / "tracking_decision_tool_requests.jsonl",
        results_path=root / "tracking_decision_tool_results.jsonl",
        request_index_path=root / "latest_tracking_decision_request_index.json",
        result_index_path=root / "latest_tracking_decision_result_index.json",
        **overrides,
    )


class TestTrackingDecisionIpcTransport(unittest.TestCase):
    def test_request_result_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transport = _new_transport(root)

            submitted = transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-1", request_id="req-1"),
                now_timestamp=1000.0,
            )
            self.assertTrue(submitted["accepted"])

            cycle = transport.run_fixture_orchestration_cycle(now_timestamp=1000.1)
            self.assertEqual(cycle["published_count"], 1)

            polled = transport.dashboard_poll_results(now_timestamp=1000.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["request_id"], "req-1")
            self.assertEqual(polled["results"][0]["status"], "ok")

    def test_duplicate_request_handling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(Path(temp_dir))
            decision = _decision(investigation_id="inv-dup", request_id="req-dup")

            first = transport.dashboard_submit_decision_request(decision=decision, now_timestamp=1000.0)
            second = transport.dashboard_submit_decision_request(decision=decision, now_timestamp=1000.1)

            self.assertTrue(first["accepted"])
            self.assertTrue(second["duplicate"])
            self.assertFalse(second["accepted"])

    def test_duplicate_result_handling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(Path(temp_dir))
            transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-rdup", request_id="req-rdup"),
                now_timestamp=1000.0,
            )
            pending = transport.orchestration_poll_pending_requests(now_timestamp=1000.1)
            request = pending["pending_requests"][0]

            first = transport.orchestration_publish_fixture_result(
                request_event=request,
                now_timestamp=1000.2,
            )
            second = transport.orchestration_publish_fixture_result(
                request_event=request,
                now_timestamp=1000.3,
            )

            self.assertTrue(first["published"])
            self.assertTrue(second["duplicate"])

            polled = transport.dashboard_poll_results(now_timestamp=1000.4)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "ok")

    def test_restart_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _new_transport(root)
            first.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-restart", request_id="req-restart"),
                now_timestamp=1000.0,
            )
            first.run_fixture_orchestration_cycle(now_timestamp=1000.1)

            restarted = _new_transport(root)
            polled = restarted.dashboard_poll_results(now_timestamp=1000.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["request_id"], "req-restart")

    def test_expired_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(Path(temp_dir), request_ttl_seconds=1.0)
            transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-exp", request_id="req-exp"),
                now_timestamp=1000.0,
            )

            pending = transport.orchestration_poll_pending_requests(now_timestamp=1002.0)
            self.assertEqual(pending["pending_count"], 0)
            self.assertEqual(pending["expired_count"], 1)

            polled = transport.dashboard_poll_results(now_timestamp=1002.1)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "expired")

    def test_delayed_results_for_terminal_investigation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(Path(temp_dir))
            transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-term", request_id="req-term"),
                now_timestamp=1000.0,
            )
            transport.mark_terminal_investigation(
                investigation_id="inv-term",
                state="closed",
                reason="investigation_closed",
                now_timestamp=1000.1,
            )

            pending = transport.orchestration_poll_pending_requests(now_timestamp=1000.2)
            self.assertEqual(pending["pending_count"], 0)
            self.assertEqual(pending["terminal_rejections"], 1)

            polled = transport.dashboard_poll_results(now_timestamp=1000.3)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "terminal")

    def test_tool_budget_exhaustion_blocks_new_requests_but_keeps_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(Path(temp_dir))
            first_submit = transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-budget", request_id="req-1"),
                now_timestamp=1000.0,
            )
            self.assertTrue(first_submit["accepted"])

            pending = transport.orchestration_poll_pending_requests(now_timestamp=1000.1)
            request = pending["pending_requests"][0]

            mark = transport.mark_tool_budget_exhausted_investigation(
                investigation_id="inv-budget",
                reason="per_investigation_call_limit",
                now_timestamp=1000.15,
            )
            self.assertTrue(mark["updated"])

            published = transport.orchestration_publish_fixture_result(
                request_event=request,
                status="ok",
                result_payload={"available": True},
                now_timestamp=1000.2,
            )
            self.assertTrue(published["published"])

            second_submit = transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-budget", request_id="req-2"),
                now_timestamp=1000.3,
            )
            self.assertTrue(second_submit["rejected"])
            self.assertEqual(second_submit["reason"], "tool_budget_exhausted")

            polled = transport.dashboard_poll_results(now_timestamp=1000.4)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "ok")

    def test_mismatched_investigation_request_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(Path(temp_dir))
            transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-match", request_id="req-match"),
                now_timestamp=1000.0,
            )

            published = transport.orchestration_publish_result_event(
                result_event={
                    "schema_version": "1.0",
                    "event_type": "decision_tool_result",
                    "request_id": "req-other",
                    "investigation_id": "inv-other",
                    "iteration_index": 1,
                    "tool_name": "get_track_evidence_history",
                    "status": "ok",
                    "result_payload": {"fixture": True},
                    "error_code": None,
                    "error": None,
                    "started_at": 1000.1,
                    "finished_at": 1000.1,
                    "idempotency_key": "inv-other:req-other:get_track_evidence_history",
                },
                now_timestamp=1000.1,
            )
            self.assertTrue(published["published"])

            polled = transport.dashboard_poll_results(now_timestamp=1000.2)
            self.assertEqual(polled["result_count"], 0)

            lines = transport.results_path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[-1])
            self.assertEqual(payload["status"], "rejected")
            self.assertEqual(payload["error_code"], "correlation_mismatch")

    def test_action_tool_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(Path(temp_dir))
            submitted = transport.dashboard_submit_decision_request(
                decision={
                    "action": "request_additional_tracking_evidence",
                    "investigation_id": "inv-obs-like",
                    "iteration_index": 1,
                    "tool_requests": [
                        {
                            "request_id": "req-obs-like",
                            "tool_name": "get_identity_snapshot",
                            "arguments": {"app_identity_id": "app-1"},
                        }
                    ],
                },
                now_timestamp=1000.0,
            )

            self.assertTrue(submitted["rejected"])
            self.assertEqual(submitted["reason"], "invalid_decision_request")
            errors = submitted.get("errors")
            self.assertIsInstance(errors, list)
            self.assertIn("decision_tool_name_mismatch_for_action", errors)

    def test_bounded_retention(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _new_transport(
                Path(temp_dir),
                max_request_records=2,
                max_result_records=2,
            )

            for index in range(3):
                transport.dashboard_submit_decision_request(
                    decision=_decision(investigation_id=f"inv-{index}", request_id=f"req-{index}"),
                    now_timestamp=1000.0 + index,
                )
                transport.run_fixture_orchestration_cycle(now_timestamp=1000.1 + index)

            request_lines = transport.requests_path.read_text(encoding="utf-8").splitlines()
            result_lines = transport.results_path.read_text(encoding="utf-8").splitlines()
            self.assertLessEqual(len(request_lines), 2)
            self.assertLessEqual(len(result_lines), 2)

    def test_partial_or_interrupted_writes_are_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transport = _new_transport(root)
            transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-partial", request_id="req-partial"),
                now_timestamp=1000.0,
            )

            with transport.requests_path.open("a", encoding="utf-8") as handle:
                handle.write('{"bad_json":')

            with transport.results_path.open("a", encoding="utf-8") as handle:
                handle.write('not-json\n')

            restarted = _new_transport(root)
            cycle = restarted.run_fixture_orchestration_cycle(now_timestamp=1000.1)
            self.assertGreaterEqual(cycle["published_count"], 1)

            polled = restarted.dashboard_poll_results(now_timestamp=1000.2)
            self.assertEqual(polled["result_count"], 1)


if __name__ == "__main__":
    unittest.main()
