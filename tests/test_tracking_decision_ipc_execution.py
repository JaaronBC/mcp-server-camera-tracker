from __future__ import annotations

import asyncio
import multiprocessing as mp
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import main
from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport


def _decision(
    *,
    investigation_id: str,
    request_id: str,
    tool_name: str = "get_track_evidence_history",
    iteration_index: int = 1,
    arguments: dict | None = None,
) -> dict:
    if arguments is None:
        if tool_name == "get_identity_snapshot":
            arguments = {"app_identity_id": "app-1"}
        else:
            arguments = {"track_id": "track-1"}
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


def _transport(root: Path, *, role: str) -> TrackingDecisionIpcTransport:
    return TrackingDecisionIpcTransport(
        requests_path=root / "tracking_decision_tool_requests.jsonl",
        results_path=root / "tracking_decision_tool_results.jsonl",
        request_index_path=root / "latest_tracking_decision_request_index.json",
        result_index_path=root / "latest_tracking_decision_result_index.json",
        writer_role=role,
    )


def _dashboard_submit_worker(root_path: str) -> None:
    root = Path(root_path)
    transport = _transport(root, role="dashboard")
    transport.dashboard_submit_decision_request(
        decision=_decision(investigation_id="inv-proc-1", request_id="req-proc-1"),
        now_timestamp=1000.0,
    )


def _orchestration_fixture_worker(root_path: str) -> None:
    root = Path(root_path)
    transport = _transport(root, role="orchestration")
    transport.run_fixture_orchestration_cycle(now_timestamp=1000.1, max_requests=4)


class _FakeSession:
    def __init__(self, *, payload: dict | None = None, sleep_seconds: float = 0.0) -> None:
        self._payload = payload if payload is not None else {"available": True}
        self._sleep_seconds = max(0.0, float(sleep_seconds))
        self.call_count = 0
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name: str, arguments: dict) -> SimpleNamespace:
        self.call_count += 1
        self.calls.append((tool_name, dict(arguments)))
        if self._sleep_seconds > 0:
            await asyncio.sleep(self._sleep_seconds)
        item = SimpleNamespace(json=dict(self._payload))
        return SimpleNamespace(content=[item])


class TestTrackingDecisionIpcProcessIntegration(unittest.TestCase):
    def test_two_process_fixture_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            context = mp.get_context("spawn")

            dashboard_process = context.Process(target=_dashboard_submit_worker, args=(str(root),))
            dashboard_process.start()
            dashboard_process.join(timeout=10)
            self.assertEqual(dashboard_process.exitcode, 0)

            orchestration_process = context.Process(target=_orchestration_fixture_worker, args=(str(root),))
            orchestration_process.start()
            orchestration_process.join(timeout=10)
            self.assertEqual(orchestration_process.exitcode, 0)

            dashboard_transport = _transport(root, role="dashboard")
            polled = dashboard_transport.dashboard_poll_results(now_timestamp=1000.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["request_id"], "req-proc-1")
            self.assertEqual(polled["results"][0]["status"], "ok")


class TestTrackingDecisionIpcWriterOwnership(unittest.TestCase):
    def test_dashboard_role_cannot_publish_orchestration_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _transport(Path(temp_dir), role="dashboard")
            response = transport.orchestration_publish_fixture_result(
                request_event={
                    "request_id": "req-x",
                    "investigation_id": "inv-x",
                    "tool_name": "get_track_evidence_history",
                    "iteration_index": 1,
                }
            )
            self.assertTrue(response["rejected"])
            self.assertEqual(response["reason"], "writer_role_violation")

    def test_orchestration_role_cannot_submit_dashboard_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            transport = _transport(Path(temp_dir), role="orchestration")
            response = transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-x", request_id="req-x"),
                now_timestamp=1000.0,
            )
            self.assertTrue(response["rejected"])
            self.assertEqual(response["reason"], "writer_role_violation")


class TestTrackingDecisionIpcBoundedMcpExecution(unittest.IsolatedAsyncioTestCase):
    async def test_enforces_one_mcp_call_per_investigation_per_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True, "state": "recorded_observation"})
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-1", request_id="req-1"),
                now_timestamp=now,
            )
            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-1", request_id="req-2"),
                now_timestamp=now + 0.01,
            )

            summary = await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            self.assertEqual(session.call_count, 1)
            self.assertEqual(summary["pending_count"], 2)

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 1.0)
            self.assertEqual(polled["result_count"], 2)
            statuses = sorted(result["status"] for result in polled["results"])
            self.assertEqual(statuses, ["ok", "rejected"])

    async def test_rejects_invalid_tool_arguments_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True})
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(
                    investigation_id="inv-invalid",
                    request_id="req-invalid",
                    tool_name="get_track_evidence_history",
                    arguments={},
                ),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            self.assertEqual(session.call_count, 0)
            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.1)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "rejected")
            self.assertEqual(polled["results"][0]["error_code"], "invalid_tool_arguments")

    async def test_timeout_is_persisted_without_blocking_future_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            slow_session = _FakeSession(payload={"available": True}, sleep_seconds=0.05)
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-timeout", request_id="req-timeout"),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=slow_session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.001,
            )

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "timeout")

    async def test_enforces_one_mcp_call_per_investigation_across_cycles_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True, "state": "recorded_observation"})
            now = time.time()

            first_submit = dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-once", request_id="req-1"),
                now_timestamp=now,
            )
            self.assertTrue(first_submit["accepted"])

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )
            self.assertEqual(session.call_count, 1)

            second_submit = dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-once", request_id="req-2"),
                now_timestamp=now + 0.1,
            )
            self.assertTrue(second_submit["rejected"])
            self.assertEqual(second_submit["reason"], "tool_budget_exhausted")

            restarted_orchestration_transport = _transport(root, role="orchestration")
            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=restarted_orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )
            self.assertEqual(session.call_count, 1)

    async def test_lookup_results_for_request_does_not_consume_dashboard_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dashboard_transport = _transport(root, role="dashboard")
            orchestration_transport = _transport(root, role="orchestration")
            session = _FakeSession(payload={"available": True, "state": "recorded_observation"})
            now = time.time()

            dashboard_transport.dashboard_submit_decision_request(
                decision=_decision(investigation_id="inv-lookup", request_id="req-lookup"),
                now_timestamp=now,
            )

            await main._run_tracking_decision_ipc_mcp_cycle(
                session=session,
                transport=orchestration_transport,
                max_requests_per_cycle=8,
                tool_timeout_seconds=0.5,
            )

            lookup = dashboard_transport.lookup_results_for_request(
                investigation_id="inv-lookup",
                request_id="req-lookup",
                tool_name="get_track_evidence_history",
                now_timestamp=now + 0.1,
            )
            self.assertTrue(lookup["available"])
            self.assertEqual(lookup["result_count"], 1)
            self.assertIsInstance(lookup.get("latest"), dict)

            polled = dashboard_transport.dashboard_poll_results(now_timestamp=now + 0.2)
            self.assertEqual(polled["result_count"], 1)
            self.assertEqual(polled["results"][0]["status"], "ok")


class TestDecisionIpcNonBlockingLoop(unittest.IsolatedAsyncioTestCase):
    async def test_monitor_movement_does_not_block_on_slow_decision_cycle(self) -> None:
        tracking_calls = 0

        async def run_tracking_cycle(_session, _address) -> None:
            nonlocal tracking_calls
            tracking_calls += 1
            if tracking_calls >= 3:
                raise KeyboardInterrupt()

        async def run_agent_ops_pipeline_fn(*_args, **_kwargs) -> dict:
            return {
                "agent_ops_enabled": False,
                "agent_ops_summary_enabled": False,
                "depth_summary_enabled": False,
                "completed_cycle_increment": 0,
            }

        async def slow_decision_cycle() -> dict[str, int]:
            await asyncio.sleep(0.2)
            return {"published_count": 0}

        started_at = time.perf_counter()
        with self.assertRaises(KeyboardInterrupt):
            await runtime_flow.monitor_movement(
                session=SimpleNamespace(),
                tracking_address=("127.0.0.1", 5005),
                analysis_interval_seconds=0.0,
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
                run_tracking_cycle_fn=run_tracking_cycle,
                run_agent_ops_pipeline_fn=run_agent_ops_pipeline_fn,
                emit=lambda _message: None,
                decision_ipc_fixture_cycle_fn=slow_decision_cycle,
            )
        elapsed = time.perf_counter() - started_at

        self.assertGreaterEqual(tracking_calls, 3)
        self.assertLess(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
