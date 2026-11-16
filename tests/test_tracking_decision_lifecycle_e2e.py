from __future__ import annotations

import json
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import main
from agents.contracts import TrackingAgentObservation, VisualAssociationUpdate
from agents.dashboard import tracking_loop
from agents.dashboard.state import SharedDashboardState
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport
from agents.tracking.agent_tracking import (
    _normalize_compact_response_payload,
    _validate_tracking_decision_payload,
)
from agents.tracking.tracking_types import GroundPosition, RadarTrackSnapshot


class _PollingRuntime:
    def __init__(self, observation: TrackingAgentObservation | None = None) -> None:
        self._observation = observation

    def poll_latest_observation(self):
        observation = self._observation
        self._observation = None
        return observation


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _track_snapshot() -> RadarTrackSnapshot:
    return RadarTrackSnapshot(
        track_id="det-1",
        timestamp=1000.0,
        pixel_position=None,
        ground_position=GroundPosition(x=1.0, z=2.0),
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        quality_flags=[],
        confidence=0.9,
    )


def _decision_payload(*, investigation_id: str, request_id: str) -> dict:
    return {
        "action": "request_identity_snapshot",
        "investigation_id": investigation_id,
        "correlation_id": "corr-e2e-1",
        "evidence_references": {
            "frame_reference": "evidence://frame/f-1",
            "detection_reference": "evidence://detection/d-1",
            "tracking_message_id": "trkmsg-1",
        },
        "snapshot_constraints": {
            "min_source_generation": 1,
            "min_source_version": 1,
        },
        "tool_requests": [
            {
                "request_id": request_id,
                "tool_name": "get_identity_snapshot",
                "arguments": {
                    "app_identity_id": "app-1",
                    "min_source_generation": 1,
                    "min_source_version": 1,
                    "investigation_id": investigation_id,
                    "correlation_id": "corr-e2e-1",
                },
            }
        ],
    }


def _semantic_decision_payload(*, investigation_id: str) -> dict:
    return {
        "action": "request_identity_snapshot",
        "investigation_id": investigation_id,
        "reason": "need_identity_snapshot",
        "app_identity_id": "app-1",
        "evidence_bounds": {
            "include_history": True,
            "max_candidates": 4,
            "min_source_generation": 1,
            "min_source_version": 1,
        },
    }


def _follow_up_observation(*, investigation_id: str, request_id: str, result_key: str, inconclusive: bool) -> TrackingAgentObservation:
    updates = [
        VisualAssociationUpdate(
            detection_id="det-1",
            resolved_track_id="track-merged",
            accepted=True,
            rejection_reasons=[],
            uncertainties=([] if not inconclusive else ["no_vlm_proposal_used"]),
            supporting_evidence=["identity_snapshot"],
        )
    ]
    return TrackingAgentObservation(
        timestamp=time.time(),
        source_agent="visual_tracking",
        vlm_model="qwen-3.8-27b",
        proposal_latency_ms=12.0,
        frame_reference=None,
        updates=updates,
        uncertainties=[
            "schema_complete_response",
            f"followup_investigation_id:{investigation_id}",
            f"followup_request_id:{request_id}",
            f"followup_result_key:{result_key}",
            "followup_result_status:ok",
        ],
        observation_id="obs-e2e-followup",
        llm_correlation_id="corr-followup-e2e",
        decision=(
            {
                "action": "inconclusive",
                "investigation_id": investigation_id,
                "reason": "insufficient_visual_evidence",
            }
            if inconclusive
            else None
        ),
    )


class TestTrackingDecisionLifecycleE2E(unittest.IsolatedAsyncioTestCase):
    async def test_e2e_semantic_identity_snapshot_decision_round_trip(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        captures_dir = project_root / "captures"
        snapshot_path = captures_dir / "latest_tracking_identity_snapshot.json"

        original_snapshot = snapshot_path.read_text(encoding="utf-8") if snapshot_path.is_file() else None
        captures_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": time.time(),
                    "freshness_window_seconds": 300.0,
                    "source_generation": 10,
                    "source_version": 20,
                    "identity_count": 1,
                    "identities": [
                        {
                            "app_identity_id": "app-1",
                            "status": "visible",
                            "provisional": False,
                            "generation": 10,
                            "identity_version": 2,
                            "updated_at": time.time(),
                            "detection_reference": "evidence://detection/d-1",
                            "observation_history": [],
                            "historical_yolo_track_ids": ["track-1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        from tempfile import TemporaryDirectory

        try:
            with TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                dashboard_transport = TrackingDecisionIpcTransport(
                    requests_path=temp_root / "tracking_decision_tool_requests.jsonl",
                    results_path=temp_root / "tracking_decision_tool_results.jsonl",
                    request_index_path=temp_root / "latest_tracking_decision_request_index.json",
                    result_index_path=temp_root / "latest_tracking_decision_result_index.json",
                    writer_role="dashboard",
                )
                orchestration_transport = TrackingDecisionIpcTransport(
                    requests_path=temp_root / "tracking_decision_tool_requests.jsonl",
                    results_path=temp_root / "tracking_decision_tool_results.jsonl",
                    request_index_path=temp_root / "latest_tracking_decision_request_index.json",
                    result_index_path=temp_root / "latest_tracking_decision_result_index.json",
                    writer_role="orchestration",
                )

                semantic_payload = _semantic_decision_payload(investigation_id="inv-semantic-e2e-1")
                normalized = _normalize_compact_response_payload(
                    dict(semantic_payload),
                    request_correlation_id="corr-semantic-e2e-1",
                    follow_up_context=None,
                    detections=[],
                )
                decision_payload, decision_errors = _validate_tracking_decision_payload(
                    normalized.get("decision") if isinstance(normalized, dict) else None
                )
                self.assertEqual(decision_errors, [])
                self.assertIsInstance(decision_payload, dict)
                assert isinstance(decision_payload, dict)

                tool_requests = decision_payload.get("tool_requests")
                self.assertIsInstance(tool_requests, list)
                assert isinstance(tool_requests, list)
                self.assertEqual(len(tool_requests), 1)
                tool_request = tool_requests[0]
                self.assertIsInstance(tool_request, dict)
                assert isinstance(tool_request, dict)
                self.assertEqual(tool_request.get("tool_name"), "get_identity_snapshot")
                self.assertIsInstance(tool_request.get("arguments"), dict)
                self.assertEqual(tool_request.get("arguments", {}).get("app_identity_id"), "app-1")

                state = SharedDashboardState()
                state.tracking_decision_ipc_enabled = True
                state.tracking_decision_ipc_transport = dashboard_transport
                state.lock = _Lock()

                initial_observation = types.SimpleNamespace(
                    decision=decision_payload,
                    llm_correlation_id="corr-semantic-e2e-1",
                    uncertainties=[],
                )

                tracking_loop._submit_tracking_decision_request_if_enabled(
                    state=state,
                    tracking_agent_observation=initial_observation,
                    now=time.time(),
                )

                request_events = [
                    json.loads(line)
                    for line in (temp_root / "tracking_decision_tool_requests.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(request_events), 1)
                self.assertEqual(request_events[0].get("tool_name"), "get_identity_snapshot")
                self.assertEqual(
                    request_events[0].get("arguments", {}).get("app_identity_id"),
                    "app-1",
                )

                snapshot_before_cycle = snapshot_path.read_text(encoding="utf-8")

                server_path = project_root / "server.py"
                params = StdioServerParameters(command=sys.executable, args=[str(server_path)])

                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        summary = await main._run_tracking_decision_ipc_mcp_cycle(
                            session=session,
                            transport=orchestration_transport,
                            max_requests_per_cycle=4,
                            tool_timeout_seconds=1.0,
                        )

                self.assertEqual(summary["published_count"], 1)

                result_events = [
                    json.loads(line)
                    for line in (temp_root / "tracking_decision_tool_results.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(result_events), 1)
                self.assertEqual(result_events[0].get("status"), "ok")
                self.assertEqual(result_events[0].get("investigation_id"), "inv-semantic-e2e-1")

                captured_follow_up_context: dict[str, object] = {}

                def _capture_submit(**kwargs):
                    follow_up = kwargs.get("follow_up_context")
                    if isinstance(follow_up, dict):
                        captured_follow_up_context.update(follow_up)
                    return True

                with patch.object(tracking_loop, "submit_tracking_runtime_candidates", side_effect=_capture_submit):
                    tracking_loop._consume_tracking_decision_results_if_enabled(
                        state=state,
                        tracking_runtime=_PollingRuntime(),
                        frame=types.SimpleNamespace(shape=(10, 10, 3)),
                        tracking_candidates=[],
                        now=time.time(),
                    )

                self.assertEqual(captured_follow_up_context.get("investigation_id"), "inv-semantic-e2e-1")
                self.assertEqual(captured_follow_up_context.get("correlation_id"), "corr-semantic-e2e-1")
                self.assertIsInstance(captured_follow_up_context.get("mcp_result"), dict)

                result_key = str(captured_follow_up_context.get("idempotency_key"))
                request_id = str(captured_follow_up_context.get("request_id"))
                runtime = _PollingRuntime(
                    _follow_up_observation(
                        investigation_id="inv-semantic-e2e-1",
                        request_id=request_id,
                        result_key=result_key,
                        inconclusive=False,
                    )
                )

                with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                    tracking_loop._apply_tracking_agent_runtime_cycle(
                        state=state,
                        tracking_runtime=runtime,
                        now=time.time(),
                        frame=types.SimpleNamespace(shape=(10, 10, 3)),
                        tracks={"det-1": _track_snapshot()},
                        tracking_candidates=[],
                        selected_event_reasons=[],
                    )

                investigation = state.tracking_decision_investigations["inv-semantic-e2e-1"]
                self.assertTrue(investigation.get("terminal", False))
                self.assertEqual(investigation["final_decision"]["action"], "propose_identity_association")

                snapshot_after_cycle = snapshot_path.read_text(encoding="utf-8")
                self.assertEqual(snapshot_before_cycle, snapshot_after_cycle)
        finally:
            if original_snapshot is None:
                try:
                    snapshot_path.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                snapshot_path.write_text(original_snapshot, encoding="utf-8")

    async def test_e2e_identity_snapshot_follow_up_round_trip(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        captures_dir = project_root / "captures"
        snapshot_path = captures_dir / "latest_tracking_identity_snapshot.json"

        original_snapshot = snapshot_path.read_text(encoding="utf-8") if snapshot_path.is_file() else None
        captures_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "state": "available",
                    "snapshot_created_at": time.time(),
                    "freshness_window_seconds": 300.0,
                    "source_generation": 10,
                    "source_version": 20,
                    "identity_count": 1,
                    "identities": [
                        {
                            "app_identity_id": "app-1",
                            "status": "visible",
                            "provisional": False,
                            "generation": 10,
                            "identity_version": 2,
                            "updated_at": time.time(),
                            "detection_reference": "evidence://detection/d-1",
                            "observation_history": [],
                            "historical_yolo_track_ids": ["track-1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        # deterministic isolated IPC artifact paths under temp directory
        from tempfile import TemporaryDirectory

        try:
            with TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                dashboard_transport = TrackingDecisionIpcTransport(
                    requests_path=temp_root / "tracking_decision_tool_requests.jsonl",
                    results_path=temp_root / "tracking_decision_tool_results.jsonl",
                    request_index_path=temp_root / "latest_tracking_decision_request_index.json",
                    result_index_path=temp_root / "latest_tracking_decision_result_index.json",
                    writer_role="dashboard",
                )
                orchestration_transport = TrackingDecisionIpcTransport(
                    requests_path=temp_root / "tracking_decision_tool_requests.jsonl",
                    results_path=temp_root / "tracking_decision_tool_results.jsonl",
                    request_index_path=temp_root / "latest_tracking_decision_request_index.json",
                    result_index_path=temp_root / "latest_tracking_decision_result_index.json",
                    writer_role="orchestration",
                )

                state = SharedDashboardState()
                state.tracking_decision_ipc_enabled = True
                state.tracking_decision_ipc_transport = dashboard_transport
                state.lock = _Lock()

                initial_observation = types.SimpleNamespace(
                    decision=_decision_payload(investigation_id="inv-e2e-1", request_id="req-e2e-1"),
                    llm_correlation_id="corr-llm-e2e-1",
                    uncertainties=[],
                )

                tracking_loop._submit_tracking_decision_request_if_enabled(
                    state=state,
                    tracking_agent_observation=initial_observation,
                    now=time.time(),
                )

                server_path = project_root / "server.py"
                params = StdioServerParameters(command=sys.executable, args=[str(server_path)])

                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        summary = await main._run_tracking_decision_ipc_mcp_cycle(
                            session=session,
                            transport=orchestration_transport,
                            max_requests_per_cycle=4,
                            tool_timeout_seconds=1.0,
                        )

                self.assertEqual(summary["published_count"], 1)

                captured_follow_up_context: dict[str, object] = {}

                def _capture_submit(**kwargs):
                    follow_up = kwargs.get("follow_up_context")
                    if isinstance(follow_up, dict):
                        captured_follow_up_context.update(follow_up)
                    return True

                with patch.object(tracking_loop, "submit_tracking_runtime_candidates", side_effect=_capture_submit):
                    tracking_loop._consume_tracking_decision_results_if_enabled(
                        state=state,
                        tracking_runtime=_PollingRuntime(),
                        frame=types.SimpleNamespace(shape=(10, 10, 3)),
                        tracking_candidates=[],
                        now=time.time(),
                    )

                self.assertEqual(captured_follow_up_context.get("investigation_id"), "inv-e2e-1")
                self.assertEqual(captured_follow_up_context.get("request_id"), "req-e2e-1")
                self.assertEqual(captured_follow_up_context.get("correlation_id"), "corr-e2e-1")
                self.assertIsInstance(captured_follow_up_context.get("evidence_references"), dict)
                self.assertIsInstance(captured_follow_up_context.get("mcp_result"), dict)

                result_key = str(captured_follow_up_context.get("idempotency_key"))
                runtime = _PollingRuntime(
                    _follow_up_observation(
                        investigation_id="inv-e2e-1",
                        request_id="req-e2e-1",
                        result_key=result_key,
                        inconclusive=False,
                    )
                )

                with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=False):
                    tracking_loop._apply_tracking_agent_runtime_cycle(
                        state=state,
                        tracking_runtime=runtime,
                        now=time.time(),
                        frame=types.SimpleNamespace(shape=(10, 10, 3)),
                        tracks={"det-1": _track_snapshot()},
                        tracking_candidates=[],
                        selected_event_reasons=[],
                    )

                investigation = state.tracking_decision_investigations["inv-e2e-1"]
                self.assertTrue(investigation.get("terminal", False))
                self.assertEqual(investigation["final_decision"]["action"], "propose_identity_association")
        finally:
            if original_snapshot is None:
                try:
                    snapshot_path.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                snapshot_path.write_text(original_snapshot, encoding="utf-8")

    async def test_restart_does_not_retrigger_follow_up_for_delivered_result(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dashboard_transport = TrackingDecisionIpcTransport(
                requests_path=temp_root / "tracking_decision_tool_requests.jsonl",
                results_path=temp_root / "tracking_decision_tool_results.jsonl",
                request_index_path=temp_root / "latest_tracking_decision_request_index.json",
                result_index_path=temp_root / "latest_tracking_decision_result_index.json",
                writer_role="dashboard",
            )
            orchestration_transport = TrackingDecisionIpcTransport(
                requests_path=temp_root / "tracking_decision_tool_requests.jsonl",
                results_path=temp_root / "tracking_decision_tool_results.jsonl",
                request_index_path=temp_root / "latest_tracking_decision_request_index.json",
                result_index_path=temp_root / "latest_tracking_decision_result_index.json",
                writer_role="orchestration",
            )

            state = SharedDashboardState()
            state.tracking_decision_ipc_enabled = True
            state.tracking_decision_ipc_transport = dashboard_transport
            state.lock = _Lock()

            observation = types.SimpleNamespace(
                decision={
                    "action": "request_additional_tracking_evidence",
                    "investigation_id": "inv-restart-1",
                    "tool_requests": [
                        {
                            "request_id": "req-restart-1",
                            "tool_name": "get_track_evidence_history",
                            "arguments": {"track_id": "track-1"},
                        }
                    ],
                },
                llm_correlation_id="corr-restart-1",
                uncertainties=[],
            )
            tracking_loop._submit_tracking_decision_request_if_enabled(
                state=state,
                tracking_agent_observation=observation,
                now=time.time(),
            )

            # publish deterministic result directly
            pending = orchestration_transport.orchestration_poll_pending_requests(max_requests=4)
            request_event = pending["pending_requests"][0]
            orchestration_transport.orchestration_publish_fixture_result(
                request_event=request_event,
                status="ok",
                result_payload={"available": True, "messages": []},
                now_timestamp=time.time(),
            )

            with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
                tracking_loop._consume_tracking_decision_results_if_enabled(
                    state=state,
                    tracking_runtime=_PollingRuntime(),
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracking_candidates=[],
                    now=time.time(),
                )
            self.assertEqual(submit_mock.call_count, 1)

            # restart dashboard state: no in-memory follow-up state
            restarted_state = SharedDashboardState()
            restarted_state.tracking_decision_ipc_enabled = True
            restarted_state.tracking_decision_ipc_transport = TrackingDecisionIpcTransport(
                requests_path=temp_root / "tracking_decision_tool_requests.jsonl",
                results_path=temp_root / "tracking_decision_tool_results.jsonl",
                request_index_path=temp_root / "latest_tracking_decision_request_index.json",
                result_index_path=temp_root / "latest_tracking_decision_result_index.json",
                writer_role="dashboard",
            )
            restarted_state.lock = _Lock()

            with patch.object(tracking_loop, "submit_tracking_runtime_candidates", return_value=True) as submit_mock:
                tracking_loop._consume_tracking_decision_results_if_enabled(
                    state=restarted_state,
                    tracking_runtime=_PollingRuntime(),
                    frame=types.SimpleNamespace(shape=(10, 10, 3)),
                    tracking_candidates=[],
                    now=time.time(),
                )
            submit_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
