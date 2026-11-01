from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agents.orchestration.runtime import flow as runtime_flow
from agents.orchestration.runtime.verification_queue import (
    VerificationQueueManager,
    build_request_from_track_id,
    build_request_from_message_id,
    extract_track_requests_from_cycle_payload,
)


class _FakeItem:
    def __init__(self, payload: dict):
        self.json = payload


class _FakeResult:
    def __init__(self, payload: dict):
        self.content = [_FakeItem(payload)]


class _FakeSession:
    def __init__(self, handler):
        self._handler = handler

    async def call_tool(self, tool_name, arguments):
        return await self._handler(tool_name, arguments)


def _envelope_payload(*, message_id: str, track_id: str, timestamp: float) -> dict:
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
            "claim_reasons": ["test"],
        },
        "frame_id": "frame-1",
        "detection_id": "det-1",
        "source_evidence_refs": ["evidence://frame/frame-1"],
        "uncertainty": [],
        "quality_flags": ["position_kind:calibrated_ground_plane"],
        "track_history": [],
    }


def _write_validated_envelope(root: Path, *, message_id: str, track_id: str, timestamp: float) -> None:
    envelopes = root / "envelopes"
    envelopes.mkdir(parents=True, exist_ok=True)
    payload = {
        "message_id": message_id,
        "stored_at": timestamp,
        "validation": {"valid": True, "errors": [], "warnings": []},
        "envelope": _envelope_payload(message_id=message_id, track_id=track_id, timestamp=timestamp),
    }
    (envelopes / f"{message_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _verification_envelope(*, message_id: str, track_id: str, timestamp: float) -> dict:
    return {
        "schema_version": "1.0",
        "observation_id": f"verify-{message_id}",
        "referenced_track_id": track_id,
        "referenced_message_id": message_id,
        "timestamp": timestamp,
        "source_evidence_refs": ["evidence://frame/frame-1"],
        "verification_status": "supported",
        "visibility_status": "visible",
        "identity_consistency": "consistent",
        "possible_id_switch": False,
        "supporting_evidence_refs": [],
        "contradicting_evidence_refs": [],
        "uncertainty": [],
        "limitations": [],
        "confidence": 0.9,
        "visual_content_inspected": True,
    }


class TestVerificationQueueLiveIntegration(unittest.IsolatedAsyncioTestCase):
    def test_cycle_payload_extracts_track_requests_only_when_actionable(self) -> None:
        payload = {
            "events": [
                {"track_id": "track-1", "evidence_requested": True},
                {"track_id": "track-2", "evidence_requested": False},
                {"track_id": "", "evidence_requested": True},
                {"evidence_requested": True},
            ]
        }
        requests = list(extract_track_requests_from_cycle_payload(payload))

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].track_id, "track-1")
        self.assertEqual(requests[0].source, "reasoning_cycle")
        self.assertIsNone(requests[0].message_id)

    async def test_monitor_loop_schedules_and_stays_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-1", track_id="track-1", timestamp=1000.0)

            verify_started = asyncio.Event()
            allow_verify_finish = asyncio.Event()
            tracking_calls = 0

            async def tool_handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-1"})
                if tool_name == "verify_tracking_visual_evidence":
                    verify_started.set()
                    await allow_verify_finish.wait()
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id=arguments["message_id"],
                                track_id="track-1",
                                timestamp=1001.0,
                            ),
                        }
                    )
                raise RuntimeError(f"unexpected tool: {tool_name}")

            async def run_tracking_cycle(_session, _address):
                nonlocal tracking_calls
                tracking_calls += 1
                if tracking_calls == 3:
                    allow_verify_finish.set()
                if tracking_calls >= 5:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            async def run_agent_ops_pipeline_fn(_session, **_kwargs):
                return {
                    "agent_ops_enabled": True,
                    "agent_ops_summary_enabled": False,
                    "depth_summary_enabled": False,
                    "completed_cycle_increment": 1,
                    "cycle_payload": {
                        "events": [
                            {
                                "track_id": "track-1",
                                "evidence_requested": True,
                            }
                        ]
                    },
                }

            manager = VerificationQueueManager(
                evidence_root_dir=evidence_root,
                request_timeout_seconds=2.0,
            )
            session = _FakeSession(tool_handler)
            await manager.start(session=session)
            try:
                with self.assertRaises(KeyboardInterrupt):
                    await runtime_flow.monitor_movement(
                        session,
                        tracking_address=("127.0.0.1", 5005),
                        analysis_interval_seconds=0,
                        enable_agent_ops_heartbeat=True,
                        agent_ops_interval_seconds=0,
                        agent_ops_max_tracks=10,
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
                    )

                await asyncio.wait_for(manager._queue.join(), timeout=2.0)
                results = manager.get_recent_results(20)

                self.assertTrue(verify_started.is_set())
                self.assertGreaterEqual(tracking_calls, 4)
                self.assertTrue(any(item["status"] == "completed" for item in results))
                completed = next(item for item in results if item["status"] == "completed")
                self.assertEqual(completed["track_id"], "track-1")
                self.assertEqual(completed["evidence_message_id"], "msg-1")
                self.assertTrue(completed["coordinator_valid"])
            finally:
                await manager.stop()

    async def test_monitor_loop_accepts_source_fn_requests_without_reasoning_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-source", track_id="track-source", timestamp=1000.0)

            tracking_calls = 0
            submitted_once = False

            async def tool_handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    return _FakeResult({"available": True, "track_id": "track-source"})
                if tool_name == "verify_tracking_visual_evidence":
                    return _FakeResult(
                        {
                            "available": True,
                            "envelope": _verification_envelope(
                                message_id=arguments["message_id"],
                                track_id="track-source",
                                timestamp=1000.5,
                            ),
                        }
                    )
                raise RuntimeError(f"unexpected tool: {tool_name}")

            async def run_tracking_cycle(_session, _address):
                nonlocal tracking_calls
                tracking_calls += 1
                if tracking_calls >= 5:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            async def run_agent_ops_pipeline_fn(_session, **_kwargs):
                return {
                    "agent_ops_enabled": True,
                    "agent_ops_summary_enabled": False,
                    "depth_summary_enabled": False,
                    "completed_cycle_increment": 1,
                    "cycle_payload": {"events": []},
                }

            def verification_request_source_fn():
                nonlocal submitted_once
                if submitted_once:
                    return []
                submitted_once = True
                return [build_request_from_track_id("track-source", source="external_source")]

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, request_timeout_seconds=2.0)
            session = _FakeSession(tool_handler)
            await manager.start(session=session)
            try:
                with self.assertRaises(KeyboardInterrupt):
                    await runtime_flow.monitor_movement(
                        session,
                        tracking_address=("127.0.0.1", 5005),
                        analysis_interval_seconds=0,
                        enable_agent_ops_heartbeat=True,
                        agent_ops_interval_seconds=0,
                        agent_ops_max_tracks=10,
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
                        verification_request_source_fn=verification_request_source_fn,
                    )

                await asyncio.wait_for(manager._queue.join(), timeout=2.0)
                results = manager.get_recent_results(20)

                self.assertGreaterEqual(tracking_calls, 4)
                self.assertTrue(any(item["source"] == "external_source" for item in results))
                self.assertTrue(any(item["status"] == "completed" for item in results))
            finally:
                await manager.stop()

    async def test_failed_verification_records_structured_failure_and_later_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence_root = Path(temp_dir) / "tracking_evidence"
            _write_validated_envelope(evidence_root, message_id="msg-fail", track_id="track-fail", timestamp=1000.0)
            _write_validated_envelope(evidence_root, message_id="msg-ok", track_id="track-ok", timestamp=1001.0)

            tracking_calls = 0
            source_calls = 0

            async def tool_handler(tool_name, arguments):
                if tool_name == "get_tracking_visual_evidence":
                    message_id = str(arguments.get("message_id", ""))
                    if message_id == "msg-fail":
                        return _FakeResult({"available": True, "track_id": "track-fail"})
                    if message_id == "msg-ok":
                        return _FakeResult({"available": True, "track_id": "track-ok"})
                    raise RuntimeError("unexpected message id")

                if tool_name == "verify_tracking_visual_evidence":
                    message_id = str(arguments.get("message_id", ""))
                    if message_id == "msg-fail":
                        return _FakeResult(
                            {
                                "available": False,
                                "error_code": "provider_failure",
                                "error": "provider down",
                                "details": {"attempts": 1},
                            }
                        )
                    if message_id == "msg-ok":
                        return _FakeResult(
                            {
                                "available": True,
                                "envelope": _verification_envelope(
                                    message_id="msg-ok",
                                    track_id="track-ok",
                                    timestamp=1001.5,
                                ),
                            }
                        )
                    raise RuntimeError("unexpected message id")

                raise RuntimeError(f"unexpected tool: {tool_name}")

            async def run_tracking_cycle(_session, _address):
                nonlocal tracking_calls
                tracking_calls += 1
                if tracking_calls >= 7:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            async def run_agent_ops_pipeline_fn(_session, **_kwargs):
                return {
                    "agent_ops_enabled": True,
                    "agent_ops_summary_enabled": False,
                    "depth_summary_enabled": False,
                    "completed_cycle_increment": 1,
                    "cycle_payload": {"events": []},
                }

            def verification_request_source_fn():
                nonlocal source_calls
                source_calls += 1
                if source_calls == 1:
                    return [
                        build_request_from_message_id(
                            "track-fail",
                            "msg-fail",
                            source="external_source",
                        )
                    ]
                if source_calls == 3:
                    return [
                        build_request_from_message_id(
                            "track-ok",
                            "msg-ok",
                            source="external_source",
                        )
                    ]
                return []

            manager = VerificationQueueManager(evidence_root_dir=evidence_root, request_timeout_seconds=2.0)
            session = _FakeSession(tool_handler)
            await manager.start(session=session)
            try:
                with self.assertRaises(KeyboardInterrupt):
                    await runtime_flow.monitor_movement(
                        session,
                        tracking_address=("127.0.0.1", 5005),
                        analysis_interval_seconds=0,
                        enable_agent_ops_heartbeat=True,
                        agent_ops_interval_seconds=0,
                        agent_ops_max_tracks=10,
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
                        verification_request_source_fn=verification_request_source_fn,
                    )

                await asyncio.wait_for(manager._queue.join(), timeout=2.0)
                results = manager.get_recent_results(50)

                fail_record = next(
                    item for item in results
                    if item["evidence_message_id"] == "msg-fail"
                )
                ok_record = next(
                    item for item in results
                    if item["evidence_message_id"] == "msg-ok"
                )

                self.assertEqual(fail_record["status"], "failed")
                self.assertEqual(fail_record["validation_outcome"], "provider_failure")
                self.assertEqual(fail_record["error_code"], "provider_failure")
                self.assertEqual(ok_record["status"], "completed")
                self.assertTrue(ok_record["coordinator_valid"])
                self.assertGreaterEqual(tracking_calls, 6)
            finally:
                await manager.stop()
