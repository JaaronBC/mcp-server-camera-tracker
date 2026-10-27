# Handles test main agent ops behavior.
from __future__ import annotations

import asyncio
import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, MagicMock, patch

import main


class FakeItem:
    def __init__(self, text: str | None = None, json_payload=None):
        # Performs init logic.
        self.text = text
        self.json = json_payload


class FakeResult:
    def __init__(self, content, is_error: bool = False):
        # Performs init logic.
        self.content = content
        self.is_error = is_error


class FakeSession:
    def __init__(self, result: FakeResult | None = None, error: Exception | None = None):
        # Performs init logic.
        self._result = result
        self._error = error

    async def call_tool(self, tool_name, arguments):
        # Performs call tool logic.
        if self._error is not None:
            raise self._error
        return self._result


class SequenceSession:
    def __init__(self, responses):
        # Performs init logic.
        self._responses = list(responses)

    async def call_tool(self, tool_name, arguments):
        # Performs call tool logic.
        if not self._responses:
            raise RuntimeError("no more responses")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeStdioContext:
    async def __aenter__(self):
        # Performs aenter logic.
        return (object(), object())

    async def __aexit__(self, exc_type, exc, tb):
        # Performs aexit logic.
        return False


class FakeClientSessionContext:
    def __init__(self, *_args, **_kwargs):
        # Performs init logic.
        pass

    async def __aenter__(self):
        # Performs aenter logic.
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Performs aexit logic.
        return False

    async def initialize(self):
        # Performs initialize logic.
        return None

    async def call_tool(self, _tool_name, _arguments):
        # Performs call tool logic.
        return FakeResult([])


class TestMainAgentOps(unittest.IsolatedAsyncioTestCase):
    async def _run_agent_with_common_patches(self, **kwargs) -> str:
        # Performs run agent with common patches logic.
        fake_client = MagicMock()
        fake_client.close = AsyncMock(return_value=None)
        output_stream = io.StringIO()

        with patch("main.load_system_skill", return_value="skill"):
            with patch("main.create_openai_client", return_value=fake_client):
                with patch("main.stdio_client", return_value=FakeStdioContext()):
                    with patch("main.ClientSession", FakeClientSessionContext):
                        with patch("main.verify_required_tools", new=AsyncMock(return_value=None)):
                            with patch("main.start_live_tracking", new=AsyncMock(return_value=None)):
                                with patch("main.monitor_movement", new=AsyncMock(return_value=None)):
                                    with redirect_stdout(output_stream):
                                        await main.run_agent(**kwargs)

        return output_stream.getvalue()

    async def test_run_agent_ops_cycle_success(self) -> None:
        # Performs test run agent ops cycle success logic.
        result = FakeResult(
            [
                FakeItem(
                    json_payload={
                        "cycle": {
                            "evaluated_tracks": 2,
                            "success_count": 2,
                            "error_count": 0,
                            "started_at": 1000.0,
                        }
                    }
                )
            ]
        )
        session = FakeSession(result=result)

        enabled = await main.run_agent_ops_cycle(session)

        self.assertTrue(enabled)

    async def test_run_agent_ops_cycle_disables_for_unknown_tool_error_result(self) -> None:
        # Performs test run agent ops cycle disables for unknown tool error result logic.
        result = FakeResult([FakeItem(text="Unknown tool: run_reasoning_cycle_and_log")], is_error=True)
        session = FakeSession(result=result)

        enabled = await main.run_agent_ops_cycle(session)

        self.assertFalse(enabled)

    async def test_run_agent_ops_cycle_keeps_enabled_for_runtime_exception(self) -> None:
        # Performs test run agent ops cycle keeps enabled for runtime exception logic.
        session = FakeSession(error=RuntimeError("temporary MCP transport issue"))

        enabled = await main.run_agent_ops_cycle(session)

        self.assertTrue(enabled)

    def test_format_agent_ops_cycle(self) -> None:
        # Performs test format agent ops cycle logic.
        text = main.format_agent_ops_cycle(
            {
                "cycle": {
                    "evaluated_tracks": 3,
                    "success_count": 2,
                    "error_count": 1,
                    "started_at": 1234.5,
                }
            }
        )
        self.assertIn("tracks=3", text)
        self.assertIn("ok=2", text)
        self.assertIn("errors=1", text)

    def test_format_agent_ops_cycle_with_degraded_event_log(self) -> None:
        # Performs test format agent ops cycle with degraded event log logic.
        text = main.format_agent_ops_cycle(
            {
                "event_log_writable": False,
                "event_log_error_count": 2,
                "cycle": {
                    "evaluated_tracks": 1,
                    "success_count": 1,
                    "error_count": 0,
                    "started_at": 1234.5,
                },
            }
        )

        self.assertIn("event_log=degraded(errors=2)", text)

    def test_format_agent_ops_cycle_evidence(self) -> None:
        # Performs test format agent ops cycle evidence logic.
        lines = main.format_agent_ops_cycle_evidence(
            {
                "events": [
                    {
                        "track_id": "track-1",
                        "evidence_trace": {
                            "latest_track_snapshot": {
                                "ground_position": {"x": 1.5, "z": 4.0},
                                "relative_depth": 0.42,
                            },
                            "tracking_agent_context": {
                                "vlm_model": "qwen-3.8-27b",
                                "selected_updates": [{"detection_id": "track-1"}],
                            },
                            "coordinate_context": {
                                "coordinate_system": "calibrated_ground_plane",
                                "calibration_status": "loaded",
                            },
                        },
                    }
                ]
            }
        )

        self.assertEqual(len(lines), 1)
        self.assertIn("track=track-1", lines[0])
        self.assertIn("latest_ground=(1.5,4.0)", lines[0])
        self.assertIn("depth=0.42", lines[0])
        self.assertIn("tracking_vlm=qwen-3.8-27b", lines[0])
        self.assertIn("selected_updates=1", lines[0])
        self.assertIn("coord=calibrated_ground_plane", lines[0])
        self.assertIn("calibration=loaded", lines[0])

    async def test_run_agent_ops_health_summary_success(self) -> None:
        # Performs test run agent ops health summary success logic.
        result = FakeResult(
            [
                FakeItem(
                    json_payload={
                        "status": {
                            "success_rate": 1.0,
                            "error_rate": 0.0,
                            "bounded_out_rate": 0.25,
                        },
                        "latency": {
                            "p50_ms": 5.0,
                            "p95_ms": 9.5,
                        },
                    }
                )
            ]
        )
        session = FakeSession(result=result)

        enabled = await main.run_agent_ops_health_summary(session)

        self.assertTrue(enabled)

    async def test_run_agent_ops_health_summary_disables_for_unknown_tool(self) -> None:
        # Performs test run agent ops health summary disables for unknown tool logic.
        result = FakeResult([FakeItem(text="Unknown tool: summarize_agent_event_health")], is_error=True)
        session = FakeSession(result=result)

        enabled = await main.run_agent_ops_health_summary(session)

        self.assertFalse(enabled)

    async def test_run_agent_ops_pipeline_skips_followups_when_cycle_disabled(self) -> None:
        # Performs test run agent ops pipeline skips followups when cycle disabled logic.
        session = FakeSession(result=FakeResult([]))
        with patch("main.run_agent_ops_cycle", new=AsyncMock(return_value=False)) as cycle_mock:
            with patch("main.run_agent_ops_health_summary", new=AsyncMock(return_value=True)) as summary_mock:
                with patch("main.run_depth_coverage_summary", new=AsyncMock(return_value=True)) as depth_mock:
                    outcome = await main._run_agent_ops_pipeline(
                        session,
                        cycle_number=3,
                        max_tracks=5,
                        agent_ops_summary_enabled=True,
                        agent_ops_summary_every_cycles=1,
                        agent_ops_summary_window_size=10,
                        handoff_degraded_warn_rate=0.1,
                        handoff_degraded_critical_rate=0.3,
                        depth_summary_enabled=True,
                        depth_coverage_summary_every_cycles=1,
                        depth_coverage_sample_limit=10,
                    )

        self.assertFalse(outcome["agent_ops_enabled"])
        self.assertEqual(outcome["completed_cycle_increment"], 0)
        cycle_mock.assert_awaited_once()
        summary_mock.assert_not_awaited()
        depth_mock.assert_not_awaited()

    async def test_monitor_movement_skips_overlapping_agent_ops_cycle(self) -> None:
        # Performs test monitor movement skips overlapping agent ops cycle logic.
        session = SequenceSession(
            responses=[
                FakeResult([FakeItem(text="frame-1")]),
                FakeResult([FakeItem(text="frame-2")]),
                KeyboardInterrupt(),
            ]
        )

        async def slow_cycle(*_args, **_kwargs):
            # Performs slow cycle logic.
            await asyncio.Event().wait()
            return True

        stream = io.StringIO()
        with patch("main.ANALYSIS_INTERVAL_SECONDS", 0):
            with patch("main.summarize_frame", new=AsyncMock(return_value="ok-summary")):
                with patch("main.run_agent_ops_cycle", new=AsyncMock(side_effect=slow_cycle)) as cycle_mock:
                    with redirect_stdout(stream):
                        with self.assertRaises(KeyboardInterrupt):
                            await main.monitor_movement(
                                session,
                                client=MagicMock(),
                                system_prompt="skill",
                                enable_agent_ops_heartbeat=True,
                                agent_ops_interval_seconds=0,
                                agent_ops_max_tracks=2,
                                agent_ops_summary_every_cycles=100,
                                depth_coverage_summary_enabled=False,
                                depth_coverage_summary_every_cycles=100,
                            )

        self.assertEqual(cycle_mock.await_count, 1)
        self.assertIn("Skipping agent ops trigger: prior cycle still running.", stream.getvalue())

    async def test_run_depth_coverage_summary_success(self) -> None:
        # Performs test run depth coverage summary success logic.
        result = FakeResult(
            [
                FakeItem(
                    json_payload={
                        "latest_coverage": {
                            "tracks_with_depth": 3,
                            "total_tracks": 4,
                            "coverage_percent": 75.0,
                        },
                        "trend": {
                            "direction": "improving",
                            "delta_coverage_percent": 10.0,
                        },
                    }
                )
            ]
        )
        session = FakeSession(result=result)

        enabled = await main.run_depth_coverage_summary(session)

        self.assertTrue(enabled)

    async def test_run_depth_coverage_summary_disables_for_unknown_tool(self) -> None:
        # Performs test run depth coverage summary disables for unknown tool logic.
        result = FakeResult([FakeItem(text="Unknown tool: dashboard_depth_coverage_metrics")], is_error=True)
        session = FakeSession(result=result)

        enabled = await main.run_depth_coverage_summary(session)

        self.assertFalse(enabled)

    def test_format_agent_ops_health_summary(self) -> None:
        # Performs test format agent ops health summary logic.
        text = main.format_agent_ops_health_summary(
            {
                "status": {
                    "success_rate": 0.9,
                    "error_rate": 0.1,
                    "bounded_out_rate": 0.2,
                },
                "latency": {
                    "p50_ms": 4.2,
                    "p95_ms": 12.5,
                },
            }
        )

        self.assertIn("success_rate=0.9", text)
        self.assertIn("error_rate=0.1", text)
        self.assertIn("p95_ms=12.5", text)

    def test_format_agent_ops_health_summary_with_error_digest(self) -> None:
        # Performs test format agent ops health summary with error digest logic.
        now = 2000.0
        text = main.format_agent_ops_health_summary(
            {
                "status": {
                    "success_rate": 0.8,
                    "error_rate": 0.2,
                    "bounded_out_rate": 0.1,
                },
                "latency": {
                    "p50_ms": 5.5,
                    "p95_ms": 15.0,
                },
                "agent_error_code_counts": {
                    "observation": {
                        "timeout": 2,
                        "runtime_error": 1,
                    },
                    "reasoning": {
                        "invalid_response_format": 3,
                    },
                    "cycle": {
                        "radar_state_stale": 4,
                    },
                },
                "agent_error_trends": {
                    "observation": {
                        "top_error_code": "timeout",
                        "top_error_count": 2,
                        "top_error_rate": 0.2,
                    },
                    "reasoning": {
                        "top_error_code": "invalid_response_format",
                        "top_error_count": 3,
                        "top_error_rate": 0.3,
                    },
                    "cycle": {
                        "top_error_code": "radar_state_stale",
                        "top_error_count": 4,
                        "top_error_rate": 0.4,
                    },
                    "handoff_validation": {
                        "top_error_code": "invalid_handoff_payload",
                        "top_error_count": 1,
                        "top_error_rate": 0.1,
                    },
                },
                "handoff_validation": {
                    "status_counts": {
                        "ok": 7,
                        "error": 3,
                    },
                    "degraded_rate": 0.3,
                },
                "returned_events": 10,
                "time_window": {
                    "last_timestamp": now,
                },
            },
            _max_age_seconds=30.0,
            _now_timestamp=now,
        )

        self.assertIn(
            "agent_error_digest=obs(timeout:2),rea(invalid_response_format:3),cyc(radar_state_stale:4)",
            text,
        )
        self.assertIn(
            "agent_error_trends=obs(timeout:2@0.200),rea(invalid_response_format:3@0.300),cyc(radar_state_stale:4@0.400)",
            text,
        )
        self.assertIn(
            "handoff_validation_trend=handoff(invalid_handoff_payload:1@0.100)",
            text,
        )
        self.assertIn(
            "handoff_status=ok:7@0.700,err:3@0.300",
            text,
        )
        self.assertIn(
            "handoff_degraded_rate=0.300",
            text,
        )
        self.assertIn(
            "handoff_severity=critical",
            text,
        )

    def test_format_agent_ops_health_summary_suppresses_digest_when_stale(self) -> None:
        # Performs test format agent ops health summary suppresses digest when stale logic.
        now = 5000.0
        text = main.format_agent_ops_health_summary(
            {
                "status": {
                    "success_rate": 0.8,
                    "error_rate": 0.2,
                    "bounded_out_rate": 0.1,
                },
                "latency": {
                    "p50_ms": 5.5,
                    "p95_ms": 15.0,
                },
                "agent_error_code_counts": {
                    "observation": {
                        "timeout": 2,
                    },
                    "reasoning": {
                        "invalid_response_format": 3,
                    },
                },
                "agent_error_trends": {
                    "observation": {
                        "top_error_code": "timeout",
                        "top_error_count": 2,
                        "top_error_rate": 0.2,
                    },
                    "reasoning": {
                        "top_error_code": "invalid_response_format",
                        "top_error_count": 3,
                        "top_error_rate": 0.3,
                    },
                    "handoff_validation": {
                        "top_error_code": "invalid_handoff_payload",
                        "top_error_count": 1,
                        "top_error_rate": 0.1,
                    },
                },
                "handoff_validation": {
                    "status_counts": {
                        "ok": 7,
                        "error": 3,
                    },
                    "degraded_rate": 0.3,
                },
                "returned_events": 10,
                "time_window": {
                    "last_timestamp": now - 60.0,
                },
            },
            _max_age_seconds=30.0,
            _now_timestamp=now,
        )

        self.assertNotIn("agent_error_digest=", text)
        self.assertNotIn("agent_error_trends=", text)
        self.assertNotIn("handoff_validation_trend=", text)
        self.assertNotIn("handoff_status=", text)
        self.assertNotIn("handoff_degraded_rate=", text)
        self.assertNotIn("handoff_severity=", text)

    def test_format_agent_error_digest_missing_counts(self) -> None:
        # Performs test format agent error digest missing counts logic.
        self.assertIsNone(main._format_agent_error_digest(None))
        self.assertIsNone(main._format_agent_error_digest({}))

        digest = main._format_agent_error_digest(
            {
                "observation": {},
                "reasoning": {},
            }
        )
        self.assertEqual(digest, "obs(none),rea(none)")

    def test_format_agent_error_trends_missing_counts(self) -> None:
        # Performs test format agent error trends missing counts logic.
        self.assertIsNone(main._format_agent_error_trends(None))
        self.assertIsNone(main._format_agent_error_trends({}))

        trends = main._format_agent_error_trends(
            {
                "observation": {},
                "reasoning": {
                    "top_error_code": "timeout",
                    "top_error_count": 1,
                    "top_error_rate": 0.5,
                },
            }
        )
        self.assertEqual(trends, "obs(none),rea(timeout:1@0.500)")

    def test_format_handoff_validation_trend(self) -> None:
        # Performs test format handoff validation trend logic.
        self.assertIsNone(main._format_handoff_validation_trend(None))
        self.assertIsNone(main._format_handoff_validation_trend({}))

        missing = main._format_handoff_validation_trend(
            {
                "handoff_validation": {},
            }
        )
        self.assertEqual(missing, "handoff(none)")

        rendered = main._format_handoff_validation_trend(
            {
                "handoff_validation": {
                    "top_error_code": "invalid_handoff_payload",
                    "top_error_count": 2,
                    "top_error_rate": 0.25,
                }
            }
        )
        self.assertEqual(rendered, "handoff(invalid_handoff_payload:2@0.250)")

    def test_format_handoff_status_summary(self) -> None:
        # Performs test format handoff status summary logic.
        self.assertIsNone(main._format_handoff_status_summary(None))
        self.assertIsNone(main._format_handoff_status_summary({}))
        self.assertIsNone(
            main._format_handoff_status_summary(
                {
                    "handoff_validation": {"status_counts": {"ok": 1}},
                }
            )
        )

        rendered = main._format_handoff_status_summary(
            {
                "handoff_validation": {
                    "status_counts": {
                        "ok": 2,
                        "error": 1,
                    }
                },
                "returned_events": 4,
            }
        )
        self.assertEqual(rendered, "ok:2@0.500,err:1@0.250")

    def test_format_handoff_degraded_rate(self) -> None:
        # Performs test format handoff degraded rate logic.
        self.assertIsNone(main._format_handoff_degraded_rate(None))
        self.assertIsNone(main._format_handoff_degraded_rate({}))
        self.assertIsNone(main._format_handoff_degraded_rate({"handoff_validation": {}}))

        rendered = main._format_handoff_degraded_rate(
            {
                "handoff_validation": {
                    "degraded_rate": 0.25,
                }
            }
        )
        self.assertEqual(rendered, "0.250")

    def test_classify_handoff_severity(self) -> None:
        # Performs test classify handoff severity logic.
        self.assertEqual(main._classify_handoff_severity(0.0), "healthy")
        self.assertEqual(main._classify_handoff_severity(0.099), "healthy")
        self.assertEqual(main._classify_handoff_severity(0.1), "warn")
        self.assertEqual(main._classify_handoff_severity(0.299), "warn")
        self.assertEqual(main._classify_handoff_severity(0.3), "critical")

    def test_format_handoff_severity(self) -> None:
        # Performs test format handoff severity logic.
        self.assertIsNone(main._format_handoff_severity(None))
        self.assertIsNone(main._format_handoff_severity({}))
        self.assertIsNone(main._format_handoff_severity({"handoff_validation": {}}))

        healthy = main._format_handoff_severity(
            {"handoff_validation": {"degraded_rate": 0.05}}
        )
        warn = main._format_handoff_severity(
            {"handoff_validation": {"degraded_rate": 0.1}}
        )
        critical = main._format_handoff_severity(
            {"handoff_validation": {"degraded_rate": 0.31}}
        )
        custom = main._format_handoff_severity(
            {"handoff_validation": {"degraded_rate": 0.31}},
            warn_rate=0.25,
            critical_rate=0.4,
        )

        self.assertEqual(healthy, "healthy")
        self.assertEqual(warn, "warn")
        self.assertEqual(critical, "critical")
        self.assertEqual(custom, "warn")

    def test_is_agent_health_payload_fresh(self) -> None:
        # Performs test is agent health payload fresh logic.
        payload = {"time_window": {"last_timestamp": 100.0}}
        self.assertTrue(
            main._is_agent_health_payload_fresh(
                payload,
                max_age_seconds=30.0,
                now_timestamp=120.0,
            )
        )
        self.assertFalse(
            main._is_agent_health_payload_fresh(
                payload,
                max_age_seconds=10.0,
                now_timestamp=120.0,
            )
        )

    def test_format_depth_coverage_summary(self) -> None:
        # Performs test format depth coverage summary logic.
        text = main.format_depth_coverage_summary(
            {
                "latest_coverage": {
                    "tracks_with_depth": 5,
                    "total_tracks": 8,
                    "coverage_percent": 62.5,
                },
                "trend": {
                    "direction": "stable",
                    "delta_coverage_percent": 0.4,
                },
            }
        )

        self.assertIn("5/8", text)
        self.assertIn("62.5%", text)
        self.assertIn("trend=stable", text)

    async def test_print_depth_coverage_startup_snapshot_prints_structured_summary(self) -> None:
        # Performs test print depth coverage startup snapshot prints structured summary logic.
        result = FakeResult(
            [
                FakeItem(
                    json_payload={
                        "latest_coverage": {
                            "tracks_with_depth": 2,
                            "total_tracks": 5,
                            "coverage_percent": 40.0,
                        },
                        "trend": {
                            "direction": "stable",
                            "delta_coverage_percent": 0.0,
                        },
                    }
                )
            ]
        )
        session = FakeSession(result=result)

        stream = io.StringIO()
        with redirect_stdout(stream):
            await main.print_depth_coverage_startup_snapshot(session, sample_limit=10)

        output = stream.getvalue()
        self.assertIn("Startup Depth coverage:", output)
        self.assertIn("2/5", output)

    async def test_print_depth_coverage_startup_snapshot_handles_tool_error(self) -> None:
        # Performs test print depth coverage startup snapshot handles tool error logic.
        result = FakeResult([FakeItem(text="Unknown tool: dashboard_depth_coverage_metrics")], is_error=True)
        session = FakeSession(result=result)

        stream = io.StringIO()
        with redirect_stdout(stream):
            await main.print_depth_coverage_startup_snapshot(session, sample_limit=10)

        output = stream.getvalue()
        self.assertIn("Depth coverage snapshot unavailable", output)

    async def test_run_agent_skips_startup_depth_snapshot_when_disabled(self) -> None:
        # Performs test run agent skips startup depth snapshot when disabled logic.
        with patch(
            "main.print_depth_coverage_startup_snapshot",
            new=AsyncMock(return_value=None),
        ) as startup_snapshot_mock:
            await self._run_agent_with_common_patches(
                show_startup_depth_snapshot=False,
                enable_agent_ops_heartbeat=False,
            )

        startup_snapshot_mock.assert_not_awaited()

    async def test_run_agent_calls_startup_depth_snapshot_when_enabled(self) -> None:
        # Performs test run agent calls startup depth snapshot when enabled logic.
        with patch(
            "main.print_depth_coverage_startup_snapshot",
            new=AsyncMock(return_value=None),
        ) as startup_snapshot_mock:
            await self._run_agent_with_common_patches(
                show_startup_depth_snapshot=True,
                depth_coverage_sample_limit=77,
                enable_agent_ops_heartbeat=False,
            )

        startup_snapshot_mock.assert_awaited_once()
        startup_snapshot_mock.assert_awaited_once_with(
            unittest.mock.ANY,
            sample_limit=77,
        )

    async def test_run_agent_skips_startup_pipeline_contract_snapshot_when_disabled(self) -> None:
        # Performs test run agent skips startup pipeline contract snapshot when disabled logic.
        with patch(
            "main.print_pipeline_contract_startup_snapshot",
            new=AsyncMock(return_value=None),
        ) as startup_contract_mock:
            await self._run_agent_with_common_patches(
                show_startup_pipeline_contract_snapshot=False,
                show_startup_depth_snapshot=False,
                enable_agent_ops_heartbeat=False,
            )

        startup_contract_mock.assert_not_awaited()

    async def test_run_agent_calls_startup_pipeline_contract_snapshot_when_enabled(self) -> None:
        # Performs test run agent calls startup pipeline contract snapshot when enabled logic.
        with patch(
            "main.print_pipeline_contract_startup_snapshot",
            new=AsyncMock(return_value=None),
        ) as startup_contract_mock:
            await self._run_agent_with_common_patches(
                show_startup_pipeline_contract_snapshot=True,
                show_startup_depth_snapshot=False,
                enable_agent_ops_heartbeat=False,
            )

        startup_contract_mock.assert_awaited_once()
        startup_contract_mock.assert_awaited_once_with(
            unittest.mock.ANY,
            max_observation_age_seconds=3.0,
            require_live_pipeline=False,
        )

    async def test_run_agent_suppresses_startup_banner_when_disabled(self) -> None:
        # Performs test run agent suppresses startup banner when disabled logic.
        output = await self._run_agent_with_common_patches(
            show_startup_banner=False,
            show_startup_depth_snapshot=False,
            enable_agent_ops_heartbeat=False,
        )
        self.assertNotIn("Initializing MCP pipeline and connecting to server.py", output)
        self.assertNotIn("Runtime config:", output)

    def test_format_runtime_config_banner_enabled_summary(self) -> None:
        # Performs test format runtime config banner enabled summary logic.
        text = main.format_runtime_config_banner(
            enable_agent_ops_heartbeat=True,
            agent_ops_interval_seconds=2.5,
            agent_ops_max_tracks=4,
            agent_ops_summary_every_cycles=3,
            agent_ops_summary_window_size=50,
            depth_coverage_summary_enabled=True,
            depth_coverage_summary_every_cycles=4,
            depth_coverage_sample_limit=25,
        )

        self.assertIn("agent_ops_heartbeat=on", text)
        self.assertIn("heartbeat_interval_s=2.5", text)
        self.assertIn("heartbeat_max_tracks=4", text)
        self.assertIn("health_summary=every_3_cycles", text)
        self.assertIn("health_summary_window=50", text)
        self.assertIn("depth_summary=every_4_cycles", text)
        self.assertIn("depth_sample_limit=25", text)

    def test_format_runtime_config_banner_disabled_summary(self) -> None:
        # Performs test format runtime config banner disabled summary logic.
        text = main.format_runtime_config_banner(
            enable_agent_ops_heartbeat=False,
            agent_ops_interval_seconds=10.0,
            agent_ops_max_tracks=10,
            agent_ops_summary_every_cycles=0,
            agent_ops_summary_window_size=100,
            depth_coverage_summary_enabled=False,
            depth_coverage_summary_every_cycles=0,
            depth_coverage_sample_limit=50,
        )

        self.assertIn("agent_ops_heartbeat=off", text)
        self.assertIn("health_summary=disabled", text)
        self.assertIn("depth_summary=disabled", text)


if __name__ == "__main__":
    unittest.main()
