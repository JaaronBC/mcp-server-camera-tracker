from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import AsyncMock, MagicMock, patch

import main


class FakeItem:
    def __init__(self, text: str | None = None, json_payload=None):
        self.text = text
        self.json = json_payload


class FakeResult:
    def __init__(self, content, is_error: bool = False):
        self.content = content
        self.is_error = is_error


class FakeSession:
    def __init__(self, result: FakeResult | None = None, error: Exception | None = None):
        self._result = result
        self._error = error

    async def call_tool(self, tool_name, arguments):
        if self._error is not None:
            raise self._error
        return self._result


class FakeStdioContext:
    async def __aenter__(self):
        return (object(), object())

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeClientSessionContext:
    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        return None


class TestMainAgentOps(unittest.IsolatedAsyncioTestCase):
    async def _run_agent_with_common_patches(self, **kwargs) -> str:
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
        result = FakeResult([FakeItem(text="Unknown tool: run_reasoning_cycle_and_log")], is_error=True)
        session = FakeSession(result=result)

        enabled = await main.run_agent_ops_cycle(session)

        self.assertFalse(enabled)

    async def test_run_agent_ops_cycle_keeps_enabled_for_runtime_exception(self) -> None:
        session = FakeSession(error=RuntimeError("temporary MCP transport issue"))

        enabled = await main.run_agent_ops_cycle(session)

        self.assertTrue(enabled)

    def test_format_agent_ops_cycle(self) -> None:
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

    async def test_run_agent_ops_health_summary_success(self) -> None:
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
        result = FakeResult([FakeItem(text="Unknown tool: summarize_agent_event_health")], is_error=True)
        session = FakeSession(result=result)

        enabled = await main.run_agent_ops_health_summary(session)

        self.assertFalse(enabled)

    async def test_run_depth_coverage_summary_success(self) -> None:
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
        result = FakeResult([FakeItem(text="Unknown tool: dashboard_depth_coverage_metrics")], is_error=True)
        session = FakeSession(result=result)

        enabled = await main.run_depth_coverage_summary(session)

        self.assertFalse(enabled)

    def test_format_agent_ops_health_summary(self) -> None:
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

    def test_format_depth_coverage_summary(self) -> None:
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
        result = FakeResult([FakeItem(text="Unknown tool: dashboard_depth_coverage_metrics")], is_error=True)
        session = FakeSession(result=result)

        stream = io.StringIO()
        with redirect_stdout(stream):
            await main.print_depth_coverage_startup_snapshot(session, sample_limit=10)

        output = stream.getvalue()
        self.assertIn("Depth coverage snapshot unavailable", output)

    async def test_run_agent_skips_startup_depth_snapshot_when_disabled(self) -> None:
        with patch(
            "main.print_depth_coverage_startup_snapshot",
            new=AsyncMock(return_value=None),
        ) as startup_snapshot_mock:
            await self._run_agent_with_common_patches(
                show_milestone_snapshot=False,
                show_startup_depth_snapshot=False,
                enable_agent_ops_heartbeat=False,
            )

        startup_snapshot_mock.assert_not_awaited()

    async def test_run_agent_calls_startup_depth_snapshot_when_enabled(self) -> None:
        with patch(
            "main.print_depth_coverage_startup_snapshot",
            new=AsyncMock(return_value=None),
        ) as startup_snapshot_mock:
            await self._run_agent_with_common_patches(
                show_milestone_snapshot=False,
                show_startup_depth_snapshot=True,
                depth_coverage_sample_limit=77,
                enable_agent_ops_heartbeat=False,
            )

        startup_snapshot_mock.assert_awaited_once()
        startup_snapshot_mock.assert_awaited_once_with(
            unittest.mock.ANY,
            sample_limit=77,
        )

    async def test_run_agent_suppresses_startup_banner_when_disabled(self) -> None:
        output = await self._run_agent_with_common_patches(
            show_startup_banner=False,
            show_milestone_snapshot=False,
            show_startup_depth_snapshot=False,
            enable_agent_ops_heartbeat=False,
        )
        self.assertNotIn("Initializing MCP pipeline and connecting to server.py", output)
        self.assertNotIn("Runtime config:", output)

    def test_format_runtime_config_banner_enabled_summary(self) -> None:
        text = main.format_runtime_config_banner(
            show_milestone_snapshot=True,
            enable_agent_ops_heartbeat=True,
            agent_ops_interval_seconds=2.5,
            agent_ops_max_tracks=4,
            agent_ops_summary_every_cycles=3,
            agent_ops_summary_window_size=50,
            depth_coverage_summary_enabled=True,
            depth_coverage_summary_every_cycles=4,
            depth_coverage_sample_limit=25,
        )

        self.assertIn("milestone_snapshot=on", text)
        self.assertIn("agent_ops_heartbeat=on", text)
        self.assertIn("heartbeat_interval_s=2.5", text)
        self.assertIn("heartbeat_max_tracks=4", text)
        self.assertIn("health_summary=every_3_cycles", text)
        self.assertIn("health_summary_window=50", text)
        self.assertIn("depth_summary=every_4_cycles", text)
        self.assertIn("depth_sample_limit=25", text)

    def test_format_runtime_config_banner_disabled_summary(self) -> None:
        text = main.format_runtime_config_banner(
            show_milestone_snapshot=False,
            enable_agent_ops_heartbeat=False,
            agent_ops_interval_seconds=10.0,
            agent_ops_max_tracks=10,
            agent_ops_summary_every_cycles=0,
            agent_ops_summary_window_size=100,
            depth_coverage_summary_enabled=False,
            depth_coverage_summary_every_cycles=0,
            depth_coverage_sample_limit=50,
        )

        self.assertIn("milestone_snapshot=off", text)
        self.assertIn("agent_ops_heartbeat=off", text)
        self.assertIn("health_summary=disabled", text)
        self.assertIn("depth_summary=disabled", text)


if __name__ == "__main__":
    unittest.main()
