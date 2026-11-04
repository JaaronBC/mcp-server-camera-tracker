# Handles test main cli behavior.
from __future__ import annotations

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import main
from agents.orchestration.runtime import app as runtime_app


class TestMainCliArgs(unittest.TestCase):
    def test_parse_cli_args_defaults(self) -> None:
        # Performs test parse cli args defaults logic.
        with patch.dict("main.os.environ", {}, clear=False):
            options = main.parse_cli_args([])
        self.assertFalse(options.no_startup_depth_snapshot)
        self.assertFalse(options.quiet_startup_banner)
        self.assertFalse(options.disable_agent_ops_heartbeat)
        self.assertEqual(options.agent_ops_interval_seconds, 10)
        self.assertEqual(options.agent_ops_max_tracks, 10)
        self.assertEqual(options.agent_ops_summary_every_cycles, 5)
        self.assertEqual(options.agent_ops_summary_window_size, 100)
        self.assertFalse(options.disable_depth_coverage_summary)
        self.assertEqual(options.depth_coverage_summary_every_cycles, 5)
        self.assertEqual(options.depth_coverage_sample_limit, 50)
        self.assertEqual(options.handoff_degraded_warn_rate, 0.1)
        self.assertEqual(options.handoff_degraded_critical_rate, 0.3)

    def test_parse_cli_args_agent_ops_overrides(self) -> None:
        # Performs test parse cli args agent ops overrides logic.
        options = main.parse_cli_args(
            [
                "--disable-agent-ops-heartbeat",
                "--agent-ops-interval-seconds",
                "2.5",
                "--agent-ops-max-tracks",
                "4",
                "--agent-ops-summary-every-cycles",
                "3",
                "--agent-ops-summary-window-size",
                "50",
                "--no-startup-depth-snapshot",
                "--quiet-startup-banner",
                "--disable-depth-coverage-summary",
                "--depth-coverage-summary-every-cycles",
                "7",
                "--depth-coverage-sample-limit",
                "120",
                "--handoff-degraded-warn-rate",
                "0.2",
                "--handoff-degraded-critical-rate",
                "0.6",
            ]
        )
        self.assertTrue(options.no_startup_depth_snapshot)
        self.assertTrue(options.quiet_startup_banner)
        self.assertTrue(options.disable_agent_ops_heartbeat)
        self.assertEqual(options.agent_ops_interval_seconds, 2.5)
        self.assertEqual(options.agent_ops_max_tracks, 4)
        self.assertEqual(options.agent_ops_summary_every_cycles, 3)
        self.assertEqual(options.agent_ops_summary_window_size, 50)
        self.assertTrue(options.disable_depth_coverage_summary)
        self.assertEqual(options.depth_coverage_summary_every_cycles, 7)
        self.assertEqual(options.depth_coverage_sample_limit, 120)
        self.assertEqual(options.handoff_degraded_warn_rate, 0.2)
        self.assertEqual(options.handoff_degraded_critical_rate, 0.6)

    def test_parse_cli_args_handoff_thresholds_from_env(self) -> None:
        # Performs test parse cli args handoff thresholds from env logic.
        with patch.dict(
            "main.os.environ",
            {
                "HANDOFF_DEGRADED_WARN_RATE": "0.15",
                "HANDOFF_DEGRADED_CRITICAL_RATE": "0.45",
            },
            clear=False,
        ):
            options = main.parse_cli_args([])

        self.assertEqual(options.handoff_degraded_warn_rate, 0.15)
        self.assertEqual(options.handoff_degraded_critical_rate, 0.45)

    def test_parse_cli_args_rejects_non_positive_values(self) -> None:
        # Performs test parse cli args rejects non positive values logic.
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--agent-ops-interval-seconds", "0"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--agent-ops-max-tracks", "0"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--agent-ops-summary-every-cycles", "-1"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--agent-ops-summary-window-size", "0"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--depth-coverage-summary-every-cycles", "-1"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--depth-coverage-sample-limit", "0"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--handoff-degraded-warn-rate", "-0.1"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(["--handoff-degraded-critical-rate", "1.1"])
        with self.assertRaises(SystemExit):
            main.parse_cli_args(
                [
                    "--handoff-degraded-warn-rate",
                    "0.5",
                    "--handoff-degraded-critical-rate",
                    "0.4",
                ]
            )

    def test_verification_request_timeout_env_config(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERIFICATION_REQUEST_TIMEOUT_SECONDS", None)
            importlib.reload(main)
            self.assertEqual(main.VERIFICATION_REQUEST_TIMEOUT_SECONDS, 8.0)

        with patch.dict(os.environ, {"VERIFICATION_REQUEST_TIMEOUT_SECONDS": "20.0"}, clear=False):
            importlib.reload(main)
            self.assertEqual(main.VERIFICATION_REQUEST_TIMEOUT_SECONDS, 20.0)

        invalid_values = ["0", "-1", "nan", "inf", "-inf", "not-a-number"]
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with patch.dict(os.environ, {"VERIFICATION_REQUEST_TIMEOUT_SECONDS": invalid}, clear=False):
                    with self.assertRaisesRegex(
                        ValueError,
                        "VERIFICATION_REQUEST_TIMEOUT_SECONDS",
                    ):
                        importlib.reload(main)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERIFICATION_REQUEST_TIMEOUT_SECONDS", None)
            importlib.reload(main)


class TestShutdownLifecycle(unittest.IsolatedAsyncioTestCase):
    async def test_run_system_with_config_attempts_services_stop_when_operator_stop_fails(self) -> None:
        config = runtime_app.RuntimeConfig(
            show_startup_banner=False,
            show_startup_depth_snapshot=False,
            show_startup_pipeline_contract_snapshot=False,
            enable_agent_ops_heartbeat=False,
            agent_ops_interval_seconds=10.0,
            agent_ops_max_tracks=10,
            agent_ops_summary_every_cycles=5,
            agent_ops_summary_window_size=100,
            handoff_degraded_warn_rate=0.1,
            handoff_degraded_critical_rate=0.3,
            depth_coverage_summary_enabled=False,
            depth_coverage_summary_every_cycles=5,
            depth_coverage_sample_limit=50,
        )

        services = SimpleNamespace(
            tracking_agent="tracking_agent",
            observation_agent="observation_agent",
            reasoning_agent="reasoning_agent",
            observation_source=object(),
            start=AsyncMock(return_value=None),
            stop=AsyncMock(return_value=None),
        )

        operator = SimpleNamespace(
            run=AsyncMock(return_value=None),
            stop=AsyncMock(side_effect=RuntimeError("operator stop failed")),
        )

        with (
            patch("main.runtime_app.configure_logging", return_value=MagicMock()),
            patch("main.runtime_app.initialize_services", AsyncMock(return_value=services)),
            patch("main.runtime_app.RuntimeLoopCoordinator", return_value=MagicMock()),
            patch("main.runtime_app.OrchestrationAgent", return_value=operator),
        ):
            with self.assertRaisesRegex(RuntimeError, "operator stop failed"):
                await main.run_system_with_config(config)

        services.stop.assert_awaited_once()
        operator.stop.assert_awaited_once()

    async def test_runtime_services_stop_attempts_stdio_after_session_stop_failure(self) -> None:
        config = runtime_app.RuntimeConfig(
            show_startup_banner=False,
            show_startup_depth_snapshot=False,
            show_startup_pipeline_contract_snapshot=False,
            enable_agent_ops_heartbeat=False,
            agent_ops_interval_seconds=10.0,
            agent_ops_max_tracks=10,
            agent_ops_summary_every_cycles=5,
            agent_ops_summary_window_size=100,
            handoff_degraded_warn_rate=0.1,
            handoff_degraded_critical_rate=0.3,
            depth_coverage_summary_enabled=False,
            depth_coverage_summary_every_cycles=5,
            depth_coverage_sample_limit=50,
        )
        services = runtime_app.RuntimeServices(
            config,
            MagicMock(),
            server_path=main.SERVER_PATH,
            python_executable="python",
            stdio_client_fn=MagicMock(),
            client_session_cls=MagicMock(),
            startup_runner=AsyncMock(return_value=None),
        )

        session_context = MagicMock()
        session_context.__aexit__ = AsyncMock(side_effect=RuntimeError("session close failed"))
        stdio_context = MagicMock()
        stdio_context.__aexit__ = AsyncMock(return_value=None)

        services._session_context = session_context
        services._stdio_context = stdio_context
        services.session = object()
        services.observation_source = object()

        with self.assertRaisesRegex(RuntimeError, "session close failed"):
            await services.stop()

        session_context.__aexit__.assert_awaited_once_with(None, None, None)
        stdio_context.__aexit__.assert_awaited_once_with(None, None, None)
        self.assertIsNone(services._session_context)
        self.assertIsNone(services._stdio_context)
        self.assertIsNone(services.session)
        self.assertIsNone(services.observation_source)


if __name__ == "__main__":
    unittest.main()
