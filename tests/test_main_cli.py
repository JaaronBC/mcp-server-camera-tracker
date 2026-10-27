from __future__ import annotations

import unittest

import main


class TestMainCliArgs(unittest.TestCase):
    def test_parse_cli_args_defaults(self) -> None:
        options = main.parse_cli_args([])
        self.assertFalse(options.no_milestone)
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

    def test_parse_cli_args_no_milestone(self) -> None:
        options = main.parse_cli_args(["--no-milestone"])
        self.assertTrue(options.no_milestone)

    def test_parse_cli_args_agent_ops_overrides(self) -> None:
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

    def test_parse_cli_args_rejects_non_positive_values(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
