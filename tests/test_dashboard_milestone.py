from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import dashboardMilestone


class TestDashboardMilestone(unittest.TestCase):
    def test_evaluate_milestone_with_missing_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = dashboardMilestone._evaluate_milestone(root)

        self.assertEqual(result["phase"], "milestone_1_dashboard_flow")
        self.assertEqual(result["total_checks"], 10)
        self.assertLess(result["completion_percent"], 50.0)
        self.assertFalse(result["milestone_finished"])

    def test_evaluate_milestone_with_expected_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools" / "components").mkdir(parents=True, exist_ok=True)
            (root / "tests").mkdir(parents=True, exist_ok=True)

            (root / "README.md").write_text(
                "Object Tracking Dashboard Demo\noptional ground-plane calibration",
                encoding="utf-8",
            )
            (root / "tools" / "components" / "liveTrackingDisplay.py").write_text(
                "cv2.hconcat\ndraw_radar_panel\nsimulation_fallback\nbuild_simulated_tracks\n"
                "GROUND_CALIBRATION_PATH\nUDP_TRACKING_PORT\nbuild_tracking_message",
                encoding="utf-8",
            )
            (root / "tools" / "components" / "dashboard_types.py").write_text("ok", encoding="utf-8")
            (root / "tools" / "components" / "calibration.py").write_text("ok", encoding="utf-8")
            (root / "tools" / "dashboardControl.py").write_text(
                "dashboard_bootstrap\ndashboard_lifecycle_status",
                encoding="utf-8",
            )
            (root / "tools" / "dashboardHealth.py").write_text(
                "dashboard_health_status\nwait_for_dashboard_ready",
                encoding="utf-8",
            )
            (root / "tests" / "test_geometry.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_calibration.py").write_text("ok", encoding="utf-8")

            result = dashboardMilestone._evaluate_milestone(root)

        self.assertEqual(result["total_checks"], 10)
        self.assertEqual(result["completed_checks"], 9)
        self.assertEqual(result["completion_percent"], 90.0)
        self.assertEqual(result["status"], "on_track")
        self.assertFalse(result["milestone_finished"])

    def test_evaluate_milestone_reports_finished_when_all_checks_complete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools" / "components").mkdir(parents=True, exist_ok=True)
            (root / "tests").mkdir(parents=True, exist_ok=True)

            (root / "README.md").write_text(
                "Object Tracking Dashboard Demo\noptional ground-plane calibration",
                encoding="utf-8",
            )
            (root / "tools" / "components" / "liveTrackingDisplay.py").write_text(
                "cv2.hconcat\ndraw_radar_panel\nsimulation_fallback\nbuild_simulated_tracks\n"
                "GROUND_CALIBRATION_PATH\nUDP_TRACKING_PORT\nbuild_tracking_message",
                encoding="utf-8",
            )
            (root / "tools" / "components" / "dashboard_types.py").write_text("ok", encoding="utf-8")
            (root / "tools" / "components" / "calibration.py").write_text("ok", encoding="utf-8")
            (root / "tools" / "dashboardControl.py").write_text(
                "dashboard_bootstrap\ndashboard_lifecycle_status",
                encoding="utf-8",
            )
            (root / "tools" / "dashboardHealth.py").write_text(
                "dashboard_health_status\nwait_for_dashboard_ready",
                encoding="utf-8",
            )
            (root / "tools" / "agent_contracts.py").write_text("ok", encoding="utf-8")
            (root / "tools" / "agent_coordinator.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_geometry.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_calibration.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_agent_coordinator.py").write_text("ok", encoding="utf-8")

            result = dashboardMilestone._evaluate_milestone(root)

        self.assertEqual(result["completed_checks"], 10)
        self.assertEqual(result["completion_percent"], 100.0)
        self.assertEqual(result["status"], "finished")
        self.assertTrue(result["milestone_finished"])

    def test_evaluate_milestone_2_with_missing_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = dashboardMilestone._evaluate_milestone_2(root)

        self.assertEqual(result["phase"], "milestone_2_agent_observability")
        self.assertEqual(result["total_checks"], 8)
        self.assertLess(result["completion_percent"], 50.0)
        self.assertFalse(result["milestone_finished"])

    def test_evaluate_milestone_2_reports_finished_when_all_checks_complete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools").mkdir(parents=True, exist_ok=True)
            (root / "tests").mkdir(parents=True, exist_ok=True)

            (root / "README.md").write_text(
                "run_reasoning_cycle_and_log\nsummarize_agent_event_health",
                encoding="utf-8",
            )
            (root / "main.py").write_text(
                "\n".join(
                    [
                        "AGENT_OPS_CYCLE_TOOL",
                        "run_agent_ops_cycle",
                        "agent_ops_interval_seconds",
                        "AGENT_OPS_HEALTH_SUMMARY_TOOL",
                        "run_agent_ops_health_summary",
                        "format_agent_ops_health_summary",
                        "--disable-agent-ops-heartbeat",
                        "--agent-ops-summary-every-cycles",
                        "--agent-ops-summary-window-size",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "tools" / "agent_contracts.py").write_text("ok", encoding="utf-8")
            (root / "tools" / "agent_coordinator.py").write_text(
                "max_evidence_requests_per_track",
                encoding="utf-8",
            )
            (root / "tools" / "dashboardReasoning.py").write_text(
                "simulate_observation_summary\nsimulate_reasoning_assessment\nsimulate_evidence_request",
                encoding="utf-8",
            )
            (root / "tools" / "dashboardAgentOps.py").write_text(
                "run_reasoning_cycle_and_log\nlatency_ms\nstatus\n"
                "summarize_agent_event_health\np95_ms\nsuccess_rate",
                encoding="utf-8",
            )
            (root / "tests" / "test_agent_coordinator.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_dashboard_reasoning.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_dashboard_agent_ops.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_main_agent_ops.py").write_text("ok", encoding="utf-8")
            (root / "tests" / "test_main_cli.py").write_text("ok", encoding="utf-8")

            result = dashboardMilestone._evaluate_milestone_2(root)

        self.assertEqual(result["completed_checks"], 8)
        self.assertEqual(result["completion_percent"], 100.0)
        self.assertEqual(result["status"], "finished")
        self.assertTrue(result["milestone_finished"])

    def test_evaluate_milestone_3_with_missing_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = dashboardMilestone._evaluate_milestone_3(root)

        self.assertEqual(result["phase"], "milestone_3_vlm_tracking_quality")
        self.assertEqual(result["total_checks"], 8)
        self.assertLess(result["completion_percent"], 50.0)
        self.assertFalse(result["milestone_finished"])

    def test_evaluate_milestone_3_reports_finished_when_all_checks_complete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tools" / "components").mkdir(parents=True, exist_ok=True)
            (root / "tests").mkdir(parents=True, exist_ok=True)

            (root / "README.md").write_text(
                "simulate_visual_tracking_association\n"
                "qwen-3.8-27b\n"
                "toggle tracking-agent diagnostics",
                encoding="utf-8",
            )
            (root / "tools" / "visual_tracking_agent.py").write_text(
                "propose_associations\nAsyncVisualTrackingRuntime",
                encoding="utf-8",
            )
            (root / "tools" / "agent_contracts.py").write_text(
                "DetectionCandidate\nrelative_depth\nuncertainties",
                encoding="utf-8",
            )
            (root / "tools" / "agent_coordinator.py").write_text(
                "validate_visual_association_proposal\ndepth_conflict_with_reliable_measurement",
                encoding="utf-8",
            )
            (root / "tools" / "components" / "liveTrackingDisplay.py").write_text(
                "AsyncVisualTrackingRuntime\n"
                "build_tracking_agent_status_text\n"
                "build_tracking_debug_toggle_hint",
                encoding="utf-8",
            )
            (root / "tools" / "dashboardVisualTracking.py").write_text(
                "simulate_visual_tracking_association",
                encoding="utf-8",
            )
            (root / "tools" / "dashboardMetrics.py").write_text(
                "dashboard_latency_continuity_metrics",
                encoding="utf-8",
            )
            (root / "tools" / "dashboardAgentOps.py").write_text(
                "summarize_agent_event_health",
                encoding="utf-8",
            )
            (root / "tests" / "test_visual_tracking_runtime.py").write_text(
                "vlm_unavailable\ndepth_conflict",
                encoding="utf-8",
            )

            result = dashboardMilestone._evaluate_milestone_3(root)

        self.assertEqual(result["completed_checks"], 8)
        self.assertEqual(result["completion_percent"], 100.0)
        self.assertEqual(result["status"], "finished")
        self.assertTrue(result["milestone_finished"])


if __name__ == "__main__":
    unittest.main()
