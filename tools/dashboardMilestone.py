from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _file_contains(path: Path, needle: str) -> bool:
    if not path.is_file():
        return False
    return needle in path.read_text(encoding="utf-8")


def _build_progress_payload(
    phase: str,
    checks: list[dict[str, Any]],
    next_actions_when_incomplete: list[str],
) -> dict[str, Any]:
    completed = sum(1 for item in checks if item["complete"])
    total = len(checks)
    percent = round((completed / total) * 100, 1) if total else 0.0
    milestone_finished = completed == total and total > 0

    if milestone_finished:
        status = "finished"
    elif percent >= 80:
        status = "on_track"
    else:
        status = "in_progress"

    if milestone_finished:
        next_actions = [
            "Milestone complete. Proceed to next milestone planning and implementation.",
        ]
    else:
        next_actions = next_actions_when_incomplete

    return {
        "phase": phase,
        "status": status,
        "milestone_finished": milestone_finished,
        "completion_percent": percent,
        "completed_checks": completed,
        "total_checks": total,
        "checks": checks,
        "next_recommended_actions": next_actions,
    }


def _evaluate_milestone(root: Path) -> dict[str, Any]:
    readme = root / "README.md"
    dashboard = root / "tools" / "components" / "liveTrackingDisplay.py"
    control = root / "tools" / "dashboardControl.py"
    health = root / "tools" / "dashboardHealth.py"
    calibration = root / "tools" / "components" / "calibration.py"
    agent_contracts = root / "tools" / "agent_contracts.py"
    agent_coordinator = root / "tools" / "agent_coordinator.py"
    geometry_tests = root / "tests" / "test_geometry.py"
    calibration_tests = root / "tests" / "test_calibration.py"
    coordinator_tests = root / "tests" / "test_agent_coordinator.py"

    checks = [
        {
            "id": "dual_panel_dashboard",
            "description": "Webcam + radar dashboard renderer exists",
            "complete": _file_contains(dashboard, "cv2.hconcat") and _file_contains(dashboard, "draw_radar_panel"),
        },
        {
            "id": "simulation_fallback",
            "description": "Simulated track fallback is implemented",
            "complete": _file_contains(dashboard, "simulation_fallback") and _file_contains(dashboard, "build_simulated_tracks"),
        },
        {
            "id": "typed_observation_contract",
            "description": "Typed dashboard observation contracts are present",
            "complete": (root / "tools" / "components" / "dashboard_types.py").is_file(),
        },
        {
            "id": "coordinate_mapping_tests",
            "description": "Coordinate mapping helper tests exist",
            "complete": geometry_tests.is_file(),
        },
        {
            "id": "calibrated_ground_plane",
            "description": "Optional homography-based calibrated mapping exists",
            "complete": calibration.is_file() and calibration_tests.is_file() and _file_contains(dashboard, "GROUND_CALIBRATION_PATH"),
        },
        {
            "id": "mcp_lifecycle_orchestration",
            "description": "MCP lifecycle control tools exist",
            "complete": _file_contains(control, "dashboard_bootstrap") and _file_contains(control, "dashboard_lifecycle_status"),
        },
        {
            "id": "health_readiness_gates",
            "description": "Health and readiness polling tools exist",
            "complete": _file_contains(health, "dashboard_health_status") and _file_contains(health, "wait_for_dashboard_ready"),
        },
        {
            "id": "udp_tracking_compatibility",
            "description": "Dashboard publishes UDP tracking messages for compatibility",
            "complete": _file_contains(dashboard, "UDP_TRACKING_PORT") and _file_contains(dashboard, "build_tracking_message"),
        },
        {
            "id": "dashboard_docs",
            "description": "README documents dashboard workflow",
            "complete": _file_contains(readme, "Object Tracking Dashboard Demo") and _file_contains(readme, "ground-plane calibration"),
        },
        {
            "id": "three_agent_pipeline",
            "description": "Tracking/Observation/Reasoning agent pipeline implemented",
            "complete": (
                agent_contracts.is_file()
                and agent_coordinator.is_file()
                and coordinator_tests.is_file()
            ),
        },
    ]

    return _build_progress_payload(
        phase="milestone_1_dashboard_flow",
        checks=checks,
        next_actions_when_incomplete=[
            "Implement Observation and Reasoning agent contracts with tests.",
            "Add depth-estimate integration path and quality flags.",
            "Add repeatable latency/continuity evaluation script for demo metrics.",
        ],
    )


def _evaluate_milestone_2(root: Path) -> dict[str, Any]:
    readme = root / "README.md"
    main_py = root / "main.py"
    agent_ops = root / "tools" / "dashboardAgentOps.py"
    reasoning = root / "tools" / "dashboardReasoning.py"
    coordinator = root / "tools" / "agent_coordinator.py"

    checks = [
        {
            "id": "agent_contracts_and_coordinator",
            "description": "Agent contracts and bounded coordinator are present",
            "complete": (
                (root / "tools" / "agent_contracts.py").is_file()
                and coordinator.is_file()
                and _file_contains(coordinator, "max_evidence_requests_per_track")
            ),
        },
        {
            "id": "reasoning_pipeline_tools",
            "description": "Observation, reasoning, and evidence-request tools are implemented",
            "complete": (
                reasoning.is_file()
                and _file_contains(reasoning, "simulate_observation_summary")
                and _file_contains(reasoning, "simulate_reasoning_assessment")
                and _file_contains(reasoning, "simulate_evidence_request")
            ),
        },
        {
            "id": "agent_event_logging",
            "description": "Reasoning cycle events are logged with status and latency",
            "complete": (
                agent_ops.is_file()
                and _file_contains(agent_ops, "run_reasoning_cycle_and_log")
                and _file_contains(agent_ops, "latency_ms")
                and _file_contains(agent_ops, "status")
            ),
        },
        {
            "id": "agent_event_health_summary",
            "description": "Event health summary metrics tool is implemented",
            "complete": (
                agent_ops.is_file()
                and _file_contains(agent_ops, "summarize_agent_event_health")
                and _file_contains(agent_ops, "p95_ms")
                and _file_contains(agent_ops, "success_rate")
            ),
        },
        {
            "id": "main_heartbeat_integration",
            "description": "Main runtime triggers periodic reasoning heartbeat cycles",
            "complete": (
                main_py.is_file()
                and _file_contains(main_py, "AGENT_OPS_CYCLE_TOOL")
                and _file_contains(main_py, "run_agent_ops_cycle")
                and _file_contains(main_py, "agent_ops_interval_seconds")
            ),
        },
        {
            "id": "main_health_summary_integration",
            "description": "Main runtime prints periodic agent health summaries",
            "complete": (
                main_py.is_file()
                and _file_contains(main_py, "AGENT_OPS_HEALTH_SUMMARY_TOOL")
                and _file_contains(main_py, "run_agent_ops_health_summary")
                and _file_contains(main_py, "format_agent_ops_health_summary")
            ),
        },
        {
            "id": "runtime_config_controls",
            "description": "CLI controls exist for heartbeat and summary cadence/window",
            "complete": (
                main_py.is_file()
                and _file_contains(main_py, "--disable-agent-ops-heartbeat")
                and _file_contains(main_py, "--agent-ops-summary-every-cycles")
                and _file_contains(main_py, "--agent-ops-summary-window-size")
            ),
        },
        {
            "id": "milestone_2_tests_and_docs",
            "description": "Tests and README documentation cover milestone 2 tools",
            "complete": (
                (root / "tests" / "test_agent_coordinator.py").is_file()
                and (root / "tests" / "test_dashboard_reasoning.py").is_file()
                and (root / "tests" / "test_dashboard_agent_ops.py").is_file()
                and (root / "tests" / "test_main_agent_ops.py").is_file()
                and (root / "tests" / "test_main_cli.py").is_file()
                and _file_contains(readme, "run_reasoning_cycle_and_log")
                and _file_contains(readme, "summarize_agent_event_health")
            ),
        },
    ]

    return _build_progress_payload(
        phase="milestone_2_agent_observability",
        checks=checks,
        next_actions_when_incomplete=[
            "Fill remaining Milestone 2 agent-observability checks.",
            "Add optional runtime archival/rotation for agent event logs.",
            "Validate heartbeat health metrics against longer real-session traces.",
        ],
    )


def _evaluate_milestone_3(root: Path) -> dict[str, Any]:
    readme = root / "README.md"
    dashboard = root / "tools" / "components" / "liveTrackingDisplay.py"
    visual_agent = root / "tools" / "visual_tracking_agent.py"
    visual_tool = root / "tools" / "dashboardVisualTracking.py"
    coordinator = root / "tools" / "agent_coordinator.py"
    contracts = root / "tools" / "agent_contracts.py"
    metrics = root / "tools" / "dashboardMetrics.py"
    agent_ops = root / "tools" / "dashboardAgentOps.py"

    checks = [
        {
            "id": "vlm_tracking_runtime",
            "description": "Vision-language tracking runtime is implemented",
            "complete": (
                visual_agent.is_file()
                and _file_contains(visual_agent, "GeminiVisualTrackingAgent")
                and _file_contains(visual_agent, "AsyncVisualTrackingRuntime")
            ),
        },
        {
            "id": "structured_perception_contracts",
            "description": "Structured detection/proposal contracts include depth and uncertainty fields",
            "complete": (
                contracts.is_file()
                and _file_contains(contracts, "DetectionCandidate")
                and _file_contains(contracts, "relative_depth")
                and _file_contains(contracts, "uncertainties")
            ),
        },
        {
            "id": "coordinator_validation_guards",
            "description": "Coordinator validates visual-association proposals with conflict safeguards",
            "complete": (
                coordinator.is_file()
                and _file_contains(coordinator, "validate_visual_association_proposal")
                and _file_contains(coordinator, "depth_conflict_with_reliable_measurement")
            ),
        },
        {
            "id": "dashboard_async_integration",
            "description": "Dashboard integrates async VLM proposals without blocking primary tracking",
            "complete": (
                dashboard.is_file()
                and _file_contains(dashboard, "AsyncVisualTrackingRuntime")
                and _file_contains(dashboard, "build_tracking_agent_status_text")
                and _file_contains(dashboard, "build_tracking_debug_toggle_hint")
            ),
        },
        {
            "id": "visual_tracking_mcp_tool",
            "description": "MCP tool exposes simulated visual tracking association flow",
            "complete": (
                visual_tool.is_file()
                and _file_contains(visual_tool, "simulate_visual_tracking_association")
            ),
        },
        {
            "id": "runtime_resilience_tests",
            "description": "VLM runtime tests cover timeout/unavailable and conflict fallback paths",
            "complete": (
                (root / "tests" / "test_visual_tracking_runtime.py").is_file()
                and _file_contains(root / "tests" / "test_visual_tracking_runtime.py", "vlm_unavailable")
                and _file_contains(root / "tests" / "test_visual_tracking_runtime.py", "depth_conflict")
            ),
        },
        {
            "id": "evaluation_metrics_tools",
            "description": "Latency/continuity metrics and event health summaries are available",
            "complete": (
                metrics.is_file()
                and _file_contains(metrics, "dashboard_latency_continuity_metrics")
                and agent_ops.is_file()
                and _file_contains(agent_ops, "summarize_agent_event_health")
            ),
        },
        {
            "id": "milestone_3_docs",
            "description": "README documents visual tracking association and diagnostics controls",
            "complete": (
                _file_contains(readme, "simulate_visual_tracking_association")
                and _file_contains(readme, "gemini-3.5-flash-lite")
                and _file_contains(readme, "toggle tracking-agent diagnostics")
            ),
        },
    ]

    return _build_progress_payload(
        phase="milestone_3_vlm_tracking_quality",
        checks=checks,
        next_actions_when_incomplete=[
            "Complete remaining VLM-tracking integration and coordinator validation checks.",
            "Increase depth-backed candidate coverage and conflict annotations in live observations.",
            "Run longer benchmark sessions and tune timeout/fallback thresholds from collected metrics.",
        ],
    )


@mcp.tool()
def milestone_progress_status() -> dict[str, Any]:
    """Return current dashboard milestone progress against design goals."""
    return _evaluate_milestone(PROJECT_ROOT)


@mcp.tool()
def milestone_2_progress_status() -> dict[str, Any]:
    """Return milestone 2 progress for agent pipeline and observability goals."""
    return _evaluate_milestone_2(PROJECT_ROOT)


@mcp.tool()
def milestone_3_progress_status() -> dict[str, Any]:
    """Return milestone 3 progress for VLM tracking and evaluation-hardening goals."""
    return _evaluate_milestone_3(PROJECT_ROOT)
