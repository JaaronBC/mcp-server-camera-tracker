# CLI/runtime configuration parsing helpers for the camera-tracker entrypoint.
from __future__ import annotations

import argparse


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def unit_interval_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def resolve_handoff_severity_thresholds(
    warn_raw: str | None,
    critical_raw: str | None,
    *,
    default_warn_rate: float,
    default_critical_rate: float,
) -> tuple[float, float]:
    warn_rate = default_warn_rate
    critical_rate = default_critical_rate

    if isinstance(warn_raw, str) and warn_raw.strip():
        try:
            warn_candidate = float(warn_raw)
        except ValueError:
            warn_candidate = default_warn_rate
        if 0 <= warn_candidate <= 1:
            warn_rate = warn_candidate

    if isinstance(critical_raw, str) and critical_raw.strip():
        try:
            critical_candidate = float(critical_raw)
        except ValueError:
            critical_candidate = default_critical_rate
        if 0 <= critical_candidate <= 1:
            critical_rate = critical_candidate

    if critical_rate < warn_rate:
        return default_warn_rate, default_critical_rate

    return warn_rate, critical_rate


def format_runtime_config_banner(
    *,
    enable_agent_ops_heartbeat: bool,
    agent_ops_interval_seconds: float,
    agent_ops_max_tracks: int,
    agent_ops_summary_every_cycles: int,
    agent_ops_summary_window_size: int,
    handoff_degraded_warn_rate: float,
    handoff_degraded_critical_rate: float,
    depth_coverage_summary_enabled: bool,
    depth_coverage_summary_every_cycles: int,
    depth_coverage_sample_limit: int,
) -> str:
    summary_mode = (
        "disabled"
        if agent_ops_summary_every_cycles == 0
        else f"every_{agent_ops_summary_every_cycles}_cycles"
    )
    depth_summary_mode = (
        "disabled"
        if (not depth_coverage_summary_enabled or depth_coverage_summary_every_cycles == 0)
        else f"every_{depth_coverage_summary_every_cycles}_cycles"
    )

    return (
        "Runtime config: "
        f"agent_ops_heartbeat={'on' if enable_agent_ops_heartbeat else 'off'} | "
        f"heartbeat_interval_s={agent_ops_interval_seconds} | "
        f"heartbeat_max_tracks={agent_ops_max_tracks} | "
        f"health_summary={summary_mode} | "
        f"health_summary_window={agent_ops_summary_window_size} | "
        f"handoff_warn_rate={handoff_degraded_warn_rate} | "
        f"handoff_critical_rate={handoff_degraded_critical_rate} | "
        f"depth_summary={depth_summary_mode} | "
        f"depth_sample_limit={depth_coverage_sample_limit}"
    )


def parse_cli_args(
    args: list[str],
    *,
    description: str,
    agent_ops_interval_seconds_default: float,
    agent_ops_max_tracks_default: int,
    agent_ops_summary_every_cycles_default: int,
    agent_ops_summary_window_size_default: int,
    depth_coverage_summary_every_cycles_default: int,
    depth_coverage_sample_limit_default: int,
    handoff_degraded_warn_rate_default: float,
    handoff_degraded_critical_rate_default: float,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--disable-agent-ops-heartbeat",
        action="store_true",
        help="Disable periodic agent-ops reasoning cycle during monitoring",
    )
    parser.add_argument(
        "--agent-ops-interval-seconds",
        type=positive_float,
        default=agent_ops_interval_seconds_default,
        help="Interval in seconds between agent-ops heartbeat cycles (default: 10)",
    )
    parser.add_argument(
        "--agent-ops-max-tracks",
        type=positive_int,
        default=agent_ops_max_tracks_default,
        help="Maximum tracks evaluated per agent-ops cycle (default: 10)",
    )
    parser.add_argument(
        "--agent-ops-summary-every-cycles",
        type=non_negative_int,
        default=agent_ops_summary_every_cycles_default,
        help=(
            "Print heartbeat health summary every N cycles (default: 5, "
            "0 disables periodic summary output)"
        ),
    )
    parser.add_argument(
        "--agent-ops-summary-window-size",
        type=positive_int,
        default=agent_ops_summary_window_size_default,
        help="Event window size for heartbeat health summary (default: 100)",
    )
    parser.add_argument(
        "--disable-depth-coverage-summary",
        action="store_true",
        help="Disable periodic relative-depth coverage summary output",
    )
    parser.add_argument(
        "--depth-coverage-summary-every-cycles",
        type=non_negative_int,
        default=depth_coverage_summary_every_cycles_default,
        help=(
            "Print depth-coverage summary every N heartbeat cycles "
            "(default: 5, 0 disables periodic summary output)"
        ),
    )
    parser.add_argument(
        "--depth-coverage-sample-limit",
        type=positive_int,
        default=depth_coverage_sample_limit_default,
        help="Sample window for depth-coverage trend metrics (default: 50)",
    )
    parser.add_argument(
        "--no-startup-depth-snapshot",
        action="store_true",
        help="Skip one-time startup depth-coverage baseline snapshot output",
    )
    parser.add_argument(
        "--quiet-startup-banner",
        action="store_true",
        help="Suppress startup banner prints in main runtime",
    )
    parser.add_argument(
        "--handoff-degraded-warn-rate",
        type=unit_interval_float,
        default=handoff_degraded_warn_rate_default,
        help=(
            "Warn threshold for handoff degraded rate in [0,1] "
            f"(default: {handoff_degraded_warn_rate_default})"
        ),
    )
    parser.add_argument(
        "--handoff-degraded-critical-rate",
        type=unit_interval_float,
        default=handoff_degraded_critical_rate_default,
        help=(
            "Critical threshold for handoff degraded rate in [0,1] "
            f"(default: {handoff_degraded_critical_rate_default})"
        ),
    )

    options = parser.parse_args(args)
    if options.handoff_degraded_critical_rate < options.handoff_degraded_warn_rate:
        parser.error("--handoff-degraded-critical-rate must be greater than or equal to --handoff-degraded-warn-rate")
    return options
