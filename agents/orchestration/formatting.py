# Orchestration-facing formatters for agent-ops cycle, health summaries, and depth coverage text.
from __future__ import annotations

import time
from typing import Any

AGENT_OPS_HEALTH_FRESHNESS_SECONDS = 30.0
HANDOFF_DEGRADED_WARN_RATE = 0.1
HANDOFF_DEGRADED_CRITICAL_RATE = 0.3


def format_agent_ops_cycle(payload: dict[str, Any]) -> str:
    # Formats agent ops cycle into concise operator-facing text.
    cycle = payload.get("cycle")
    if not isinstance(cycle, dict):
        return "Agent ops cycle: unavailable"

    evaluated_tracks = cycle.get("evaluated_tracks", "?")
    success_count = cycle.get("success_count", "?")
    error_count = cycle.get("error_count", "?")
    started_at = cycle.get("started_at", "?")
    event_log_writable = payload.get("event_log_writable")
    event_log_error_count = payload.get("event_log_error_count")

    suffix = ""
    if event_log_writable is False:
        suffix = f" | event_log=degraded(errors={event_log_error_count})"

    return (
        "Agent ops cycle: "
        f"tracks={evaluated_tracks} | ok={success_count} | "
        f"errors={error_count} | started_at={started_at}{suffix}"
    )


def format_agent_ops_cycle_evidence(payload: dict[str, Any]) -> list[str]:
    # Formats agent ops cycle evidence into concise operator-facing text.
    events = payload.get("events")
    if not isinstance(events, list):
        return []

    rendered: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue

        evidence_trace = event.get("evidence_trace")
        if not isinstance(evidence_trace, dict):
            continue

        latest_track_snapshot = evidence_trace.get("latest_track_snapshot")
        tracking_agent_context = evidence_trace.get("tracking_agent_context")
        coordinate_context = evidence_trace.get("coordinate_context")
        if not isinstance(latest_track_snapshot, dict):
            latest_track_snapshot = {}
        if not isinstance(tracking_agent_context, dict):
            tracking_agent_context = {}
        if not isinstance(coordinate_context, dict):
            coordinate_context = {}

        track_id = event.get("track_id", "?")

        ground_position = latest_track_snapshot.get("ground_position")
        if isinstance(ground_position, dict):
            ground_x = ground_position.get("x", "?")
            ground_z = ground_position.get("z", "?")
            ground_text = f"({ground_x},{ground_z})"
        else:
            ground_text = "n/a"

        relative_depth = latest_track_snapshot.get("relative_depth")
        depth_text = relative_depth if isinstance(relative_depth, (int, float)) else "n/a"

        vlm_model = tracking_agent_context.get("vlm_model", "n/a")
        selected_updates = tracking_agent_context.get("selected_updates")
        selected_updates_count = len(selected_updates) if isinstance(selected_updates, list) else 0

        coordinate_system = coordinate_context.get("coordinate_system", "n/a")
        calibration_status = coordinate_context.get("calibration_status", "n/a")

        rendered.append(
            "Agent ops evidence: "
            f"track={track_id} | latest_ground={ground_text} | depth={depth_text} | "
            f"tracking_vlm={vlm_model} | selected_updates={selected_updates_count} | "
            f"coord={coordinate_system} | calibration={calibration_status}"
        )

    return rendered


def format_agent_ops_health_summary(
    payload: dict[str, Any],
    _max_age_seconds: float = AGENT_OPS_HEALTH_FRESHNESS_SECONDS,
    _now_timestamp: float | None = None,
    _handoff_warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    _handoff_critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
) -> str:
    # Formats agent ops health summary into concise operator-facing text.
    status = payload.get("status")
    latency = payload.get("latency")
    if not isinstance(status, dict) or not isinstance(latency, dict):
        return "Agent ops health: unavailable"

    success_rate = status.get("success_rate", "?")
    error_rate = status.get("error_rate", "?")
    bounded_out_rate = status.get("bounded_out_rate", "?")
    p50_ms = latency.get("p50_ms", "?")
    p95_ms = latency.get("p95_ms", "?")

    health_is_fresh = _is_agent_health_payload_fresh(
        payload,
        max_age_seconds=_max_age_seconds,
        now_timestamp=_now_timestamp,
    )

    agent_error_digest = None
    agent_error_trends = None
    handoff_validation_trend = None
    handoff_status = None
    handoff_degraded_rate = None
    handoff_severity = None
    collaboration_signals = None
    if health_is_fresh:
        agent_error_digest = _format_agent_error_digest(payload.get("agent_error_code_counts"))
        agent_error_trends = _format_agent_error_trends(payload.get("agent_error_trends"))
        handoff_validation_trend = _format_handoff_validation_trend(payload.get("agent_error_trends"))
        handoff_status = _format_handoff_status_summary(payload)
        handoff_degraded_rate = _format_handoff_degraded_rate(payload)
        handoff_severity = _format_handoff_severity(
            payload,
            warn_rate=_handoff_warn_rate,
            critical_rate=_handoff_critical_rate,
        )
        collaboration_signals = _format_collaboration_signals(payload)

    suffix_parts: list[str] = []
    if agent_error_digest is not None:
        suffix_parts.append(f"agent_error_digest={agent_error_digest}")
    if agent_error_trends is not None:
        suffix_parts.append(f"agent_error_trends={agent_error_trends}")
    if handoff_validation_trend is not None:
        suffix_parts.append(f"handoff_validation_trend={handoff_validation_trend}")
    if handoff_status is not None:
        suffix_parts.append(f"handoff_status={handoff_status}")
    if handoff_degraded_rate is not None:
        suffix_parts.append(f"handoff_degraded_rate={handoff_degraded_rate}")
    if handoff_severity is not None:
        suffix_parts.append(f"handoff_severity={handoff_severity}")
    if collaboration_signals is not None:
        suffix_parts.append(f"collaboration_signals={collaboration_signals}")

    suffix = ""
    if suffix_parts:
        suffix = " | " + " | ".join(suffix_parts)

    return (
        "Agent ops health: "
        f"success_rate={success_rate} | error_rate={error_rate} | "
        f"bounded_out_rate={bounded_out_rate} | p50_ms={p50_ms} | p95_ms={p95_ms}"
        f"{suffix}"
    )


def _format_agent_error_digest(agent_error_code_counts: Any) -> str | None:
    # Formats agent error digest into concise operator-facing text.
    if not isinstance(agent_error_code_counts, dict):
        return None

    observation = agent_error_code_counts.get("observation")
    reasoning = agent_error_code_counts.get("reasoning")
    cycle = agent_error_code_counts.get("cycle")
    if not isinstance(observation, dict) and not isinstance(reasoning, dict) and not isinstance(cycle, dict):
        return None

    def _top_code(code_counts: Any) -> str:
        if not isinstance(code_counts, dict):
            return "none"

        best_code = None
        best_count = -1
        for code, count in code_counts.items():
            if not isinstance(code, str) or not code.strip():
                continue
            if not isinstance(count, int):
                continue
            if count > best_count:
                best_code = code
                best_count = count

        if best_code is None:
            return "none"
        return f"{best_code}:{best_count}"

    observation_text = _top_code(observation)
    reasoning_text = _top_code(reasoning)
    cycle_text = _top_code(cycle)
    if isinstance(cycle, dict):
        return f"obs({observation_text}),rea({reasoning_text}),cyc({cycle_text})"
    return f"obs({observation_text}),rea({reasoning_text})"


def _format_agent_error_trends(agent_error_trends: Any) -> str | None:
    # Formats agent error trends into concise operator-facing text.
    if not isinstance(agent_error_trends, dict):
        return None

    observation = agent_error_trends.get("observation")
    reasoning = agent_error_trends.get("reasoning")
    cycle = agent_error_trends.get("cycle")
    if not isinstance(observation, dict) and not isinstance(reasoning, dict) and not isinstance(cycle, dict):
        return None

    def _format_side(value: Any, prefix: str) -> str:
        if not isinstance(value, dict):
            return f"{prefix}(none)"

        top_error_code = value.get("top_error_code")
        top_error_count = value.get("top_error_count")
        top_error_rate = value.get("top_error_rate")

        if not isinstance(top_error_code, str) or not top_error_code.strip():
            return f"{prefix}(none)"
        if not isinstance(top_error_count, int):
            return f"{prefix}(none)"
        if not isinstance(top_error_rate, (int, float)):
            return f"{prefix}(none)"

        return f"{prefix}({top_error_code}:{top_error_count}@{top_error_rate:.3f})"

    observation_text = _format_side(observation, "obs")
    reasoning_text = _format_side(reasoning, "rea")
    if isinstance(cycle, dict):
        cycle_text = _format_side(cycle, "cyc")
        return f"{observation_text},{reasoning_text},{cycle_text}"
    return f"{observation_text},{reasoning_text}"


def _format_handoff_validation_trend(agent_error_trends: Any) -> str | None:
    # Formats handoff validation trend into concise operator-facing text.
    if not isinstance(agent_error_trends, dict):
        return None

    handoff = agent_error_trends.get("handoff_validation")
    if not isinstance(handoff, dict):
        return None

    top_error_code = handoff.get("top_error_code")
    top_error_count = handoff.get("top_error_count")
    top_error_rate = handoff.get("top_error_rate")

    if not isinstance(top_error_code, str) or not top_error_code.strip():
        return "handoff(none)"
    if not isinstance(top_error_count, int):
        return "handoff(none)"
    if not isinstance(top_error_rate, (int, float)):
        return "handoff(none)"

    return f"handoff({top_error_code}:{top_error_count}@{top_error_rate:.3f})"


def _format_handoff_status_summary(payload: dict[str, Any]) -> str | None:
    # Formats handoff status summary into concise operator-facing text.
    if not isinstance(payload, dict):
        return None

    handoff_validation = payload.get("handoff_validation")
    if not isinstance(handoff_validation, dict):
        return None

    status_counts = handoff_validation.get("status_counts")
    if not isinstance(status_counts, dict):
        return None

    returned_events = payload.get("returned_events")
    if not isinstance(returned_events, int) or returned_events <= 0:
        return None

    ok_count = status_counts.get("ok")
    error_count = status_counts.get("error")
    if not isinstance(ok_count, int):
        ok_count = 0
    if not isinstance(error_count, int):
        error_count = 0

    ok_rate = ok_count / returned_events
    error_rate = error_count / returned_events
    return f"ok:{ok_count}@{ok_rate:.3f},err:{error_count}@{error_rate:.3f}"


def _format_handoff_degraded_rate(payload: dict[str, Any]) -> str | None:
    # Formats handoff degraded rate into concise operator-facing text.
    if not isinstance(payload, dict):
        return None

    handoff_validation = payload.get("handoff_validation")
    if not isinstance(handoff_validation, dict):
        return None

    degraded_rate = handoff_validation.get("degraded_rate")
    if not isinstance(degraded_rate, (int, float)):
        return None

    return f"{float(degraded_rate):.3f}"


def _format_collaboration_signals(payload: dict[str, Any]) -> str | None:
    # Formats collaboration signals into concise operator-facing text.
    if not isinstance(payload, dict):
        return None

    collaboration_signals = payload.get("collaboration_signals")
    if not isinstance(collaboration_signals, dict):
        return None

    conflict_event_count = collaboration_signals.get("conflict_event_count")
    conflict_event_rate = collaboration_signals.get("conflict_event_rate")
    limitation_event_count = collaboration_signals.get("limitation_event_count")
    limitation_event_rate = collaboration_signals.get("limitation_event_rate")

    if not isinstance(conflict_event_count, int):
        return None
    if not isinstance(conflict_event_rate, (int, float)):
        return None
    if not isinstance(limitation_event_count, int):
        return None
    if not isinstance(limitation_event_rate, (int, float)):
        return None

    return (
        f"conflicts={conflict_event_count}@{float(conflict_event_rate):.3f},"
        f"limitations={limitation_event_count}@{float(limitation_event_rate):.3f}"
    )


def _classify_handoff_severity(
    degraded_rate: float,
    warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
) -> str:
    # Classifies handoff severity into normalized categories.
    if degraded_rate >= critical_rate:
        return "critical"
    if degraded_rate >= warn_rate:
        return "warn"
    return "healthy"


def _format_handoff_severity(
    payload: dict[str, Any],
    warn_rate: float = HANDOFF_DEGRADED_WARN_RATE,
    critical_rate: float = HANDOFF_DEGRADED_CRITICAL_RATE,
) -> str | None:
    # Formats handoff severity into concise operator-facing text.
    if not isinstance(payload, dict):
        return None

    handoff_validation = payload.get("handoff_validation")
    if not isinstance(handoff_validation, dict):
        return None

    degraded_rate = handoff_validation.get("degraded_rate")
    if not isinstance(degraded_rate, (int, float)):
        return None

    return _classify_handoff_severity(
        float(degraded_rate),
        warn_rate=warn_rate,
        critical_rate=critical_rate,
    )


def _is_agent_health_payload_fresh(
    payload: dict[str, Any],
    max_age_seconds: float = AGENT_OPS_HEALTH_FRESHNESS_SECONDS,
    now_timestamp: float | None = None,
) -> bool:
    # Determines whether the health payload is fresh enough for trends.
    time_window = payload.get("time_window")
    if not isinstance(time_window, dict):
        return False

    last_timestamp = time_window.get("last_timestamp")
    if not isinstance(last_timestamp, (int, float)):
        return False

    current_timestamp = now_timestamp
    if current_timestamp is None:
        current_timestamp = time.time()

    age_seconds = float(current_timestamp) - float(last_timestamp)
    if age_seconds < 0:
        return True
    return age_seconds <= max_age_seconds


def format_depth_coverage_summary(payload: dict[str, Any]) -> str:
    # Formats depth coverage summary into concise operator-facing text.
    latest = payload.get("latest_coverage")
    trend = payload.get("trend")
    if not isinstance(latest, dict) or not isinstance(trend, dict):
        return "Depth coverage: unavailable"

    tracks_with_depth = latest.get("tracks_with_depth", "?")
    total_tracks = latest.get("total_tracks", "?")
    coverage_percent = latest.get("coverage_percent", "?")

    direction = trend.get("direction", "unknown")
    delta = trend.get("delta_coverage_percent")
    delta_text = "n/a" if not isinstance(delta, (int, float)) else f"{delta:+.1f}%"

    return (
        "Depth coverage: "
        f"{tracks_with_depth}/{total_tracks} ({coverage_percent}%) | "
        f"trend={direction} ({delta_text})"
    )
