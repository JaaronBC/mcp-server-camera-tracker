# MCP tools for reasoning-cycle event logging, health summaries, and event log maintenance.
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..reasoning import tools_reasoning_assessment as dashboardReasoning
from ..registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_EVENTS_PATH = PROJECT_ROOT / "captures" / "agent_events.jsonl"
AGENT_EVENT_HEALTH_FRESHNESS_SECONDS = 30.0
MAX_EVIDENCE_STR_LIST_ITEMS = 5
MAX_EVIDENCE_SELECTED_UPDATES = 3


def _append_event(event: dict[str, Any]) -> None:
    # Implements append event behavior for this module.
    AGENT_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AGENT_EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(event)}\n")


def _extract_track_ids(radar_state: dict[str, Any]) -> list[str]:
    # Extracts track ids from structured payloads.
    tracks = radar_state.get("tracks")
    if not isinstance(tracks, list):
        return []

    track_ids: list[str] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_id = track.get("track_id")
        if isinstance(track_id, str) and track_id.strip():
            track_ids.append(track_id)

    return track_ids


def _load_events(limit: int) -> tuple[list[dict[str, Any]], int]:
    # Loads events data from storage for downstream processing.
    lines = AGENT_EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    selected = lines[-limit:]
    events: list[dict[str, Any]] = []
    dropped_lines = 0

    for line in selected:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            dropped_lines += 1
            continue
        if isinstance(payload, dict):
            events.append(payload)
        else:
            dropped_lines += 1

    return events, dropped_lines


def _percentile(values: list[float], percentile: float) -> float | None:
    # Implements percentile behavior for this module.
    if not values:
        return None

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = percentile * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * fraction


def _top_count_entry(counts: dict[str, int]) -> tuple[str | None, int]:
    # Implements top count entry behavior for this module.
    best_key = None
    best_count = 0
    for key, count in counts.items():
        if count > best_count:
            best_key = key
            best_count = count
    return best_key, best_count


def _archive_path_for_now(base_path: Path) -> Path:
    # Implements archive path for now behavior for this module.
    timestamp_label = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return base_path.with_name(f"{base_path.stem}_archive_{timestamp_label}{base_path.suffix}")


def _format_agent_error_digest(
    observation_error_code: Any,
    reasoning_error_code: Any,
) -> str:
    # Formats agent error digest into concise operator-facing text.
    observation = observation_error_code if isinstance(observation_error_code, str) and observation_error_code.strip() else "none"
    reasoning = reasoning_error_code if isinstance(reasoning_error_code, str) and reasoning_error_code.strip() else "none"
    return f"obs({observation}),rea({reasoning})"


def _derive_handoff_quality(
    handoff_validation_status: Any,
    handoff_validation_error_code: Any,
) -> str:
    # Implements derive handoff quality behavior for this module.
    if isinstance(handoff_validation_status, str) and handoff_validation_status.strip().lower() == "error":
        return "degraded"
    if isinstance(handoff_validation_error_code, str) and handoff_validation_error_code.strip():
        return "degraded"
    return "healthy"


def _coerce_nonempty_str(value: Any) -> str | None:
    # Implements coerce nonempty str behavior for this module.
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_str_list(value: Any, max_items: int = MAX_EVIDENCE_STR_LIST_ITEMS) -> list[str]:
    # Implements coerce str list behavior for this module.
    if not isinstance(value, list):
        return []

    result: list[str] = []
    for item in value:
        normalized = _coerce_nonempty_str(item)
        if normalized is None:
            continue
        result.append(normalized)
        if len(result) >= max_items:
            break

    return result


def _extract_ground_position(value: Any) -> dict[str, Any] | None:
    # Extracts ground position from structured payloads.
    if not isinstance(value, dict):
        return None
    return {
        "x": value.get("x"),
        "z": value.get("z"),
    }


def _extract_evidence_trace(summary: Any) -> dict[str, Any]:
    # Extracts evidence trace from structured payloads.
    if not isinstance(summary, dict):
        return {
            "latest_track_snapshot": {"available": False},
            "tracking_agent_context": {"available": False},
            "coordinate_context": {"available": False},
        }

    latest_track_snapshot = summary.get("latest_track_snapshot")
    tracking_agent_context = summary.get("tracking_agent_context")
    coordinate_context = summary.get("coordinate_context")

    if not isinstance(latest_track_snapshot, dict):
        latest_track_snapshot = {"available": False}
    if not isinstance(tracking_agent_context, dict):
        tracking_agent_context = {"available": False}
    if not isinstance(coordinate_context, dict):
        coordinate_context = {"available": False}

    normalized_snapshot = {
        "available": bool(latest_track_snapshot.get("available")),
        "track_id": _coerce_nonempty_str(latest_track_snapshot.get("track_id")),
        "timestamp": latest_track_snapshot.get("timestamp"),
        "ground_position": _extract_ground_position(latest_track_snapshot.get("ground_position")),
        "ground_coordinate_unit": _coerce_nonempty_str(latest_track_snapshot.get("ground_coordinate_unit")),
        "position_kind": _coerce_nonempty_str(latest_track_snapshot.get("position_kind")),
        "relative_depth": latest_track_snapshot.get("relative_depth"),
        "confidence": latest_track_snapshot.get("confidence"),
        "quality_flags": _coerce_str_list(latest_track_snapshot.get("quality_flags")),
    }

    raw_selected_updates = tracking_agent_context.get("selected_updates")
    normalized_selected_updates: list[dict[str, Any]] = []
    if isinstance(raw_selected_updates, list):
        for item in raw_selected_updates:
            if not isinstance(item, dict):
                continue

            normalized_selected_updates.append(
                {
                    "detection_id": _coerce_nonempty_str(item.get("detection_id")),
                    "resolved_track_id": _coerce_nonempty_str(item.get("resolved_track_id")),
                    "accepted": item.get("accepted"),
                    "rejection_reasons": _coerce_str_list(item.get("rejection_reasons")),
                    "uncertainties": _coerce_str_list(item.get("uncertainties")),
                    "supporting_evidence": _coerce_str_list(item.get("supporting_evidence")),
                }
            )
            if len(normalized_selected_updates) >= MAX_EVIDENCE_SELECTED_UPDATES:
                break

    normalized_tracking_context = {
        "available": bool(tracking_agent_context.get("available")),
        "source_agent": _coerce_nonempty_str(tracking_agent_context.get("source_agent")),
        "vlm_model": _coerce_nonempty_str(tracking_agent_context.get("vlm_model")),
        "timestamp": tracking_agent_context.get("timestamp"),
        "proposal_latency_ms": tracking_agent_context.get("proposal_latency_ms"),
        "uncertainties": _coerce_str_list(tracking_agent_context.get("uncertainties")),
        "selected_updates": normalized_selected_updates,
    }

    normalized_coordinate_context = {
        "available": bool(coordinate_context.get("available")),
        "coordinate_system": _coerce_nonempty_str(coordinate_context.get("coordinate_system")),
        "calibration_status": _coerce_nonempty_str(coordinate_context.get("calibration_status")),
        "camera_status": _coerce_nonempty_str(coordinate_context.get("camera_status")),
        "detector_status": _coerce_nonempty_str(coordinate_context.get("detector_status")),
        "last_observation_timestamp": coordinate_context.get("last_observation_timestamp"),
    }

    return {
        "latest_track_snapshot": normalized_snapshot,
        "tracking_agent_context": normalized_tracking_context,
        "coordinate_context": normalized_coordinate_context,
    }


@mcp.tool()
def run_reasoning_cycle_and_log(
    request_evidence_on_uncertainty: bool = True,
    max_tracks: int = 10,
) -> dict[str, Any]:
    # Runs reasoning cycle and log workflow steps and returns status outcomes.
    """Run a bounded reasoning cycle for current tracks and append structured events."""
    if max_tracks <= 0:
        raise ValueError("max_tracks must be greater than zero")

    radar_state = dashboardReasoning._load_radar_state()
    if radar_state is None:
        return {
            "available": False,
            "error": "radar_state_unavailable",
            "event_log_path": str(AGENT_EVENTS_PATH),
        }

    cycle_started_at = time.time()
    track_ids = _extract_track_ids(radar_state)[:max_tracks]

    events: list[dict[str, Any]] = []
    success_count = 0
    error_count = 0
    event_log_error_count = 0
    event_log_last_error: str | None = None

    for track_id in track_ids:
        started = time.perf_counter()
        result = dashboardReasoning.simulate_reasoning_assessment(
            track_id,
            request_evidence_on_uncertainty=request_evidence_on_uncertainty,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        available = bool(result.get("available"))
        status = "ok" if available else "error"
        if available:
            success_count += 1
        else:
            error_count += 1

        assessment = result.get("assessment")
        assessment_label = None
        if isinstance(assessment, dict):
            label = assessment.get("assessment_label")
            if isinstance(label, str):
                assessment_label = label

        agents_payload = result.get("agents")
        observation_agent = agents_payload.get("observation") if isinstance(agents_payload, dict) else None
        reasoning_agent = agents_payload.get("reasoning") if isinstance(agents_payload, dict) else None
        observation_error_code = (
            observation_agent.get("error_code")
            if isinstance(observation_agent, dict)
            else None
        )
        reasoning_error_code = (
            reasoning_agent.get("error_code")
            if isinstance(reasoning_agent, dict)
            else None
        )
        handoff_validation = result.get("handoff_validation")
        handoff_validation_status = (
            handoff_validation.get("status")
            if isinstance(handoff_validation, dict)
            else None
        )
        handoff_validation_error_code = (
            handoff_validation.get("error_code")
            if isinstance(handoff_validation, dict)
            else None
        )
        handoff_validation_error = (
            handoff_validation.get("error")
            if isinstance(handoff_validation, dict)
            else None
        )
        handoff_quality = _derive_handoff_quality(
            handoff_validation_status,
            handoff_validation_error_code,
        )
        cycle_error_code = None
        raw_cycle_error = result.get("error")
        if isinstance(raw_cycle_error, str) and raw_cycle_error.strip():
            cycle_error_code = raw_cycle_error.strip()

        event = {
            "event_type": "reasoning_cycle",
            "timestamp": cycle_started_at,
            "track_id": track_id,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "assessment_label": assessment_label,
            "evidence_requested": result.get("evidence_request") is not None,
            "bounded_out": bool(
                result.get("evidence_request") is None
                and bool(result.get("summary", {}).get("needs_more_observations", False))
                and request_evidence_on_uncertainty
            ),
            "observation_agent_model": (
                observation_agent.get("model")
                if isinstance(observation_agent, dict)
                else None
            ),
            "observation_agent_status": (
                observation_agent.get("status")
                if isinstance(observation_agent, dict)
                else None
            ),
            "observation_agent_latency_ms": (
                observation_agent.get("latency_ms")
                if isinstance(observation_agent, dict)
                else None
            ),
            "observation_agent_error_code": (
                observation_error_code
            ),
            "observation_agent_attempts": (
                observation_agent.get("attempts")
                if isinstance(observation_agent, dict)
                else None
            ),
            "observation_agent_retry_count": (
                observation_agent.get("retry_count")
                if isinstance(observation_agent, dict)
                else None
            ),
            "reasoning_agent_model": (
                reasoning_agent.get("model")
                if isinstance(reasoning_agent, dict)
                else None
            ),
            "reasoning_agent_status": (
                reasoning_agent.get("status")
                if isinstance(reasoning_agent, dict)
                else None
            ),
            "reasoning_agent_latency_ms": (
                reasoning_agent.get("latency_ms")
                if isinstance(reasoning_agent, dict)
                else None
            ),
            "reasoning_agent_error_code": (
                reasoning_error_code
            ),
            "reasoning_agent_attempts": (
                reasoning_agent.get("attempts")
                if isinstance(reasoning_agent, dict)
                else None
            ),
            "reasoning_agent_retry_count": (
                reasoning_agent.get("retry_count")
                if isinstance(reasoning_agent, dict)
                else None
            ),
            "handoff_validation_status": handoff_validation_status,
            "handoff_validation_error_code": handoff_validation_error_code,
            "handoff_validation_error": handoff_validation_error,
            "handoff_quality": handoff_quality,
            "agent_error_digest": _format_agent_error_digest(
                observation_error_code,
                reasoning_error_code,
            ),
            "cycle_error_code": cycle_error_code,
            "evidence_trace": _extract_evidence_trace(result.get("summary")),
            "error": result.get("error"),
        }

        try:
            _append_event(event)
        except OSError as error:
            event_log_error_count += 1
            event_log_last_error = f"{type(error).__name__}:{error}"
        events.append(event)

    return {
        "available": True,
        "event_log_path": str(AGENT_EVENTS_PATH),
        "event_log_writable": event_log_error_count == 0,
        "event_log_error_count": event_log_error_count,
        "event_log_last_error": event_log_last_error,
        "cycle": {
            "started_at": cycle_started_at,
            "evaluated_tracks": len(track_ids),
            "success_count": success_count,
            "error_count": error_count,
            "request_evidence_on_uncertainty": request_evidence_on_uncertainty,
        },
        "events": events,
    }


@mcp.tool()
def get_recent_agent_events(limit: int = 20) -> dict[str, Any]:
    # Gets recent agent events values for downstream consumers.
    """Return recent structured agent events from captures/agent_events.jsonl."""
    if limit <= 0:
        raise ValueError("limit must be greater than zero")

    if not AGENT_EVENTS_PATH.is_file():
        return {
            "available": False,
            "event_log_path": str(AGENT_EVENTS_PATH),
            "error": "agent_events_missing",
            "events": [],
        }

    events, dropped_lines = _load_events(limit)

    return {
        "available": True,
        "event_log_path": str(AGENT_EVENTS_PATH),
        "requested_limit": limit,
        "returned_events": len(events),
        "dropped_lines": dropped_lines,
        "events": events,
    }


@mcp.tool()
def summarize_agent_event_health(window_size: int = 100) -> dict[str, Any]:
    # Implements summarize agent event health behavior for this module.
    """Summarize recent agent event health from captures/agent_events.jsonl."""
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero")

    if not AGENT_EVENTS_PATH.is_file():
        return {
            "available": False,
            "event_log_path": str(AGENT_EVENTS_PATH),
            "error": "agent_events_missing",
        }

    events, dropped_lines = _load_events(window_size)
    if not events:
        return {
            "available": False,
            "event_log_path": str(AGENT_EVENTS_PATH),
            "error": "no_valid_events",
            "dropped_lines": dropped_lines,
        }

    ok_count = 0
    error_count = 0
    bounded_out_count = 0
    latency_values: list[float] = []
    timestamps: list[float] = []
    assessment_counts: dict[str, int] = {}
    observation_error_code_counts: dict[str, int] = {}
    reasoning_error_code_counts: dict[str, int] = {}
    cycle_error_code_counts: dict[str, int] = {}
    handoff_validation_status_counts: dict[str, int] = {}
    handoff_validation_error_code_counts: dict[str, int] = {}
    handoff_quality_counts: dict[str, int] = {}

    for event in events:
        status = event.get("status")
        if status == "ok":
            ok_count += 1
        elif status == "error":
            error_count += 1

        if bool(event.get("bounded_out")):
            bounded_out_count += 1

        latency_ms = event.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            latency_values.append(float(latency_ms))

        timestamp = event.get("timestamp")
        if isinstance(timestamp, (int, float)):
            timestamps.append(float(timestamp))

        label = event.get("assessment_label")
        if isinstance(label, str) and label.strip():
            assessment_counts[label] = assessment_counts.get(label, 0) + 1

        observation_error_code = event.get("observation_agent_error_code")
        if isinstance(observation_error_code, str) and observation_error_code.strip():
            observation_error_code_counts[observation_error_code] = (
                observation_error_code_counts.get(observation_error_code, 0) + 1
            )

        reasoning_error_code = event.get("reasoning_agent_error_code")
        if isinstance(reasoning_error_code, str) and reasoning_error_code.strip():
            reasoning_error_code_counts[reasoning_error_code] = (
                reasoning_error_code_counts.get(reasoning_error_code, 0) + 1
            )

        cycle_error_code = event.get("cycle_error_code")
        if isinstance(cycle_error_code, str) and cycle_error_code.strip():
            cycle_error_code_counts[cycle_error_code] = (
                cycle_error_code_counts.get(cycle_error_code, 0) + 1
            )
        else:
            fallback_error = event.get("error")
            if isinstance(fallback_error, str) and fallback_error.strip():
                cycle_error_code_counts[fallback_error] = (
                    cycle_error_code_counts.get(fallback_error, 0) + 1
                )

        handoff_validation_status = event.get("handoff_validation_status")
        if isinstance(handoff_validation_status, str) and handoff_validation_status.strip():
            handoff_validation_status_counts[handoff_validation_status] = (
                handoff_validation_status_counts.get(handoff_validation_status, 0) + 1
            )

        handoff_validation_error_code = event.get("handoff_validation_error_code")
        if isinstance(handoff_validation_error_code, str) and handoff_validation_error_code.strip():
            handoff_validation_error_code_counts[handoff_validation_error_code] = (
                handoff_validation_error_code_counts.get(handoff_validation_error_code, 0) + 1
            )

        handoff_quality = event.get("handoff_quality")
        if not isinstance(handoff_quality, str) or not handoff_quality.strip():
            handoff_quality = _derive_handoff_quality(
                handoff_validation_status,
                handoff_validation_error_code,
            )
        handoff_quality_counts[handoff_quality] = handoff_quality_counts.get(handoff_quality, 0) + 1

    total_events = len(events)
    success_rate = ok_count / total_events
    error_rate = error_count / total_events
    bounded_out_rate = bounded_out_count / total_events

    observation_top_code, observation_top_count = _top_count_entry(observation_error_code_counts)
    reasoning_top_code, reasoning_top_count = _top_count_entry(reasoning_error_code_counts)
    cycle_top_code, cycle_top_count = _top_count_entry(cycle_error_code_counts)
    handoff_top_code, handoff_top_count = _top_count_entry(handoff_validation_error_code_counts)

    observation_top_rate = (
        observation_top_count / total_events if observation_top_code is not None else 0.0
    )
    reasoning_top_rate = (
        reasoning_top_count / total_events if reasoning_top_code is not None else 0.0
    )
    cycle_top_rate = (
        cycle_top_count / total_events if cycle_top_code is not None else 0.0
    )
    handoff_top_rate = (
        handoff_top_count / total_events if handoff_top_code is not None else 0.0
    )
    handoff_degraded_count = handoff_quality_counts.get("degraded", 0)
    handoff_degraded_rate = handoff_degraded_count / total_events

    latency_summary = {
        "count": len(latency_values),
        "p50_ms": _percentile(latency_values, 0.5),
        "p95_ms": _percentile(latency_values, 0.95),
        "max_ms": max(latency_values) if latency_values else None,
    }

    time_window = {
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }

    latest_timestamp = time_window["last_timestamp"]
    freshness_age_seconds: float | None = None
    freshness_is_fresh = False
    if isinstance(latest_timestamp, (int, float)):
        freshness_age_seconds = max(0.0, time.time() - float(latest_timestamp))
        freshness_is_fresh = freshness_age_seconds <= AGENT_EVENT_HEALTH_FRESHNESS_SECONDS

    return {
        "available": True,
        "event_log_path": str(AGENT_EVENTS_PATH),
        "window_size": window_size,
        "returned_events": total_events,
        "dropped_lines": dropped_lines,
        "status": {
            "ok_count": ok_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "bounded_out_count": bounded_out_count,
            "bounded_out_rate": bounded_out_rate,
        },
        "latency": latency_summary,
        "assessment_label_counts": assessment_counts,
        "agent_error_code_counts": {
            "observation": observation_error_code_counts,
            "reasoning": reasoning_error_code_counts,
            "cycle": cycle_error_code_counts,
        },
        "handoff_validation": {
            "status_counts": handoff_validation_status_counts,
            "error_code_counts": handoff_validation_error_code_counts,
            "quality_counts": handoff_quality_counts,
            "degraded_count": handoff_degraded_count,
            "degraded_rate": handoff_degraded_rate,
        },
        "agent_error_trends": {
            "observation": {
                "top_error_code": observation_top_code,
                "top_error_count": observation_top_count,
                "top_error_rate": observation_top_rate,
            },
            "reasoning": {
                "top_error_code": reasoning_top_code,
                "top_error_count": reasoning_top_count,
                "top_error_rate": reasoning_top_rate,
            },
            "cycle": {
                "top_error_code": cycle_top_code,
                "top_error_count": cycle_top_count,
                "top_error_rate": cycle_top_rate,
            },
            "handoff_validation": {
                "top_error_code": handoff_top_code,
                "top_error_count": handoff_top_count,
                "top_error_rate": handoff_top_rate,
            },
        },
        "freshness": {
            "is_fresh": freshness_is_fresh,
            "age_seconds": freshness_age_seconds,
            "threshold_seconds": AGENT_EVENT_HEALTH_FRESHNESS_SECONDS,
        },
        "time_window": time_window,
    }


@mcp.tool()
def prune_agent_event_log(
    max_events_keep: int = 5000,
    archive_removed_events: bool = True,
) -> dict[str, Any]:
    # Implements prune agent event log behavior for this module.
    """Prune old agent event log lines and optionally archive removed lines."""
    if max_events_keep < 0:
        raise ValueError("max_events_keep must be zero or greater")

    if not AGENT_EVENTS_PATH.is_file():
        return {
            "available": False,
            "event_log_path": str(AGENT_EVENTS_PATH),
            "error": "agent_events_missing",
        }

    lines = AGENT_EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    total_before = len(lines)

    if total_before <= max_events_keep:
        return {
            "available": True,
            "event_log_path": str(AGENT_EVENTS_PATH),
            "total_lines_before": total_before,
            "total_lines_after": total_before,
            "removed_lines": 0,
            "archive_path": None,
            "pruned": False,
        }

    cut_index = total_before - max_events_keep
    removed_lines = lines[:cut_index]
    kept_lines = lines[cut_index:]

    archive_path: Path | None = None
    if archive_removed_events and removed_lines:
        archive_path = _archive_path_for_now(AGENT_EVENTS_PATH)
        archive_path.write_text("\n".join(removed_lines) + "\n", encoding="utf-8")

    if kept_lines:
        AGENT_EVENTS_PATH.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    else:
        AGENT_EVENTS_PATH.write_text("", encoding="utf-8")

    return {
        "available": True,
        "event_log_path": str(AGENT_EVENTS_PATH),
        "total_lines_before": total_before,
        "total_lines_after": len(kept_lines),
        "removed_lines": len(removed_lines),
        "archive_path": None if archive_path is None else str(archive_path),
        "pruned": True,
    }
