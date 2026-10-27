from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import dashboardReasoning
from .registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_EVENTS_PATH = PROJECT_ROOT / "captures" / "agent_events.jsonl"


def _append_event(event: dict[str, Any]) -> None:
    AGENT_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AGENT_EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(event)}\n")


def _extract_track_ids(radar_state: dict[str, Any]) -> list[str]:
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


def _archive_path_for_now(base_path: Path) -> Path:
    timestamp_label = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return base_path.with_name(f"{base_path.stem}_archive_{timestamp_label}{base_path.suffix}")


@mcp.tool()
def run_reasoning_cycle_and_log(
    request_evidence_on_uncertainty: bool = True,
    max_tracks: int = 10,
) -> dict[str, Any]:
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
                and bool(result.get("summary", {}).get("speed") is None)
                and request_evidence_on_uncertainty
            ),
            "error": result.get("error"),
        }

        _append_event(event)
        events.append(event)

    return {
        "available": True,
        "event_log_path": str(AGENT_EVENTS_PATH),
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

    total_events = len(events)
    success_rate = ok_count / total_events
    error_rate = error_count / total_events
    bounded_out_rate = bounded_out_count / total_events

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
        "time_window": time_window,
    }


@mcp.tool()
def prune_agent_event_log(
    max_events_keep: int = 5000,
    archive_removed_events: bool = True,
) -> dict[str, Any]:
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
