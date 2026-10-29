# Handles dashboardMetrics behavior.
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import ops
from ...registry import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = PROJECT_ROOT / "captures"
RADAR_STATE_PATH = CAPTURES_DIR / "latest_radar_state.json"
TRACKING_FRAME_PATH = CAPTURES_DIR / "latest_tracking_frame.json"
DEPTH_COVERAGE_HISTORY_PATH = CAPTURES_DIR / "radar_depth_coverage_history.jsonl"


def _load_json_file(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    # Performs load json file logic.
    if not path.is_file():
        return None, "missing"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, "invalid_json"

    if not isinstance(payload, dict):
        return None, "not_object"

    return payload, None


def _latest_tracking_history_file(captures_dir: Path) -> Path | None:
    # Performs latest tracking history file logic.
    candidates = sorted(captures_dir.glob("tracking_*.json"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _compute_continuity_metrics(history_payload: dict[str, Any]) -> dict[str, Any]:
    # Performs compute continuity metrics logic.
    evaluated_tracks = 0
    transitions = 0
    gap_transitions = 0
    tracks_with_gaps = 0

    for _track_id, records in history_payload.items():
        if not isinstance(records, list):
            continue

        frames: list[int] = []
        for record in records:
            if not isinstance(record, dict):
                continue

            frame = record.get("frame")
            if isinstance(frame, int):
                frames.append(frame)

        if not frames:
            continue

        evaluated_tracks += 1
        frames.sort()

        track_has_gap = False
        for index in range(1, len(frames)):
            transitions += 1
            if frames[index] - frames[index - 1] > 1:
                gap_transitions += 1
                track_has_gap = True

        if track_has_gap:
            tracks_with_gaps += 1

    continuity_ratio = None
    if transitions > 0:
        continuity_ratio = max(0.0, 1.0 - (gap_transitions / transitions))

    return {
        "available": evaluated_tracks > 0,
        "evaluated_tracks": evaluated_tracks,
        "transitions": transitions,
        "gap_transitions": gap_transitions,
        "tracks_with_gaps": tracks_with_gaps,
        "continuity_ratio": continuity_ratio,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    # Performs safe int logic.
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    # Performs safe float logic.
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _load_depth_coverage_history_samples(
    path: Path,
    sample_limit: int,
) -> tuple[list[dict[str, Any]], str | None]:
    # Performs load depth coverage history samples logic.
    if sample_limit <= 0:
        return [], "invalid_sample_limit"

    if not path.is_file():
        return [], "missing"

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], "unreadable"

    samples: list[dict[str, Any]] = []
    for line in lines[-sample_limit:]:
        raw = line.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(entry, dict):
            continue

        timestamp = entry.get("timestamp")
        coverage = entry.get("coverage")
        if not isinstance(timestamp, (int, float)) or not isinstance(coverage, dict):
            continue

        samples.append(
            {
                "timestamp": float(timestamp),
                "tracks_with_depth": _safe_int(coverage.get("tracks_with_depth"), default=0),
                "total_tracks": _safe_int(coverage.get("total_tracks"), default=0),
                "coverage_ratio": _safe_float(coverage.get("coverage_ratio"), default=0.0),
                "coverage_percent": _safe_float(coverage.get("coverage_percent"), default=0.0),
            }
        )

    if not samples:
        return [], "empty_or_invalid"

    return samples, None


@mcp.tool()
def dashboard_depth_coverage_metrics(sample_limit: int = 50) -> dict[str, Any]:
    # Performs dashboard depth coverage metrics logic.
    """Return latest relative-depth coverage and trend over recent radar snapshots."""
    now = time.time()
    errors: list[str] = []

    radar_state, radar_error = _load_json_file(RADAR_STATE_PATH)
    if radar_error is not None:
        errors.append(f"radar_state_{radar_error}")

    history_samples, history_error = _load_depth_coverage_history_samples(
        DEPTH_COVERAGE_HISTORY_PATH,
        sample_limit=sample_limit,
    )
    if history_error is not None:
        errors.append(f"depth_coverage_history_{history_error}")

    latest_coverage = {
        "tracks_with_depth": 0,
        "total_tracks": 0,
        "coverage_ratio": 0.0,
        "coverage_percent": 0.0,
        "source": "default",
    }

    if radar_state is not None:
        raw = radar_state.get("relative_depth_coverage")
        if isinstance(raw, dict):
            latest_coverage = {
                "tracks_with_depth": _safe_int(raw.get("tracks_with_depth"), default=0),
                "total_tracks": _safe_int(raw.get("total_tracks"), default=0),
                "coverage_ratio": _safe_float(raw.get("coverage_ratio"), default=0.0),
                "coverage_percent": _safe_float(raw.get("coverage_percent"), default=0.0),
                "source": "latest_radar_state",
            }
        elif history_samples:
            latest_sample = history_samples[-1]
            latest_coverage = {
                "tracks_with_depth": latest_sample["tracks_with_depth"],
                "total_tracks": latest_sample["total_tracks"],
                "coverage_ratio": latest_sample["coverage_ratio"],
                "coverage_percent": latest_sample["coverage_percent"],
                "source": "depth_coverage_history",
            }

    trend = {
        "available": len(history_samples) >= 2,
        "samples_analyzed": len(history_samples),
        "window_seconds": None,
        "start_coverage_percent": None,
        "end_coverage_percent": None,
        "delta_coverage_percent": None,
        "direction": "unknown",
    }

    if len(history_samples) >= 2:
        start = history_samples[0]
        end = history_samples[-1]
        delta = end["coverage_percent"] - start["coverage_percent"]

        if delta > 1.0:
            direction = "improving"
        elif delta < -1.0:
            direction = "declining"
        else:
            direction = "stable"

        trend = {
            "available": True,
            "samples_analyzed": len(history_samples),
            "window_seconds": max(0.0, end["timestamp"] - start["timestamp"]),
            "start_coverage_percent": round(start["coverage_percent"], 1),
            "end_coverage_percent": round(end["coverage_percent"], 1),
            "delta_coverage_percent": round(delta, 1),
            "direction": direction,
        }

    available = radar_state is not None or bool(history_samples)

    return {
        "available": available,
        "timestamp": now,
        "sample_limit": sample_limit,
        "source_files": {
            "radar_state_path": str(RADAR_STATE_PATH),
            "depth_coverage_history_path": str(DEPTH_COVERAGE_HISTORY_PATH),
            "radar_state_loaded": radar_state is not None,
            "history_samples_loaded": len(history_samples),
        },
        "latest_coverage": latest_coverage,
        "trend": trend,
        "errors": errors,
    }


@mcp.tool()
def dashboard_latency_continuity_metrics() -> dict[str, Any]:
    # Performs dashboard latency continuity metrics logic.
    """Return dashboard latency and continuity metrics from capture snapshots."""
    now = time.time()
    errors: list[str] = []

    radar_state, radar_error = _load_json_file(RADAR_STATE_PATH)
    if radar_error is not None:
        errors.append(f"radar_state_{radar_error}")

    tracking_frame, tracking_error = _load_json_file(TRACKING_FRAME_PATH)
    if tracking_error is not None:
        errors.append(f"tracking_frame_{tracking_error}")

    radar_timestamp = None
    last_observation_timestamp = None
    active_track_count = 0
    if radar_state is not None:
        timestamp_value = radar_state.get("timestamp")
        if isinstance(timestamp_value, (int, float)):
            radar_timestamp = float(timestamp_value)

        observation_timestamp = radar_state.get("last_observation_timestamp")
        if isinstance(observation_timestamp, (int, float)):
            last_observation_timestamp = float(observation_timestamp)
        else:
            last_observation_timestamp = radar_timestamp

        count_value = radar_state.get("active_track_count")
        if isinstance(count_value, int):
            active_track_count = count_value
        else:
            tracks = radar_state.get("tracks")
            active_track_count = len(tracks) if isinstance(tracks, list) else 0

    tracking_timestamp = None
    detection_count = 0
    if tracking_frame is not None:
        time_value = tracking_frame.get("time")
        if isinstance(time_value, (int, float)):
            tracking_timestamp = float(time_value)

        detections = tracking_frame.get("detections")
        detection_count = len(detections) if isinstance(detections, list) else 0

    state_update_latency_seconds = None
    if radar_timestamp is not None:
        state_update_latency_seconds = max(0.0, now - radar_timestamp)

    tracking_update_latency_seconds = None
    if tracking_timestamp is not None:
        tracking_update_latency_seconds = max(0.0, now - tracking_timestamp)

    end_to_end_delay_seconds = None
    if last_observation_timestamp is not None:
        end_to_end_delay_seconds = max(0.0, now - last_observation_timestamp)

    continuity: dict[str, Any] = {
        "available": False,
        "history_file": None,
        "reason": "history_unavailable",
    }

    history_path = _latest_tracking_history_file(CAPTURES_DIR)
    if history_path is not None:
        history_payload, history_error = _load_json_file(history_path)
        if history_error is None and history_payload is not None:
            continuity = _compute_continuity_metrics(history_payload)
            continuity["history_file"] = str(history_path)
        else:
            continuity = {
                "available": False,
                "history_file": str(history_path),
                "reason": f"history_{history_error}",
            }

    available = radar_state is not None or tracking_frame is not None
    if not available and not errors:
        errors.append("no_capture_snapshots")

    return {
        "available": available,
        "captures_dir": str(CAPTURES_DIR),
        "source_files": {
            "radar_state_path": str(RADAR_STATE_PATH),
            "tracking_frame_path": str(TRACKING_FRAME_PATH),
            "radar_state_loaded": radar_state is not None,
            "tracking_frame_loaded": tracking_frame is not None,
        },
        "timestamps": {
            "now": now,
            "radar_timestamp": radar_timestamp,
            "tracking_timestamp": tracking_timestamp,
            "last_observation_timestamp": last_observation_timestamp,
        },
        "counts": {
            "active_track_count": active_track_count,
            "detection_count": detection_count,
        },
        "latency": {
            "state_update_latency_seconds": state_update_latency_seconds,
            "tracking_update_latency_seconds": tracking_update_latency_seconds,
            "end_to_end_delay_seconds": end_to_end_delay_seconds,
        },
        "continuity": continuity,
        "errors": errors,
    }


@mcp.tool()
def dashboard_tracking_quality_report(
    event_window_size: int = 100,
    depth_sample_limit: int = 50,
) -> dict[str, Any]:
    # Performs dashboard tracking quality report logic.
    """Return consolidated tracking quality report from latency, continuity, depth, and event health metrics."""
    if event_window_size <= 0:
        raise ValueError("event_window_size must be greater than zero")
    if depth_sample_limit <= 0:
        raise ValueError("depth_sample_limit must be greater than zero")

    latency_continuity = dashboard_latency_continuity_metrics()
    depth_coverage = dashboard_depth_coverage_metrics(sample_limit=depth_sample_limit)
    event_health = ops.summarize_agent_event_health(window_size=event_window_size)

    continuity_ratio = latency_continuity.get("continuity", {}).get("continuity_ratio")
    end_to_end_delay = latency_continuity.get("latency", {}).get("end_to_end_delay_seconds")
    depth_percent = depth_coverage.get("latest_coverage", {}).get("coverage_percent")
    event_success_rate = event_health.get("status", {}).get("success_rate")

    signals: list[str] = []
    if isinstance(continuity_ratio, (int, float)):
        if continuity_ratio < 0.6:
            signals.append("low_continuity")
        elif continuity_ratio >= 0.85:
            signals.append("good_continuity")

    if isinstance(end_to_end_delay, (int, float)):
        if end_to_end_delay > 3.0:
            signals.append("high_end_to_end_delay")
        elif end_to_end_delay <= 1.0:
            signals.append("low_end_to_end_delay")

    if isinstance(depth_percent, (int, float)):
        if depth_percent < 30.0:
            signals.append("low_depth_coverage")
        elif depth_percent >= 70.0:
            signals.append("good_depth_coverage")

    if isinstance(event_success_rate, (int, float)):
        if event_success_rate < 0.75:
            signals.append("low_agent_success_rate")
        elif event_success_rate >= 0.9:
            signals.append("good_agent_success_rate")

    quality_status = "degraded"
    if {
        "good_continuity",
        "low_end_to_end_delay",
        "good_depth_coverage",
        "good_agent_success_rate",
    }.issubset(set(signals)):
        quality_status = "healthy"
    elif any(
        signal in signals
        for signal in {
            "low_continuity",
            "high_end_to_end_delay",
            "low_depth_coverage",
            "low_agent_success_rate",
        }
    ):
        quality_status = "degraded"
    elif signals:
        quality_status = "warning"

    errors: list[str] = []
    errors.extend(latency_continuity.get("errors", []))
    errors.extend(depth_coverage.get("errors", []))
    event_error = event_health.get("error")
    if isinstance(event_error, str) and event_error:
        errors.append(f"event_health_{event_error}")

    available = bool(
        latency_continuity.get("available")
        or depth_coverage.get("available")
        or event_health.get("available")
    )

    return {
        "available": available,
        "quality_status": quality_status,
        "signals": signals,
        "inputs": {
            "event_window_size": event_window_size,
            "depth_sample_limit": depth_sample_limit,
        },
        "summary": {
            "continuity_ratio": continuity_ratio,
            "end_to_end_delay_seconds": end_to_end_delay,
            "depth_coverage_percent": depth_percent,
            "event_success_rate": event_success_rate,
        },
        "latency_continuity": latency_continuity,
        "depth_coverage": depth_coverage,
        "event_health": event_health,
        "errors": errors,
    }


@mcp.tool()
def orchestration_depth_coverage_metrics(sample_limit: int = 50) -> dict[str, Any]:
    # Performs orchestration depth coverage metrics logic.
    return dashboard_depth_coverage_metrics(sample_limit=sample_limit)


@mcp.tool()
def orchestration_latency_continuity_metrics() -> dict[str, Any]:
    # Performs orchestration latency continuity metrics logic.
    return dashboard_latency_continuity_metrics()


@mcp.tool()
def orchestration_tracking_quality_report(
    event_window_size: int = 100,
    depth_sample_limit: int = 50,
) -> dict[str, Any]:
    # Performs orchestration tracking quality report logic.
    return dashboard_tracking_quality_report(
        event_window_size=event_window_size,
        depth_sample_limit=depth_sample_limit,
    )
