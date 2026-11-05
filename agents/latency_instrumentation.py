from __future__ import annotations

import json
import os
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "captures" / "latency_events.jsonl"

_correlation_id_var: ContextVar[str | None] = ContextVar("latency_correlation_id", default=None)
_cycle_id_var: ContextVar[str | None] = ContextVar("latency_cycle_id", default=None)
_track_id_var: ContextVar[str | None] = ContextVar("latency_track_id", default=None)
_llm_attempt_context_var: ContextVar[dict[str, Any] | None] = ContextVar("latency_llm_attempt_context", default=None)


def is_enabled() -> bool:
    value = os.getenv("PHASE16E_LATENCY_INSTRUMENTATION_ENABLED", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def events_path() -> Path:
    configured = os.getenv("PHASE16E_LATENCY_EVENTS_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_EVENTS_PATH


def new_correlation_id(prefix: str = "corr") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def now_perf_counter() -> float:
    return time.perf_counter()


def duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - float(started_at)) * 1000.0, 3)


def set_context(*, correlation_id: str | None = None, cycle_id: str | None = None, track_id: str | None = None) -> dict[str, Any]:
    previous = {
        "correlation_id": _correlation_id_var.get(),
        "cycle_id": _cycle_id_var.get(),
        "track_id": _track_id_var.get(),
    }
    if correlation_id is not None:
        _correlation_id_var.set(correlation_id)
    if cycle_id is not None:
        _cycle_id_var.set(cycle_id)
    if track_id is not None:
        _track_id_var.set(track_id)
    return previous


def restore_context(previous: dict[str, Any]) -> None:
    _correlation_id_var.set(previous.get("correlation_id"))
    _cycle_id_var.set(previous.get("cycle_id"))
    _track_id_var.set(previous.get("track_id"))


def get_context() -> dict[str, Any]:
    return {
        "correlation_id": _correlation_id_var.get(),
        "cycle_id": _cycle_id_var.get(),
        "track_id": _track_id_var.get(),
    }


def set_llm_attempt_context(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    previous = _llm_attempt_context_var.get()
    _llm_attempt_context_var.set(payload)
    return previous


def get_llm_attempt_context() -> dict[str, Any] | None:
    value = _llm_attempt_context_var.get()
    if isinstance(value, dict):
        return dict(value)
    return None


def emit_event(event_type: str, **fields: Any) -> None:
    if not is_enabled():
        return

    payload: dict[str, Any] = {
        "event_type": event_type,
        "timestamp": time.time(),
    }
    payload.update(get_context())
    payload.update(fields)

    try:
        path = events_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(payload)}\n")
    except Exception:
        return


def summarize_events(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_values: dict[str, list[float]] = {}
    cycle_status_counts = {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "timed_out": 0,
    }

    for record in records:
        if not isinstance(record, dict):
            continue
        stage = record.get("stage")
        duration = record.get("duration_ms")
        if isinstance(stage, str) and isinstance(duration, (int, float)):
            stage_values.setdefault(stage, []).append(float(duration))

        if record.get("event_type") == "reasoning_cycle_completed":
            status = record.get("cycle_status")
            if status in cycle_status_counts:
                cycle_status_counts[str(status)] += 1

    stage_summary: dict[str, dict[str, float | int | None]] = {}
    for stage, values in stage_values.items():
        stage_summary[stage] = {
            "count": len(values),
            "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95),
            "max_ms": max(values) if values else None,
        }

    return {
        "stages": stage_summary,
        "cycle_status_counts": cycle_status_counts,
    }


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    rank = ratio * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return round(lower + (upper - lower) * fraction, 3)


def load_events(path: Path | None = None) -> list[dict[str, Any]]:
    target = events_path() if path is None else path
    if not target.is_file():
        return []

    results: list[dict[str, Any]] = []
    try:
        raw_text = target.read_text(encoding="utf-8")
        chunks: list[str] = []
        for line in raw_text.splitlines():
            chunks.extend(line.split("\\n"))
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                payload = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                results.append(payload)
    except OSError:
        return []
    return results
