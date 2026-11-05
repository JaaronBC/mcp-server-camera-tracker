from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


def _env_instrumentation_enabled() -> bool:
    value = os.getenv("PHASE16E_LATENCY_INSTRUMENTATION_ENABLED", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_read_json_object(
    path: Path,
    *,
    max_retries: int,
    retry_delay_ms: int,
    sleep_fn: Callable[[float], None],
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"

    attempts = max(1, int(max_retries))
    for index in range(attempts):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if index + 1 < attempts:
                sleep_fn(max(0.0, retry_delay_ms / 1000.0))
                continue
            return None, "invalid_json"
        except OSError:
            return None, "unreadable"

        if not isinstance(payload, dict):
            return None, "not_object"
        return payload, None

    return None, "invalid_json"


def classify_failure_category(uncertainties: list[str]) -> str | None:
    values = set(uncertainties)
    if "vlm_timeout" in values:
        return "timeout"
    if "vlm_output_truncated" in values:
        return "truncation"
    if "vlm_invalid_json_response" in values:
        return "invalid_json"
    if any(value.startswith("vlm_") for value in values):
        return "other_failure"
    return None


def _observation_key(observation: dict[str, Any]) -> str | None:
    observation_id = observation.get("observation_id")
    if isinstance(observation_id, str) and observation_id.strip():
        return f"id:{observation_id.strip()}"

    ts = observation.get("timestamp")
    corr = observation.get("llm_correlation_id")
    latency = observation.get("proposal_latency_ms")
    if isinstance(ts, (int, float)) and isinstance(corr, str) and corr.strip():
        return f"fallback:{float(ts):.6f}:{corr.strip()}:{latency}"
    return None


def _metric(
    *,
    value: Any,
    source: str,
    quality: str,
    available: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "available": bool(available),
        "value": value,
        "source": source,
        "quality": quality,
        "details": {} if details is None else details,
    }


def _latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    ordered = sorted(values)

    def _percentile(ratio: float) -> float:
        if len(ordered) == 1:
            return round(ordered[0], 3)
        rank = ratio * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        frac = rank - lower
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * frac, 3)

    mean = round(sum(ordered) / len(ordered), 3)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 3),
        "p50": _percentile(0.5),
        "p95": _percentile(0.95),
        "max": round(ordered[-1], 3),
        "mean": mean,
    }


class EventLogTailer:
    def __init__(
        self,
        *,
        path: Path,
        start_time: float,
        instrumentation_enabled: bool,
    ) -> None:
        self.path = path
        self.start_time = float(start_time)
        self.instrumentation_enabled = bool(instrumentation_enabled)
        self.start_offset = 0
        self.current_offset = 0
        self.initial_inode: int | None = None
        self.current_inode: int | None = None
        self.rotation_detected = False
        self.truncation_detected = False
        self.malformed_lines = 0
        self.missing_file_observed = False
        self.pending_fragment = ""
        self.completion_events: list[dict[str, Any]] = []

    def initialize(self) -> None:
        if not self.path.is_file():
            self.missing_file_observed = True
            self.start_offset = 0
            self.current_offset = 0
            self.initial_inode = None
            self.current_inode = None
            return

        stat = self.path.stat()
        self.start_offset = int(stat.st_size)
        self.current_offset = int(stat.st_size)
        self.initial_inode = int(stat.st_ino)
        self.current_inode = int(stat.st_ino)

    def _parse_line(self, line: str) -> None:
        if not line.strip():
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self.malformed_lines += 1
            return

        if not isinstance(payload, dict):
            self.malformed_lines += 1
            return

        if payload.get("event_type") != "llm_request_completed":
            return
        if payload.get("stage") != "tracking_agent_llm_request":
            return

        timestamp = payload.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            self.malformed_lines += 1
            return

        if float(timestamp) < self.start_time:
            return

        self.completion_events.append(payload)

    def poll(self) -> None:
        if not self.path.is_file():
            self.missing_file_observed = True
            return

        stat = self.path.stat()
        inode = int(stat.st_ino)
        size = int(stat.st_size)

        if self.current_inode is None:
            self.current_inode = inode
        elif inode != self.current_inode:
            self.rotation_detected = True
            self.current_inode = inode
            self.current_offset = 0
            self.pending_fragment = ""

        if size < self.current_offset:
            self.truncation_detected = True
            self.current_offset = 0
            self.pending_fragment = ""

        if size == self.current_offset:
            return

        with self.path.open("rb") as handle:
            handle.seek(self.current_offset)
            chunk = handle.read(size - self.current_offset)
        self.current_offset = size

        text = chunk.decode("utf-8", errors="replace")
        combined = self.pending_fragment + text
        lines = combined.split("\n")
        if combined and not combined.endswith("\n"):
            self.pending_fragment = lines.pop()
        else:
            self.pending_fragment = ""

        for line in lines:
            self._parse_line(line)

    def finalize(self) -> None:
        if self.pending_fragment.strip():
            self._parse_line(self.pending_fragment)
        self.pending_fragment = ""

    def completion_count(self) -> int:
        return len(self.completion_events)

    def timeout_count(self) -> int:
        count = 0
        for event in self.completion_events:
            if event.get("status") == "error" and event.get("error_code") == "timeout":
                count += 1
        return count

    def latency_values_ms(self) -> list[float]:
        values: list[float] = []
        for event in self.completion_events:
            duration = event.get("duration_ms")
            if isinstance(duration, (int, float)):
                values.append(float(duration))
        return values

    def coverage_quality(self) -> str:
        if not self.instrumentation_enabled:
            return "observed_event_count"
        if self.missing_file_observed:
            return "observed_event_count"
        if self.rotation_detected or self.truncation_detected or self.malformed_lines > 0:
            return "observed_event_count"
        return "exact_counter"

    def coverage_complete(self) -> bool:
        return self.coverage_quality() == "exact_counter"


class ObservationAccumulator:
    def __init__(self, *, start_time: float) -> None:
        self.start_time = float(start_time)
        self.seen_keys: set[str] = set()
        self.duplicate_dropped = 0
        self.partial_read_errors = 0
        self.missing_radar_snapshots = 0
        self.sample_count = 0

        self.valid_responses = 0
        self.schema_complete = 0
        self.schema_incomplete = 0
        self.failures = {
            "timeout": 0,
            "truncation": 0,
            "invalid_json": 0,
            "other_failure": 0,
        }
        self.fallback_usage = 0

        self.observation_latency_ms: list[float] = []
        self.dashboard_observation_age_seconds: list[float] = []

        self.latest_status_counters: dict[str, Any] = {}
        self.status_counter_rollbacks = 0
        self.segments: list[dict[str, Any]] = []
        self._current_segment = {
            "segment_id": 1,
            "started_at": self.start_time,
            "restart_reason": None,
        }
        self._last_completions_counter: int | None = None

        self.applied_counter_available = False
        self.applied_counter_start: int | None = None
        self.applied_counter_end: int | None = None

    def _start_new_segment(self, now_value: float, reason: str) -> None:
        segment = dict(self._current_segment)
        segment["ended_at"] = now_value
        self.segments.append(segment)
        self._current_segment = {
            "segment_id": int(segment["segment_id"]) + 1,
            "started_at": now_value,
            "restart_reason": reason,
        }

    def process_snapshot(self, payload: dict[str, Any], *, sampled_at: float) -> None:
        self.sample_count += 1

        last_observation_timestamp = payload.get("last_observation_timestamp")
        if isinstance(last_observation_timestamp, (int, float)):
            self.dashboard_observation_age_seconds.append(
                max(0.0, float(sampled_at) - float(last_observation_timestamp))
            )

        status = payload.get("tracking_agent_status")
        if isinstance(status, dict):
            self.latest_status_counters = dict(status)
            completions = status.get("completions")
            if isinstance(completions, int):
                if self._last_completions_counter is not None and completions < self._last_completions_counter:
                    self.status_counter_rollbacks += 1
                    self._start_new_segment(sampled_at, "status_counter_rollback")
                self._last_completions_counter = completions

            applied_counter_value = status.get("tracking_agent_llm_applied_update_count")
            if isinstance(applied_counter_value, int):
                self.applied_counter_available = True
                if self.applied_counter_start is None:
                    self.applied_counter_start = applied_counter_value
                self.applied_counter_end = applied_counter_value

        observation = payload.get("tracking_agent_observation")
        if not isinstance(observation, dict):
            return

        key = _observation_key(observation)
        if key is None:
            return
        if key in self.seen_keys:
            self.duplicate_dropped += 1
            return
        self.seen_keys.add(key)

        latency = observation.get("proposal_latency_ms")
        if isinstance(latency, (int, float)):
            self.observation_latency_ms.append(float(latency))

        uncertainties_raw = observation.get("uncertainties")
        uncertainties = [str(value) for value in uncertainties_raw] if isinstance(uncertainties_raw, list) else []

        failure = classify_failure_category(uncertainties)
        if failure is None:
            self.valid_responses += 1
        else:
            self.failures[failure] += 1

        if "schema_complete_response" in uncertainties:
            self.schema_complete += 1
        elif "schema_incomplete_response" in uncertainties:
            self.schema_incomplete += 1

        updates_raw = observation.get("updates")
        if isinstance(updates_raw, list):
            for update in updates_raw:
                if not isinstance(update, dict):
                    continue
                update_uncertainties_raw = update.get("uncertainties")
                update_uncertainties = (
                    [str(value) for value in update_uncertainties_raw]
                    if isinstance(update_uncertainties_raw, list)
                    else []
                )
                if "no_vlm_proposal_used" in update_uncertainties:
                    self.fallback_usage += 1

    def finalize(self, *, ended_at: float) -> None:
        segment = dict(self._current_segment)
        segment["ended_at"] = ended_at
        self.segments.append(segment)


def _classify_metric_quality_from_observations(accumulator: ObservationAccumulator) -> str:
    return "sampled_observation_count"


def _build_report(
    *,
    started_at: float,
    ended_at: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    instrumentation_enabled: bool,
    event_tailer: EventLogTailer,
    accumulator: ObservationAccumulator,
    radar_state_path: Path,
    latency_events_path: Path,
) -> dict[str, Any]:
    event_completion_count = event_tailer.completion_count()
    event_coverage_quality = event_tailer.coverage_quality()

    if event_completion_count > 0:
        completed_metric = _metric(
            value=event_completion_count,
            source="latency_events.jsonl",
            quality=event_coverage_quality,
            details={
                "instrumentation_enabled": instrumentation_enabled,
                "start_offset": event_tailer.start_offset,
                "rotation_detected": event_tailer.rotation_detected,
                "truncation_detected": event_tailer.truncation_detected,
                "malformed_lines": event_tailer.malformed_lines,
            },
        )
    elif len(accumulator.seen_keys) > 0:
        completed_metric = _metric(
            value=len(accumulator.seen_keys),
            source="latest_radar_state.json",
            quality="sampled_observation_count",
            details={
                "missed_observation_possible": True,
                "reason": "no_tracking_completion_events_in_window",
            },
        )
    else:
        completed_metric = _metric(
            value=None,
            source="none",
            quality="unavailable",
            available=False,
            details={"reason": "no_events_and_no_observations"},
        )

    timeout_metric: dict[str, Any]
    if event_completion_count > 0:
        timeout_metric = _metric(
            value=event_tailer.timeout_count(),
            source="latency_events.jsonl",
            quality=event_coverage_quality,
            details={"classification": "error_code=timeout"},
        )
    else:
        timeout_metric = _metric(
            value=accumulator.failures["timeout"],
            source="latest_radar_state.json",
            quality=_classify_metric_quality_from_observations(accumulator),
            details={"classification": "observation_uncertainties"},
        )

    valid_metric = _metric(
        value=accumulator.valid_responses,
        source="latest_radar_state.json",
        quality=_classify_metric_quality_from_observations(accumulator),
        details={
            "rule": "observation has no vlm_* uncertainty flag",
            "missed_observation_possible": True,
        },
    )

    schema_complete_metric = _metric(
        value=accumulator.schema_complete,
        source="latest_radar_state.json",
        quality=_classify_metric_quality_from_observations(accumulator),
        details={
            "rule": "observation uncertainty includes schema_complete_response",
            "missed_observation_possible": True,
        },
    )

    truncation_metric = _metric(
        value=accumulator.failures["truncation"],
        source="latest_radar_state.json",
        quality=_classify_metric_quality_from_observations(accumulator),
        details={"classification": "observation_uncertainties"},
    )
    invalid_json_metric = _metric(
        value=accumulator.failures["invalid_json"],
        source="latest_radar_state.json",
        quality=_classify_metric_quality_from_observations(accumulator),
        details={"classification": "observation_uncertainties"},
    )
    other_failure_metric = _metric(
        value=accumulator.failures["other_failure"],
        source="latest_radar_state.json",
        quality=_classify_metric_quality_from_observations(accumulator),
        details={"classification": "observation_uncertainties"},
    )

    fallback_metric = _metric(
        value=accumulator.fallback_usage,
        source="latest_radar_state.json",
        quality=_classify_metric_quality_from_observations(accumulator),
        details={
            "rule": "count updates with no_vlm_proposal_used",
            "missed_observation_possible": True,
        },
    )

    if accumulator.applied_counter_available and accumulator.applied_counter_start is not None and accumulator.applied_counter_end is not None:
        applied_metric = _metric(
            value=max(0, accumulator.applied_counter_end - accumulator.applied_counter_start),
            source="tracking_agent_status",
            quality="exact_counter",
            details={
                "start": accumulator.applied_counter_start,
                "end": accumulator.applied_counter_end,
            },
        )
    else:
        applied_metric = _metric(
            value=None,
            source="tracking_agent_status",
            quality="unavailable",
            available=False,
            details={
                "reason": "authoritative_applied_counter_not_exposed",
            },
        )

    event_latency_values = event_tailer.latency_values_ms()
    if event_latency_values:
        request_latency_metric = _metric(
            value=_latency_stats(event_latency_values),
            source="latency_events.jsonl",
            quality=event_coverage_quality,
            details={"field": "duration_ms"},
        )
    else:
        request_latency_metric = _metric(
            value=_latency_stats(accumulator.observation_latency_ms),
            source="latest_radar_state.json",
            quality="sampled_observation_count",
            details={
                "field": "tracking_agent_observation.proposal_latency_ms",
                "missed_observation_possible": True,
            },
        )

    if accumulator.dashboard_observation_age_seconds:
        dashboard_age_metric = _metric(
            value=_latency_stats(accumulator.dashboard_observation_age_seconds),
            source="latest_radar_state.json",
            quality="sampled_observation_count",
            details={
                "definition": "sample_time - last_observation_timestamp",
            },
        )
    else:
        dashboard_age_metric = _metric(
            value=None,
            source="latest_radar_state.json",
            quality="unavailable",
            available=False,
            details={"reason": "last_observation_timestamp_unavailable"},
        )

    llm_result_apply_age_metric = _metric(
        value=None,
        source="none",
        quality="unavailable",
        available=False,
        details={"reason": "llm_application_timestamp_not_exposed"},
    )

    return {
        "meta": {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(duration_seconds, 3),
            "sample_interval_seconds": sample_interval_seconds,
            "timeout_config_expected_seconds": 1.0,
            "read_only": True,
        },
        "sources": {
            "radar_state_path": str(radar_state_path),
            "latency_events_path": str(latency_events_path),
            "instrumentation_enabled": instrumentation_enabled,
            "event_log_start_offset": event_tailer.start_offset,
            "event_log_end_offset": event_tailer.current_offset,
            "event_log_rotation_detected": event_tailer.rotation_detected,
            "event_log_truncation_detected": event_tailer.truncation_detected,
            "event_log_malformed_lines": event_tailer.malformed_lines,
        },
        "segments": accumulator.segments,
        "metrics": {
            "completed_requests_total": completed_metric,
            "valid_responses_total": valid_metric,
            "schema_complete_responses_total": schema_complete_metric,
            "failures": {
                "timeout": timeout_metric,
                "truncation": truncation_metric,
                "invalid_json": invalid_json_metric,
                "other": other_failure_metric,
            },
            "fallback_usage_total": fallback_metric,
            "applied_non_fallback_llm_updates_total": applied_metric,
            "request_latency_ms": request_latency_metric,
            "dashboard_observation_freshness_age_seconds": dashboard_age_metric,
            "llm_result_application_age_seconds": llm_result_apply_age_metric,
        },
        "measurement_quality": {
            "sample_count": accumulator.sample_count,
            "deduplicated_observation_count": len(accumulator.seen_keys),
            "duplicate_observations_dropped": accumulator.duplicate_dropped,
            "radar_snapshot_missing_count": accumulator.missing_radar_snapshots,
            "radar_snapshot_partial_read_errors": accumulator.partial_read_errors,
            "status_counter_rollbacks": accumulator.status_counter_rollbacks,
            "event_coverage_quality": event_coverage_quality,
            "event_coverage_complete": event_tailer.coverage_complete(),
            "notes": [
                "completed request totals are exact only when event_coverage_complete=true",
                "observation-derived metrics are sampled and may miss intermediate observations",
                "applied update metric remains unavailable unless authoritative counter is exposed",
            ],
        },
    }


def run_tracking_agent_baseline(
    *,
    duration_seconds: float,
    sample_interval_seconds: float,
    captures_dir: Path,
    radar_state_path: Path,
    latency_events_path: Path,
    output_json: Path,
    max_parse_retries: int = 3,
    parse_retry_delay_ms: int = 50,
    now_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be greater than zero")

    output_resolved = output_json.expanduser().resolve()
    radar_resolved = radar_state_path.expanduser().resolve()
    latency_resolved = latency_events_path.expanduser().resolve()
    if output_resolved in {radar_resolved, latency_resolved}:
        raise ValueError("output_json must not overwrite a production input file")

    started_at = float(now_fn())
    ended_at_target = started_at + float(duration_seconds)

    instrumentation_enabled = _env_instrumentation_enabled()

    event_tailer = EventLogTailer(
        path=latency_events_path,
        start_time=started_at,
        instrumentation_enabled=instrumentation_enabled,
    )
    event_tailer.initialize()

    accumulator = ObservationAccumulator(start_time=started_at)

    while True:
        now_value = float(now_fn())
        if now_value > ended_at_target:
            break

        radar_payload, radar_error = _safe_read_json_object(
            radar_state_path,
            max_retries=max_parse_retries,
            retry_delay_ms=parse_retry_delay_ms,
            sleep_fn=sleep_fn,
        )
        if radar_payload is None:
            if radar_error == "missing":
                accumulator.missing_radar_snapshots += 1
            else:
                accumulator.partial_read_errors += 1
        else:
            accumulator.process_snapshot(radar_payload, sampled_at=now_value)

        event_tailer.poll()

        remaining = ended_at_target - float(now_fn())
        if remaining <= 0:
            break
        sleep_fn(min(sample_interval_seconds, remaining))

    event_tailer.poll()
    event_tailer.finalize()

    ended_at = float(now_fn())
    accumulator.finalize(ended_at=ended_at)

    report = _build_report(
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=ended_at - started_at,
        sample_interval_seconds=sample_interval_seconds,
        instrumentation_enabled=instrumentation_enabled,
        event_tailer=event_tailer,
        accumulator=accumulator,
        radar_state_path=radar_state_path,
        latency_events_path=latency_events_path,
    )

    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    output_resolved.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Tracking Agent baseline runner")
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.2)
    parser.add_argument("--captures-dir", type=Path, default=Path("captures"))
    parser.add_argument("--radar-state-path", type=Path, default=None)
    parser.add_argument("--latency-events-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-parse-retries", type=int, default=3)
    parser.add_argument("--parse-retry-delay-ms", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    captures_dir = args.captures_dir
    radar_state_path = args.radar_state_path or (captures_dir / "latest_radar_state.json")
    latency_events_path = args.latency_events_path or (captures_dir / "latency_events.jsonl")

    run_tracking_agent_baseline(
        duration_seconds=float(args.duration_seconds),
        sample_interval_seconds=float(args.sample_interval_seconds),
        captures_dir=captures_dir,
        radar_state_path=radar_state_path,
        latency_events_path=latency_events_path,
        output_json=args.output_json,
        max_parse_retries=int(args.max_parse_retries),
        parse_retry_delay_ms=int(args.parse_retry_delay_ms),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
