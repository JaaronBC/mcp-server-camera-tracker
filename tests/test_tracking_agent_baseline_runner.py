from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.tracking_agent_baseline_runner import (
    EventLogTailer,
    ObservationAccumulator,
    classify_failure_category,
    run_tracking_agent_baseline,
)


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += max(0.0, float(seconds))


class _AppendingClock(_FakeClock):
    def __init__(self, *, events_path: Path, event_payload: dict[str, object], start: float = 1000.0) -> None:
        super().__init__(start=start)
        self._events_path = events_path
        self._event_payload = event_payload
        self._did_append = False

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if self._did_append:
            return
        self._did_append = True
        self._events_path.write_text(
            json.dumps(self._event_payload) + "\n",
            encoding="utf-8",
        )


class TestTrackingAgentBaselineRunner(unittest.TestCase):
    def test_failure_classification_precedence(self) -> None:
        self.assertEqual(classify_failure_category(["vlm_timeout", "vlm_output_truncated"]), "timeout")
        self.assertEqual(classify_failure_category(["vlm_output_truncated", "vlm_invalid_json_response"]), "truncation")
        self.assertEqual(classify_failure_category(["vlm_invalid_json_response"]), "invalid_json")
        self.assertEqual(classify_failure_category(["vlm_unavailable"]), "other_failure")
        self.assertIsNone(classify_failure_category(["schema_complete_response"]))

    def test_deduplicates_repeated_observation_ids(self) -> None:
        accumulator = ObservationAccumulator(start_time=1000.0)
        snapshot = {
            "last_observation_timestamp": 1000.0,
            "tracking_agent_status": {"completions": 1},
            "tracking_agent_observation": {
                "observation_id": "obs-1",
                "timestamp": 1000.0,
                "proposal_latency_ms": 10.0,
                "uncertainties": ["schema_complete_response"],
                "updates": [{"uncertainties": ["no_vlm_proposal_used"]}],
            },
        }
        accumulator.process_snapshot(snapshot, sampled_at=1000.1)
        accumulator.process_snapshot(snapshot, sampled_at=1000.2)

        self.assertEqual(len(accumulator.seen_keys), 1)
        self.assertEqual(accumulator.duplicate_dropped, 1)
        self.assertEqual(accumulator.schema_complete, 1)
        self.assertEqual(accumulator.fallback_usage, 1)

    def test_detects_restart_on_completion_counter_rollback(self) -> None:
        accumulator = ObservationAccumulator(start_time=1000.0)

        accumulator.process_snapshot(
            {
                "tracking_agent_status": {"completions": 5},
                "tracking_agent_observation": {
                    "observation_id": "obs-a",
                    "timestamp": 1000.0,
                    "uncertainties": [],
                    "updates": [],
                },
            },
            sampled_at=1000.1,
        )
        accumulator.process_snapshot(
            {
                "tracking_agent_status": {"completions": 1},
                "tracking_agent_observation": {
                    "observation_id": "obs-b",
                    "timestamp": 1000.2,
                    "uncertainties": [],
                    "updates": [],
                },
            },
            sampled_at=1000.3,
        )
        accumulator.finalize(ended_at=1001.0)

        self.assertEqual(accumulator.status_counter_rollbacks, 1)
        self.assertGreaterEqual(len(accumulator.segments), 2)
        self.assertEqual(accumulator.segments[1]["restart_reason"], "status_counter_rollback")

    def test_event_tailer_handles_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latency_events.jsonl"
            path.write_text("", encoding="utf-8")

            tailer = EventLogTailer(
                path=path,
                start_time=1000.0,
                instrumentation_enabled=True,
            )
            tailer.initialize()

            path.write_text(
                json.dumps(
                    {
                        "event_type": "llm_request_completed",
                        "stage": "tracking_agent_llm_request",
                        "timestamp": 1001.0,
                        "status": "ok",
                        "duration_ms": 12.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            tailer.poll()
            self.assertEqual(tailer.completion_count(), 1)

            path.write_text("", encoding="utf-8")
            tailer.poll()
            self.assertTrue(tailer.truncation_detected)
            self.assertEqual(tailer.coverage_quality(), "observed_event_count")

    def test_event_tailer_handles_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latency_events.jsonl"
            path.write_text("", encoding="utf-8")

            tailer = EventLogTailer(
                path=path,
                start_time=1000.0,
                instrumentation_enabled=True,
            )
            tailer.initialize()

            rotated = Path(temp_dir) / "latency_events.rotated.jsonl"
            path.rename(rotated)
            path.write_text(
                json.dumps(
                    {
                        "event_type": "llm_request_completed",
                        "stage": "tracking_agent_llm_request",
                        "timestamp": 1001.0,
                        "status": "ok",
                        "duration_ms": 11.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            tailer.poll()
            self.assertTrue(tailer.rotation_detected)
            self.assertEqual(tailer.completion_count(), 1)
            self.assertEqual(tailer.coverage_quality(), "observed_event_count")

    def test_missing_sources_do_not_crash_and_report_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_path = root / "report.json"
            clock = _FakeClock()

            report = run_tracking_agent_baseline(
                duration_seconds=0.5,
                sample_interval_seconds=0.25,
                captures_dir=root,
                radar_state_path=root / "latest_radar_state.json",
                latency_events_path=root / "latency_events.jsonl",
                output_json=output_path,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            self.assertTrue(output_path.is_file())
            self.assertFalse(report["metrics"]["completed_requests_total"]["available"])
            self.assertFalse(report["metrics"]["applied_non_fallback_llm_updates_total"]["available"])
            self.assertIn("authoritative_applied_counter_not_exposed", report["metrics"]["applied_non_fallback_llm_updates_total"]["details"]["reason"])

    def test_incomplete_logging_is_labeled_observed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            radar_path = root / "latest_radar_state.json"
            events_path = root / "latency_events.jsonl"
            output_path = root / "report.json"

            radar_payload = {
                "last_observation_timestamp": 1000.0,
                "tracking_agent_status": {"completions": 1},
                "tracking_agent_observation": {
                    "observation_id": "obs-1",
                    "timestamp": 1000.0,
                    "proposal_latency_ms": 15.0,
                    "uncertainties": ["schema_complete_response"],
                    "updates": [],
                },
            }
            radar_path.write_text(json.dumps(radar_payload), encoding="utf-8")
            events_path.write_text("", encoding="utf-8")
            clock = _AppendingClock(
                events_path=events_path,
                event_payload={
                    "event_type": "llm_request_completed",
                    "stage": "tracking_agent_llm_request",
                    "timestamp": 1000.1,
                    "status": "error",
                    "error_code": "timeout",
                    "duration_ms": 1001.0,
                },
            )

            report = run_tracking_agent_baseline(
                duration_seconds=0.5,
                sample_interval_seconds=0.25,
                captures_dir=root,
                radar_state_path=radar_path,
                latency_events_path=events_path,
                output_json=output_path,
                now_fn=clock.now,
                sleep_fn=clock.sleep,
            )

            completed = report["metrics"]["completed_requests_total"]
            timeout_metric = report["metrics"]["failures"]["timeout"]
            self.assertEqual(completed["source"], "latency_events.jsonl")
            self.assertEqual(completed["quality"], "observed_event_count")
            self.assertEqual(timeout_metric["source"], "latency_events.jsonl")
            self.assertEqual(timeout_metric["quality"], "observed_event_count")

    def test_rejects_output_path_collision_with_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            radar_path = root / "latest_radar_state.json"
            latency_path = root / "latency_events.jsonl"
            radar_path.write_text("{}", encoding="utf-8")
            latency_path.write_text("", encoding="utf-8")
            clock = _FakeClock()

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                run_tracking_agent_baseline(
                    duration_seconds=0.5,
                    sample_interval_seconds=0.25,
                    captures_dir=root,
                    radar_state_path=radar_path,
                    latency_events_path=latency_path,
                    output_json=radar_path,
                    now_fn=clock.now,
                    sleep_fn=clock.sleep,
                )

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                run_tracking_agent_baseline(
                    duration_seconds=0.5,
                    sample_interval_seconds=0.25,
                    captures_dir=root,
                    radar_state_path=radar_path,
                    latency_events_path=latency_path,
                    output_json=latency_path,
                    now_fn=clock.now,
                    sleep_fn=clock.sleep,
                )


if __name__ == "__main__":
    unittest.main()
