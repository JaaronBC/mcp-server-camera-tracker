from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from agents import latency_instrumentation


class TestLatencyInstrumentation(unittest.TestCase):
    def test_emit_event_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latency_events.jsonl"
            previous_enabled = os.environ.pop("PHASE16E_LATENCY_INSTRUMENTATION_ENABLED", None)
            previous_path = os.environ.get("PHASE16E_LATENCY_EVENTS_PATH")
            os.environ["PHASE16E_LATENCY_EVENTS_PATH"] = str(path)
            try:
                latency_instrumentation.emit_event("test_event", stage="x", duration_ms=1.0)
            finally:
                if previous_enabled is not None:
                    os.environ["PHASE16E_LATENCY_INSTRUMENTATION_ENABLED"] = previous_enabled
                else:
                    os.environ.pop("PHASE16E_LATENCY_INSTRUMENTATION_ENABLED", None)
                if previous_path is not None:
                    os.environ["PHASE16E_LATENCY_EVENTS_PATH"] = previous_path
                else:
                    os.environ.pop("PHASE16E_LATENCY_EVENTS_PATH", None)

            self.assertFalse(path.exists())

    def test_emit_event_enabled_with_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latency_events.jsonl"
            previous_enabled = os.environ.get("PHASE16E_LATENCY_INSTRUMENTATION_ENABLED")
            previous_path = os.environ.get("PHASE16E_LATENCY_EVENTS_PATH")
            os.environ["PHASE16E_LATENCY_INSTRUMENTATION_ENABLED"] = "1"
            os.environ["PHASE16E_LATENCY_EVENTS_PATH"] = str(path)
            previous = latency_instrumentation.set_context(
                correlation_id="corr-1",
                cycle_id="cycle-1",
                track_id="track-1",
            )
            try:
                latency_instrumentation.emit_event(
                    "test_event",
                    stage="test_stage",
                    duration_ms=12.5,
                    status="ok",
                )
                self.assertTrue(path.exists())
                text = path.read_text(encoding="utf-8")
                self.assertIn('"event_type": "test_event"', text)
                self.assertIn('"stage": "test_stage"', text)
                self.assertIn('"correlation_id": "corr-1"', text)
                self.assertIn('"cycle_id": "cycle-1"', text)
                self.assertIn('"track_id": "track-1"', text)
            finally:
                latency_instrumentation.restore_context(previous)
                if previous_enabled is not None:
                    os.environ["PHASE16E_LATENCY_INSTRUMENTATION_ENABLED"] = previous_enabled
                else:
                    os.environ.pop("PHASE16E_LATENCY_INSTRUMENTATION_ENABLED", None)
                if previous_path is not None:
                    os.environ["PHASE16E_LATENCY_EVENTS_PATH"] = previous_path
                else:
                    os.environ.pop("PHASE16E_LATENCY_EVENTS_PATH", None)

    def test_summarize_events_counts_and_percentiles(self) -> None:
        summary = latency_instrumentation.summarize_events(
            [
                {
                    "event_type": "reasoning_cycle_completed",
                    "cycle_status": "completed",
                    "stage": "orchestration_cycle",
                    "duration_ms": 10.0,
                },
                {
                    "event_type": "reasoning_cycle_completed",
                    "cycle_status": "timed_out",
                    "stage": "orchestration_cycle",
                    "duration_ms": 30.0,
                },
                {
                    "event_type": "reasoning_cycle_completed",
                    "cycle_status": "failed",
                    "stage": "orchestration_cycle",
                    "duration_ms": 50.0,
                },
            ]
        )

        self.assertEqual(summary["cycle_status_counts"]["completed"], 1)
        self.assertEqual(summary["cycle_status_counts"]["timed_out"], 1)
        self.assertEqual(summary["cycle_status_counts"]["failed"], 1)
        stage_summary = summary["stages"]["orchestration_cycle"]
        self.assertEqual(stage_summary["count"], 3)
        self.assertEqual(stage_summary["p50_ms"], 30.0)
        self.assertEqual(stage_summary["max_ms"], 50.0)
