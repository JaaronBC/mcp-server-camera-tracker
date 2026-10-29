from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from agents.dashboard import rendering


class TestDashboardRenderingAssessmentSummary(unittest.TestCase):
    def test_format_assessment_summary_unavailable(self) -> None:
        summary, limits, color = rendering._format_assessment_summary(time.time(), None)

        self.assertIn("assessment: unavailable", summary)
        self.assertIn("event_log=missing_or_empty", limits)
        self.assertEqual(color, (170, 170, 170))

    def test_format_assessment_summary_fresh(self) -> None:
        now = time.time()
        payload = {
            "timestamp": now - 2.0,
            "status": "ok",
            "assessment_label": "stable_motion",
            "handoff_quality": "healthy",
            "observation_agent_error_code": None,
            "reasoning_agent_error_code": None,
            "cycle_error_code": None,
        }

        summary, limits, color = rendering._format_assessment_summary(now, payload)

        self.assertIn("status=ok", summary)
        self.assertIn("label=stable_motion", summary)
        self.assertIn("handoff=healthy", summary)
        self.assertIn("fresh", summary)
        self.assertIn("obs_err=none", limits)
        self.assertIn("rea_err=none", limits)
        self.assertIn("cycle_err=none", limits)
        self.assertEqual(color, (180, 220, 180))

    def test_format_assessment_summary_stale(self) -> None:
        now = time.time()
        payload = {
            "timestamp": now - (rendering.AGENT_EVENT_FRESHNESS_SECONDS + 5.0),
            "status": "ok",
            "assessment_label": "n/a",
            "handoff_quality": "healthy",
        }

        summary, _limits, color = rendering._format_assessment_summary(now, payload)

        self.assertIn("stale", summary)
        self.assertEqual(color, (0, 180, 255))

    def test_load_latest_agent_event_uses_last_valid_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events_path = Path(temp_dir) / "agent_events.jsonl"
            events_path.write_text(
                "\n".join(
                    [
                        "not-json",
                        json.dumps({"timestamp": 1.0, "status": "ok", "assessment_label": "old"}),
                        json.dumps({"timestamp": 2.0, "status": "error", "assessment_label": "latest"}),
                    ]
                ),
                encoding="utf-8",
            )

            mtime, payload = rendering._load_latest_agent_event(0.0, None, events_path=events_path)

            self.assertGreater(mtime, 0.0)
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertIn("latest", payload)
            self.assertIsInstance(payload["latest"], dict)
            self.assertEqual(payload["latest"].get("assessment_label"), "latest")

    def test_select_assessment_event_prefers_selected_track(self) -> None:
        snapshot = {
            "latest": {"track_id": "track-b", "assessment_label": "global-latest"},
            "latest_by_track": {
                "track-a": {"track_id": "track-a", "assessment_label": "track-a-latest"},
            },
        }

        selected = rendering._select_assessment_event(snapshot, "track-a")

        self.assertIsInstance(selected, dict)
        assert isinstance(selected, dict)
        self.assertEqual(selected.get("assessment_label"), "track-a-latest")

    def test_select_assessment_event_falls_back_to_global(self) -> None:
        snapshot = {
            "latest": {"track_id": "track-b", "assessment_label": "global-latest"},
            "latest_by_track": {
                "track-a": {"track_id": "track-a", "assessment_label": "track-a-latest"},
            },
        }

        selected = rendering._select_assessment_event(snapshot, "track-z")

        self.assertIsInstance(selected, dict)
        assert isinstance(selected, dict)
        self.assertEqual(selected.get("assessment_label"), "global-latest")


if __name__ == "__main__":
    unittest.main()
