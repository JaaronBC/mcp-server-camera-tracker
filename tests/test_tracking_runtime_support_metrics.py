from __future__ import annotations

import types
import unittest

from agents.contracts import TrackingAgentObservation, VisualAssociationUpdate
from agents.tracking.runtime_support import (
    apply_tracking_agent_metric_update_to_state,
    summarize_tracking_agent_observation_metrics,
)


class TestTrackingRuntimeSupportMetrics(unittest.TestCase):
    def _build_observation(self, uncertainties: list[str]) -> TrackingAgentObservation:
        return TrackingAgentObservation(
            timestamp=1000.0,
            source_agent="visual_tracking",
            vlm_model="qwen-3.8-27b",
            proposal_latency_ms=None,
            frame_reference=None,
            updates=[
                VisualAssociationUpdate(
                    detection_id="det-1",
                    resolved_track_id="yolo-1",
                    accepted=True,
                    rejection_reasons=[],
                    uncertainties=[],
                    supporting_evidence=["yolo_candidate_track"],
                )
            ],
            uncertainties=uncertainties,
        )

    def test_missing_key_observation_updates_unavailable_metrics(self) -> None:
        observation = self._build_observation(["vlm_api_key_missing"])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 1, "rejected": 0, "unmatched": 0},
        )

        self.assertEqual(summary["completions_increment"], 1)
        self.assertEqual(summary["unavailable_increment"], 1)
        self.assertEqual(summary["last_error"], "vlm_api_key_missing")

    def test_failed_observation_updates_unavailable_metrics(self) -> None:
        observation = self._build_observation(["vlm_unavailable"])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 0, "rejected": 1, "unmatched": 0},
        )

        self.assertEqual(summary["completions_increment"], 1)
        self.assertEqual(summary["unavailable_increment"], 1)
        self.assertEqual(summary["last_error"], "vlm_unavailable")

    def test_apply_metric_update_mutates_status_counters(self) -> None:
        state = types.SimpleNamespace(
            lock=types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, *args: False),
            tracking_agent_completions=0,
            tracking_agent_last_latency_ms=None,
            tracking_agent_total_updates_accepted=0,
            tracking_agent_total_updates_rejected=0,
            tracking_agent_total_updates_unmatched=0,
            tracking_agent_unavailable_count=0,
            tracking_agent_last_error=None,
        )

        class _Lock:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        state.lock = _Lock()

        apply_tracking_agent_metric_update_to_state(
            state=state,
            metric_update={
                "completions_increment": 1,
                "last_latency_ms": 7.5,
                "accepted": 2,
                "rejected": 1,
                "unmatched": 0,
                "unavailable_increment": 1,
                "last_error": "vlm_api_key_missing",
            },
        )

        self.assertEqual(state.tracking_agent_completions, 1)
        self.assertEqual(state.tracking_agent_last_latency_ms, 7.5)
        self.assertEqual(state.tracking_agent_total_updates_accepted, 2)
        self.assertEqual(state.tracking_agent_total_updates_rejected, 1)
        self.assertEqual(state.tracking_agent_total_updates_unmatched, 0)
        self.assertEqual(state.tracking_agent_unavailable_count, 1)
        self.assertEqual(state.tracking_agent_last_error, "vlm_api_key_missing")


if __name__ == "__main__":
    unittest.main()
