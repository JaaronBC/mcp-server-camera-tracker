from __future__ import annotations

import types
import unittest

from agents.contracts import TrackingAgentObservation, VisualAssociationUpdate
from agents.tracking.tracking_types import GroundPosition, RadarTrackSnapshot
from agents.tracking.runtime_support import (
    apply_tracking_agent_updates_to_tracks,
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

    def test_timeout_observation_updates_timeout_metrics(self) -> None:
        observation = self._build_observation(["vlm_timeout"])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 0, "rejected": 1, "unmatched": 0, "fallback_used": 1, "llm_applied": 0},
        )

        self.assertEqual(summary["completions_increment"], 1)
        self.assertEqual(summary["timeout_increment"], 1)
        self.assertEqual(summary["truncation_increment"], 0)
        self.assertEqual(summary["invalid_json_increment"], 0)
        self.assertEqual(summary["other_failure_increment"], 0)
        self.assertEqual(summary["valid_response_increment"], 0)
        self.assertEqual(summary["schema_complete_increment"], 0)
        self.assertEqual(summary["schema_incomplete_increment"], 0)
        self.assertEqual(summary["unavailable_increment"], 1)
        self.assertEqual(summary["last_error"], "vlm_timeout")

    def test_truncation_observation_updates_truncation_metrics(self) -> None:
        observation = self._build_observation(["vlm_output_truncated"])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 0, "rejected": 1, "unmatched": 0, "fallback_used": 1, "llm_applied": 0},
        )

        self.assertEqual(summary["timeout_increment"], 0)
        self.assertEqual(summary["truncation_increment"], 1)
        self.assertEqual(summary["invalid_json_increment"], 0)
        self.assertEqual(summary["other_failure_increment"], 0)
        self.assertEqual(summary["valid_response_increment"], 0)
        self.assertEqual(summary["schema_complete_increment"], 0)
        self.assertEqual(summary["schema_incomplete_increment"], 0)
        self.assertEqual(summary["unavailable_increment"], 1)

    def test_invalid_json_observation_updates_invalid_json_metrics(self) -> None:
        observation = self._build_observation(["vlm_invalid_json_response"])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 0, "rejected": 1, "unmatched": 0, "fallback_used": 1, "llm_applied": 0},
        )

        self.assertEqual(summary["timeout_increment"], 0)
        self.assertEqual(summary["truncation_increment"], 0)
        self.assertEqual(summary["invalid_json_increment"], 1)
        self.assertEqual(summary["other_failure_increment"], 0)
        self.assertEqual(summary["valid_response_increment"], 0)
        self.assertEqual(summary["schema_complete_increment"], 0)
        self.assertEqual(summary["schema_incomplete_increment"], 0)
        self.assertEqual(summary["unavailable_increment"], 1)

    def test_other_failure_observation_updates_other_failure_metrics(self) -> None:
        observation = self._build_observation(["vlm_api_key_missing"])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 0, "rejected": 1, "unmatched": 0, "fallback_used": 1, "llm_applied": 0},
        )

        self.assertEqual(summary["timeout_increment"], 0)
        self.assertEqual(summary["truncation_increment"], 0)
        self.assertEqual(summary["invalid_json_increment"], 0)
        self.assertEqual(summary["other_failure_increment"], 1)
        self.assertEqual(summary["valid_response_increment"], 0)
        self.assertEqual(summary["schema_complete_increment"], 0)
        self.assertEqual(summary["schema_incomplete_increment"], 0)
        self.assertEqual(summary["unavailable_increment"], 1)
        self.assertEqual(summary["last_error"], "vlm_api_key_missing")

    def test_valid_response_increments_valid_counter(self) -> None:
        observation = self._build_observation(["schema_complete_response"])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 1, "rejected": 0, "unmatched": 0, "fallback_used": 0, "llm_applied": 1},
        )

        self.assertEqual(summary["valid_response_increment"], 1)
        self.assertEqual(summary["schema_complete_increment"], 1)
        self.assertEqual(summary["schema_incomplete_increment"], 0)
        self.assertEqual(summary["timeout_increment"], 0)
        self.assertEqual(summary["truncation_increment"], 0)
        self.assertEqual(summary["invalid_json_increment"], 0)
        self.assertEqual(summary["other_failure_increment"], 0)
        self.assertEqual(summary["unavailable_increment"], 0)
        self.assertIsNone(summary["last_error"])

    def test_multiple_failure_flags_increment_only_one_failure_category(self) -> None:
        observation = self._build_observation([
            "vlm_timeout",
            "vlm_output_truncated",
            "vlm_invalid_json_response",
        ])
        summary = summarize_tracking_agent_observation_metrics(
            observation,
            {"accepted": 0, "rejected": 1, "unmatched": 0, "fallback_used": 1, "llm_applied": 0},
        )

        self.assertEqual(summary["unavailable_increment"], 1)
        self.assertEqual(summary["timeout_increment"], 1)
        self.assertEqual(summary["truncation_increment"], 0)
        self.assertEqual(summary["invalid_json_increment"], 0)
        self.assertEqual(summary["other_failure_increment"], 0)

        failure_total = (
            summary["timeout_increment"]
            + summary["truncation_increment"]
            + summary["invalid_json_increment"]
            + summary["other_failure_increment"]
        )
        self.assertEqual(failure_total, 1)

    def test_apply_updates_counts_fallback_and_actual_llm_applied(self) -> None:
        tracks = {
            "det-1": RadarTrackSnapshot(
                track_id="det-1",
                timestamp=1000.0,
                pixel_position=None,
                ground_position=GroundPosition(x=1.0, z=2.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=[],
                confidence=0.9,
            ),
            "det-2": RadarTrackSnapshot(
                track_id="det-2",
                timestamp=1000.0,
                pixel_position=None,
                ground_position=GroundPosition(x=1.0, z=2.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=[],
                confidence=0.9,
            ),
        }
        observation = TrackingAgentObservation(
            timestamp=1000.0,
            source_agent="visual_tracking",
            vlm_model="qwen-3.8-27b",
            proposal_latency_ms=9.0,
            frame_reference=None,
            updates=[
                VisualAssociationUpdate(
                    detection_id="det-1",
                    resolved_track_id="vlm-track-1",
                    accepted=True,
                    rejection_reasons=[],
                    uncertainties=[],
                    supporting_evidence=["appearance_match"],
                ),
                VisualAssociationUpdate(
                    detection_id="det-2",
                    resolved_track_id="det-2",
                    accepted=True,
                    rejection_reasons=[],
                    uncertainties=["no_vlm_proposal_used"],
                    supporting_evidence=["yolo_candidate_track"],
                ),
                VisualAssociationUpdate(
                    detection_id="det-missing",
                    resolved_track_id="vlm-track-missing",
                    accepted=True,
                    rejection_reasons=[],
                    uncertainties=[],
                    supporting_evidence=["appearance_match"],
                ),
            ],
            uncertainties=[],
        )

        summary = apply_tracking_agent_updates_to_tracks(tracks, observation)
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["rejected"], 0)
        self.assertEqual(summary["unmatched"], 1)
        self.assertEqual(summary["fallback_used"], 1)
        self.assertEqual(summary["llm_applied"], 1)

    def test_apply_metric_update_mutates_status_counters(self) -> None:
        state = types.SimpleNamespace(
            lock=types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, *args: False),
            tracking_agent_completions=0,
            tracking_agent_last_latency_ms=None,
            tracking_agent_total_updates_accepted=0,
            tracking_agent_total_updates_rejected=0,
            tracking_agent_total_updates_unmatched=0,
            tracking_agent_fallback_usage_count=0,
            tracking_agent_llm_applied_update_count=0,
            tracking_agent_llm_valid_response_count=0,
            tracking_agent_schema_complete_count=0,
            tracking_agent_schema_incomplete_count=0,
            tracking_agent_llm_timeout_count=0,
            tracking_agent_llm_truncation_count=0,
            tracking_agent_llm_invalid_json_count=0,
            tracking_agent_llm_other_failure_count=0,
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
                "fallback_used": 1,
                "llm_applied": 1,
                "valid_response_increment": 0,
                "schema_complete_increment": 0,
                "schema_incomplete_increment": 1,
                "timeout_increment": 1,
                "truncation_increment": 0,
                "invalid_json_increment": 0,
                "other_failure_increment": 0,
                "unavailable_increment": 1,
                "last_error": "vlm_api_key_missing",
            },
        )

        self.assertEqual(state.tracking_agent_completions, 1)
        self.assertEqual(state.tracking_agent_last_latency_ms, 7.5)
        self.assertEqual(state.tracking_agent_total_updates_accepted, 2)
        self.assertEqual(state.tracking_agent_total_updates_rejected, 1)
        self.assertEqual(state.tracking_agent_total_updates_unmatched, 0)
        self.assertEqual(state.tracking_agent_fallback_usage_count, 1)
        self.assertEqual(state.tracking_agent_llm_applied_update_count, 1)
        self.assertEqual(state.tracking_agent_llm_valid_response_count, 0)
        self.assertEqual(state.tracking_agent_schema_complete_count, 0)
        self.assertEqual(state.tracking_agent_schema_incomplete_count, 1)
        self.assertEqual(state.tracking_agent_llm_timeout_count, 1)
        self.assertEqual(state.tracking_agent_llm_truncation_count, 0)
        self.assertEqual(state.tracking_agent_llm_invalid_json_count, 0)
        self.assertEqual(state.tracking_agent_llm_other_failure_count, 0)
        self.assertEqual(state.tracking_agent_unavailable_count, 1)
        self.assertEqual(state.tracking_agent_last_error, "vlm_api_key_missing")


if __name__ == "__main__":
    unittest.main()
