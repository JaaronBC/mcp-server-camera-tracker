from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.contracts import BoundingBox, DetectionCandidate
from agents.provider_call_budget import ProviderCallReservation
from agents.tracking.agent_tracking import LLMVisualTrackingAgent


class _FakeCompletions:
    def __init__(self, *, content: str | None = None, error: Exception | None = None, finish_reason: str = "stop") -> None:
        self._content = content
        self._error = error
        self._finish_reason = finish_reason

    async def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self._content),
            finish_reason=self._finish_reason,
        )
        return SimpleNamespace(choices=[choice], usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _make_detection(detection_id: str) -> DetectionCandidate:
    return DetectionCandidate(
        detection_id=detection_id,
        timestamp=1000.0,
        yolo_track_id=f"track-{detection_id}",
        label="person",
        detection_confidence=0.9,
        pixel_bbox=BoundingBox(x1=1.0, y1=2.0, x2=20.0, y2=30.0),
        pixel_center_x=10.0,
        pixel_center_y=30.0,
        relative_depth=0.25,
        ground_x=1.0,
        ground_z=2.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
        app_identity_id=None,
        detection_reference=f"fdet-1000000-0-{detection_id}-track-{detection_id}",
    )


def _reservation(attempt_id: str) -> ProviderCallReservation:
    return ProviderCallReservation(
        enabled=True,
        allowed=True,
        role="tracking_vlm",
        attempt_id=attempt_id,
        reserved_at=1000.0,
    )


def _read_single_jsonl(path: Path) -> dict:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    return json.loads(lines[0])


class TestTrackingVlmAttemptDiagnostics(unittest.TestCase):
    def test_compact_minimal_json_with_empty_proposals_parses(self) -> None:
        compact_payload = '{"proposals":[],"uncertainty_flags":["no_supported_identity_association"],"supporting_evidence":[]}'
        agent = LLMVisualTrackingAgent(client=_FakeClient(_FakeCompletions(content=compact_payload)))

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_vlm_attempt_diagnostics.jsonl"
            env = {
                "TRACKING_VLM_ATTEMPT_DIAGNOSTICS_PATH": str(diagnostics_path),
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
            }
            with patch.dict(os.environ, env, clear=False), patch(
                "agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise",
                return_value=_reservation("attempt-compact-empty"),
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt",
                return_value=None,
            ), patch(
                "agents.tracking.agent_tracking.latency_instrumentation.new_correlation_id",
                return_value="tracking-llm-correlation-compact-empty",
            ):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection("d1")],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertTrue(result.available)
            self.assertEqual(len(result.proposals), 0)
            self.assertIn("schema_complete_response", result.uncertainty_flags)
            record = _read_single_jsonl(diagnostics_path)
            self.assertEqual(record["parse_status"], "parsed")
            self.assertEqual(record["schema_status"], "complete")
            self.assertEqual(record["raw_identity_candidate_count"], 0)
            self.assertEqual(record["normalized_identity_candidate_count"], 0)

    def test_successful_parse_emits_complete_diagnostic_record(self) -> None:
        proposal_payload = json.dumps(
            {
                "proposals": [
                    {
                        "detection_id": "d1",
                        "proposed_track_id": "track-d1",
                        "association_confidence": 0.8,
                        "occlusion_likelihood": 0.1,
                        "id_switch_risk": 0.2,
                        "depth_conflict_flag": False,
                        "position_conflict_flag": False,
                        "uncertainties": [],
                        "supporting_evidence": ["motion_continuity"],
                    }
                ],
                "uncertainty_flags": [],
                "supporting_evidence": ["ok"],
            }
        )
        agent = LLMVisualTrackingAgent(client=_FakeClient(_FakeCompletions(content=proposal_payload)))

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_vlm_attempt_diagnostics.jsonl"
            env = {
                "TRACKING_VLM_ATTEMPT_DIAGNOSTICS_PATH": str(diagnostics_path),
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
            }
            with patch.dict(os.environ, env, clear=False), patch(
                "agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise",
                return_value=_reservation("attempt-success"),
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt",
                return_value=None,
            ), patch(
                "agents.tracking.agent_tracking.latency_instrumentation.new_correlation_id",
                return_value="tracking-llm-correlation-success",
            ):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection("d1")],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertTrue(result.available)
            record = _read_single_jsonl(diagnostics_path)
            self.assertEqual(record["attempt_id"], "attempt-success")
            self.assertEqual(record["llm_correlation_id"], "tracking-llm-correlation-success")
            self.assertEqual(record["provider_outcome"], "ok")
            self.assertEqual(record["parse_status"], "parsed")
            self.assertIsNone(record["parse_error_code"])
            self.assertEqual(record["schema_status"], "complete")
            self.assertTrue(record["result_available"])
            self.assertIsNone(record["result_error_code"])
            self.assertEqual(record["raw_identity_candidate_count"], 1)
            self.assertEqual(record["normalized_identity_candidate_count"], 1)

    def test_parse_failure_emits_parse_failed_not_provider_error(self) -> None:
        agent = LLMVisualTrackingAgent(client=_FakeClient(_FakeCompletions(content="not-json")))

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_vlm_attempt_diagnostics.jsonl"
            with patch.dict(
                os.environ,
                {
                    "TRACKING_VLM_ATTEMPT_DIAGNOSTICS_PATH": str(diagnostics_path),
                    "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
                },
                clear=False,
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise",
                return_value=_reservation("attempt-parse-fail"),
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt",
                return_value=None,
            ), patch(
                "agents.tracking.agent_tracking.latency_instrumentation.new_correlation_id",
                return_value="tracking-llm-correlation-parse-fail",
            ):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection("d1")],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertFalse(result.available)
            self.assertEqual(result.error, "vlm_invalid_json_response")
            record = _read_single_jsonl(diagnostics_path)
            self.assertEqual(record["provider_outcome"], "ok")
            self.assertEqual(record["parse_status"], "parse_failed")
            self.assertEqual(record["parse_error_code"], "vlm_invalid_json_response")
            self.assertEqual(record["schema_status"], "invalid")
            self.assertFalse(record["result_available"])
            self.assertEqual(record["result_error_code"], "vlm_invalid_json_response")

    def test_provider_runtime_error_is_distinguished_from_parse_failure(self) -> None:
        agent = LLMVisualTrackingAgent(client=_FakeClient(_FakeCompletions(error=RuntimeError("provider down"))))

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_vlm_attempt_diagnostics.jsonl"
            with patch.dict(
                os.environ,
                {
                    "TRACKING_VLM_ATTEMPT_DIAGNOSTICS_PATH": str(diagnostics_path),
                    "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
                },
                clear=False,
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise",
                return_value=_reservation("attempt-runtime-error"),
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt",
                return_value=None,
            ), patch(
                "agents.tracking.agent_tracking.latency_instrumentation.new_correlation_id",
                return_value="tracking-llm-correlation-runtime-error",
            ):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection("d1")],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            self.assertFalse(result.available)
            self.assertEqual(result.error, "vlm_unavailable")
            record = _read_single_jsonl(diagnostics_path)
            self.assertEqual(record["provider_outcome"], "error")
            self.assertEqual(record["parse_status"], "not_attempted")
            self.assertIsNone(record["parse_error_code"])
            self.assertEqual(record["schema_status"], "unknown")
            self.assertFalse(record["result_available"])
            self.assertEqual(record["result_error_code"], "vlm_unavailable")

    def test_rejected_identity_candidates_counted_separately_from_absent_candidates(self) -> None:
        payload = json.dumps(
            {
                "proposals": [
                    {
                        "detection_id": "d1",
                        "proposed_track_id": "track-d1",
                        "association_confidence": 0.75,
                        "occlusion_likelihood": 0.2,
                        "id_switch_risk": 0.3,
                        "depth_conflict_flag": False,
                        "position_conflict_flag": False,
                        "uncertainties": [],
                        "supporting_evidence": ["history_match"],
                        "proposed_app_identity_id": "app-123",
                        "identity_resolution_status": "identity_proposed",
                    },
                    {
                        "detection_id": "d2",
                        "proposed_track_id": "track-d2",
                        "association_confidence": 0.65,
                        "occlusion_likelihood": 0.1,
                        "id_switch_risk": 0.2,
                        "depth_conflict_flag": False,
                        "position_conflict_flag": False,
                        "uncertainties": [],
                        "supporting_evidence": ["motion_continuity"],
                    },
                ],
                "uncertainty_flags": [],
                "supporting_evidence": ["mixed"],
            }
        )
        agent = LLMVisualTrackingAgent(client=_FakeClient(_FakeCompletions(content=payload)))

        with tempfile.TemporaryDirectory() as temp_dir:
            diagnostics_path = Path(temp_dir) / "tracking_vlm_attempt_diagnostics.jsonl"
            with patch.dict(
                os.environ,
                {
                    "TRACKING_VLM_ATTEMPT_DIAGNOSTICS_PATH": str(diagnostics_path),
                    "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
                },
                clear=False,
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise",
                return_value=_reservation("attempt-rejection-semantics"),
            ), patch(
                "agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt",
                return_value=None,
            ), patch(
                "agents.tracking.agent_tracking.latency_instrumentation.new_correlation_id",
                return_value="tracking-llm-correlation-rejection-semantics",
            ):
                result = asyncio.run(
                    agent.propose_associations(
                        frame_reference=None,
                        detections=[_make_detection("d1"), _make_detection("d2")],
                        event_reasons=["possible_id_switch"],
                        request_timeout_seconds=1.0,
                    )
                )

            proposal_by_id = {item.detection_id: item for item in result.proposals}
            self.assertIn("d1", proposal_by_id)
            self.assertIsNone(proposal_by_id["d1"].proposed_app_identity_id)
            self.assertEqual(proposal_by_id["d1"].identity_resolution_status, "unresolved_identity_evidence_unusable")

            record = _read_single_jsonl(diagnostics_path)
            self.assertEqual(record["raw_identity_candidate_count"], 2)
            self.assertEqual(record["normalized_identity_candidate_count"], 2)
            self.assertEqual(record["usable_identity_proposal_count"], 0)
            self.assertEqual(record["rejected_identity_candidate_count"], 1)
            self.assertIn("missing_proposed_app_identity_id", record["rejection_reasons"])

    def test_diagnostics_persistence_failure_never_interrupts_tracking(self) -> None:
        payload = json.dumps(
            {
                "proposals": [
                    {
                        "detection_id": "d1",
                        "proposed_track_id": "track-d1",
                        "association_confidence": 0.8,
                        "occlusion_likelihood": 0.1,
                        "id_switch_risk": 0.2,
                        "depth_conflict_flag": False,
                        "position_conflict_flag": False,
                        "uncertainties": [],
                        "supporting_evidence": ["motion_continuity"],
                    }
                ],
                "uncertainty_flags": [],
                "supporting_evidence": ["ok"],
            }
        )
        agent = LLMVisualTrackingAgent(client=_FakeClient(_FakeCompletions(content=payload)))

        with patch.dict(
            os.environ,
            {
                "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
            },
            clear=False,
        ), patch(
            "agents.tracking.agent_tracking.provider_call_budget.reserve_or_raise",
            return_value=_reservation("attempt-diagnostics-failure"),
        ), patch(
            "agents.tracking.agent_tracking.provider_call_budget.finalize_provider_call_attempt",
            return_value=None,
        ), patch(
            "agents.tracking.agent_tracking.latency_instrumentation.new_correlation_id",
            return_value="tracking-llm-correlation-diagnostics-failure",
        ), patch(
            "agents.tracking.agent_tracking._append_tracking_vlm_attempt_diagnostic_record",
            side_effect=OSError("disk full"),
        ):
            result = asyncio.run(
                agent.propose_associations(
                    frame_reference=None,
                    detections=[_make_detection("d1")],
                    event_reasons=["possible_id_switch"],
                    request_timeout_seconds=1.0,
                )
            )

        self.assertTrue(result.available)
        self.assertEqual(len(result.proposals), 1)


if __name__ == "__main__":
    unittest.main()
