from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile

from agents.contracts import BoundingBox, DetectionCandidate
from agents.tracking.agent_tracking import (
    LLMVisualTrackingAgent,
    _normalize_compact_response_payload,
    _validate_tracking_decision_payload,
)
from agents.orchestration.runtime.tracking_decision_ipc import TrackingDecisionIpcTransport


class _CaptureCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create(self, **kwargs):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self._content),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice], usage=None)


class _CaptureClient:
    def __init__(self, completions: _CaptureCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _make_detection() -> DetectionCandidate:
    return DetectionCandidate(
        detection_id="det-1",
        timestamp=1000.0,
        yolo_track_id="track-1",
        label="person",
        detection_confidence=0.91,
        pixel_bbox=BoundingBox(x1=10.0, y1=20.0, x2=50.0, y2=80.0),
        pixel_center_x=30.0,
        pixel_center_y=80.0,
        relative_depth=0.4,
        ground_x=1.0,
        ground_z=2.0,
        ground_coordinate_unit="relative_units",
        position_kind="relative_ground_estimate",
        motion_history=[],
        app_identity_id=None,
        detection_reference="fdet-1000000-0-det-1-track-1",
    )


def _build_response(*, decision: dict | None = None, proposals: list[dict] | None = None) -> str:
    if proposals is None:
        proposals = [
            {
                "detection_id": "det-1",
                "proposed_track_id": "track-1",
                "association_confidence": 0.81,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.3,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": [],
            }
        ]
    payload = {
        "proposals": proposals,
        "uncertainty_flags": [],
        "supporting_evidence": [],
    }
    if decision is not None:
        payload["decision"] = decision
    return json.dumps(payload)


def _invoke(
    response_payload: str,
    *,
    env_overrides: dict[str, str] | None = None,
    follow_up_context: dict | None = None,
):
    completions = _CaptureCompletions(response_payload)
    client = _CaptureClient(completions)
    agent = LLMVisualTrackingAgent(client=client)
    env = {
        "PROVIDER_CALL_BUDGET_ENABLED": "0",
        "TRACKING_IDENTITY_EVIDENCE_ENABLED": "0",
    }
    if isinstance(env_overrides, dict):
        env.update(env_overrides)
    with patch.dict(os.environ, env, clear=False):
        return asyncio.run(
            agent.propose_associations(
                frame_reference=None,
                detections=[_make_detection()],
                event_reasons=["possible_id_switch"],
                follow_up_context=follow_up_context,
                request_timeout_seconds=1.0,
            )
        )


class TestTrackingDecisionContract(unittest.TestCase):
    def test_accepts_all_supported_decision_actions(self) -> None:
        cases = {
            "request_identity_snapshot": {
                "action": "request_identity_snapshot",
                "investigation_id": "inv-1",
                "correlation_id": "corr-1",
                "snapshot_constraints": {
                    "min_source_generation": 10,
                    "min_source_version": 20,
                },
                "evidence_references": {
                    "frame_reference": "frame-1",
                    "detection_reference": "fdet-1000000-0-det-1-track-1",
                },
                "provenance": {"source": "tracking_agent", "attempt": 1},
                "tool_requests": [
                    {
                        "request_id": "req-1",
                        "tool_name": "get_identity_snapshot",
                        "arguments": {
                            "app_identity_id": "app-1",
                            "min_source_generation": 10,
                            "min_source_version": 20,
                            "investigation_id": "inv-1",
                        },
                    }
                ],
                "_proposals": [],
            },
            "request_additional_tracking_evidence": {
                "action": "request_additional_tracking_evidence",
                "investigation_id": "inv-2",
                "tool_requests": [
                    {
                        "request_id": "req-2",
                        "tool_name": "get_track_evidence_history",
                        "arguments": {
                            "track_id": "track-1",
                            "max_messages": 5,
                            "investigation_id": "inv-2",
                        },
                    }
                ],
                "_proposals": [],
            },
            "propose_identity_association": {
                "action": "propose_identity_association",
                "investigation_id": "inv-3",
                "uncertainty": ["occlusion"],
                "provenance": {"policy": "conservative"},
            },
            "inconclusive": {
                "action": "inconclusive",
                "investigation_id": "inv-4",
                "reason": "insufficient visual evidence",
                "uncertainty": ["insufficient_data"],
                "_proposals": [],
            },
        }

        for action, decision in cases.items():
            with self.subTest(action=action):
                decision_payload = dict(decision)
                proposals_override = decision_payload.pop("_proposals", None)
                result = _invoke(_build_response(decision=decision_payload, proposals=proposals_override))
                self.assertTrue(result.available)
                self.assertIsNotNone(result.decision)
                self.assertEqual(result.decision.get("action"), action)

    def test_rejects_disallowed_tool_name(self) -> None:
        decision = {
            "action": "request_identity_snapshot",
            "investigation_id": "inv-1",
            "tool_requests": [
                {
                    "request_id": "req-1",
                    "tool_name": "mutate_identity_registry",
                    "arguments": {"app_identity_id": "app-1"},
                }
            ],
        }
        result = _invoke(_build_response(decision=decision))
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_invalid_decision_contract")
        self.assertIn("decision_tool_name_not_allowed", result.uncertainty_flags)

    def test_rejects_invalid_tool_arguments(self) -> None:
        decision = {
            "action": "request_additional_tracking_evidence",
            "investigation_id": "inv-1",
            "tool_requests": [
                {
                    "request_id": "req-1",
                    "tool_name": "get_track_evidence_history",
                    "arguments": {
                        "max_messages": 5,
                    },
                }
            ],
        }
        result = _invoke(_build_response(decision=decision))
        self.assertFalse(result.available)
        self.assertIn("decision_track_history_track_id_required", result.uncertainty_flags)

    def test_rejects_missing_request_id(self) -> None:
        decision = {
            "action": "request_identity_snapshot",
            "investigation_id": "inv-1",
            "tool_requests": [
                {
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                }
            ],
        }
        result = _invoke(_build_response(decision=decision))
        self.assertFalse(result.available)
        self.assertIn("decision_tool_request_id_required", result.uncertainty_flags)

    def test_rejects_more_than_one_tool_request(self) -> None:
        decision = {
            "action": "request_identity_snapshot",
            "investigation_id": "inv-1",
            "tool_requests": [
                {
                    "request_id": "req-1",
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                },
                {
                    "request_id": "req-2",
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-2"},
                },
            ],
        }
        result = _invoke(_build_response(decision=decision))
        self.assertFalse(result.available)
        self.assertIn("decision_tool_requests_exceeds_max", result.uncertainty_flags)

    def test_rejects_invalid_snapshot_constraints(self) -> None:
        decision = {
            "action": "propose_identity_association",
            "snapshot_constraints": {
                "min_source_generation": -1,
                "min_source_version": 1,
            },
        }
        result = _invoke(_build_response(decision=decision))
        self.assertFalse(result.available)
        self.assertIn("decision_min_source_generation_invalid", result.uncertainty_flags)

    def test_rejects_inconclusive_without_reason(self) -> None:
        decision = {
            "action": "inconclusive",
            "investigation_id": "inv-4",
            "uncertainty": ["insufficient_data"],
        }
        result = _invoke(_build_response(decision=decision))
        self.assertFalse(result.available)
        self.assertIn("decision_inconclusive_reason_required", result.uncertainty_flags)

    def test_no_decision_payload_remains_backward_compatible(self) -> None:
        result = _invoke(_build_response())
        self.assertTrue(result.available)
        self.assertIsNone(result.decision)

    def test_request_action_rejects_actionable_proposals(self) -> None:
        decision = {
            "action": "request_identity_snapshot",
            "investigation_id": "inv-1",
            "tool_requests": [
                {
                    "request_id": "req-1",
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                }
            ],
        }
        proposals = [
            {
                "detection_id": "det-1",
                "proposed_track_id": "track-1",
                "association_confidence": 0.81,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.3,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": [],
            }
        ]
        result = _invoke(_build_response(decision=decision, proposals=proposals))
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_decision_proposal_conflict")
        self.assertIn("vlm_decision_proposal_conflict", result.uncertainty_flags)

    def test_inconclusive_rejects_actionable_proposals(self) -> None:
        decision = {
            "action": "inconclusive",
            "investigation_id": "inv-2",
            "reason": "insufficient evidence",
        }
        proposals = [
            {
                "detection_id": "det-1",
                "proposed_track_id": "track-1",
                "association_confidence": 0.81,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.3,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": [],
            }
        ]
        result = _invoke(_build_response(decision=decision, proposals=proposals))
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_decision_proposal_conflict")
        self.assertIn("vlm_decision_proposal_conflict", result.uncertainty_flags)

    def test_compact_mode_normalizes_valid_evidence_request(self) -> None:
        compact = json.dumps(
            {
                "action": "request_additional_tracking_evidence",
                "investigation_id": "inv-compact-1",
                "uncertainty": ["possible_id_switch"],
                "tool_request": {
                    "request_id": "req-compact-1",
                    "tool_name": "get_track_evidence_history",
                    "arguments": {
                        "track_id": "track-1",
                        "max_messages": 5,
                        "investigation_id": "inv-compact-1",
                    },
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertTrue(result.available)
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.get("action"), "request_additional_tracking_evidence")

    def test_compact_mode_rejects_invalid_tool_arguments(self) -> None:
        compact = json.dumps(
            {
                "action": "request_additional_tracking_evidence",
                "investigation_id": "inv-compact-2",
                "tool_request": {
                    "request_id": "req-compact-2",
                    "tool_name": "get_track_evidence_history",
                    "arguments": {"max_messages": 3},
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_invalid_decision_contract")
        self.assertIn("decision_track_history_track_id_required", result.uncertainty_flags)

    def test_compact_mode_missing_investigation_id_is_runtime_bound(self) -> None:
        compact = json.dumps(
            {
                "action": "request_identity_snapshot",
                "tool_request": {
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertTrue(result.available)
        self.assertIsNotNone(result.decision)
        assert isinstance(result.decision, dict)
        self.assertEqual(result.decision.get("action"), "request_identity_snapshot")
        self.assertIsInstance(result.decision.get("investigation_id"), str)
        self.assertTrue(str(result.decision.get("investigation_id", "")).startswith("inv-tracking-llm-"))
        self.assertEqual(result.decision.get("correlation_id"), result.llm_correlation_id)

        requests = result.decision.get("tool_requests")
        self.assertIsInstance(requests, list)
        assert isinstance(requests, list)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertIsInstance(request.get("request_id"), str)
        self.assertEqual(request.get("tool_name"), "get_identity_snapshot")

    def test_compact_mode_runtime_bound_decision_reaches_ipc_dispatch(self) -> None:
        compact = json.dumps(
            {
                "action": "request_additional_tracking_evidence",
                "tool_request": {
                    "arguments": {"track_id": "track-1", "max_messages": 3},
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertTrue(result.available)
        self.assertIsNotNone(result.decision)
        assert isinstance(result.decision, dict)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transport = TrackingDecisionIpcTransport(
                requests_path=root / "tracking_decision_tool_requests.jsonl",
                results_path=root / "tracking_decision_tool_results.jsonl",
                request_index_path=root / "latest_tracking_decision_request_index.json",
                result_index_path=root / "latest_tracking_decision_result_index.json",
            )
            submitted = transport.dashboard_submit_decision_request(
                decision=result.decision,
                llm_correlation_id=result.llm_correlation_id,
                now_timestamp=1000.0,
            )
            self.assertTrue(submitted.get("accepted", False), submitted)
            pending = transport.orchestration_poll_pending_requests(max_requests=2, now_timestamp=1000.1)
            self.assertEqual(pending.get("pending_count"), 1)

    def test_compact_mode_semantic_identity_snapshot_maps_to_canonical_request(self) -> None:
        compact = json.dumps(
            {
                "action": "request_identity_snapshot",
                "detection_reference": "fdet-1000000-0-det-1-track-1",
                "evidence_bounds": {
                    "include_history": True,
                    "max_candidates": 4,
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertTrue(result.available)
        assert isinstance(result.decision, dict)
        self.assertEqual(result.decision.get("action"), "request_identity_snapshot")
        requests = result.decision.get("tool_requests")
        self.assertIsInstance(requests, list)
        assert isinstance(requests, list)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.get("tool_name"), "get_identity_snapshot")
        self.assertIsInstance(request.get("request_id"), str)
        self.assertEqual(request.get("arguments", {}).get("detection_reference"), "fdet-1000000-0-det-1-track-1")
        self.assertEqual(request.get("arguments", {}).get("max_candidates"), 4)

    def test_compact_mode_semantic_track_request_duplicate_submission_is_suppressed(self) -> None:
        compact = json.dumps(
            {
                "action": "request_additional_tracking_evidence",
                "track_id": "track-1",
                "evidence_bounds": {
                    "max_messages": 3,
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertTrue(result.available)
        assert isinstance(result.decision, dict)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transport = TrackingDecisionIpcTransport(
                requests_path=root / "tracking_decision_tool_requests.jsonl",
                results_path=root / "tracking_decision_tool_results.jsonl",
                request_index_path=root / "latest_tracking_decision_request_index.json",
                result_index_path=root / "latest_tracking_decision_result_index.json",
            )
            first = transport.dashboard_submit_decision_request(
                decision=result.decision,
                llm_correlation_id=result.llm_correlation_id,
                now_timestamp=1000.0,
            )
            second = transport.dashboard_submit_decision_request(
                decision=result.decision,
                llm_correlation_id=result.llm_correlation_id,
                now_timestamp=1000.1,
            )
            self.assertTrue(first.get("accepted", False), first)
            self.assertTrue(second.get("duplicate", False), second)

    def test_compact_mode_semantic_identity_snapshot_missing_selector_is_rejected(self) -> None:
        compact = json.dumps(
            {
                "action": "request_identity_snapshot",
                "evidence_bounds": {
                    "include_history": True,
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_invalid_decision_contract")
        self.assertIn("decision_identity_snapshot_selector_required", result.uncertainty_flags)

    def test_compact_mode_semantic_conflicting_explicit_tool_name_is_rejected(self) -> None:
        compact = json.dumps(
            {
                "action": "request_identity_snapshot",
                "detection_reference": "fdet-1000000-0-det-1-track-1",
                "tool_name": "get_track_evidence_history",
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_invalid_decision_contract")
        self.assertIn("decision_tool_action_mismatch", result.uncertainty_flags)

    def test_compact_mode_semantic_unsupported_argument_keys_and_types_are_rejected(self) -> None:
        compact = json.dumps(
            {
                "action": "request_additional_tracking_evidence",
                "track_id": "track-1",
                "evidence_bounds": {
                    "max_messages": "five",
                    "unsupported_key": 1,
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_invalid_decision_contract")
        self.assertIn("decision_track_history_max_messages_invalid", result.uncertainty_flags)
        self.assertIn("decision_track_history_unknown_arguments", result.uncertainty_flags)

    def test_compact_mode_semantic_followup_context_binds_trusted_request_and_investigation(self) -> None:
        compact = json.dumps(
            {
                "action": "request_additional_tracking_evidence",
                "track_id": "track-1",
                "proposals": [],
            }
        )
        result = _invoke(
            compact,
            env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"},
            follow_up_context={"investigation_id": "inv-followup-trusted", "request_id": "req-followup-trusted"},
        )
        self.assertTrue(result.available)
        assert isinstance(result.decision, dict)
        self.assertEqual(result.decision.get("investigation_id"), "inv-followup-trusted")
        requests = result.decision.get("tool_requests")
        self.assertIsInstance(requests, list)
        assert isinstance(requests, list)
        self.assertEqual(requests[0].get("request_id"), "req-followup-trusted")

    def test_runtime_bound_request_id_varies_by_action_with_shared_correlation(self) -> None:
        snapshot_compact = {
            "action": "request_identity_snapshot",
            "detection_reference": "fdet-1000000-0-det-1-track-1",
            "proposals": [],
            "uncertainty_flags": [],
            "supporting_evidence": [],
        }
        history_compact = {
            "action": "request_additional_tracking_evidence",
            "track_id": "track-1",
            "proposals": [],
            "uncertainty_flags": [],
            "supporting_evidence": [],
        }

        normalized_snapshot = _normalize_compact_response_payload(
            dict(snapshot_compact),
            request_correlation_id="tracking-llm-shared",
            follow_up_context=None,
            detections=[_make_detection()],
        )
        normalized_history = _normalize_compact_response_payload(
            dict(history_compact),
            request_correlation_id="tracking-llm-shared",
            follow_up_context=None,
            detections=[_make_detection()],
        )

        snapshot_request_id = normalized_snapshot["decision"]["tool_requests"][0]["request_id"]
        history_request_id = normalized_history["decision"]["tool_requests"][0]["request_id"]
        self.assertNotEqual(snapshot_request_id, history_request_id)

    def test_compact_mode_maps_tool_aliases_and_string_uncertainty(self) -> None:
        compact = json.dumps(
            {
                "action": "request_additional_tracking_evidence",
                "uncertainty": "possible_id_switch",
                "tool_request": {
                    "id": "req-alias-1",
                    "name": "get_track_evidence_history",
                    "args": {"track_id": "track-1"},
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertTrue(result.available)
        self.assertIsNotNone(result.decision)
        assert isinstance(result.decision, dict)
        self.assertEqual(result.decision.get("uncertainty"), ["possible_id_switch"])
        requests = result.decision.get("tool_requests")
        self.assertIsInstance(requests, list)
        assert isinstance(requests, list)
        self.assertEqual(requests[0].get("tool_name"), "get_track_evidence_history")
        self.assertEqual(requests[0].get("request_id"), "req-alias-1")

    def test_compact_mode_conflicting_actionable_proposals_are_rejected(self) -> None:
        compact = json.dumps(
            {
                "action": "inconclusive",
                "investigation_id": "inv-compact-4",
                "reason": "insufficient evidence",
                "proposals": [
                    {
                        "detection_id": "det-1",
                        "proposed_track_id": "track-1",
                    }
                ],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_decision_proposal_conflict")

    def test_compact_mode_followup_inconclusive_is_accepted(self) -> None:
        compact = json.dumps(
            {
                "action": "inconclusive",
                "investigation_id": "inv-followup-1",
                "reason": "evidence_still_conflicting",
                "uncertainty": ["needs_more_evidence"],
                "proposals": [],
            }
        )
        result = _invoke(
            compact,
            env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"},
            follow_up_context={"investigation_id": "inv-followup-1", "request_id": "req-followup-1"},
        )
        self.assertTrue(result.available)
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.get("action"), "inconclusive")
        self.assertEqual(result.decision.get("investigation_id"), "inv-followup-1")

    def test_compact_mode_followup_conflicting_investigation_id_is_overridden_by_trusted_context(self) -> None:
        compact = json.dumps(
            {
                "action": "inconclusive",
                "investigation_id": "inv-untrusted-model-id",
                "reason": "insufficient_evidence",
                "proposals": [],
            }
        )
        result = _invoke(
            compact,
            env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"},
            follow_up_context={"investigation_id": "inv-followup-trusted", "request_id": "req-followup-trusted"},
        )
        self.assertTrue(result.available)
        self.assertIsNotNone(result.decision)
        assert isinstance(result.decision, dict)
        self.assertEqual(result.decision.get("investigation_id"), "inv-followup-trusted")

    def test_compact_mode_followup_identity_proposal_preserves_safeguards(self) -> None:
        compact = json.dumps(
            {
                "action": "propose_identity_association",
                "investigation_id": "inv-followup-2",
                "proposals": [
                    {
                        "detection_id": "det-1",
                        "proposed_track_id": "track-1",
                        "proposed_app_identity_id": "app-unsafe",
                        "identity_resolution_status": "identity_proposed",
                        "uncertainties": [],
                        "supporting_evidence": [],
                    }
                ],
                "uncertainty_flags": [],
                "supporting_evidence": [],
            }
        )
        result = _invoke(
            compact,
            env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"},
            follow_up_context={"investigation_id": "inv-followup-2", "request_id": "req-followup-2"},
        )
        self.assertTrue(result.available)
        self.assertEqual(len(result.proposals), 1)
        proposal = result.proposals[0]
        self.assertIsNone(proposal.proposed_app_identity_id)
        self.assertEqual(proposal.identity_resolution_status, "unresolved_identity_evidence_unusable")

    def test_identity_claim_without_evidence_still_normalizes_conservatively(self) -> None:
        proposals = [
            {
                "detection_id": "det-1",
                "proposed_track_id": "track-1",
                "association_confidence": 0.81,
                "occlusion_likelihood": 0.2,
                "id_switch_risk": 0.3,
                "depth_conflict_flag": False,
                "position_conflict_flag": False,
                "uncertainties": [],
                "supporting_evidence": [],
                "proposed_app_identity_id": "app-claimed",
                "identity_evidence_refs": [],
            }
        ]
        result = _invoke(_build_response(proposals=proposals))
        self.assertTrue(result.available)
        self.assertEqual(len(result.proposals), 1)
        self.assertIsNone(result.proposals[0].proposed_app_identity_id)
        self.assertIn("identity_claim_without_usable_evidence", result.proposals[0].uncertainties)
        self.assertEqual(result.proposals[0].identity_resolution_status, "unresolved_identity_evidence_unusable")

    def test_compact_mode_rejects_ambiguous_uncertainty_type(self) -> None:
        compact = json.dumps(
            {
                "action": "request_identity_snapshot",
                "uncertainty": {"reason": "possible_id_switch"},
                "tool_request": {
                    "tool_name": "get_identity_snapshot",
                    "arguments": {"app_identity_id": "app-1"},
                },
                "proposals": [],
            }
        )
        result = _invoke(compact, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(result.available)
        self.assertEqual(result.error, "vlm_invalid_decision_contract")
        self.assertIn("decision_uncertainty_invalid", result.uncertainty_flags)

    def test_compact_mode_rejects_unsupported_or_mismatched_tool_names(self) -> None:
        unsupported = json.dumps(
            {
                "action": "request_identity_snapshot",
                "tool_request": {
                    "tool_name": "mutate_identity_registry",
                    "arguments": {"app_identity_id": "app-1"},
                },
                "proposals": [],
            }
        )
        unsupported_result = _invoke(unsupported, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(unsupported_result.available)
        self.assertIn("decision_tool_name_not_allowed", unsupported_result.uncertainty_flags)

        mismatch = json.dumps(
            {
                "action": "request_identity_snapshot",
                "tool_request": {
                    "tool_name": "get_track_evidence_history",
                    "arguments": {"track_id": "track-1"},
                },
                "proposals": [],
            }
        )
        mismatch_result = _invoke(mismatch, env_overrides={"TRACKING_AGENT_COMPACT_DECISION_MODE": "1"})
        self.assertFalse(mismatch_result.available)
        self.assertIn("decision_tool_action_mismatch", mismatch_result.uncertainty_flags)

    def test_validator_reproduces_observed_four_error_codes(self) -> None:
        payload = {
            "action": "request_additional_tracking_evidence",
            "uncertainty": {"not": "supported"},
            "tool_requests": [
                {
                    "arguments": {"track_id": "track-1"},
                }
            ],
        }
        decision, errors = _validate_tracking_decision_payload(payload)
        self.assertIsNone(decision)
        self.assertIn("decision_investigation_id_required", errors)
        self.assertIn("decision_tool_name_required", errors)
        self.assertIn("decision_tool_request_id_required", errors)
        self.assertIn("decision_uncertainty_invalid", errors)

    def test_compact_normalization_missing_trusted_context_remains_rejected(self) -> None:
        parsed = {
            "action": "request_additional_tracking_evidence",
            "tool_request": {
                "arguments": {"track_id": "track-1"},
            },
            "uncertainty": [],
            "proposals": [],
            "uncertainty_flags": [],
            "supporting_evidence": [],
        }
        normalized = _normalize_compact_response_payload(
            parsed,
            request_correlation_id=None,
            follow_up_context=None,
            detections=None,
        )
        decision, errors = _validate_tracking_decision_payload(normalized.get("decision"))
        self.assertIsNone(decision)
        self.assertIn("decision_investigation_id_required", errors)
        self.assertIn("decision_tool_request_id_required", errors)

    def test_runtime_bound_request_id_supports_idempotent_retry(self) -> None:
        compact = {
            "action": "request_additional_tracking_evidence",
            "tool_request": {
                "arguments": {"track_id": "track-1"},
            },
            "proposals": [],
            "uncertainty_flags": [],
            "supporting_evidence": [],
        }
        normalized_a = _normalize_compact_response_payload(
            dict(compact),
            request_correlation_id="tracking-llm-abc123",
            follow_up_context=None,
            detections=[_make_detection()],
        )
        normalized_b = _normalize_compact_response_payload(
            dict(compact),
            request_correlation_id="tracking-llm-abc123",
            follow_up_context=None,
            detections=[_make_detection()],
        )

        req_a = normalized_a["decision"]["tool_requests"][0]["request_id"]
        req_b = normalized_b["decision"]["tool_requests"][0]["request_id"]
        inv_a = normalized_a["decision"]["investigation_id"]
        inv_b = normalized_b["decision"]["investigation_id"]
        self.assertEqual(req_a, req_b)
        self.assertEqual(inv_a, inv_b)


if __name__ == "__main__":
    unittest.main()
