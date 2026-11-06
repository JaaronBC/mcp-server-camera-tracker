from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents.contracts import BoundingBox, DetectionCandidate, VisualAssociationProposal
from agents.coordinator import AgentCoordinator
from agents import provider_call_budget
from agents.observation.agent_observation import (
    DEFAULT_OBSERVATION_VISUAL_MODEL,
    DEFAULT_OBSERVATION_VISUAL_SYSTEM_PROMPT,
    invoke_visual_verification_llm_response,
)
from agents.tracking.agent_tracking import (
    DEFAULT_OPENAI_BASE_URL,
    LLMVisualTrackingAgent,
    TRACKING_VLM_MODEL,
    _extract_json_object,
    encode_frame_as_data_url,
    load_tracking_agent_skill,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_EVENTS_PATH = PROJECT_ROOT / "captures" / "latency_events.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "captures" / "diagnostics" / "tracking_agent_phase1_validation_report.json"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "captures" / "diagnostics" / "tracking_agent_phase1_validation_events.jsonl"
STRUCTURED_INPUT_MARKER = "Structured input:\n"
REQUIRED_RESPONSE_KEYS = ("proposals", "uncertainty_flags", "supporting_evidence")
REQUIRED_PROPOSAL_FIELDS = (
    "detection_id",
    "proposed_track_id",
    "association_confidence",
    "occlusion_likelihood",
    "id_switch_risk",
    "depth_conflict_flag",
    "position_conflict_flag",
    "uncertainties",
    "supporting_evidence",
)
SCENARIO_BASELINE = "baseline"
SCENARIO_YOLO_ID_SWAP = "yolo-id-swap"
SCENARIO_CHOICES = (SCENARIO_BASELINE, SCENARIO_YOLO_ID_SWAP)
INDEPENDENT_TRACK_HISTORY_KEY = "independent_track_history_evidence"
INDEPENDENT_HISTORY_FORBIDDEN_TOKENS = (
    "expected_identity",
    "ground_truth",
    "canonical_track_id",
    "expected_track",
    "expected_correct_associations",
)
OBSERVATION_VERIFICATION_MODE_OFF = "off"
OBSERVATION_VERIFICATION_MODE_MOCK = "mock"
OBSERVATION_VERIFICATION_MODE_LIVE = "live"
OBSERVATION_VERIFICATION_MODE_CHOICES = (
    OBSERVATION_VERIFICATION_MODE_OFF,
    OBSERVATION_VERIFICATION_MODE_MOCK,
    OBSERVATION_VERIFICATION_MODE_LIVE,
)
TRACKING_RECONSIDERATION_MODE_OFF = "off"
TRACKING_RECONSIDERATION_MODE_LIVE = "live"
TRACKING_RECONSIDERATION_MODE_CHOICES = (
    TRACKING_RECONSIDERATION_MODE_OFF,
    TRACKING_RECONSIDERATION_MODE_LIVE,
)
DEFAULT_OBSERVATION_VISUAL_TIMEOUT_SECONDS = 6.0
DEFAULT_OBSERVATION_VISUAL_MAX_OUTPUT_TOKENS = 768
DEFAULT_OBSERVATION_VISUAL_REASONING_EFFORT = "none"
DIAGNOSTIC_PROVIDER_CALL_META_KEY = "__diagnostic_provider_call"
OBSERVATION_REQUIRED_KEYS = (
    "verification_status",
    "visibility_status",
    "identity_consistency",
    "possible_id_switch",
    "uncertainty",
    "limitations",
    "supporting_evidence_refs",
    "contradicting_evidence_refs",
    "visual_content_inspected",
)
EVIDENCE_MODE_SYNTHETIC_WITH_HISTORY = "synthetic_imagery_with_structured_tracking_history"
EVIDENCE_MODE_SYNTHETIC_NO_HISTORY = "synthetic_imagery_without_structured_tracking_history"
VERIFICATION_DIMENSION_SUPPORTED = "supported"
VERIFICATION_DIMENSION_UNSUPPORTED = "unsupported"
VERIFICATION_DIMENSION_UNAVAILABLE = "unavailable"
DIAGNOSTIC_ADJUDICATION_CLASSIFICATIONS = ("accepted", "rejected", "unresolved")
DIAGNOSTIC_ADJUDICATION_PROVISIONAL_MIN_CONFIDENCE = 0.55
DIAGNOSTIC_RECOVERY_MAX_HISTORY_AGE_SECONDS = 2.5
TRACKING_RECONSIDERATION_REQUIRED_KEYS = (
    "revisions",
    "declined",
)


@dataclass(slots=True)
class ResponseCapture:
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    usage_details: dict[str, Any] | None = None
    content_text: str | None = None
    content_text_char_length: int | None = None
    content_text_sha256: str | None = None
    content_text_head: str | None = None
    content_text_tail: str | None = None
    response_object_keys: list[str] | None = None
    message_object_keys: list[str] | None = None
    reasoning_effort_requested: str | None = None
    reasoning_effort_applied_in_request: str | None = None
    original_user_instruction: str | None = None
    compressed_user_instruction: str | None = None
    compression_applied: bool = False


def _build_compressed_instruction_from_original(
    original_text: str,
    independent_track_history_override: list[dict[str, Any]] | None = None,
) -> str:
    structured_payload = "{}"
    marker_index = original_text.find(STRUCTURED_INPUT_MARKER)
    if marker_index >= 0:
        structured_payload = original_text[marker_index + len(STRUCTURED_INPUT_MARKER):].strip() or "{}"

    parsed_payload: dict[str, Any] | None = None
    if isinstance(structured_payload, str):
        try:
            payload_candidate = json.loads(structured_payload)
            if isinstance(payload_candidate, dict):
                parsed_payload = payload_candidate
        except Exception:
            parsed_payload = None

    if isinstance(parsed_payload, dict) and independent_track_history_override:
        parsed_payload[INDEPENDENT_TRACK_HISTORY_KEY] = independent_track_history_override

    if isinstance(parsed_payload, dict):
        structured_payload = json.dumps(parsed_payload, separators=(",", ":"))

    schema_keys = ", ".join(REQUIRED_RESPONSE_KEYS)
    proposal_fields = ", ".join(REQUIRED_PROPOSAL_FIELDS)

    return (
        "Task: Propose conservative visual track associations from detections + frame evidence.\n"
        f"Return one JSON object with keys: {schema_keys}. JSON only; no markdown or prose.\n"
        "For each detection that requires evaluation, include exactly one proposal entry.\n"
        f"Each proposal must include fields: {proposal_fields}.\n"
        "Use concise evidence and uncertainty tags: max 2 short items per proposal list, max 3 global flags/evidence.\n"
        "Keep occlusion, identity-ambiguity, and insufficient-evidence signaling explicit; do not invent measurements or IDs.\n"
        f"{STRUCTURED_INPUT_MARKER}{structured_payload}"
    )


def _build_instruction_structure_view(instruction_text: str) -> dict[str, Any]:
    marker_index = instruction_text.find(STRUCTURED_INPUT_MARKER)
    has_structured_input = marker_index >= 0
    structured_input_chars = 0
    if has_structured_input:
        structured_input_chars = len(instruction_text[marker_index + len(STRUCTURED_INPUT_MARKER):].strip())
    return {
        "chars_total": len(instruction_text),
        "has_structured_input": has_structured_input,
        "structured_input_chars": structured_input_chars,
        "includes_required_response_keys": all(key in instruction_text for key in REQUIRED_RESPONSE_KEYS),
        "includes_required_proposal_fields": all(field in instruction_text for field in REQUIRED_PROPOSAL_FIELDS),
    }


def _validate_compressed_instruction_template() -> dict[str, Any]:
    sample_original = (
        "Analyze visual continuity and propose track associations. "
        "Return JSON object with keys: proposals, uncertainty_flags, supporting_evidence.\n"
        f"{STRUCTURED_INPUT_MARKER}{{\"event_reasons\":[],\"detections\":[]}}"
    )
    compressed = _build_compressed_instruction_from_original(sample_original)
    checks = {
        "has_structured_input_marker": STRUCTURED_INPUT_MARKER in compressed,
        "includes_required_response_keys": all(key in compressed for key in REQUIRED_RESPONSE_KEYS),
        "includes_required_proposal_fields": all(field in compressed for field in REQUIRED_PROPOSAL_FIELDS),
    }
    checks["all_passed"] = all(checks.values())
    return {
        "checks": checks,
        "compressed_structure": _build_instruction_structure_view(compressed),
    }


def _extract_structured_payload_from_instruction(instruction_text: str) -> tuple[dict[str, Any] | None, str | None]:
    marker_index = instruction_text.find(STRUCTURED_INPUT_MARKER)
    if marker_index < 0:
        return None, "structured_input_marker_missing"

    payload_text = instruction_text[marker_index + len(STRUCTURED_INPUT_MARKER):].strip()
    if not payload_text:
        return None, "structured_payload_empty"

    try:
        parsed = json.loads(payload_text)
    except Exception as error:
        return None, f"structured_payload_parse_error:{type(error).__name__}"

    if not isinstance(parsed, dict):
        return None, "structured_payload_not_object"

    return parsed, None


def _build_independent_track_history_evidence(now: float) -> list[dict[str, Any]]:
    return [
        {
            "track_id": "track-A",
            "history_window_seconds": 1.8,
            "history_sample_count": 3,
            "recent_ground_path": [
                {"sample_age_s": 1.4, "ground_x": 0.7, "ground_z": 4.5},
                {"sample_age_s": 0.9, "ground_x": 0.95, "ground_z": 4.2},
                {"sample_age_s": 0.3, "ground_x": 1.2, "ground_z": 4.0},
            ],
            "recent_relative_depth": {"median": 0.52, "range": [0.5, 0.54]},
            "evidence_timestamp": now - 0.2,
            "source": "independent_track_memory",
        },
        {
            "track_id": "track-B",
            "history_window_seconds": 1.8,
            "history_sample_count": 3,
            "recent_ground_path": [
                {"sample_age_s": 1.5, "ground_x": 2.2, "ground_z": 4.6},
                {"sample_age_s": 0.95, "ground_x": 2.0, "ground_z": 4.3},
                {"sample_age_s": 0.35, "ground_x": 1.85, "ground_z": 4.1},
            ],
            "recent_relative_depth": {"median": 0.49, "range": [0.47, 0.51]},
            "evidence_timestamp": now - 0.2,
            "source": "independent_track_memory",
        },
    ]


def _build_independent_history_consistency_audit(
    independent_history: list[dict[str, Any]],
    detection_ids: list[str],
) -> dict[str, Any]:
    independent_history_text = json.dumps(independent_history, separators=(",", ":"))
    leaked_detection_ids = [detection_id for detection_id in detection_ids if detection_id in independent_history_text]
    leaked_forbidden_tokens = [
        token
        for token in INDEPENDENT_HISTORY_FORBIDDEN_TOKENS
        if token in independent_history_text
    ]
    return {
        "evidence_present": len(independent_history) > 0,
        "evidence_entry_count": len(independent_history),
        "references_detection_ids": len(leaked_detection_ids) > 0,
        "referenced_detection_ids": leaked_detection_ids,
        "forbidden_tokens_checked": list(INDEPENDENT_HISTORY_FORBIDDEN_TOKENS),
        "forbidden_tokens_found": leaked_forbidden_tokens,
        "leakage_detected": bool(leaked_detection_ids or leaked_forbidden_tokens),
    }


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float, bool)) and str(item).strip()]


def _get_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _find_independent_history_record(
    independent_history: list[dict[str, Any]],
    track_id: str | None,
) -> dict[str, Any] | None:
    if not isinstance(track_id, str) or not track_id.strip():
        return None
    for item in independent_history:
        if not isinstance(item, dict):
            continue
        if item.get("track_id") == track_id:
            return item
    return None


def _extract_tracking_proposal_view(raw_proposal: dict[str, Any]) -> dict[str, Any]:
    alternative_track_candidates: list[dict[str, Any]] = []
    raw_alternative_track_candidates = raw_proposal.get("alternative_track_candidates")
    if isinstance(raw_alternative_track_candidates, list):
        for item in raw_alternative_track_candidates:
            if not isinstance(item, dict):
                continue
            track_id_value = item.get("track_id")
            confidence_value = _get_numeric(item.get("association_confidence"))
            if not isinstance(track_id_value, str) or not track_id_value.strip():
                continue
            candidate_view: dict[str, Any] = {"track_id": track_id_value}
            if confidence_value is not None:
                candidate_view["association_confidence"] = confidence_value
            alternative_track_candidates.append(candidate_view)

    return {
        "detection_id": raw_proposal.get("detection_id"),
        "proposed_track_id": raw_proposal.get("proposed_track_id"),
        "association_confidence": raw_proposal.get("association_confidence"),
        "occlusion_likelihood": raw_proposal.get("occlusion_likelihood"),
        "id_switch_risk": raw_proposal.get("id_switch_risk"),
        "depth_conflict_flag": bool(raw_proposal.get("depth_conflict_flag", False)),
        "position_conflict_flag": bool(raw_proposal.get("position_conflict_flag", False)),
        "uncertainties": _coerce_str_list(raw_proposal.get("uncertainties")),
        "supporting_evidence": _coerce_str_list(raw_proposal.get("supporting_evidence")),
        "alternative_track_candidates": alternative_track_candidates,
    }


def _build_observation_verification_payload(
    *,
    detection: DetectionCandidate,
    tracking_proposal: dict[str, Any],
    scenario_metadata: dict[str, Any],
    event_reasons: list[str],
) -> dict[str, Any]:
    independent_history = scenario_metadata.get(INDEPENDENT_TRACK_HISTORY_KEY)
    if not isinstance(independent_history, list):
        independent_history = []

    proposal_track_id = tracking_proposal.get("proposed_track_id")
    selected_history = _find_independent_history_record(
        independent_history,
        proposal_track_id if isinstance(proposal_track_id, str) else None,
    )
    structured_history_available = len(independent_history) > 0
    evidence_mode = {
        "mode": (
            EVIDENCE_MODE_SYNTHETIC_WITH_HISTORY
            if structured_history_available
            else EVIDENCE_MODE_SYNTHETIC_NO_HISTORY
        ),
        "frame_source": "synthetic_frame",
        "visual_identity_verifiability": VERIFICATION_DIMENSION_UNAVAILABLE,
        "structured_history_available": structured_history_available,
        "structured_history_source": (
            "independent_track_memory"
            if structured_history_available
            else None
        ),
    }

    return {
        "task": "independent_tracking_association_verification",
        "requirements": {
            "do_not_invent_measurements": True,
            "assess_motion_depth_visual_consistency": True,
            "report_supported_contradicted_or_inconclusive": True,
        },
        "event_context": {
            "scenario": scenario_metadata.get("scenario"),
            "event_reasons": list(event_reasons),
        },
        "evidence_mode": evidence_mode,
        "tracking_proposal": _extract_tracking_proposal_view(tracking_proposal),
        "detection_measurements": detection.to_dict(),
        "independent_history": {
            "all_tracks": independent_history,
            "for_proposed_track": selected_history,
        },
    }


def _run_deterministic_observation_audit(
    *,
    detection: DetectionCandidate,
    tracking_proposal: dict[str, Any],
    scenario_metadata: dict[str, Any],
) -> dict[str, Any]:
    independent_history = scenario_metadata.get(INDEPENDENT_TRACK_HISTORY_KEY)
    if not isinstance(independent_history, list):
        independent_history = []

    proposed_track_id = tracking_proposal.get("proposed_track_id")
    proposal_supporting_evidence = _coerce_str_list(tracking_proposal.get("supporting_evidence"))
    depth_conflict_flag = bool(tracking_proposal.get("depth_conflict_flag", False))
    depth_consistency_claimed = "depth_consistency" in proposal_supporting_evidence

    selected_history = _find_independent_history_record(
        independent_history,
        proposed_track_id if isinstance(proposed_track_id, str) else None,
    )

    history_depth_median = None
    if isinstance(selected_history, dict):
        recent_relative_depth = selected_history.get("recent_relative_depth")
        if isinstance(recent_relative_depth, dict):
            history_depth_median = _get_numeric(recent_relative_depth.get("median"))

    detection_relative_depth = _get_numeric(detection.relative_depth)
    depth_delta = None
    depth_measurement_consistent = None
    if detection_relative_depth is not None and history_depth_median is not None:
        depth_delta = abs(detection_relative_depth - history_depth_median)
        depth_measurement_consistent = depth_delta <= 0.08

    history_path_latest: tuple[float, float] | None = None
    if isinstance(selected_history, dict):
        recent_ground_path = selected_history.get("recent_ground_path")
        if isinstance(recent_ground_path, list) and len(recent_ground_path) > 0:
            latest = recent_ground_path[-1]
            if isinstance(latest, dict):
                hx = _get_numeric(latest.get("ground_x"))
                hz = _get_numeric(latest.get("ground_z"))
                if hx is not None and hz is not None:
                    history_path_latest = (hx, hz)

    detection_motion_latest: tuple[float, float] | None = None
    if isinstance(detection.motion_history, list) and len(detection.motion_history) > 0:
        last_motion = detection.motion_history[-1]
        if isinstance(last_motion, dict):
            mx = _get_numeric(last_motion.get("ground_x"))
            mz = _get_numeric(last_motion.get("ground_z"))
            if mx is not None and mz is not None:
                detection_motion_latest = (mx, mz)
    if detection_motion_latest is None:
        gx = _get_numeric(detection.ground_x)
        gz = _get_numeric(detection.ground_z)
        if gx is not None and gz is not None:
            detection_motion_latest = (gx, gz)

    motion_distance_to_history_endpoint = None
    motion_measurement_consistent = None
    if history_path_latest is not None and detection_motion_latest is not None:
        motion_distance_to_history_endpoint = (
            (history_path_latest[0] - detection_motion_latest[0]) ** 2
            + (history_path_latest[1] - detection_motion_latest[1]) ** 2
        ) ** 0.5
        motion_measurement_consistent = motion_distance_to_history_endpoint <= 0.6

    audit_flags: list[str] = []
    if depth_conflict_flag and depth_consistency_claimed:
        audit_flags.append("depth_conflict_flag_and_depth_consistency_claimed")

    return {
        "audit_flags": audit_flags,
        "measurement_checks": {
            "depth_consistency_claimed": depth_consistency_claimed,
            "depth_conflict_flag": depth_conflict_flag,
            "detection_relative_depth": detection_relative_depth,
            "history_depth_median": history_depth_median,
            "depth_delta": depth_delta,
            "depth_measurement_consistent": depth_measurement_consistent,
            "motion_distance_to_history_endpoint": motion_distance_to_history_endpoint,
            "motion_measurement_consistent": motion_measurement_consistent,
        },
        "schema_checks": {
            "proposal_has_required_identity_fields": all(
                key in tracking_proposal
                for key in ("detection_id", "proposed_track_id", "association_confidence", "depth_conflict_flag")
            ),
        },
    }


def _validate_observation_verdict_schema(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [key for key in OBSERVATION_REQUIRED_KEYS if key not in payload]
    return len(missing) == 0, missing


def _classify_expected_identity_status(
    *,
    expected_track_id: str | None,
    proposed_track_id: str | None,
) -> tuple[str, bool | None]:
    if not isinstance(expected_track_id, str) or not expected_track_id.strip():
        return "expected_identity_unavailable", None
    if not isinstance(proposed_track_id, str) or not proposed_track_id.strip():
        return "proposed_identity_missing", False
    if proposed_track_id == expected_track_id:
        return "matched_expected_identity", True
    return "mismatched_expected_identity", False


def _classify_structured_motion_depth_consistency(
    deterministic_audit: dict[str, Any] | None,
) -> tuple[str, bool | None, bool | None]:
    measurement_checks = (
        deterministic_audit.get("measurement_checks")
        if isinstance(deterministic_audit, dict) and isinstance(deterministic_audit.get("measurement_checks"), dict)
        else {}
    )
    motion_consistent = measurement_checks.get("motion_measurement_consistent")
    depth_consistent = measurement_checks.get("depth_measurement_consistent")

    if motion_consistent is False or depth_consistent is False:
        return VERIFICATION_DIMENSION_UNSUPPORTED, motion_consistent, depth_consistent
    if motion_consistent is True and depth_consistent is True:
        return VERIFICATION_DIMENSION_SUPPORTED, motion_consistent, depth_consistent
    return VERIFICATION_DIMENSION_UNAVAILABLE, motion_consistent, depth_consistent


def _build_observation_verification_dimensions(
    *,
    observation_payload: dict[str, Any],
    deterministic_audit: dict[str, Any],
    observation_invocation: dict[str, Any] | None,
    observation_verdict: dict[str, Any] | None,
) -> dict[str, Any]:
    structured_status, motion_consistent, depth_consistent = _classify_structured_motion_depth_consistency(
        deterministic_audit
    )

    evidence_mode = observation_payload.get("evidence_mode")
    evidence_mode_name = None
    real_camera_frame_available = None
    visual_identity_verifiability = VERIFICATION_DIMENSION_UNAVAILABLE
    if isinstance(evidence_mode, dict):
        mode_value = evidence_mode.get("mode")
        if isinstance(mode_value, str):
            evidence_mode_name = mode_value
        frame_source = evidence_mode.get("frame_source")
        if isinstance(frame_source, str):
            real_camera_frame_available = frame_source == "camera_frame"
        verifiability_value = evidence_mode.get("visual_identity_verifiability")
        if verifiability_value in {
            VERIFICATION_DIMENSION_SUPPORTED,
            VERIFICATION_DIMENSION_UNSUPPORTED,
            VERIFICATION_DIMENSION_UNAVAILABLE,
        }:
            visual_identity_verifiability = str(verifiability_value)

    visual_content_inspected = None
    if isinstance(observation_verdict, dict):
        visual_content_inspected = bool(observation_verdict.get("visual_content_inspected", False))

    verification_status = None
    overall_status = VERIFICATION_DIMENSION_UNAVAILABLE
    if isinstance(observation_verdict, dict):
        status_value = observation_verdict.get("verification_status")
        if isinstance(status_value, str):
            verification_status = status_value
            if status_value == "supported":
                overall_status = VERIFICATION_DIMENSION_SUPPORTED
            elif status_value in {"contradicted", "inconclusive"}:
                overall_status = VERIFICATION_DIMENSION_UNSUPPORTED

    invocation_status = observation_invocation.get("status") if isinstance(observation_invocation, dict) else None

    return {
        "structured_motion_depth_consistency": {
            "status": structured_status,
            "motion_measurement_consistent": motion_consistent,
            "depth_measurement_consistent": depth_consistent,
        },
        "visual_identity_verifiability": {
            "status": visual_identity_verifiability,
            "evidence_mode": evidence_mode_name,
            "visual_content_inspected": visual_content_inspected,
            "real_camera_frame_available": real_camera_frame_available,
            "invocation_status": invocation_status,
        },
        "overall_observation_verification": {
            "status": overall_status,
            "verification_status": verification_status,
        },
    }


def _collect_conflicting_proposed_track_ids(
    per_detection: dict[str, dict[str, Any]],
) -> set[str]:
    detection_ids_by_track: dict[str, list[str]] = {}
    for detection_id, payload in per_detection.items():
        if not isinstance(payload, dict):
            continue
        tracking_proposal = payload.get("tracking_proposal")
        if not isinstance(tracking_proposal, dict):
            continue
        proposed_track_id = tracking_proposal.get("proposed_track_id")
        if not isinstance(proposed_track_id, str) or not proposed_track_id.strip():
            continue
        detection_ids_by_track.setdefault(proposed_track_id, []).append(detection_id)

    return {
        track_id
        for track_id, detection_ids in detection_ids_by_track.items()
        if len(detection_ids) > 1
    }


def _classify_diagnostic_adjudication(
    *,
    detection_id: str,
    per_detection_payload: dict[str, Any],
    conflicting_proposed_track_ids: set[str],
    provisional_min_confidence: float = DIAGNOSTIC_ADJUDICATION_PROVISIONAL_MIN_CONFIDENCE,
) -> dict[str, Any]:
    tracking_proposal = per_detection_payload.get("tracking_proposal")
    if not isinstance(tracking_proposal, dict):
        return {
            "classification": "unresolved",
            "reason_codes": ["missing_tracking_proposal"],
            "provisional_min_confidence": provisional_min_confidence,
            "inputs": {
                "detection_id": detection_id,
                "association_confidence": None,
                "observation_verification_status": None,
                "deterministic_motion_consistent": None,
                "deterministic_depth_consistent": None,
                "coordinator_accepted": None,
                "coordinator_rejection_reasons": [],
                "conflicting_assignment_to_same_track": False,
                "retained_yolo_identity": None,
            },
        }

    proposed_track_id = tracking_proposal.get("proposed_track_id")
    association_confidence = _get_numeric(tracking_proposal.get("association_confidence"))

    observation_verification_status = per_detection_payload.get("observation_verification_status")
    if not isinstance(observation_verification_status, str):
        observation_verification_status = None

    deterministic_audit = per_detection_payload.get("deterministic_audit")
    deterministic_checks = (
        deterministic_audit.get("measurement_checks")
        if isinstance(deterministic_audit, dict)
        and isinstance(deterministic_audit.get("measurement_checks"), dict)
        else {}
    )
    deterministic_motion_consistent = deterministic_checks.get("motion_measurement_consistent")
    deterministic_depth_consistent = deterministic_checks.get("depth_measurement_consistent")

    coordinator_decision = per_detection_payload.get("coordinator_decision")
    coordinator_accepted = None
    coordinator_rejection_reasons: list[str] = []
    if isinstance(coordinator_decision, dict):
        accepted_value = coordinator_decision.get("accepted")
        if isinstance(accepted_value, bool):
            coordinator_accepted = accepted_value
        raw_reasons = coordinator_decision.get("rejection_reasons")
        if isinstance(raw_reasons, list):
            coordinator_rejection_reasons = [str(item) for item in raw_reasons]

    observation_request_payload = per_detection_payload.get("observation_request_payload")
    yolo_track_id = None
    if isinstance(observation_request_payload, dict):
        detection_measurements = observation_request_payload.get("detection_measurements")
        if isinstance(detection_measurements, dict):
            value = detection_measurements.get("yolo_track_id")
            if isinstance(value, str):
                yolo_track_id = value

    retained_yolo_identity = (
        isinstance(yolo_track_id, str)
        and isinstance(proposed_track_id, str)
        and yolo_track_id == proposed_track_id
    )
    conflicting_assignment_to_same_track = (
        isinstance(proposed_track_id, str)
        and proposed_track_id in conflicting_proposed_track_ids
    )

    reason_codes: list[str] = []

    if coordinator_accepted is False:
        reason_codes.append("coordinator_rejected")
        return {
            "classification": "rejected",
            "reason_codes": reason_codes,
            "provisional_min_confidence": provisional_min_confidence,
            "inputs": {
                "detection_id": detection_id,
                "association_confidence": association_confidence,
                "observation_verification_status": observation_verification_status,
                "deterministic_motion_consistent": deterministic_motion_consistent,
                "deterministic_depth_consistent": deterministic_depth_consistent,
                "coordinator_accepted": coordinator_accepted,
                "coordinator_rejection_reasons": coordinator_rejection_reasons,
                "conflicting_assignment_to_same_track": conflicting_assignment_to_same_track,
                "retained_yolo_identity": retained_yolo_identity,
            },
        }

    if conflicting_assignment_to_same_track:
        reason_codes.append("conflicting_assignment_to_same_track")

    if association_confidence is None:
        reason_codes.append("missing_association_confidence")
    elif association_confidence < provisional_min_confidence:
        reason_codes.append("provisional_low_confidence")

    if observation_verification_status not in {"supported", "contradicted", "inconclusive"}:
        reason_codes.append("observation_verification_missing_or_malformed")

    if deterministic_motion_consistent is False:
        reason_codes.append("deterministic_motion_inconsistent")
    if deterministic_depth_consistent is False:
        reason_codes.append("deterministic_depth_inconsistent")

    if retained_yolo_identity:
        reason_codes.append("retained_yolo_identity")

    classification = "unresolved"

    if observation_verification_status == "contradicted":
        if deterministic_motion_consistent is False or deterministic_depth_consistent is False:
            classification = "rejected"
            reason_codes.append("observation_contradicted_with_deterministic_conflict")
        else:
            classification = "rejected"
            reason_codes.append("observation_contradicted")
    elif observation_verification_status == "inconclusive":
        classification = "unresolved"
        reason_codes.append("observation_inconclusive")
    elif observation_verification_status == "supported":
        if (
            isinstance(association_confidence, float)
            and association_confidence >= provisional_min_confidence
            and coordinator_accepted is True
            and deterministic_motion_consistent is True
            and deterministic_depth_consistent is not False
            and not conflicting_assignment_to_same_track
        ):
            classification = "accepted"
            reason_codes.append("multi_source_consistent_support")
        else:
            classification = "unresolved"
            reason_codes.append("supported_but_insufficient_consistency")

    if (
        classification == "accepted"
        and observation_verification_status in {"contradicted", "inconclusive"}
        and deterministic_motion_consistent is False
    ):
        classification = "unresolved"
        reason_codes.append("accepted_guardrail_disagreement_with_motion_inconsistency")

    if retained_yolo_identity and classification == "accepted":
        reason_codes.append("accepted_with_retained_yolo_identity_independently_supported")
    elif retained_yolo_identity and classification != "accepted":
        reason_codes.append("retained_yolo_identity_not_independently_verified")

    return {
        "classification": classification,
        "reason_codes": reason_codes,
        "provisional_min_confidence": provisional_min_confidence,
        "inputs": {
            "detection_id": detection_id,
            "association_confidence": association_confidence,
            "observation_verification_status": observation_verification_status,
            "deterministic_motion_consistent": deterministic_motion_consistent,
            "deterministic_depth_consistent": deterministic_depth_consistent,
            "coordinator_accepted": coordinator_accepted,
            "coordinator_rejection_reasons": coordinator_rejection_reasons,
            "conflicting_assignment_to_same_track": conflicting_assignment_to_same_track,
            "retained_yolo_identity": retained_yolo_identity,
        },
    }


def _collect_recovery_candidate_track_ids(
    tracking_proposal: dict[str, Any],
) -> list[tuple[str, str, float | None]]:
    candidate_rows: list[tuple[str, str, float | None]] = []
    seen: set[str] = set()

    proposed_track_id = tracking_proposal.get("proposed_track_id")
    proposed_confidence = _get_numeric(tracking_proposal.get("association_confidence"))
    if isinstance(proposed_track_id, str) and proposed_track_id.strip():
        seen.add(proposed_track_id)
        candidate_rows.append((proposed_track_id, "proposed", proposed_confidence))

    alternatives = tracking_proposal.get("alternative_track_candidates")
    if isinstance(alternatives, list):
        for item in alternatives:
            if not isinstance(item, dict):
                continue
            track_id = item.get("track_id")
            confidence = _get_numeric(item.get("association_confidence"))
            if not isinstance(track_id, str) or not track_id.strip() or track_id in seen:
                continue
            seen.add(track_id)
            candidate_rows.append((track_id, "alternative", confidence))

    return candidate_rows


def _evaluate_recovery_candidate_against_history(
    *,
    detection: DetectionCandidate,
    candidate_track_id: str,
    candidate_source: str,
    candidate_association_confidence: float | None,
    independent_history: list[dict[str, Any]],
    max_history_age_seconds: float = DIAGNOSTIC_RECOVERY_MAX_HISTORY_AGE_SECONDS,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    score = 0.0

    history_record = _find_independent_history_record(independent_history, candidate_track_id)
    if not isinstance(history_record, dict):
        return {
            "track_id": candidate_track_id,
            "candidate_source": candidate_source,
            "candidate_association_confidence": candidate_association_confidence,
            "history_status": "missing",
            "history_age_seconds": None,
            "motion_distance_to_history_endpoint": None,
            "motion_consistent": None,
            "depth_delta": None,
            "depth_consistent": None,
            "score": score,
            "eligible_for_correction": False,
            "reason_codes": ["missing_history_record"],
        }

    history_status = "ok"
    history_age_seconds = None

    evidence_timestamp = _get_numeric(history_record.get("evidence_timestamp"))
    detection_timestamp = _get_numeric(detection.timestamp)
    if evidence_timestamp is None or detection_timestamp is None:
        history_status = "malformed"
        reason_codes.append("history_timestamp_missing_or_non_numeric")
    else:
        history_age_seconds = detection_timestamp - evidence_timestamp
        if history_age_seconds < 0:
            history_status = "malformed"
            reason_codes.append("history_timestamp_in_future")
        elif history_age_seconds > max_history_age_seconds:
            history_status = "stale"
            reason_codes.append("history_timestamp_stale")

    history_path_latest: tuple[float, float] | None = None
    recent_ground_path = history_record.get("recent_ground_path")
    if isinstance(recent_ground_path, list) and len(recent_ground_path) > 0:
        latest = recent_ground_path[-1]
        if isinstance(latest, dict):
            hx = _get_numeric(latest.get("ground_x"))
            hz = _get_numeric(latest.get("ground_z"))
            if hx is not None and hz is not None:
                history_path_latest = (hx, hz)
    if history_path_latest is None:
        reason_codes.append("history_motion_endpoint_missing_or_malformed")

    history_depth_median = None
    recent_relative_depth = history_record.get("recent_relative_depth")
    if isinstance(recent_relative_depth, dict):
        history_depth_median = _get_numeric(recent_relative_depth.get("median"))
    if history_depth_median is None:
        reason_codes.append("history_depth_median_missing_or_malformed")

    detection_motion_latest: tuple[float, float] | None = None
    if isinstance(detection.motion_history, list) and len(detection.motion_history) > 0:
        motion_item = detection.motion_history[-1]
        if isinstance(motion_item, dict):
            mx = _get_numeric(motion_item.get("ground_x"))
            mz = _get_numeric(motion_item.get("ground_z"))
            if mx is not None and mz is not None:
                detection_motion_latest = (mx, mz)
    if detection_motion_latest is None:
        gx = _get_numeric(detection.ground_x)
        gz = _get_numeric(detection.ground_z)
        if gx is not None and gz is not None:
            detection_motion_latest = (gx, gz)

    motion_distance = None
    motion_consistent = None
    if history_path_latest is not None and detection_motion_latest is not None:
        motion_distance = (
            (history_path_latest[0] - detection_motion_latest[0]) ** 2
            + (history_path_latest[1] - detection_motion_latest[1]) ** 2
        ) ** 0.5
        motion_consistent = motion_distance <= 0.6
        if motion_consistent:
            score += 2.0
            reason_codes.append("motion_supported_by_history")
        else:
            score -= 2.0
            reason_codes.append("motion_contradicted_by_history")
    else:
        reason_codes.append("motion_comparison_unavailable")

    depth_delta = None
    depth_consistent = None
    detection_depth = _get_numeric(detection.relative_depth)
    if detection_depth is not None and history_depth_median is not None:
        depth_delta = abs(detection_depth - history_depth_median)
        depth_consistent = depth_delta <= 0.08
        if depth_consistent:
            score += 2.0
            reason_codes.append("depth_supported_by_history")
        else:
            score -= 2.0
            reason_codes.append("depth_contradicted_by_history")
    else:
        reason_codes.append("depth_comparison_unavailable")

    if candidate_association_confidence is not None:
        score += min(max(candidate_association_confidence, 0.0), 1.0) * 0.5

    if history_status == "ok":
        score += 1.0
        reason_codes.append("history_timestamp_fresh")

    eligible_for_correction = (
        history_status == "ok"
        and motion_consistent is True
        and depth_consistent is True
    )
    if not eligible_for_correction:
        reason_codes.append("insufficient_deterministic_support_for_correction")

    return {
        "track_id": candidate_track_id,
        "candidate_source": candidate_source,
        "candidate_association_confidence": candidate_association_confidence,
        "history_status": history_status,
        "history_age_seconds": history_age_seconds,
        "motion_distance_to_history_endpoint": motion_distance,
        "motion_consistent": motion_consistent,
        "depth_delta": depth_delta,
        "depth_consistent": depth_consistent,
        "score": round(score, 4),
        "eligible_for_correction": eligible_for_correction,
        "reason_codes": reason_codes,
    }


def _select_group_consistent_recovery_assignment(
    candidate_evaluations_by_detection: dict[str, list[dict[str, Any]]],
) -> dict[str, str | None]:
    detection_ids = sorted(candidate_evaluations_by_detection.keys())
    best_assignment: dict[str, str | None] = {detection_id: None for detection_id in detection_ids}
    best_objective = (-1, float("-inf"))

    def _search(
        index: int,
        used_track_ids: set[str],
        assignment: dict[str, str | None],
        assigned_count: int,
        score_total: float,
    ) -> None:
        nonlocal best_assignment
        nonlocal best_objective
        if index >= len(detection_ids):
            objective = (assigned_count, score_total)
            if objective > best_objective:
                best_objective = objective
                best_assignment = dict(assignment)
            return

        detection_id = detection_ids[index]
        assignment[detection_id] = None
        _search(index + 1, used_track_ids, assignment, assigned_count, score_total)

        for item in candidate_evaluations_by_detection.get(detection_id, []):
            if not isinstance(item, dict):
                continue
            if item.get("eligible_for_correction") is not True:
                continue
            track_id = item.get("track_id")
            if not isinstance(track_id, str) or not track_id.strip() or track_id in used_track_ids:
                continue
            candidate_score = _get_numeric(item.get("score"))
            if candidate_score is None:
                candidate_score = 0.0
            assignment[detection_id] = track_id
            used_track_ids.add(track_id)
            _search(index + 1, used_track_ids, assignment, assigned_count + 1, score_total + candidate_score)
            used_track_ids.remove(track_id)

    _search(0, set(), {}, 0, 0.0)
    return best_assignment


def _run_diagnostic_alternative_identity_recovery(
    *,
    detections_by_id: dict[str, DetectionCandidate],
    per_detection: dict[str, dict[str, Any]],
    scenario_metadata: dict[str, Any],
) -> dict[str, Any]:
    independent_history = scenario_metadata.get(INDEPENDENT_TRACK_HISTORY_KEY)
    if not isinstance(independent_history, list):
        independent_history = []

    target_detection_ids: list[str] = []
    candidate_evaluations_by_detection: dict[str, list[dict[str, Any]]] = {}
    proposal_track_by_detection: dict[str, str | None] = {}

    for detection_id, payload in per_detection.items():
        if not isinstance(payload, dict):
            continue
        adjudication = payload.get("diagnostic_adjudication")
        if not isinstance(adjudication, dict):
            continue
        classification = adjudication.get("classification")
        if classification not in {"rejected", "unresolved"}:
            continue

        tracking_proposal = payload.get("tracking_proposal")
        if not isinstance(tracking_proposal, dict):
            continue
        detection = detections_by_id.get(detection_id)
        if detection is None:
            continue

        target_detection_ids.append(detection_id)
        proposed_track = tracking_proposal.get("proposed_track_id")
        proposal_track_by_detection[detection_id] = proposed_track if isinstance(proposed_track, str) else None

        candidate_rows = _collect_recovery_candidate_track_ids(tracking_proposal)
        candidate_evals: list[dict[str, Any]] = []
        for track_id, source, candidate_confidence in candidate_rows:
            candidate_evals.append(
                _evaluate_recovery_candidate_against_history(
                    detection=detection,
                    candidate_track_id=track_id,
                    candidate_source=source,
                    candidate_association_confidence=candidate_confidence,
                    independent_history=independent_history,
                )
            )
        candidate_evaluations_by_detection[detection_id] = candidate_evals

    assignment = _select_group_consistent_recovery_assignment(candidate_evaluations_by_detection)

    corrections_by_detection: dict[str, dict[str, Any]] = {}
    unresolved_by_detection: dict[str, dict[str, Any]] = {}
    proposed_corrections_count = 0

    for detection_id in target_detection_ids:
        payload = per_detection.get(detection_id)
        if not isinstance(payload, dict):
            continue

        candidate_evals = candidate_evaluations_by_detection.get(detection_id, [])
        selected_track = assignment.get(detection_id)
        proposed_track = proposal_track_by_detection.get(detection_id)

        selected_eval = None
        for item in candidate_evals:
            if isinstance(item, dict) and item.get("track_id") == selected_track:
                selected_eval = item
                break

        recommended_outcome = "unresolved"
        reason_codes: list[str] = []
        proposed_correction = None

        if not candidate_evals:
            reason_codes.append("no_candidates_available")
        elif selected_track is None:
            reason_codes.append("no_group_consistent_candidate_assignment")
        elif selected_eval is None:
            reason_codes.append("selected_candidate_not_found")
        elif selected_track == proposed_track:
            reason_codes.append("best_supported_candidate_matches_original_proposal")
        elif selected_eval.get("eligible_for_correction") is not True:
            reason_codes.append("selected_candidate_not_eligible_for_correction")
        else:
            recommended_outcome = "correction_proposed"
            proposed_corrections_count += 1
            proposed_correction = {
                "from_track_id": proposed_track,
                "to_track_id": selected_track,
                "group_consistency_enforced": True,
                "deterministic_evidence": {
                    "history_status": selected_eval.get("history_status"),
                    "history_age_seconds": selected_eval.get("history_age_seconds"),
                    "motion_distance_to_history_endpoint": selected_eval.get("motion_distance_to_history_endpoint"),
                    "motion_consistent": selected_eval.get("motion_consistent"),
                    "depth_delta": selected_eval.get("depth_delta"),
                    "depth_consistent": selected_eval.get("depth_consistent"),
                    "score": selected_eval.get("score"),
                },
                "reason_codes": selected_eval.get("reason_codes"),
            }

        output_payload = {
            "considered": True,
            "original_adjudication_classification": payload.get("diagnostic_adjudication", {}).get("classification"),
            "candidate_evaluations": candidate_evals,
            "group_selected_track_id": selected_track,
            "recommended_outcome": recommended_outcome,
            "proposed_correction": proposed_correction,
            "reason_codes": reason_codes,
            "no_fabricated_identities": True,
            "live_state_mutation": False,
        }

        payload["diagnostic_alternative_identity_recovery"] = output_payload
        if recommended_outcome == "correction_proposed":
            corrections_by_detection[detection_id] = output_payload
        else:
            unresolved_by_detection[detection_id] = output_payload

    for detection_id, payload in per_detection.items():
        if detection_id in target_detection_ids:
            continue
        if not isinstance(payload, dict):
            continue
        payload["diagnostic_alternative_identity_recovery"] = {
            "considered": False,
            "reason_codes": ["adjudication_not_rejected_or_unresolved"],
            "live_state_mutation": False,
        }

    assigned_tracks = [
        item.get("proposed_correction", {}).get("to_track_id")
        for item in corrections_by_detection.values()
        if isinstance(item.get("proposed_correction"), dict)
    ]
    assigned_tracks = [item for item in assigned_tracks if isinstance(item, str)]
    duplicate_assignment_detected = len(assigned_tracks) != len(set(assigned_tracks))

    return {
        "target_detection_count": len(target_detection_ids),
        "proposed_corrections_count": proposed_corrections_count,
        "unresolved_count": len(unresolved_by_detection),
        "correction_detection_ids": sorted(corrections_by_detection.keys()),
        "unresolved_detection_ids": sorted(unresolved_by_detection.keys()),
        "duplicate_assignment_detected": duplicate_assignment_detected,
        "group_consistency_enforced": True,
    }


async def _mock_observation_visual_invoker(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    image_data_url: str,
    timeout_seconds: float,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    _ = (model, system_prompt, user_payload, image_data_url, timeout_seconds, max_output_tokens, reasoning_effort)
    return {
        "verification_status": "inconclusive",
        "visibility_status": "unknown",
        "identity_consistency": "unknown",
        "possible_id_switch": False,
        "confidence": None,
        "uncertainty": ["mocked_no_network_invocation"],
        "limitations": ["observation_agent_invocation_mocked"],
        "supporting_evidence_refs": [],
        "contradicting_evidence_refs": [],
        "recommended_follow_up": "run_live_observation_verification_with_approval",
        "visual_content_inspected": True,
    }


def _consume_provider_request_budget(
    budget: dict[str, int] | None,
) -> bool:
    if budget is None:
        return True
    remaining = budget.get("remaining")
    if not isinstance(remaining, int):
        return False
    if remaining <= 0:
        return False
    budget["remaining"] = remaining - 1
    budget["consumed"] = int(budget.get("consumed", 0)) + 1
    return True


def _extract_usage_fields_from_response(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    payload: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if isinstance(value, int):
            payload[field] = value
    return payload or None


def _build_provider_call_telemetry_entry(
    *,
    role: str,
    round_name: str,
    model: str,
    detection_id: str | None,
    attempted: bool,
    status: str,
    token_usage: dict[str, int] | None,
    latency_ms: float | None,
    error_reason: str | None,
    usage_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    input_tokens = token_usage.get("prompt_tokens") if isinstance(token_usage, dict) else None
    output_tokens = token_usage.get("completion_tokens") if isinstance(token_usage, dict) else None
    total_tokens = token_usage.get("total_tokens") if isinstance(token_usage, dict) else None
    usage_available = isinstance(token_usage, dict)
    return {
        "role": role,
        "round": round_name,
        "detection_id": detection_id,
        "model": model,
        "attempted": attempted,
        "status": status,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_available": usage_available,
        "usage_unavailable_reason": (None if usage_available else usage_unavailable_reason),
        "latency_ms": latency_ms,
        "error_reason": error_reason,
    }


def _read_persisted_provider_budget_state() -> dict[str, Any]:
    state_path, _, _ = provider_call_budget._resolve_paths()
    state_file_exists = state_path.is_file()
    try:
        totals_view = provider_call_budget.read_budget_totals()
    except Exception as error:
        return {
            "state_path": str(state_path),
            "available": False,
            "error": f"state_unavailable:{type(error).__name__}",
            "limits": None,
            "totals": None,
            "state_file_exists": state_file_exists,
        }
    if not isinstance(totals_view, dict):
        return {
            "state_path": str(state_path),
            "available": False,
            "error": "state_payload_not_object",
            "limits": None,
            "totals": None,
            "state_file_exists": state_file_exists,
        }
    limits = totals_view.get("limits") if isinstance(totals_view.get("limits"), dict) else None
    totals = totals_view.get("totals") if isinstance(totals_view.get("totals"), dict) else None
    return {
        "state_path": str(state_path),
        "available": isinstance(limits, dict) and isinstance(totals, dict),
        "error": None if isinstance(limits, dict) and isinstance(totals, dict) else "state_payload_missing_limits_or_totals",
        "limits": limits,
        "totals": totals,
        "state_file_exists": state_file_exists,
    }


def _compute_persisted_budget_preflight(
    *,
    detection_count: int,
    observation_verification_mode: str,
    tracking_reconsideration_mode: str,
) -> dict[str, Any]:
    required_tracking = 1
    required_observation_visual = 0
    if tracking_reconsideration_mode == TRACKING_RECONSIDERATION_MODE_LIVE:
        required_tracking += 1
    if observation_verification_mode == OBSERVATION_VERIFICATION_MODE_LIVE:
        required_observation_visual += detection_count
        if tracking_reconsideration_mode == TRACKING_RECONSIDERATION_MODE_LIVE:
            required_observation_visual += detection_count
    required_total = required_tracking + required_observation_visual

    persisted = _read_persisted_provider_budget_state()
    limits = persisted.get("limits") if isinstance(persisted.get("limits"), dict) else {}
    totals = persisted.get("totals") if isinstance(persisted.get("totals"), dict) else {}
    per_role_limits = limits.get("per_role_limits") if isinstance(limits.get("per_role_limits"), dict) else {}
    per_role_attempted = totals.get("per_role_attempted") if isinstance(totals.get("per_role_attempted"), dict) else {}

    def _remaining(limit_value: Any, attempted_value: Any) -> int | None:
        if not isinstance(limit_value, int) or not isinstance(attempted_value, int):
            return None
        return max(limit_value - attempted_value, 0)

    tracking_remaining = _remaining(
        per_role_limits.get(provider_call_budget.ROLE_TRACKING_VLM),
        per_role_attempted.get(provider_call_budget.ROLE_TRACKING_VLM),
    )
    observation_visual_remaining = _remaining(
        per_role_limits.get(provider_call_budget.ROLE_OBSERVATION_VISUAL),
        per_role_attempted.get(provider_call_budget.ROLE_OBSERVATION_VISUAL),
    )
    total_remaining = _remaining(
        limits.get("total_limit"),
        totals.get("attempted"),
    )

    shortfall_tracking = None if tracking_remaining is None else max(required_tracking - tracking_remaining, 0)
    shortfall_observation_visual = None if observation_visual_remaining is None else max(required_observation_visual - observation_visual_remaining, 0)
    shortfall_total = None if total_remaining is None else max(required_total - total_remaining, 0)

    sufficient = (
        persisted.get("available") is True
        and isinstance(shortfall_tracking, int)
        and isinstance(shortfall_observation_visual, int)
        and isinstance(shortfall_total, int)
        and shortfall_tracking == 0
        and shortfall_observation_visual == 0
        and shortfall_total == 0
    )

    return {
        "required": {
            "tracking_vlm": required_tracking,
            "observation_visual": required_observation_visual,
            "total": required_total,
        },
        "persisted": {
            "state_path": persisted.get("state_path"),
            "available": persisted.get("available"),
            "error": persisted.get("error"),
            "limits": limits if isinstance(limits, dict) else None,
            "totals": totals if isinstance(totals, dict) else None,
            "remaining": {
                "tracking_vlm": tracking_remaining,
                "observation_visual": observation_visual_remaining,
                "total": total_remaining,
            },
        },
        "shortfall": {
            "tracking_vlm": shortfall_tracking,
            "observation_visual": shortfall_observation_visual,
            "total": shortfall_total,
        },
        "sufficient": sufficient,
    }


def _build_tracking_reconsideration_prompt(
    user_payload: dict[str, Any],
) -> str:
    return (
        "Analyze the provided reconsideration targets and return strict JSON only. "
        "Return keys: revisions, declined. Optionally include decline_reason. "
        "Each revision must include detection_id, proposed_track_id, association_confidence, "
        "occlusion_likelihood, id_switch_risk, depth_conflict_flag, position_conflict_flag, "
        "uncertainties, supporting_evidence, alternative_track_candidates. "
        "Do not invent measurements, identities, or candidate IDs.\n"
        f"{STRUCTURED_INPUT_MARKER}{json.dumps(user_payload)}"
    )


async def _invoke_tracking_reconsideration_live(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    image_data_url: str | None,
    timeout_seconds: float,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("OPENROUTER_API_KEY is required for live reconsideration")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _build_tracking_reconsideration_prompt(user_payload),
        }
    ]
    if isinstance(image_data_url, str) and image_data_url.strip():
        user_content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    reservation = provider_call_budget.reserve_or_raise(provider_call_budget.ROLE_TRACKING_VLM)
    started = time.perf_counter()
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    try:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            request_kwargs["max_tokens"] = max_output_tokens
        if isinstance(reasoning_effort, str) and reasoning_effort.strip():
            request_kwargs["reasoning_effort"] = reasoning_effort.strip()

        response = await asyncio.wait_for(
            client.chat.completions.create(**request_kwargs),
            timeout=timeout_seconds,
        )
        usage = _extract_usage_fields_from_response(response)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="ok",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            usage=usage,
        )
    except asyncio.TimeoutError:
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code="timeout",
        )
        raise
    except Exception:
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code="runtime_error",
        )
        raise

    if not getattr(response, "choices", None):
        raise ValueError("tracking_reconsideration_empty_response")

    content = getattr(getattr(response.choices[0], "message", None), "content", "") or ""
    if getattr(response.choices[0], "finish_reason", None) == "length":
        raise ValueError("tracking_reconsideration_output_truncated")

    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict):
        raise ValueError("tracking_reconsideration_invalid_json_response")
    parsed[DIAGNOSTIC_PROVIDER_CALL_META_KEY] = {
        "role": provider_call_budget.ROLE_TRACKING_VLM,
        "model": model,
        "token_usage": usage,
        "usage_unavailable_reason": None if isinstance(usage, dict) else "provider_usage_not_returned",
    }
    return parsed


async def _invoke_observation_visual_verification_live(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    image_data_url: str,
    timeout_seconds: float,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("OPENROUTER_API_KEY is required for live observation verification")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    reservation = provider_call_budget.reserve_or_raise(provider_call_budget.ROLE_OBSERVATION_VISUAL)
    started = time.perf_counter()
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    try:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(user_payload)},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        }
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            request_kwargs["max_tokens"] = max_output_tokens
        if isinstance(reasoning_effort, str) and reasoning_effort.strip():
            request_kwargs["reasoning_effort"] = reasoning_effort.strip()
        response = await asyncio.wait_for(
            client.chat.completions.create(**request_kwargs),
            timeout=timeout_seconds,
        )
        usage = _extract_usage_fields_from_response(response)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="ok",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            usage=usage,
        )
    except asyncio.TimeoutError:
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code="timeout",
        )
        raise
    except Exception:
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="error",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error_code="runtime_error",
        )
        raise
    finally:
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_result = close_method()
            if asyncio.iscoroutine(close_result):
                await close_result

    if not getattr(response, "choices", None):
        raise ValueError("observation_response_empty")
    if getattr(response.choices[0], "finish_reason", None) == "length":
        raise ValueError("observation_response_truncated")

    content = getattr(getattr(response.choices[0], "message", None), "content", "") or ""
    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict):
        raise ValueError("observation_response_invalid_json")
    parsed[DIAGNOSTIC_PROVIDER_CALL_META_KEY] = {
        "role": provider_call_budget.ROLE_OBSERVATION_VISUAL,
        "model": model,
        "token_usage": usage,
        "usage_unavailable_reason": None if isinstance(usage, dict) else "provider_usage_not_returned",
    }
    return parsed


async def _invoke_observation_agent_verdict(
    *,
    mode: str,
    observation_model: str,
    observation_system_prompt: str,
    user_payload: dict[str, Any],
    frame_reference: str | None,
    timeout_seconds: float,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    invoker: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    provider_request_budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    if mode == OBSERVATION_VERIFICATION_MODE_OFF:
        return {
            "invocation": {"status": "skipped", "mode": mode, "error": None},
            "verdict": None,
        }

    if not isinstance(frame_reference, str) or not frame_reference.strip():
        return {
            "invocation": {
                "status": "error",
                "mode": mode,
                "error": "frame_reference_unavailable_for_observation_verification",
            },
            "verdict": None,
        }

    selected_invoker = invoker
    if selected_invoker is None:
        if mode == OBSERVATION_VERIFICATION_MODE_MOCK:
            selected_invoker = _mock_observation_visual_invoker
        else:
            selected_invoker = _invoke_observation_visual_verification_live

    if mode == OBSERVATION_VERIFICATION_MODE_LIVE:
        if not _consume_provider_request_budget(provider_request_budget):
            return {
                "invocation": {
                    "status": "error",
                    "mode": mode,
                    "error": "provider_request_limit_exhausted",
                },
                "verdict": None,
            }

    started = time.perf_counter()
    try:
        raw_payload = await selected_invoker(
            model=observation_model,
            system_prompt=observation_system_prompt,
            user_payload=user_payload,
            image_data_url=frame_reference,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    except Exception as error:
        return {
            "invocation": {
                "status": "error",
                "mode": mode,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error": f"{type(error).__name__}:{error}",
            },
            "verdict": None,
        }

    if not isinstance(raw_payload, dict):
        return {
            "invocation": {
                "status": "error",
                "mode": mode,
                "latency_ms": latency_ms,
                "error": "observation_response_not_object",
            },
            "verdict": None,
        }

    provider_meta = raw_payload.get(DIAGNOSTIC_PROVIDER_CALL_META_KEY)
    if isinstance(provider_meta, dict):
        raw_payload = dict(raw_payload)
        raw_payload.pop(DIAGNOSTIC_PROVIDER_CALL_META_KEY, None)

    usage = provider_meta.get("token_usage") if isinstance(provider_meta, dict) and isinstance(provider_meta.get("token_usage"), dict) else None
    usage_unavailable_reason = (
        provider_meta.get("usage_unavailable_reason")
        if isinstance(provider_meta, dict) and isinstance(provider_meta.get("usage_unavailable_reason"), str)
        else ("usage_not_exposed_by_invoker" if not isinstance(usage, dict) else None)
    )

    schema_valid, missing_keys = _validate_observation_verdict_schema(raw_payload)
    if not schema_valid:
        response_top_level_keys = sorted(
            str(key)
            for key in raw_payload.keys()
            if isinstance(key, str)
        )
        return {
            "invocation": {
                "status": "error",
                "mode": mode,
                "latency_ms": latency_ms,
                "error": "observation_response_schema_invalid",
                "missing_keys": missing_keys,
                "response_top_level_keys": response_top_level_keys,
            },
            "verdict": None,
        }

    verdict = {
        "verification_status": raw_payload.get("verification_status"),
        "visibility_status": raw_payload.get("visibility_status"),
        "identity_consistency": raw_payload.get("identity_consistency"),
        "possible_id_switch": bool(raw_payload.get("possible_id_switch", False)),
        "confidence": raw_payload.get("confidence"),
        "uncertainty": _coerce_str_list(raw_payload.get("uncertainty")),
        "limitations": _coerce_str_list(raw_payload.get("limitations")),
        "supporting_evidence_refs": _coerce_str_list(raw_payload.get("supporting_evidence_refs")),
        "contradicting_evidence_refs": _coerce_str_list(raw_payload.get("contradicting_evidence_refs")),
        "recommended_follow_up": raw_payload.get("recommended_follow_up"),
        "visual_content_inspected": bool(raw_payload.get("visual_content_inspected", False)),
    }

    return {
        "invocation": {
            "status": "ok",
            "mode": mode,
            "latency_ms": latency_ms,
            "error": None,
            "token_usage": usage,
            "usage_unavailable_reason": usage_unavailable_reason,
        },
        "verdict": verdict,
    }


def _build_visual_association_proposal_from_payload(
    *,
    detection_id: str,
    payload: dict[str, Any],
) -> VisualAssociationProposal:
    proposed_track_id = payload.get("proposed_track_id")
    return VisualAssociationProposal(
        detection_id=detection_id,
        proposed_track_id=(proposed_track_id if isinstance(proposed_track_id, str) and proposed_track_id.strip() else None),
        association_confidence=_get_numeric(payload.get("association_confidence")),
        occlusion_likelihood=_get_numeric(payload.get("occlusion_likelihood")),
        id_switch_risk=_get_numeric(payload.get("id_switch_risk")),
        depth_conflict_flag=bool(payload.get("depth_conflict_flag", False)),
        position_conflict_flag=bool(payload.get("position_conflict_flag", False)),
        uncertainties=_coerce_str_list(payload.get("uncertainties")),
        supporting_evidence=_coerce_str_list(payload.get("supporting_evidence")),
    )


def _build_tracking_reconsideration_payload(
    *,
    eligible_detection_ids: list[str],
    detections_by_id: dict[str, DetectionCandidate],
    per_detection: dict[str, dict[str, Any]],
    scenario_metadata: dict[str, Any],
    event_reasons: list[str],
) -> dict[str, Any]:
    independent_history = scenario_metadata.get(INDEPENDENT_TRACK_HISTORY_KEY)
    if not isinstance(independent_history, list):
        independent_history = []

    reconsideration_targets: list[dict[str, Any]] = []
    for detection_id in eligible_detection_ids:
        payload = per_detection.get(detection_id)
        detection = detections_by_id.get(detection_id)
        if not isinstance(payload, dict) or detection is None:
            continue
        tracking_proposal = payload.get("tracking_proposal")
        observation_verdict = payload.get("observation_agent_verdict")
        adjudication = payload.get("diagnostic_adjudication")
        if not isinstance(tracking_proposal, dict):
            continue

        reconsideration_targets.append(
            {
                "detection_id": detection_id,
                "detection_measurements": detection.to_dict(),
                "original_tracking_proposal": tracking_proposal,
                "original_observation_feedback": observation_verdict,
                "original_diagnostic_adjudication": adjudication,
                "available_alternative_track_candidates": tracking_proposal.get("alternative_track_candidates", []),
            }
        )

    return {
        "task": "diagnostic_tracking_identity_reconsideration",
        "requirements": {
            "evaluate_all_targets_jointly": True,
            "do_not_invent_candidate_track_ids": True,
            "do_not_use_deterministic_recovery_answer_as_source": True,
            "return_revisions_or_decline": True,
            "preserve_uncertainty_when_insufficient_evidence": True,
        },
        "event_context": {
            "scenario": scenario_metadata.get("scenario"),
            "event_reasons": list(event_reasons),
        },
        "independent_track_history_evidence": independent_history,
        "targets": reconsideration_targets,
    }


async def _invoke_tracking_reconsideration(
    *,
    mode: str | None = None,
    enabled: bool,
    tracking_model: str,
    user_payload: dict[str, Any],
    frame_reference: str | None,
    timeout_seconds: float,
    max_output_tokens: int | None = None,
    reasoning_effort: str | None = None,
    invoker: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    provider_request_budget: dict[str, int] | None = None,
) -> dict[str, Any]:
    if mode == TRACKING_RECONSIDERATION_MODE_OFF or not enabled:
        return {
            "invocation": {
                "status": "skipped",
                "error": None,
                "mode": "disabled" if mode is None else mode,
            },
            "revisions_by_detection": {},
            "declined": False,
            "decline_reason": None,
        }

    selected_invoker = invoker
    if mode == TRACKING_RECONSIDERATION_MODE_LIVE and selected_invoker is None:
        selected_invoker = _invoke_tracking_reconsideration_live

    if selected_invoker is None:
        return {
            "invocation": {
                "status": "skipped",
                "error": "tracking_reconsideration_invoker_missing",
                "mode": "diagnostic_legacy",
            },
            "revisions_by_detection": {},
            "declined": False,
            "decline_reason": None,
        }

    if mode == TRACKING_RECONSIDERATION_MODE_LIVE:
        if not _consume_provider_request_budget(provider_request_budget):
            return {
                "invocation": {
                    "status": "error",
                    "mode": "diagnostic",
                    "error": "provider_request_limit_exhausted",
                },
                "revisions_by_detection": {},
                "declined": False,
                "decline_reason": None,
            }

    started = time.perf_counter()
    try:
        raw_payload = await selected_invoker(
            model=tracking_model,
            system_prompt=load_tracking_agent_skill(),
            user_payload=user_payload,
            image_data_url=frame_reference,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    except Exception as error:
        return {
            "invocation": {
                "status": "error",
                "mode": "diagnostic",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error": f"{type(error).__name__}:{error}",
            },
            "revisions_by_detection": {},
            "declined": False,
            "decline_reason": None,
        }

    if not isinstance(raw_payload, dict):
        return {
            "invocation": {
                "status": "error",
                "mode": "diagnostic",
                "latency_ms": latency_ms,
                "error": "tracking_reconsideration_response_not_object",
            },
            "revisions_by_detection": {},
            "declined": False,
            "decline_reason": None,
        }

    provider_meta = raw_payload.get(DIAGNOSTIC_PROVIDER_CALL_META_KEY)
    if isinstance(provider_meta, dict):
        raw_payload = dict(raw_payload)
        raw_payload.pop(DIAGNOSTIC_PROVIDER_CALL_META_KEY, None)

    usage = provider_meta.get("token_usage") if isinstance(provider_meta, dict) and isinstance(provider_meta.get("token_usage"), dict) else None
    usage_unavailable_reason = (
        provider_meta.get("usage_unavailable_reason")
        if isinstance(provider_meta, dict) and isinstance(provider_meta.get("usage_unavailable_reason"), str)
        else ("usage_not_exposed_by_invoker" if not isinstance(usage, dict) else None)
    )

    missing_keys = [key for key in TRACKING_RECONSIDERATION_REQUIRED_KEYS if key not in raw_payload]
    if missing_keys:
        return {
            "invocation": {
                "status": "error",
                "mode": "diagnostic",
                "latency_ms": latency_ms,
                "error": "tracking_reconsideration_schema_invalid",
                "missing_keys": missing_keys,
            },
            "revisions_by_detection": {},
            "declined": False,
            "decline_reason": None,
        }

    revisions_by_detection: dict[str, dict[str, Any]] = {}
    revisions = raw_payload.get("revisions")
    if isinstance(revisions, list):
        for item in revisions:
            if not isinstance(item, dict):
                continue
            detection_id = item.get("detection_id")
            if not isinstance(detection_id, str) or not detection_id.strip():
                continue
            revisions_by_detection[detection_id] = {
                "detection_id": detection_id,
                "proposed_track_id": item.get("proposed_track_id"),
                "association_confidence": _get_numeric(item.get("association_confidence")),
                "occlusion_likelihood": _get_numeric(item.get("occlusion_likelihood")),
                "id_switch_risk": _get_numeric(item.get("id_switch_risk")),
                "depth_conflict_flag": bool(item.get("depth_conflict_flag", False)),
                "position_conflict_flag": bool(item.get("position_conflict_flag", False)),
                "uncertainties": _coerce_str_list(item.get("uncertainties")),
                "supporting_evidence": _coerce_str_list(item.get("supporting_evidence")),
                "alternative_track_candidates": item.get("alternative_track_candidates", []),
            }

    declined = bool(raw_payload.get("declined", False))
    decline_reason = raw_payload.get("decline_reason")

    return {
        "invocation": {
            "status": "ok",
            "mode": "diagnostic",
            "latency_ms": latency_ms,
            "error": None,
            "token_usage": usage,
            "usage_unavailable_reason": usage_unavailable_reason,
        },
        "revisions_by_detection": revisions_by_detection,
        "declined": declined,
        "decline_reason": (decline_reason if isinstance(decline_reason, str) else None),
    }


def _collect_duplicate_assigned_track_ids(
    assignment_track_by_detection: dict[str, str | None],
) -> dict[str, list[str]]:
    detection_ids_by_track: dict[str, list[str]] = {}
    for detection_id, track_id in assignment_track_by_detection.items():
        if not isinstance(track_id, str) or not track_id.strip():
            continue
        detection_ids_by_track.setdefault(track_id, []).append(detection_id)
    return {
        track_id: detection_ids
        for track_id, detection_ids in detection_ids_by_track.items()
        if len(detection_ids) > 1
    }


async def _run_observation_collaboration_diagnostic(
    *,
    detections: list[DetectionCandidate],
    proposal_by_detection: dict[str, dict[str, Any]],
    updates: list[dict[str, Any]],
    scenario_metadata: dict[str, Any],
    event_reasons: list[str],
    frame_reference: str | None,
    mode: str,
    observation_model: str,
    observation_timeout_seconds: float,
    observation_max_output_tokens: int | None = None,
    observation_reasoning_effort: str | None = None,
    invoker: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    tracking_reconsideration_model: str | None = None,
    tracking_reconsideration_timeout_seconds: float | None = None,
    tracking_reconsideration_max_output_tokens: int | None = None,
    tracking_reconsideration_reasoning_effort: str | None = None,
    reconsideration_invoker: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    tracking_reconsideration_mode: str | None = None,
    provider_request_budget: dict[str, int] | None = None,
    provider_call_telemetry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    detection_by_id = {item.detection_id: item for item in detections}
    updates_by_detection: dict[str, dict[str, Any]] = {}
    for item in updates:
        if not isinstance(item, dict):
            continue
        detection_id = item.get("detection_id")
        if isinstance(detection_id, str):
            updates_by_detection[detection_id] = item

    per_detection: dict[str, Any] = {}
    verification_status_counts: dict[str, int] = {}
    invocation_status_counts: dict[str, int] = {}
    reconsideration_verification_status_counts: dict[str, int] = {}
    reconsideration_invocation_status_counts: dict[str, int] = {}
    contradiction_count = 0
    modality_dimension_status_counts = {
        "structured_motion_depth_consistency": {},
        "visual_identity_verifiability": {},
        "overall_observation_verification": {},
    }
    raw_expected_associations = scenario_metadata.get("expected_correct_associations") if isinstance(scenario_metadata, dict) else None
    expected_correct_associations: dict[str, str] = {}
    if isinstance(raw_expected_associations, dict):
        for key, value in raw_expected_associations.items():
            if isinstance(key, str) and isinstance(value, str):
                expected_correct_associations[key] = value

    for detection_id, tracking_proposal in proposal_by_detection.items():
        detection = detection_by_id.get(detection_id)
        if detection is None:
            continue

        proposal_view = _extract_tracking_proposal_view(tracking_proposal)
        observation_payload = _build_observation_verification_payload(
            detection=detection,
            tracking_proposal=tracking_proposal,
            scenario_metadata=scenario_metadata,
            event_reasons=event_reasons,
        )
        deterministic_audit = _run_deterministic_observation_audit(
            detection=detection,
            tracking_proposal=tracking_proposal,
            scenario_metadata=scenario_metadata,
        )
        if "depth_conflict_flag_and_depth_consistency_claimed" in deterministic_audit.get("audit_flags", []):
            contradiction_count += 1

        observation_result = await _invoke_observation_agent_verdict(
            mode=mode,
            observation_model=observation_model,
            observation_system_prompt=DEFAULT_OBSERVATION_VISUAL_SYSTEM_PROMPT,
            user_payload=observation_payload,
            frame_reference=frame_reference,
            timeout_seconds=observation_timeout_seconds,
            max_output_tokens=observation_max_output_tokens,
            reasoning_effort=observation_reasoning_effort,
            invoker=invoker,
            provider_request_budget=provider_request_budget,
        )

        if isinstance(provider_call_telemetry, list):
            invocation_payload = observation_result.get("invocation") if isinstance(observation_result, dict) else {}
            status_value = invocation_payload.get("status") if isinstance(invocation_payload, dict) else None
            error_reason = invocation_payload.get("error") if isinstance(invocation_payload, dict) else None
            token_usage = invocation_payload.get("token_usage") if isinstance(invocation_payload, dict) and isinstance(invocation_payload.get("token_usage"), dict) else None
            usage_unavailable_reason = invocation_payload.get("usage_unavailable_reason") if isinstance(invocation_payload, dict) and isinstance(invocation_payload.get("usage_unavailable_reason"), str) else "usage_not_exposed_by_invoker"
            attempted = status_value in {"ok", "error"} and error_reason != "provider_request_limit_exhausted"
            if attempted:
                provider_call_telemetry.append(
                    _build_provider_call_telemetry_entry(
                        role="observation_visual",
                        round_name="observation_round1",
                        model=observation_model,
                        detection_id=detection_id,
                        attempted=True,
                        status=("success" if status_value == "ok" else "failure"),
                        token_usage=token_usage,
                        latency_ms=(invocation_payload.get("latency_ms") if isinstance(invocation_payload, dict) else None),
                        error_reason=(None if status_value == "ok" else error_reason),
                        usage_unavailable_reason=usage_unavailable_reason,
                    )
                )

        invocation_status = str(observation_result.get("invocation", {}).get("status", "unknown"))
        invocation_status_counts[invocation_status] = invocation_status_counts.get(invocation_status, 0) + 1

        verdict = observation_result.get("verdict")
        verification_status = None
        if isinstance(verdict, dict):
            status_value = verdict.get("verification_status")
            if isinstance(status_value, str):
                verification_status = status_value
                verification_status_counts[verification_status] = verification_status_counts.get(verification_status, 0) + 1

        observation_dimensions = _build_observation_verification_dimensions(
            observation_payload=observation_payload,
            deterministic_audit=deterministic_audit,
            observation_invocation=(observation_result.get("invocation") if isinstance(observation_result, dict) else None),
            observation_verdict=(verdict if isinstance(verdict, dict) else None),
        )
        for dimension_key in (
            "structured_motion_depth_consistency",
            "visual_identity_verifiability",
            "overall_observation_verification",
        ):
            dimension_payload = observation_dimensions.get(dimension_key)
            if not isinstance(dimension_payload, dict):
                continue
            status_value = dimension_payload.get("status")
            if not isinstance(status_value, str):
                continue
            counts = modality_dimension_status_counts[dimension_key]
            counts[status_value] = counts.get(status_value, 0) + 1

        expected_track_id = expected_correct_associations.get(detection_id)
        initial_proposed_track_id = (
            proposal_view.get("proposed_track_id")
            if isinstance(proposal_view.get("proposed_track_id"), str)
            else None
        )
        initial_expected_identity_status, initial_matches_expected_identity = _classify_expected_identity_status(
            expected_track_id=expected_track_id,
            proposed_track_id=initial_proposed_track_id,
        )
        observation_independently_supported_initial_association = verification_status == "supported"

        per_detection[detection_id] = {
            "tracking_proposal": proposal_view,
            "observation_request_payload": observation_payload,
            "observation_agent_verdict": verdict,
            "observation_agent_invocation": observation_result.get("invocation"),
            "deterministic_audit": deterministic_audit,
            "coordinator_decision": updates_by_detection.get(detection_id),
            "observation_verification_status": verification_status,
            "observation_verification_dimensions": observation_dimensions,
            "diagnostic_tracking_evaluation": {
                "expected_track_id": expected_track_id,
                "initial_proposed_track_id": initial_proposed_track_id,
                "initial_expected_identity_status": initial_expected_identity_status,
                "initial_matches_expected_identity": initial_matches_expected_identity,
                "final_proposed_track_id": initial_proposed_track_id,
                "final_expected_identity_status": initial_expected_identity_status,
                "final_matches_expected_identity": initial_matches_expected_identity,
                "reconsideration_proposed_track_id_changed": False,
                "incorrect_initial_corrected_after_feedback": False,
                "observation_independently_supported_initial_association": (
                    observation_independently_supported_initial_association
                ),
                "observation_independently_supported_final_association": (
                    observation_independently_supported_initial_association
                ),
                "deterministic_recovery_correction_proposed": False,
                "deterministic_recovery_contributed": False,
            },
        }

    conflicting_proposed_track_ids = _collect_conflicting_proposed_track_ids(per_detection)
    adjudication_counts: dict[str, int] = {
        value: 0 for value in DIAGNOSTIC_ADJUDICATION_CLASSIFICATIONS
    }
    for detection_id, payload in per_detection.items():
        adjudication = _classify_diagnostic_adjudication(
            detection_id=detection_id,
            per_detection_payload=payload,
            conflicting_proposed_track_ids=conflicting_proposed_track_ids,
        )
        classification = adjudication.get("classification")
        if isinstance(classification, str) and classification in adjudication_counts:
            adjudication_counts[classification] += 1
        payload["diagnostic_adjudication"] = adjudication
        payload["diagnostic_adjudication_final"] = dict(adjudication)
        payload["identity_origin"] = "initial_tracking"
        payload["tracking_reconsideration"] = {
            "considered": False,
            "revised": False,
            "reason_codes": ["not_eligible_for_reconsideration"],
            "live_state_mutation": False,
        }
        payload["evidence_provenance"] = {
            "yolo_bytetrack_measurements": "observation_request_payload.detection_measurements",
            "historical_track_evidence": "observation_request_payload.independent_history",
            "tracking_llm_proposal": "tracking_proposal",
            "observation_llm_verdict": "observation_agent_verdict",
            "deterministic_validation": "deterministic_audit",
            "coordinator_decision": "coordinator_decision",
            "diagnostic_adjudication": "diagnostic_adjudication",
            "tracking_reconsideration": "tracking_reconsideration",
            "observation_llm_verdict_round2": "tracking_reconsideration.observation_agent_verdict_round2",
            "deterministic_validation_round2": "tracking_reconsideration.deterministic_audit_round2",
            "coordinator_decision_round2": "tracking_reconsideration.coordinator_decision_round2",
            "diagnostic_adjudication_final": "diagnostic_adjudication_final",
            "diagnostic_alternative_identity_recovery": "diagnostic_alternative_identity_recovery",
        }

    eligible_detection_ids = sorted(
        detection_id
        for detection_id, payload in per_detection.items()
        if isinstance(payload, dict)
        and isinstance(payload.get("diagnostic_adjudication"), dict)
        and payload["diagnostic_adjudication"].get("classification") in {"rejected", "unresolved"}
    )

    reconsideration_payload = _build_tracking_reconsideration_payload(
        eligible_detection_ids=eligible_detection_ids,
        detections_by_id=detection_by_id,
        per_detection=per_detection,
        scenario_metadata=scenario_metadata,
        event_reasons=event_reasons,
    )

    reconsideration_result = await _invoke_tracking_reconsideration(
        mode=tracking_reconsideration_mode,
        enabled=len(eligible_detection_ids) > 0,
        tracking_model=(tracking_reconsideration_model or observation_model),
        user_payload=reconsideration_payload,
        frame_reference=frame_reference,
        timeout_seconds=(tracking_reconsideration_timeout_seconds or observation_timeout_seconds),
        max_output_tokens=tracking_reconsideration_max_output_tokens,
        reasoning_effort=tracking_reconsideration_reasoning_effort,
        invoker=reconsideration_invoker,
        provider_request_budget=provider_request_budget,
    )

    if isinstance(provider_call_telemetry, list):
        reconsideration_invocation = reconsideration_result.get("invocation") if isinstance(reconsideration_result, dict) else {}
        reconsideration_status = reconsideration_invocation.get("status") if isinstance(reconsideration_invocation, dict) else None
        reconsideration_error = reconsideration_invocation.get("error") if isinstance(reconsideration_invocation, dict) else None
        reconsideration_usage = reconsideration_invocation.get("token_usage") if isinstance(reconsideration_invocation, dict) and isinstance(reconsideration_invocation.get("token_usage"), dict) else None
        reconsideration_usage_unavailable_reason = reconsideration_invocation.get("usage_unavailable_reason") if isinstance(reconsideration_invocation, dict) and isinstance(reconsideration_invocation.get("usage_unavailable_reason"), str) else "usage_not_exposed_by_invoker"
        attempted = reconsideration_status in {"ok", "error"} and reconsideration_error != "provider_request_limit_exhausted"
        if attempted:
            provider_call_telemetry.append(
                _build_provider_call_telemetry_entry(
                    role="tracking_vlm",
                    round_name="tracking_reconsideration",
                    model=(tracking_reconsideration_model or observation_model),
                    detection_id=None,
                    attempted=True,
                    status=("success" if reconsideration_status == "ok" else "failure"),
                    token_usage=reconsideration_usage,
                    latency_ms=(reconsideration_invocation.get("latency_ms") if isinstance(reconsideration_invocation, dict) else None),
                    error_reason=(None if reconsideration_status == "ok" else reconsideration_error),
                    usage_unavailable_reason=reconsideration_usage_unavailable_reason,
                )
            )

    revisions_by_detection = reconsideration_result.get("revisions_by_detection")
    if not isinstance(revisions_by_detection, dict):
        revisions_by_detection = {}

    coordinator = AgentCoordinator()
    for detection_id in eligible_detection_ids:
        payload = per_detection.get(detection_id)
        detection = detection_by_id.get(detection_id)
        if not isinstance(payload, dict) or detection is None:
            continue

        reconsideration_block = {
            "considered": True,
            "invocation": reconsideration_result.get("invocation"),
            "declined": reconsideration_result.get("declined"),
            "decline_reason": reconsideration_result.get("decline_reason"),
            "live_state_mutation": False,
        }

        revised_payload = revisions_by_detection.get(detection_id)
        if not isinstance(revised_payload, dict):
            reason_codes = ["tracking_declined_or_no_revision_for_detection"]
            reconsideration_block["revised"] = False
            reconsideration_block["reason_codes"] = reason_codes
            payload["tracking_reconsideration"] = reconsideration_block
            payload["identity_origin"] = "initial_tracking"
            continue

        revised_proposal_view = _extract_tracking_proposal_view(revised_payload)
        round2_observation_payload = _build_observation_verification_payload(
            detection=detection,
            tracking_proposal=revised_payload,
            scenario_metadata=scenario_metadata,
            event_reasons=event_reasons,
        )
        round2_observation_result = await _invoke_observation_agent_verdict(
            mode=mode,
            observation_model=observation_model,
            observation_system_prompt=DEFAULT_OBSERVATION_VISUAL_SYSTEM_PROMPT,
            user_payload=round2_observation_payload,
            frame_reference=frame_reference,
            timeout_seconds=observation_timeout_seconds,
            max_output_tokens=observation_max_output_tokens,
            reasoning_effort=observation_reasoning_effort,
            invoker=invoker,
            provider_request_budget=provider_request_budget,
        )

        if isinstance(provider_call_telemetry, list):
            round2_invocation_payload = round2_observation_result.get("invocation") if isinstance(round2_observation_result, dict) else {}
            round2_status = round2_invocation_payload.get("status") if isinstance(round2_invocation_payload, dict) else None
            round2_error = round2_invocation_payload.get("error") if isinstance(round2_invocation_payload, dict) else None
            round2_usage = round2_invocation_payload.get("token_usage") if isinstance(round2_invocation_payload, dict) and isinstance(round2_invocation_payload.get("token_usage"), dict) else None
            round2_usage_unavailable_reason = round2_invocation_payload.get("usage_unavailable_reason") if isinstance(round2_invocation_payload, dict) and isinstance(round2_invocation_payload.get("usage_unavailable_reason"), str) else "usage_not_exposed_by_invoker"
            attempted = round2_status in {"ok", "error"} and round2_error != "provider_request_limit_exhausted"
            if attempted:
                provider_call_telemetry.append(
                    _build_provider_call_telemetry_entry(
                        role="observation_visual",
                        round_name="observation_round2",
                        model=observation_model,
                        detection_id=detection_id,
                        attempted=True,
                        status=("success" if round2_status == "ok" else "failure"),
                        token_usage=round2_usage,
                        latency_ms=(round2_invocation_payload.get("latency_ms") if isinstance(round2_invocation_payload, dict) else None),
                        error_reason=(None if round2_status == "ok" else round2_error),
                        usage_unavailable_reason=round2_usage_unavailable_reason,
                    )
                )

        round2_invocation_status = str(round2_observation_result.get("invocation", {}).get("status", "unknown"))
        reconsideration_invocation_status_counts[round2_invocation_status] = (
            reconsideration_invocation_status_counts.get(round2_invocation_status, 0) + 1
        )

        round2_verification_status = None
        round2_verdict = round2_observation_result.get("verdict")
        if isinstance(round2_verdict, dict):
            status_value = round2_verdict.get("verification_status")
            if isinstance(status_value, str):
                round2_verification_status = status_value
                reconsideration_verification_status_counts[round2_verification_status] = (
                    reconsideration_verification_status_counts.get(round2_verification_status, 0) + 1
                )

        round2_deterministic_audit = _run_deterministic_observation_audit(
            detection=detection,
            tracking_proposal=revised_payload,
            scenario_metadata=scenario_metadata,
        )

        round2_proposal = _build_visual_association_proposal_from_payload(
            detection_id=detection_id,
            payload=revised_payload,
        )
        round2_coordinator_decision = coordinator.validate_visual_association_proposal(detection, round2_proposal).to_dict()

        reconsideration_block["revised"] = True
        reconsideration_block["reason_codes"] = ["tracking_revised_identity_proposal"]
        reconsideration_block["revised_tracking_proposal"] = revised_proposal_view
        reconsideration_block["observation_request_payload_round2"] = round2_observation_payload
        reconsideration_block["observation_agent_verdict_round2"] = round2_verdict
        reconsideration_block["observation_agent_invocation_round2"] = round2_observation_result.get("invocation")
        reconsideration_block["deterministic_audit_round2"] = round2_deterministic_audit
        reconsideration_block["coordinator_decision_round2"] = round2_coordinator_decision
        payload["tracking_reconsideration"] = reconsideration_block
        payload["identity_origin"] = "revised_tracking"

        final_adjudication_input = {
            "tracking_proposal": revised_proposal_view,
            "observation_request_payload": round2_observation_payload,
            "observation_agent_verdict": round2_verdict,
            "observation_agent_invocation": round2_observation_result.get("invocation"),
            "deterministic_audit": round2_deterministic_audit,
            "coordinator_decision": round2_coordinator_decision,
            "observation_verification_status": round2_verification_status,
        }
        round2_dimensions = _build_observation_verification_dimensions(
            observation_payload=round2_observation_payload,
            deterministic_audit=round2_deterministic_audit,
            observation_invocation=(
                round2_observation_result.get("invocation")
                if isinstance(round2_observation_result, dict)
                else None
            ),
            observation_verdict=(round2_verdict if isinstance(round2_verdict, dict) else None),
        )
        payload["observation_verification_dimensions_round2"] = round2_dimensions
        for dimension_key in (
            "structured_motion_depth_consistency",
            "visual_identity_verifiability",
            "overall_observation_verification",
        ):
            dimension_payload = round2_dimensions.get(dimension_key)
            if not isinstance(dimension_payload, dict):
                continue
            status_value = dimension_payload.get("status")
            if not isinstance(status_value, str):
                continue
            counts = modality_dimension_status_counts[dimension_key]
            counts[status_value] = counts.get(status_value, 0) + 1

        payload["diagnostic_adjudication_final"] = _classify_diagnostic_adjudication(
            detection_id=detection_id,
            per_detection_payload=final_adjudication_input,
            conflicting_proposed_track_ids=set(),
        )

        tracking_evaluation = payload.get("diagnostic_tracking_evaluation")
        if isinstance(tracking_evaluation, dict):
            initial_proposed_track_id = (
                tracking_evaluation.get("initial_proposed_track_id")
                if isinstance(tracking_evaluation.get("initial_proposed_track_id"), str)
                else None
            )
            revised_proposed_track_id = (
                revised_proposal_view.get("proposed_track_id")
                if isinstance(revised_proposal_view.get("proposed_track_id"), str)
                else None
            )
            track_id_changed = revised_proposed_track_id != initial_proposed_track_id
            expected_track_id = (
                tracking_evaluation.get("expected_track_id")
                if isinstance(tracking_evaluation.get("expected_track_id"), str)
                else None
            )
            final_expected_identity_status, final_matches_expected_identity = _classify_expected_identity_status(
                expected_track_id=expected_track_id,
                proposed_track_id=revised_proposed_track_id,
            )
            initial_matches_expected_identity = tracking_evaluation.get("initial_matches_expected_identity")
            corrected_after_feedback = (
                initial_matches_expected_identity is False
                and final_matches_expected_identity is True
                and track_id_changed
            )
            tracking_evaluation["final_proposed_track_id"] = revised_proposed_track_id
            tracking_evaluation["final_expected_identity_status"] = final_expected_identity_status
            tracking_evaluation["final_matches_expected_identity"] = final_matches_expected_identity
            tracking_evaluation["reconsideration_proposed_track_id_changed"] = track_id_changed
            tracking_evaluation["incorrect_initial_corrected_after_feedback"] = corrected_after_feedback
            tracking_evaluation["observation_independently_supported_final_association"] = (
                round2_verification_status == "supported"
            )
            payload["diagnostic_tracking_evaluation"] = tracking_evaluation

    final_track_assignment_by_detection: dict[str, str | None] = {}
    for detection_id, payload in per_detection.items():
        if not isinstance(payload, dict):
            continue
        revised_block = payload.get("tracking_reconsideration")
        if isinstance(revised_block, dict) and revised_block.get("revised") is True:
            revised_proposal = revised_block.get("revised_tracking_proposal")
            if isinstance(revised_proposal, dict):
                revised_track_id = revised_proposal.get("proposed_track_id")
                final_track_assignment_by_detection[detection_id] = (
                    revised_track_id if isinstance(revised_track_id, str) else None
                )
                continue
        tracking_proposal = payload.get("tracking_proposal")
        if isinstance(tracking_proposal, dict):
            track_id = tracking_proposal.get("proposed_track_id")
            final_track_assignment_by_detection[detection_id] = (track_id if isinstance(track_id, str) else None)
        else:
            final_track_assignment_by_detection[detection_id] = None

    duplicate_assignment_map = _collect_duplicate_assigned_track_ids(final_track_assignment_by_detection)
    duplicate_detection_ids = {
        detection_id
        for detection_ids in duplicate_assignment_map.values()
        for detection_id in detection_ids
    }
    for detection_id in duplicate_detection_ids:
        payload = per_detection.get(detection_id)
        if not isinstance(payload, dict):
            continue
        final_adjudication = payload.get("diagnostic_adjudication_final")
        if not isinstance(final_adjudication, dict):
            continue
        reason_codes = final_adjudication.get("reason_codes")
        if not isinstance(reason_codes, list):
            reason_codes = []
        if "duplicate_track_identity_assignment_global" not in reason_codes:
            reason_codes.append("duplicate_track_identity_assignment_global")
        final_adjudication["reason_codes"] = reason_codes
        final_adjudication["classification"] = "unresolved"
        payload["diagnostic_adjudication_final"] = final_adjudication

    final_adjudication_counts: dict[str, int] = {value: 0 for value in DIAGNOSTIC_ADJUDICATION_CLASSIFICATIONS}
    for payload in per_detection.values():
        if not isinstance(payload, dict):
            continue
        final_adjudication = payload.get("diagnostic_adjudication_final")
        if not isinstance(final_adjudication, dict):
            continue
        classification = final_adjudication.get("classification")
        if isinstance(classification, str) and classification in final_adjudication_counts:
            final_adjudication_counts[classification] += 1

    recovery_summary = _run_diagnostic_alternative_identity_recovery(
        detections_by_id=detection_by_id,
        per_detection=per_detection,
        scenario_metadata=scenario_metadata,
    )

    tracking_evaluation_summary = {
        "initial_expected_identity_status_counts": {},
        "final_expected_identity_status_counts": {},
        "reconsideration_track_id_changed_count": 0,
        "incorrect_initial_corrected_after_feedback_count": 0,
        "observation_supported_final_association_count": 0,
        "deterministic_recovery_correction_proposed_count": 0,
        "deterministic_recovery_contributed_count": 0,
    }

    for payload in per_detection.values():
        if not isinstance(payload, dict):
            continue
        tracking_evaluation = payload.get("diagnostic_tracking_evaluation")
        if not isinstance(tracking_evaluation, dict):
            continue

        recovery_payload = payload.get("diagnostic_alternative_identity_recovery")
        recovery_correction_proposed = (
            isinstance(recovery_payload, dict)
            and isinstance(recovery_payload.get("proposed_correction"), dict)
        )
        tracking_evaluation["deterministic_recovery_correction_proposed"] = recovery_correction_proposed
        tracking_evaluation["deterministic_recovery_contributed"] = False

        initial_status = tracking_evaluation.get("initial_expected_identity_status")
        if isinstance(initial_status, str):
            counts = tracking_evaluation_summary["initial_expected_identity_status_counts"]
            counts[initial_status] = counts.get(initial_status, 0) + 1

        final_status = tracking_evaluation.get("final_expected_identity_status")
        if isinstance(final_status, str):
            counts = tracking_evaluation_summary["final_expected_identity_status_counts"]
            counts[final_status] = counts.get(final_status, 0) + 1

        if tracking_evaluation.get("reconsideration_proposed_track_id_changed") is True:
            tracking_evaluation_summary["reconsideration_track_id_changed_count"] += 1
        if tracking_evaluation.get("incorrect_initial_corrected_after_feedback") is True:
            tracking_evaluation_summary["incorrect_initial_corrected_after_feedback_count"] += 1
        if tracking_evaluation.get("observation_independently_supported_final_association") is True:
            tracking_evaluation_summary["observation_supported_final_association_count"] += 1
        if recovery_correction_proposed:
            tracking_evaluation_summary["deterministic_recovery_correction_proposed_count"] += 1
        if tracking_evaluation.get("deterministic_recovery_contributed") is True:
            tracking_evaluation_summary["deterministic_recovery_contributed_count"] += 1
        payload["diagnostic_tracking_evaluation"] = tracking_evaluation

    return {
        "enabled": mode != OBSERVATION_VERIFICATION_MODE_OFF,
        "mode": mode,
        "observation_model": observation_model,
        "observation_timeout_seconds": observation_timeout_seconds,
        "observation_max_output_tokens": observation_max_output_tokens,
        "observation_reasoning_effort": observation_reasoning_effort,
        "per_detection": per_detection,
        "summary": {
            "proposal_count": len(proposal_by_detection),
            "observation_invocation_status_counts": invocation_status_counts,
            "observation_verification_status_counts": verification_status_counts,
            "reconsideration_observation_invocation_status_counts": reconsideration_invocation_status_counts,
            "reconsideration_observation_verification_status_counts": reconsideration_verification_status_counts,
            "depth_conflict_vs_depth_consistency_contradictions": contradiction_count,
            "diagnostic_adjudication_counts": adjudication_counts,
            "diagnostic_final_adjudication_counts": final_adjudication_counts,
            "conflicting_proposed_track_ids": sorted(conflicting_proposed_track_ids),
            "conflicting_final_assigned_track_ids": sorted(duplicate_assignment_map.keys()),
            "diagnostic_provisional_min_confidence": DIAGNOSTIC_ADJUDICATION_PROVISIONAL_MIN_CONFIDENCE,
            "tracking_reconsideration": {
                "mode": tracking_reconsideration_mode,
                "eligible_detection_count": len(eligible_detection_ids),
                "eligible_detection_ids": eligible_detection_ids,
                "invocation": reconsideration_result.get("invocation"),
                "declined": bool(reconsideration_result.get("declined", False)),
                "decline_reason": reconsideration_result.get("decline_reason"),
                "revised_detection_count": len(
                    [
                        detection_id
                        for detection_id, payload in per_detection.items()
                        if isinstance(payload, dict)
                        and isinstance(payload.get("tracking_reconsideration"), dict)
                        and payload["tracking_reconsideration"].get("revised") is True
                    ]
                ),
                "max_tracking_requests_per_scenario": 1 + (1 if len(eligible_detection_ids) > 0 else 0),
                "max_observation_requests_per_scenario": len(proposal_by_detection) + len(eligible_detection_ids),
                "max_total_provider_requests_per_scenario": (
                    (1 + (1 if len(eligible_detection_ids) > 0 else 0))
                    + (len(proposal_by_detection) + len(eligible_detection_ids))
                ),
            },
            "observation_verification_dimensions_status_counts": modality_dimension_status_counts,
            "diagnostic_tracking_evaluation": tracking_evaluation_summary,
            "diagnostic_alternative_identity_recovery": recovery_summary,
            "provider_request_budget": (
                {
                    "max_total": provider_request_budget.get("max_total"),
                    "consumed": provider_request_budget.get("consumed"),
                    "remaining": provider_request_budget.get("remaining"),
                }
                if isinstance(provider_request_budget, dict)
                else None
            ),
        },
    }


def _compute_expected_provider_request_upper_bound(
    *,
    detection_count: int,
    observation_verification_mode: str,
    tracking_reconsideration_mode: str,
) -> dict[str, int]:
    tracking_requests = 1
    if tracking_reconsideration_mode == TRACKING_RECONSIDERATION_MODE_LIVE:
        tracking_requests += 1

    observation_requests = 0
    if observation_verification_mode == OBSERVATION_VERIFICATION_MODE_LIVE:
        observation_requests += detection_count
        if tracking_reconsideration_mode == TRACKING_RECONSIDERATION_MODE_LIVE:
            observation_requests += detection_count

    total = tracking_requests + observation_requests
    return {
        "tracking_requests": tracking_requests,
        "observation_requests": observation_requests,
        "total_requests": total,
    }


class _CapturingCompletions:
    def __init__(
        self,
        inner: Any,
        capture: ResponseCapture,
        compress_user_instruction: bool = False,
        reasoning_effort_override: str | None = None,
        independent_track_history_override: list[dict[str, Any]] | None = None,
    ) -> None:
        self._inner = inner
        self._capture = capture
        self._compress_user_instruction = compress_user_instruction
        self._reasoning_effort_override = reasoning_effort_override
        self._independent_track_history_override = independent_track_history_override

    def _maybe_compress_user_instruction(self, kwargs: dict[str, Any]) -> None:
        if not self._compress_user_instruction:
            return

        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return

        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if not isinstance(content, list):
                continue

            for entry in content:
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") != "text":
                    continue

                text = entry.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue

                compressed = _build_compressed_instruction_from_original(
                    text,
                    independent_track_history_override=self._independent_track_history_override,
                )
                self._capture.original_user_instruction = text
                self._capture.compressed_user_instruction = compressed
                self._capture.compression_applied = compressed != text
                entry["text"] = compressed
                return

    def _maybe_apply_reasoning_effort_override(self, kwargs: dict[str, Any]) -> None:
        if not isinstance(self._reasoning_effort_override, str):
            return
        requested = self._reasoning_effort_override.strip()
        if not requested:
            return
        kwargs["reasoning_effort"] = requested

    def _prepare_request_kwargs(self, kwargs: dict[str, Any]) -> None:
        self._maybe_compress_user_instruction(kwargs)
        self._maybe_apply_reasoning_effort_override(kwargs)

    async def create(self, **kwargs):
        self._prepare_request_kwargs(kwargs)
        self._capture.reasoning_effort_requested = self._reasoning_effort_override
        applied_effort = kwargs.get("reasoning_effort")
        self._capture.reasoning_effort_applied_in_request = applied_effort if isinstance(applied_effort, str) else None
        response = await self._inner.create(**kwargs)

        if hasattr(response, "model_dump"):
            try:
                self._capture.response_object_keys = sorted(list(response.model_dump().keys()))
            except Exception:
                self._capture.response_object_keys = None

        usage_payload: dict[str, int] = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, field, None)
                if isinstance(value, int):
                    usage_payload[field] = value
        self._capture.usage = usage_payload or None

        usage_details_payload: dict[str, Any] = {}
        if usage is not None:
            for field in ("completion_tokens_details", "prompt_tokens_details"):
                value = getattr(usage, field, None)
                if value is None:
                    continue
                if hasattr(value, "model_dump"):
                    try:
                        usage_details_payload[field] = value.model_dump()
                        continue
                    except Exception:
                        pass
                if isinstance(value, dict):
                    usage_details_payload[field] = value
        self._capture.usage_details = usage_details_payload or None

        if getattr(response, "choices", None):
            first_choice = response.choices[0]
            self._capture.finish_reason = getattr(first_choice, "finish_reason", None)
            message = getattr(first_choice, "message", None)
            if hasattr(message, "model_dump"):
                try:
                    self._capture.message_object_keys = sorted(list(message.model_dump().keys()))
                except Exception:
                    self._capture.message_object_keys = None

            content_text = getattr(message, "content", None)
            self._capture.content_text = content_text
            if isinstance(content_text, str):
                self._capture.content_text_char_length = len(content_text)
                self._capture.content_text_sha256 = hashlib.sha256(content_text.encode("utf-8")).hexdigest()
                self._capture.content_text_head = content_text[:300]
                self._capture.content_text_tail = content_text[-300:]
        return response


class _CapturingClient:
    def __init__(
        self,
        inner_client: AsyncOpenAI,
        capture: ResponseCapture,
        compress_user_instruction: bool = False,
        reasoning_effort_override: str | None = None,
        independent_track_history_override: list[dict[str, Any]] | None = None,
    ) -> None:
        self.chat = type("ChatWrapper", (), {})()
        self.chat.completions = _CapturingCompletions(
            inner_client.chat.completions,
            capture,
            compress_user_instruction=compress_user_instruction,
            reasoning_effort_override=reasoning_effort_override,
            independent_track_history_override=independent_track_history_override,
        )


def _build_synthetic_frame() -> Any:
    frame = np.full((180, 320, 3), 240, dtype=np.uint8)
    cv2.rectangle(frame, (28, 44), (128, 158), (0, 0, 255), thickness=-1)
    cv2.rectangle(frame, (118, 34), (218, 148), (0, 255, 0), thickness=-1)
    cv2.circle(frame, (252, 92), 28, (255, 0, 0), thickness=-1)
    return frame


def _build_detection_scenario(now: float, scenario: str) -> tuple[list[DetectionCandidate], dict[str, str], dict[str, Any]]:
    if scenario not in SCENARIO_CHOICES:
        raise ValueError(f"unsupported scenario: {scenario}")

    detection_payloads = [
        {
            "detection_id": "det-amb-1",
            "timestamp": now,
            "yolo_track_id": "track-A",
            "label": "person",
            "detection_confidence": 0.81,
            "pixel_bbox": BoundingBox(x1=42.0, y1=44.0, x2=136.0, y2=162.0),
            "pixel_center_x": 89.0,
            "pixel_center_y": 162.0,
            "relative_depth": 0.52,
            "ground_x": 1.4,
            "ground_z": 3.8,
            "ground_coordinate_unit": "relative_units",
            "position_kind": "relative_ground_estimate",
            "motion_history": [
                {"history_index": 0, "ground_x": 0.7, "ground_z": 4.5},
                {"history_index": 1, "ground_x": 0.95, "ground_z": 4.2},
                {"history_index": 2, "ground_x": 1.2, "ground_z": 4.0},
            ],
        },
        {
            "detection_id": "det-amb-2",
            "timestamp": now,
            "yolo_track_id": "track-B",
            "label": "person",
            "detection_confidence": 0.79,
            "pixel_bbox": BoundingBox(x1=114.0, y1=40.0, x2=214.0, y2=156.0),
            "pixel_center_x": 164.0,
            "pixel_center_y": 156.0,
            "relative_depth": 0.49,
            "ground_x": 1.7,
            "ground_z": 3.9,
            "ground_coordinate_unit": "relative_units",
            "position_kind": "relative_ground_estimate",
            "motion_history": [
                {"history_index": 0, "ground_x": 2.2, "ground_z": 4.6},
                {"history_index": 1, "ground_x": 2.0, "ground_z": 4.3},
                {"history_index": 2, "ground_x": 1.85, "ground_z": 4.1},
            ],
        },
    ]

    expected_correct_associations: dict[str, str] = {
        "det-amb-1": "track-A",
        "det-amb-2": "track-B",
    }

    independent_track_history: list[dict[str, Any]] = []

    if scenario == SCENARIO_YOLO_ID_SWAP:
        detection_payloads[0]["yolo_track_id"] = "track-B"
        detection_payloads[1]["yolo_track_id"] = "track-A"
        independent_track_history = _build_independent_track_history_evidence(now)

    detections: list[DetectionCandidate] = []
    for item in detection_payloads:
        detections.append(
            DetectionCandidate(
                detection_id=str(item["detection_id"]),
                timestamp=float(item["timestamp"]),
                yolo_track_id=str(item["yolo_track_id"]),
                label=str(item["label"]),
                detection_confidence=float(item["detection_confidence"]),
                pixel_bbox=item["pixel_bbox"],
                pixel_center_x=float(item["pixel_center_x"]),
                pixel_center_y=float(item["pixel_center_y"]),
                relative_depth=float(item["relative_depth"]),
                ground_x=float(item["ground_x"]),
                ground_z=float(item["ground_z"]),
                ground_coordinate_unit=str(item["ground_coordinate_unit"]),
                position_kind=str(item["position_kind"]),
                motion_history=list(item["motion_history"]),
            )
        )

    detection_payload_for_audit = [item.to_dict() for item in detections]
    payload_text = json.dumps(detection_payload_for_audit)
    forbidden_tokens = ["expected_identity", "ground_truth", "canonical_track_id", "expected_track"]
    leaked_tokens = [token for token in forbidden_tokens if token in payload_text]
    independent_history_audit = _build_independent_history_consistency_audit(
        independent_track_history,
        [item.detection_id for item in detections],
    )

    scenario_metadata = {
        "scenario": scenario,
        "input_yolo_track_assignments": {
            item.detection_id: item.yolo_track_id
            for item in detections
        },
        "expected_correct_associations": dict(expected_correct_associations),
        INDEPENDENT_TRACK_HISTORY_KEY: independent_track_history,
        "independent_history_consistency_audit": independent_history_audit,
        "leakage_audit": {
            "forbidden_tokens_checked": forbidden_tokens,
            "leaked_tokens_found": leaked_tokens,
            "leakage_detected": len(leaked_tokens) > 0,
        },
    }

    return detections, expected_correct_associations, scenario_metadata


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 isolated Tracking Agent validator.")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--model", type=str, default=TRACKING_VLM_MODEL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--events-jsonl", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--max-total-requests", type=int, default=1)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Diagnostic-only reasoning_effort forwarded to the Tracking Agent chat completion request.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIO_CHOICES),
        default=SCENARIO_BASELINE,
        help="Diagnostic scenario selector. 'yolo-id-swap' swaps YOLO track IDs while preserving motion/depth/image evidence.",
    )
    parser.add_argument(
        "--observation-verification-mode",
        type=str,
        choices=list(OBSERVATION_VERIFICATION_MODE_CHOICES),
        default=OBSERVATION_VERIFICATION_MODE_OFF,
        help="Diagnostic-only Observation Agent verification mode: off, mock (no network), or live.",
    )
    parser.add_argument(
        "--observation-model",
        type=str,
        default=DEFAULT_OBSERVATION_VISUAL_MODEL,
        help="Observation visual verification model (used when observation verification mode is mock/live).",
    )
    parser.add_argument(
        "--observation-timeout-seconds",
        type=float,
        default=DEFAULT_OBSERVATION_VISUAL_TIMEOUT_SECONDS,
        help="Observation visual verification timeout in seconds for diagnostic adapter calls.",
    )
    parser.add_argument(
        "--observation-max-output-tokens",
        type=int,
        default=DEFAULT_OBSERVATION_VISUAL_MAX_OUTPUT_TOKENS,
        help="Diagnostic-only Observation visual max output tokens override.",
    )
    parser.add_argument(
        "--observation-reasoning-effort",
        type=str,
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=DEFAULT_OBSERVATION_VISUAL_REASONING_EFFORT,
        help="Diagnostic-only Observation visual reasoning_effort override.",
    )
    parser.add_argument(
        "--tracking-reconsideration-mode",
        type=str,
        choices=list(TRACKING_RECONSIDERATION_MODE_CHOICES),
        default=TRACKING_RECONSIDERATION_MODE_OFF,
        help="Diagnostic-only Tracking reconsideration mode: off (default) or live (explicit opt-in).",
    )
    parser.add_argument("--dry-run-no-api", action="store_true")
    return parser.parse_args()


async def _run_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be greater than zero")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be greater than zero")
    if args.max_total_requests < 1:
        raise ValueError("--max-total-requests must be >= 1")
    if args.observation_timeout_seconds <= 0:
        raise ValueError("--observation-timeout-seconds must be greater than zero")
    if args.observation_max_output_tokens <= 0:
        raise ValueError("--observation-max-output-tokens must be greater than zero")

    request_count = 1
    if request_count > args.max_total_requests:
        raise ValueError(
            f"Requested {request_count} request exceeds --max-total-requests={args.max_total_requests}"
        )

    output_path = args.output_json.expanduser().resolve()
    events_path = args.events_jsonl.expanduser().resolve()
    if output_path == events_path:
        raise ValueError("--output-json and --events-jsonl must be different")
    if output_path == PRODUCTION_EVENTS_PATH.resolve() or events_path == PRODUCTION_EVENTS_PATH.resolve():
        raise ValueError("paths must not target production latency events path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    os.environ["PHASE16E_LATENCY_INSTRUMENTATION_ENABLED"] = "1"
    os.environ["PHASE16E_LATENCY_EVENTS_PATH"] = str(events_path)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)

    frame = _build_synthetic_frame()
    frame_reference = encode_frame_as_data_url(frame)
    now = time.time()
    detections, expected_correct_associations, scenario_metadata = _build_detection_scenario(now, args.scenario)
    expected_provider_requests = _compute_expected_provider_request_upper_bound(
        detection_count=len(detections),
        observation_verification_mode=args.observation_verification_mode,
        tracking_reconsideration_mode=args.tracking_reconsideration_mode,
    )
    if expected_provider_requests["total_requests"] > args.max_total_requests:
        raise ValueError(
            "Configured diagnostic request upper-bound exceeds --max-total-requests; "
            f"expected={expected_provider_requests['total_requests']} max={args.max_total_requests}"
        )
    persisted_budget_preflight = _compute_persisted_budget_preflight(
        detection_count=len(detections),
        observation_verification_mode=args.observation_verification_mode,
        tracking_reconsideration_mode=args.tracking_reconsideration_mode,
    )
    if not persisted_budget_preflight.get("sufficient", False):
        shortfall = persisted_budget_preflight.get("shortfall", {})
        raise ValueError(
            "Persisted provider budget insufficient for diagnostic worst-case request plan; "
            f"shortfall_tracking_vlm={shortfall.get('tracking_vlm')} "
            f"shortfall_observation_visual={shortfall.get('observation_visual')} "
            f"shortfall_total={shortfall.get('total')}"
        )
    event_reasons = ["possible_id_switch", "overlapping_tracks", "occlusion_risk"]

    capture = ResponseCapture()
    raw_client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    independent_track_history_override = scenario_metadata.get(INDEPENDENT_TRACK_HISTORY_KEY)
    client = _CapturingClient(
        raw_client,
        capture,
        compress_user_instruction=True,
        reasoning_effort_override=args.reasoning_effort,
        independent_track_history_override=(
            independent_track_history_override
            if isinstance(independent_track_history_override, list) and len(independent_track_history_override) > 0
            else None
        ),
    )
    agent = LLMVisualTrackingAgent(model=args.model, openai_base_url=base_url, client=client)

    old_max_tokens = os.environ.get("TRACKING_AGENT_MAX_OUTPUT_TOKENS")
    os.environ["TRACKING_AGENT_MAX_OUTPUT_TOKENS"] = str(int(args.max_output_tokens))

    started_at = time.time()
    started_perf = time.perf_counter()

    _append_event(
        events_path,
        {
            "event_type": "phase1_validation_started",
            "timestamp": started_at,
            "model": args.model,
            "scenario": args.scenario,
            "timeout_seconds": args.timeout_seconds,
            "max_output_tokens_override": int(args.max_output_tokens),
            "reasoning_effort": args.reasoning_effort,
            "request_count": request_count,
            "retry_limit": 0,
        },
    )

    try:
        result = await agent.propose_associations(
            frame_reference=frame_reference,
            detections=detections,
            event_reasons=event_reasons,
            request_timeout_seconds=float(args.timeout_seconds),
        )
    finally:
        if old_max_tokens is None:
            os.environ.pop("TRACKING_AGENT_MAX_OUTPUT_TOKENS", None)
        else:
            os.environ["TRACKING_AGENT_MAX_OUTPUT_TOKENS"] = old_max_tokens

    wall_latency_ms = round((time.perf_counter() - started_perf) * 1000.0, 3)
    ended_at = time.time()

    provider_request_budget = {
        "max_total": int(args.max_total_requests),
        "consumed": 1,
        "remaining": int(args.max_total_requests) - 1,
    }
    provider_call_telemetry: list[dict[str, Any]] = []
    provider_call_telemetry.append(
        _build_provider_call_telemetry_entry(
            role="tracking_vlm",
            round_name="tracking_initial",
            model=args.model,
            detection_id=None,
            attempted=True,
            status=("success" if result.available else "failure"),
            token_usage=(capture.usage if isinstance(capture.usage, dict) else None),
            latency_ms=(result.latency_ms if isinstance(result.latency_ms, (int, float)) else wall_latency_ms),
            error_reason=(None if result.available else (result.error or "tracking_initial_failed")),
            usage_unavailable_reason=(None if isinstance(capture.usage, dict) else "tracking_usage_not_returned"),
        )
    )

    schema_complete = "schema_complete_response" in result.uncertainty_flags
    schema_incomplete = "schema_incomplete_response" in result.uncertainty_flags
    parsed_response_payload = _extract_json_object(capture.content_text or "")
    valid_json = parsed_response_payload is not None
    timeout_flag = result.error == "vlm_timeout" or "vlm_timeout" in result.uncertainty_flags
    truncation_flag = result.error == "vlm_output_truncated" or "vlm_output_truncated" in result.uncertainty_flags
    fallback_or_error_classification = result.error or (
        "success"
        if result.available
        else "unavailable"
    )

    coordinator = AgentCoordinator()
    observation = coordinator.build_tracking_agent_observation(
        detections=detections,
        proposals=result.proposals,
        timestamp=now,
        vlm_model=args.model,
        frame_reference=frame_reference,
        proposal_latency_ms=result.latency_ms,
        additional_uncertainties=result.uncertainty_flags,
    )
    observation_dict = observation.to_dict()
    updates = observation_dict.get("updates", []) if isinstance(observation_dict, dict) else []
    fallback_updates = 0
    evidence_backed_updates = 0
    accepted_updates = 0
    rejected_updates = 0
    rejection_reasons_by_detection: dict[str, list[str]] = {}
    for item in updates if isinstance(updates, list) else []:
        if not isinstance(item, dict):
            continue
        accepted = item.get("accepted")
        if accepted is True:
            accepted_updates += 1
        elif accepted is False:
            rejected_updates += 1
            detection_id = item.get("detection_id")
            if isinstance(detection_id, str):
                rejection_reasons = item.get("rejection_reasons")
                if isinstance(rejection_reasons, list):
                    rejection_reasons_by_detection[detection_id] = [str(reason) for reason in rejection_reasons]
        uncertainties = item.get("uncertainties")
        supporting_evidence = item.get("supporting_evidence")
        if isinstance(uncertainties, list) and "no_vlm_proposal_used" in uncertainties:
            fallback_updates += 1
        if isinstance(supporting_evidence, list) and len(supporting_evidence) > 0:
            evidence_backed_updates += 1

    proposal_by_detection: dict[str, dict[str, Any]] = {}
    parsed_response_for_eval = _extract_json_object(capture.content_text or "")
    if isinstance(parsed_response_for_eval, dict):
        raw_proposals_for_eval = parsed_response_for_eval.get("proposals")
        if isinstance(raw_proposals_for_eval, list):
            for item in raw_proposals_for_eval:
                if not isinstance(item, dict):
                    continue
                detection_id = item.get("detection_id")
                if isinstance(detection_id, str):
                    proposal_by_detection[detection_id] = item

    per_detection_evaluation: dict[str, Any] = {}
    corrected_count = 0
    preserved_swapped_count = 0
    uncertain_count = 0
    for detection in detections:
        detection_id = detection.detection_id
        expected_track_id = expected_correct_associations.get(detection_id)
        input_yolo_track_id = detection.yolo_track_id
        raw_proposal = proposal_by_detection.get(detection_id)
        proposed_track_id = raw_proposal.get("proposed_track_id") if isinstance(raw_proposal, dict) else None
        uncertainties = raw_proposal.get("uncertainties") if isinstance(raw_proposal, dict) else []
        if not isinstance(uncertainties, list):
            uncertainties = []

        if isinstance(proposed_track_id, str) and isinstance(expected_track_id, str) and proposed_track_id == expected_track_id:
            outcome = "matched_expected_identity"
            corrected_count += 1
        elif (
            args.scenario == SCENARIO_YOLO_ID_SWAP
            and isinstance(proposed_track_id, str)
            and isinstance(input_yolo_track_id, str)
            and proposed_track_id == input_yolo_track_id
        ):
            outcome = "preserved_swapped_yolo_identity"
            preserved_swapped_count += 1
        elif "possible_id_switch" in [str(item) for item in uncertainties] or "crossing_targets" in [str(item) for item in uncertainties]:
            outcome = "uncertain"
            uncertain_count += 1
        else:
            outcome = "unresolved"

        per_detection_evaluation[detection_id] = {
            "input_yolo_track_id": input_yolo_track_id,
            "expected_track_id": expected_track_id,
            "proposed_track_id": proposed_track_id,
            "outcome": outcome,
            "uncertainties": [str(item) for item in uncertainties],
        }

    if corrected_count == len(expected_correct_associations) and corrected_count > 0:
        scenario_outcome = "corrected_swap"
    elif preserved_swapped_count == len(expected_correct_associations) and preserved_swapped_count > 0:
        scenario_outcome = "preserved_swapped_ids"
    elif uncertain_count > 0:
        scenario_outcome = "reported_uncertainty"
    else:
        scenario_outcome = "mixed_or_unresolved"

    compressed_payload, compressed_payload_error = _extract_structured_payload_from_instruction(
        capture.compressed_user_instruction or ""
    )
    independent_history_from_prompt = []
    if isinstance(compressed_payload, dict):
        candidate = compressed_payload.get(INDEPENDENT_TRACK_HISTORY_KEY)
        if isinstance(candidate, list):
            independent_history_from_prompt = candidate
    prompt_independent_history_audit = _build_independent_history_consistency_audit(
        independent_history_from_prompt,
        [item.detection_id for item in detections],
    )
    expected_mapping_in_prompt = False
    if isinstance(compressed_payload, dict):
        expected_mapping_in_prompt = "expected_correct_associations" in compressed_payload
    structured_input_audit = {
        "structured_payload_parsed": isinstance(compressed_payload, dict),
        "structured_payload_parse_error": compressed_payload_error,
        "includes_expected_correct_associations_key": expected_mapping_in_prompt,
        "includes_independent_track_history_evidence": len(independent_history_from_prompt) > 0,
        "independent_history_consistency_audit": prompt_independent_history_audit,
    }

    observation_collaboration = await _run_observation_collaboration_diagnostic(
        detections=detections,
        proposal_by_detection=proposal_by_detection,
        updates=(updates if isinstance(updates, list) else []),
        scenario_metadata=scenario_metadata,
        event_reasons=event_reasons,
        frame_reference=frame_reference,
        mode=args.observation_verification_mode,
        observation_model=args.observation_model,
        observation_timeout_seconds=float(args.observation_timeout_seconds),
        observation_max_output_tokens=int(args.observation_max_output_tokens),
        observation_reasoning_effort=args.observation_reasoning_effort,
        invoker=None,
        tracking_reconsideration_model=args.model,
        tracking_reconsideration_timeout_seconds=float(args.timeout_seconds),
        tracking_reconsideration_max_output_tokens=int(args.max_output_tokens),
        tracking_reconsideration_reasoning_effort=args.reasoning_effort,
        reconsideration_invoker=None,
        tracking_reconsideration_mode=args.tracking_reconsideration_mode,
        provider_request_budget=provider_request_budget,
        provider_call_telemetry=provider_call_telemetry,
    )

    report = {
        "meta": {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(ended_at - started_at, 3),
            "read_only": True,
            "production_settings_changed": False,
            "dashboard_state_mutation": False,
            "total_requests_expected": request_count,
        },
        "config": {
            "resolved_base_url": base_url,
            "resolved_model": args.model,
            "scenario": args.scenario,
            "timeout_seconds": float(args.timeout_seconds),
            "max_output_tokens_override": int(args.max_output_tokens),
            "reasoning_effort_requested": args.reasoning_effort,
            "reasoning_effort_applied_in_request": capture.reasoning_effort_applied_in_request,
            "observation_verification_mode": args.observation_verification_mode,
            "observation_model": args.observation_model,
            "observation_timeout_seconds": float(args.observation_timeout_seconds),
            "observation_max_output_tokens": int(args.observation_max_output_tokens),
            "observation_reasoning_effort": args.observation_reasoning_effort,
            "tracking_reconsideration_mode": args.tracking_reconsideration_mode,
            "provider_request_upper_bound": expected_provider_requests,
            "persisted_provider_budget_preflight": persisted_budget_preflight,
            "production_timeout_default_unchanged": True,
            "production_output_default_unchanged": True,
            "retry_limit": 0,
        },
        "input_summary": {
            "event_reasons": event_reasons,
            "detection_count": len(detections),
            "detection_ids": [item.detection_id for item in detections],
            "input_yolo_track_assignments": scenario_metadata["input_yolo_track_assignments"],
            "expected_correct_associations": scenario_metadata["expected_correct_associations"],
            "scenario_leakage_audit": scenario_metadata["leakage_audit"],
            "independent_track_history_evidence": scenario_metadata[INDEPENDENT_TRACK_HISTORY_KEY],
            "independent_history_consistency_audit": scenario_metadata["independent_history_consistency_audit"],
            "structured_input_audit": structured_input_audit,
            "frame_reference_included": True,
            "visual_evidence_limitation": (
                "Synthetic frame only approximates detection geometry; historical motion/depth identity evidence is "
                "provided as structured metadata, not visually demonstrated in pixels."
            ),
        },
        "result": {
            "available": result.available,
            "error": result.error,
            "uncertainty_flags": result.uncertainty_flags,
            "proposal_count": len(result.proposals),
            "valid_json": valid_json,
            "json_parsed": valid_json,
            "schema_complete": schema_complete,
            "schema_incomplete": schema_incomplete,
            "timeout": timeout_flag,
            "truncation": truncation_flag,
            "finish_reason": capture.finish_reason,
            "token_usage": capture.usage,
            "token_usage_details": capture.usage_details,
            "response_text": capture.content_text,
            "response_text_char_length": capture.content_text_char_length,
            "response_text_sha256": capture.content_text_sha256,
            "response_text_head": capture.content_text_head,
            "response_text_tail": capture.content_text_tail,
            "response_text_logged_without_local_truncation": True,
            "response_object_keys": capture.response_object_keys,
            "message_object_keys": capture.message_object_keys,
            "wall_latency_ms": wall_latency_ms,
            "agent_reported_latency_ms": result.latency_ms,
            "fallback_or_error_classification": fallback_or_error_classification,
            "observation_fallback_updates": fallback_updates,
            "observation_evidence_backed_updates": evidence_backed_updates,
            "observation_accepted_updates": accepted_updates,
            "observation_rejected_updates": rejected_updates,
            "observation_rejection_reasons_by_detection": rejection_reasons_by_detection,
        },
        "scenario_evaluation": {
            "scenario": args.scenario,
            "outcome": scenario_outcome,
            "per_detection": per_detection_evaluation,
        },
        "tracking_observation_collaboration": observation_collaboration,
        "provider_call_telemetry": provider_call_telemetry,
        "instruction_compression": {
            "applied": capture.compression_applied,
            "compression_scope": "diagnostic_request_only",
            "system_prompt_unchanged": True,
            "original_structure": _build_instruction_structure_view(capture.original_user_instruction or ""),
            "compressed_structure": _build_instruction_structure_view(capture.compressed_user_instruction or ""),
            "removed_items_summary": [
                "Removed duplicated long-form wording while preserving required response keys/fields.",
                "Added strict concision caps for evidence/uncertainty list verbosity.",
                "Retained explicit occlusion, identity ambiguity, and insufficient-evidence signaling requirements.",
            ],
        },
        "local_schema_checks": _validate_compressed_instruction_template(),
    }

    _append_event(
        events_path,
        {
            "event_type": "phase1_validation_completed",
            "timestamp": time.time(),
            "model": args.model,
            "scenario": args.scenario,
            "status": "ok" if result.available else "error",
            "error": result.error,
            "wall_latency_ms": wall_latency_ms,
            "finish_reason": capture.finish_reason,
            "token_usage": capture.usage,
            "schema_complete": schema_complete,
            "schema_incomplete": schema_incomplete,
            "fallback_or_error_classification": fallback_or_error_classification,
            "fallback_updates": fallback_updates,
        },
    )

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    if args.dry_run_no_api:
        try:
            from openai.types.chat.chat_completion import ChatCompletion
            from openai.types.chat.chat_completion_message import ChatCompletionMessage
            from openai.types.completion_usage import CompletionUsage

            sdk_introspection = {
                "chat_completion_fields": sorted(list(ChatCompletion.model_fields.keys())),
                "chat_completion_message_fields": sorted(list(ChatCompletionMessage.model_fields.keys())),
                "completion_usage_fields": sorted(list(CompletionUsage.model_fields.keys())),
                "has_completion_tokens_details": "completion_tokens_details" in CompletionUsage.model_fields,
                "has_prompt_tokens_details": "prompt_tokens_details" in CompletionUsage.model_fields,
                "has_reasoning_content_field": "reasoning_content" in ChatCompletionMessage.model_fields,
                "content_annotation": str(ChatCompletionMessage.model_fields["content"].annotation),
            }
        except Exception as error:
            sdk_introspection = {
                "error": f"{type(error).__name__}:{error}",
            }

        preview_now = time.time()
        preview_detections, preview_expected_associations, preview_scenario_metadata = _build_detection_scenario(
            preview_now,
            args.scenario,
        )

        preview_capture = ResponseCapture()
        preview_wrapper = _CapturingCompletions(
            inner=None,
            capture=preview_capture,
            compress_user_instruction=True,
            reasoning_effort_override=args.reasoning_effort,
            independent_track_history_override=(
                preview_scenario_metadata[INDEPENDENT_TRACK_HISTORY_KEY]
                if isinstance(preview_scenario_metadata.get(INDEPENDENT_TRACK_HISTORY_KEY), list)
                and len(preview_scenario_metadata[INDEPENDENT_TRACK_HISTORY_KEY]) > 0
                else None
            ),
        )
        preview_kwargs: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze visual continuity and propose track associations. "
                                "Return JSON object with keys: proposals, uncertainty_flags, supporting_evidence.\n"
                                f"{STRUCTURED_INPUT_MARKER}{{\"event_reasons\":[],\"detections\":[]}}"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
                    ],
                }
            ]
        }
        preview_wrapper._prepare_request_kwargs(preview_kwargs)
        preview_user_text = (
            preview_kwargs["messages"][0]["content"][0].get("text")
            if isinstance(preview_kwargs.get("messages"), list)
            else None
        )
        preview_structured_payload, preview_structured_payload_error = _extract_structured_payload_from_instruction(
            preview_user_text if isinstance(preview_user_text, str) else ""
        )
        preview_independent_history = []
        if isinstance(preview_structured_payload, dict):
            candidate = preview_structured_payload.get(INDEPENDENT_TRACK_HISTORY_KEY)
            if isinstance(candidate, list):
                preview_independent_history = candidate
        preview_prompt_independent_history_audit = _build_independent_history_consistency_audit(
            preview_independent_history,
            [item.detection_id for item in preview_detections],
        )

        dry_run_payload = {
            "dry_run_no_api": True,
            "local_schema_checks": _validate_compressed_instruction_template(),
            "sdk_introspection": sdk_introspection,
            "observation_verification_preview": {
                "mode": args.observation_verification_mode,
                "model": args.observation_model,
                "timeout_seconds": float(args.observation_timeout_seconds),
                "max_output_tokens": int(args.observation_max_output_tokens),
                "reasoning_effort": args.observation_reasoning_effort,
                "uses_live_observation_llm": args.observation_verification_mode == OBSERVATION_VERIFICATION_MODE_LIVE,
                "uses_mock_observation_invoker": args.observation_verification_mode == OBSERVATION_VERIFICATION_MODE_MOCK,
            },
            "request_override_preview": {
                "reasoning_effort_requested": args.reasoning_effort,
                "reasoning_effort_applied": preview_kwargs.get("reasoning_effort"),
                "compression_applied": preview_capture.compression_applied,
                "compressed_user_instruction_structure": _build_instruction_structure_view(
                    preview_user_text if isinstance(preview_user_text, str) else ""
                ),
                "structured_payload_parsed": isinstance(preview_structured_payload, dict),
                "structured_payload_parse_error": preview_structured_payload_error,
                "includes_expected_correct_associations_key": (
                    isinstance(preview_structured_payload, dict)
                    and "expected_correct_associations" in preview_structured_payload
                ),
                "includes_independent_track_history_evidence": len(preview_independent_history) > 0,
                "independent_history_consistency_audit": preview_prompt_independent_history_audit,
            },
            "scenario_preview": {
                "scenario": args.scenario,
                "detection_ids": [item.detection_id for item in preview_detections],
                "input_yolo_track_assignments": preview_scenario_metadata["input_yolo_track_assignments"],
                "expected_correct_associations": preview_expected_associations,
                "independent_track_history_evidence": preview_scenario_metadata[INDEPENDENT_TRACK_HISTORY_KEY],
                "independent_history_consistency_audit": preview_scenario_metadata["independent_history_consistency_audit"],
                "leakage_audit": preview_scenario_metadata["leakage_audit"],
            },
            "max_total_requests": int(args.max_total_requests),
            "retry_limit": 0,
            "model": args.model,
            "timeout_seconds": float(args.timeout_seconds),
            "max_output_tokens": int(args.max_output_tokens),
        }
        print(json.dumps(dry_run_payload, indent=2))
        return

    report = asyncio.run(_run_validation(args))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "events_jsonl": str(args.events_jsonl),
                "total_requests_expected": report["meta"]["total_requests_expected"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
