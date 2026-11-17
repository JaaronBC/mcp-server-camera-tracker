# Visual tracking agent runtime that invokes VLM association proposals and converts them into validated tracking observations.
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import re
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ..contracts import DetectionCandidate, VisualAssociationProposal
from ..coordinator import AgentCoordinator
from .. import latency_instrumentation
from .. import llm_call
from .. import provider_call_budget
from ..orchestration.runtime.identity_snapshot_service import (
    DEFAULT_EVIDENCE_ROOT_DIR,
    DEFAULT_SNAPSHOT_PATH,
    find_identity_candidates,
    load_identity_snapshot,
)

TRACKING_VLM_MODEL = "qwen-3.8-27b"
DEFAULT_OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
DEFAULT_TRACKING_AGENT_MAX_OUTPUT_TOKENS = 384
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_VLM_DIAGNOSTICS_PATH = (
    PROJECT_ROOT / "captures" / "diagnostics" / "tracking_vlm_attempt_diagnostics.jsonl"
)
DEFAULT_TRACKING_SKILL_PATH = PROJECT_ROOT / "skills" / "object-identity-tracking" / "runtime.md"
TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK = (
    "You are a visual tracking agent for continuity-preserving identity association. "
    "Given structured detections and optional frame evidence, return conservative, evidence-grounded "
    "association proposals with calibrated uncertainty and explicit conflict flags. "
    "Never invent measurements or unsupported identities, and always return strict JSON only."
)
DEFAULT_TRACKING_IDENTITY_EVIDENCE_MAX_CANDIDATES = 8
DEFAULT_TRACKING_IDENTITY_EVIDENCE_MAX_IDENTITY_POOL = 16
DEFAULT_TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS = 3.0
TRACKING_DECISION_ACTIONS = {
    "request_identity_snapshot",
    "request_additional_tracking_evidence",
    "propose_identity_association",
    "inconclusive",
}
TRACKING_DECISION_TOOL_ALLOWLIST = {
    "get_identity_snapshot",
    "get_track_evidence_history",
}

_PROVIDER_ERROR_CODE_ALLOWED_PATTERN = re.compile(r"[^a-z0-9._:-]+")


def _sanitize_provider_error_code(raw_code: object) -> str | None:
    if not isinstance(raw_code, str):
        return None
    normalized = raw_code.strip().lower()
    if not normalized:
        return None
    normalized = _PROVIDER_ERROR_CODE_ALLOWED_PATTERN.sub("_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        return None
    return normalized[:64]


def _extract_exception_telemetry_fields(error: Exception) -> dict[str, object]:
    fields: dict[str, object] = {
        "sdk_exception_class": type(error).__name__,
    }

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        fields["http_status_code"] = status_code

    provider_error_code = _sanitize_provider_error_code(getattr(error, "code", None))
    if provider_error_code is not None:
        fields["provider_error_code"] = provider_error_code

    return fields


def _env_flag_enabled(name: str) -> bool:
    raw_value = os.getenv(name)
    if not isinstance(raw_value, str):
        return False
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int, *, minimum: int = 1, maximum: int = 512) -> int:
    raw_value = os.getenv(name)
    if not isinstance(raw_value, str):
        return default
    try:
        parsed = int(raw_value.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _env_positive_float(name: str, default: float, *, minimum: float = 0.1, maximum: float = 120.0) -> float:
    raw_value = os.getenv(name)
    if not isinstance(raw_value, str):
        return default
    try:
        parsed = float(raw_value.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


def _tracking_identity_evidence_enabled() -> bool:
    return _env_flag_enabled("TRACKING_IDENTITY_EVIDENCE_ENABLED")


def _resolve_tracking_identity_snapshot_path() -> Path:
    configured = os.getenv("TRACKING_IDENTITY_SNAPSHOT_PATH")
    if isinstance(configured, str) and configured.strip():
        path = Path(configured.strip()).expanduser()
        if not path.is_absolute():
            return (PROJECT_ROOT / path).resolve()
        return path.resolve()
    return DEFAULT_SNAPSHOT_PATH


def _resolve_tracking_identity_evidence_root_dir() -> Path:
    configured = os.getenv("TRACKING_IDENTITY_EVIDENCE_ROOT_DIR")
    if isinstance(configured, str) and configured.strip():
        path = Path(configured.strip()).expanduser()
        if not path.is_absolute():
            return (PROJECT_ROOT / path).resolve()
        return path.resolve()
    return DEFAULT_EVIDENCE_ROOT_DIR


def _extract_identity_pool(snapshot_result: dict[str, object], *, max_identities: int) -> list[dict[str, object]]:
    snapshot = snapshot_result.get("snapshot") if isinstance(snapshot_result, dict) else None
    identities = snapshot.get("identities") if isinstance(snapshot, dict) else None
    if not isinstance(identities, list):
        return []

    pool: list[dict[str, object]] = []
    for item in identities:
        if not isinstance(item, dict):
            continue
        app_identity_id = item.get("app_identity_id")
        lifecycle_status = item.get("status")
        if not isinstance(app_identity_id, str) or not app_identity_id.strip():
            continue
        if lifecycle_status == "retired":
            continue
        pool.append(
            {
                "app_identity_id": app_identity_id,
                "lifecycle_status": lifecycle_status,
                "provisional": bool(item.get("provisional", False)),
                "generation": item.get("generation"),
                "identity_version": item.get("identity_version"),
                "updated_at": item.get("updated_at"),
                "detection_id": item.get("detection_id"),
                "detection_reference": item.get("detection_reference"),
                "yolo_track_id": item.get("yolo_track_id"),
                "historical_yolo_track_ids": (
                    list(item.get("historical_yolo_track_ids", []))
                    if isinstance(item.get("historical_yolo_track_ids"), list)
                    else []
                ),
            }
        )
        if len(pool) >= max_identities:
            break
    return pool


def _gather_tracking_identity_evidence(
    *,
    detections: list[DetectionCandidate],
) -> dict[str, object] | None:
    if not _tracking_identity_evidence_enabled():
        return None

    max_candidates = _env_positive_int(
        "TRACKING_IDENTITY_EVIDENCE_MAX_CANDIDATES",
        DEFAULT_TRACKING_IDENTITY_EVIDENCE_MAX_CANDIDATES,
    )
    max_identity_pool = _env_positive_int(
        "TRACKING_IDENTITY_EVIDENCE_MAX_IDENTITY_POOL",
        DEFAULT_TRACKING_IDENTITY_EVIDENCE_MAX_IDENTITY_POOL,
    )
    stale_after_seconds = _env_positive_float(
        "TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS",
        DEFAULT_TRACKING_IDENTITY_EVIDENCE_STALE_SECONDS,
    )

    snapshot_path = _resolve_tracking_identity_snapshot_path()
    evidence_root_dir = _resolve_tracking_identity_evidence_root_dir()
    snapshot_result = load_identity_snapshot(
        path=snapshot_path,
        enabled=True,
        stale_after_seconds=stale_after_seconds,
    )

    available = bool(snapshot_result.get("available", False))
    state = snapshot_result.get("state")
    snapshot_generation = snapshot_result.get("source_generation")
    snapshot_version = snapshot_result.get("source_version")
    identity_pool = _extract_identity_pool(snapshot_result, max_identities=max_identity_pool)

    per_detection: list[dict[str, object]] = []
    for detection in detections:
        detection_reference = (
            detection.detection_reference
            if isinstance(detection.detection_reference, str)
            else None
        )

        if not isinstance(detection_reference, str) or not detection_reference.strip():
            per_detection.append(
                {
                    "detection_id": detection.detection_id,
                    "detection_reference": detection_reference,
                    "lookup": {
                        "available": False,
                        "candidate_lookup_state": "invalid_request",
                        "error": "detection_reference_missing",
                        "reference_kind": "missing",
                        "resolved_detection_reference": None,
                        "ambiguity": {"ambiguous": False, "reason": None},
                        "candidate_count": 0,
                        "candidates": [],
                    },
                    "evidence_usable": False,
                    "evidence_unusable_reason": "detection_reference_missing",
                }
            )
            continue

        lookup = find_identity_candidates(
            detection_reference=detection_reference,
            max_candidates=max_candidates,
            path=snapshot_path,
            evidence_root_dir=evidence_root_dir,
            enabled=True,
            stale_after_seconds=stale_after_seconds,
        )
        lookup_state = lookup.get("candidate_lookup_state")
        lookup_ambiguity = lookup.get("ambiguity")
        ambiguity = bool(
            isinstance(lookup_ambiguity, dict) and lookup_ambiguity.get("ambiguous") is True
        )
        lookup_error = lookup.get("error")
        evidence_usable = available and lookup_state in {"found", "none"} and not ambiguity
        unusable_reason = None
        if not available:
            unusable_reason = f"snapshot_{state or 'unavailable'}"
        elif ambiguity:
            unusable_reason = "candidate_reference_ambiguous"
        elif lookup_state not in {"found", "none"}:
            if isinstance(lookup_error, str) and lookup_error:
                unusable_reason = lookup_error
            else:
                unusable_reason = f"lookup_{lookup_state}"

        per_detection.append(
            {
                "detection_id": detection.detection_id,
                "detection_reference": detection_reference,
                "lookup": {
                    "available": bool(lookup.get("available", False)),
                    "state": lookup.get("state"),
                    "candidate_lookup_state": lookup_state,
                    "error": lookup_error,
                    "reference_kind": lookup.get("reference_kind"),
                    "resolved_detection_reference": lookup.get("resolved_detection_reference"),
                    "ambiguity": lookup_ambiguity,
                    "candidate_count": lookup.get("candidate_count"),
                    "candidates": (
                        list(lookup.get("candidates", []))
                        if isinstance(lookup.get("candidates"), list)
                        else []
                    ),
                    "provenance": lookup.get("provenance"),
                },
                "evidence_usable": evidence_usable,
                "evidence_unusable_reason": unusable_reason,
            }
        )

    return {
        "enabled": True,
        "snapshot": {
            "available": available,
            "state": state,
            "error": snapshot_result.get("error"),
            "fresh": snapshot_result.get("fresh"),
            "snapshot_age_seconds": snapshot_result.get("snapshot_age_seconds"),
            "freshness_window_seconds": snapshot_result.get("freshness_window_seconds"),
            "source_generation": snapshot_generation,
            "source_version": snapshot_version,
            "identity_count": snapshot_result.get("identity_count"),
            "snapshot_path": snapshot_result.get("snapshot_path"),
            "evidence_root_dir": str(evidence_root_dir),
            "stale_after_seconds": stale_after_seconds,
        },
        "identity_pool": identity_pool,
        "identity_pool_count": len(identity_pool),
        "max_candidates": max_candidates,
        "per_detection": per_detection,
        "provenance": {
            "retrieval_mode": "deterministic_read_only_evidence_gathering",
            "identity_selection_authorized": False,
            "mutation_authorized": False,
        },
    }


def _resolve_tracking_skill_path() -> Path:
    configured_path = os.getenv("TRACKING_SKILL_PATH")
    if isinstance(configured_path, str) and configured_path.strip():
        candidate = Path(configured_path.strip()).expanduser()
        if not candidate.is_absolute():
            return (PROJECT_ROOT / candidate).resolve()
        return candidate.resolve()
    return DEFAULT_TRACKING_SKILL_PATH


def _load_text_prompt_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        loaded = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if not isinstance(loaded, str):
        return None
    normalized = loaded.strip()
    if not normalized:
        return None
    return normalized


def load_tracking_agent_skill() -> str:
    # Loads tracking agent skill data from storage for downstream processing.
    base_prompt = TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK
    if not _env_flag_enabled("TRACKING_SKILL_ENABLED"):
        return base_prompt

    skill_text = _load_text_prompt_file(_resolve_tracking_skill_path())
    if not isinstance(skill_text, str):
        return base_prompt

    if skill_text == base_prompt or skill_text in base_prompt:
        return base_prompt

    return (
        f"{base_prompt}\n\n"
        "Additional behavioral skill guidance is provided below. "
        "If there is any conflict, follow the existing JSON-only and response-schema requirements above.\n\n"
        f"{skill_text}"
    )


def _resolve_tracking_max_output_tokens() -> int | None:
    return llm_call.resolve_max_output_tokens(
        "TRACKING_AGENT_MAX_OUTPUT_TOKENS",
        DEFAULT_TRACKING_AGENT_MAX_OUTPUT_TOKENS,
    )


def _tracking_compact_decision_mode_enabled() -> bool:
    return _env_flag_enabled("TRACKING_AGENT_COMPACT_DECISION_MODE")


def _resolve_tracking_vlm_diagnostics_path() -> Path:
    configured = os.getenv("TRACKING_VLM_ATTEMPT_DIAGNOSTICS_PATH")
    if isinstance(configured, str) and configured.strip():
        candidate = Path(configured.strip()).expanduser()
        if not candidate.is_absolute():
            return (PROJECT_ROOT / candidate).resolve()
        return candidate.resolve()
    return DEFAULT_TRACKING_VLM_DIAGNOSTICS_PATH


def _append_tracking_vlm_attempt_diagnostic_record(payload: dict[str, object]) -> None:
    path = _resolve_tracking_vlm_diagnostics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _is_usable_identity_proposal(proposal: VisualAssociationProposal) -> bool:
    if not isinstance(proposal.proposed_app_identity_id, str) or not proposal.proposed_app_identity_id.strip():
        return False
    if not isinstance(proposal.detection_reference, str) or not proposal.detection_reference.strip():
        return False
    if not isinstance(proposal.identity_snapshot_generation, int):
        return False
    if not isinstance(proposal.identity_snapshot_version, int):
        return False
    if not isinstance(proposal.identity_resolution_status, str) or not proposal.identity_resolution_status.strip():
        return False
    if proposal.identity_resolution_status.strip().lower().startswith("unresolved"):
        return False
    return True


def _collect_identity_diagnostic_details(
    *,
    raw_proposals: list[object] | None,
    normalized_proposals: list[VisualAssociationProposal] | None,
) -> dict[str, object]:
    raw_items = raw_proposals if isinstance(raw_proposals, list) else []
    normalized_items = normalized_proposals if isinstance(normalized_proposals, list) else []

    raw_identity_candidate_count = sum(1 for item in raw_items if isinstance(item, dict))
    normalized_identity_candidate_count = len(normalized_items)
    usable_identity_proposal_count = sum(1 for item in normalized_items if _is_usable_identity_proposal(item))

    normalized_by_detection_id: dict[str, VisualAssociationProposal] = {}
    for proposal in normalized_items:
        detection_id = proposal.detection_id
        if isinstance(detection_id, str) and detection_id and detection_id not in normalized_by_detection_id:
            normalized_by_detection_id[detection_id] = proposal

    rejected_identity_candidate_count = 0
    rejection_reasons: set[str] = set()

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        has_identity_keys = any(
            key in raw_item
            for key in (
                "proposed_app_identity_id",
                "identity_evidence_refs",
                "identity_snapshot_generation",
                "identity_snapshot_version",
                "identity_resolution_status",
                "rejected_app_identity_ids",
            )
        )
        if not has_identity_keys:
            continue

        detection_id = raw_item.get("detection_id")
        normalized = (
            normalized_by_detection_id.get(detection_id)
            if isinstance(detection_id, str)
            else None
        )

        if normalized is None:
            rejected_identity_candidate_count += 1
            rejection_reasons.add("identity_candidate_failed_normalization")
            continue

        if _is_usable_identity_proposal(normalized):
            continue

        rejected_identity_candidate_count += 1
        if not isinstance(normalized.proposed_app_identity_id, str) or not normalized.proposed_app_identity_id.strip():
            rejection_reasons.add("missing_proposed_app_identity_id")
        if not isinstance(normalized.identity_snapshot_generation, int):
            rejection_reasons.add("missing_identity_snapshot_generation")
        if not isinstance(normalized.identity_snapshot_version, int):
            rejection_reasons.add("missing_identity_snapshot_version")
        if isinstance(normalized.identity_resolution_status, str) and normalized.identity_resolution_status:
            if normalized.identity_resolution_status.startswith("unresolved"):
                rejection_reasons.add(normalized.identity_resolution_status)
        for uncertainty in normalized.uncertainties:
            if isinstance(uncertainty, str) and uncertainty.startswith("identity_claim_"):
                rejection_reasons.add(uncertainty)

    return {
        "raw_identity_candidate_count": raw_identity_candidate_count,
        "normalized_identity_candidate_count": normalized_identity_candidate_count,
        "usable_identity_proposal_count": usable_identity_proposal_count,
        "rejected_identity_candidate_count": rejected_identity_candidate_count,
        "rejection_reasons": sorted(rejection_reasons),
    }


def _sanitize_runtime_exception_class_name(error: Exception) -> str | None:
    return _sanitize_provider_error_code(type(error).__name__)


def _classify_runtime_exception_category(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {401, 403}:
            return "auth_error"
        if status_code == 429:
            return "rate_limit"
        if 500 <= status_code <= 599:
            return "provider_server_error"
        return "provider_http_error"

    provider_code = _sanitize_provider_error_code(getattr(error, "code", None))
    if provider_code is not None:
        if "timeout" in provider_code:
            return "provider_timeout"
        if "rate" in provider_code and "limit" in provider_code:
            return "rate_limit"
        return "provider_error"

    if isinstance(error, ConnectionError):
        return "network_error"
    if isinstance(error, OSError):
        return "io_error"
    return "runtime_exception"


def _sanitize_validation_error_codes(error_codes: list[str] | None) -> list[str]:
    if not isinstance(error_codes, list):
        return []

    output: list[str] = []
    seen: set[str] = set()
    for item in error_codes:
        sanitized = _sanitize_provider_error_code(item)
        if sanitized is None or sanitized in seen:
            continue
        seen.add(sanitized)
        output.append(sanitized)
    return output


def _coerce_timestamp(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _sanitize_timing_markers(payload: dict[str, Any] | None) -> dict[str, float | None] | None:
    if not isinstance(payload, dict):
        return None

    output: dict[str, float | None] = {}
    for key in (
        "runtime_candidate_submitted_at",
        "runtime_worker_started_at",
        "provider_request_started_at",
        "provider_response_completed_at",
        "provider_timeout_at",
        "provider_error_at",
        "observation_published_at",
    ):
        output[key] = _coerce_timestamp(payload.get(key))

    if any(value is not None for value in output.values()):
        return output
    return None


def _sanitize_request_metadata(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    output: dict[str, Any] = {}
    model_identifier = payload.get("model_identifier")
    if isinstance(model_identifier, str) and model_identifier.strip():
        output["model_identifier"] = model_identifier.strip()

    for key in (
        "image_count",
        "detection_count",
        "output_token_limit",
        "request_size_estimate_bytes",
    ):
        value = payload.get(key)
        if isinstance(value, int):
            output[key] = value

    configured_timeout_seconds = payload.get("configured_timeout_seconds")
    if isinstance(configured_timeout_seconds, (int, float)):
        output["configured_timeout_seconds"] = float(configured_timeout_seconds)

    for key in (
        "follow_up_mcp_result_present",
        "follow_up_evidence_refs_present",
    ):
        value = payload.get(key)
        if isinstance(value, bool):
            output[key] = value

    for key in (
        "follow_up_investigation_id",
        "follow_up_request_id",
        "follow_up_result_status",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            output[key] = value.strip()

    return output if output else None


def _emit_tracking_vlm_attempt_diagnostic(
    *,
    attempt_id: str | None,
    llm_correlation_id: str | None,
    provider_outcome: str,
    parse_status: str,
    parse_error_code: str | None,
    schema_status: str,
    result_available: bool,
    result_error_code: str | None,
    finish_reason: str | None = None,
    latency_ms: float | None = None,
    usage: dict[str, int] | None = None,
    raw_proposals: list[object] | None = None,
    normalized_proposals: list[VisualAssociationProposal] | None = None,
    decision_validation_error_codes: list[str] | None = None,
    runtime_exception_class: str | None = None,
    runtime_exception_category: str | None = None,
    timing_markers: dict[str, Any] | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> None:
    identity_metrics = _collect_identity_diagnostic_details(
        raw_proposals=raw_proposals,
        normalized_proposals=normalized_proposals,
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "event_type": "tracking_vlm_attempt_completed",
        "timestamp": time.time(),
        "attempt_id": attempt_id,
        "llm_correlation_id": llm_correlation_id,
        "provider_outcome": provider_outcome,
        "parse_status": parse_status,
        "parse_error_code": parse_error_code,
        "finish_reason": finish_reason,
        "schema_status": schema_status,
        "result_available": bool(result_available),
        "result_error_code": result_error_code,
        "latency_ms": latency_ms,
        "usage": usage,
        "decision_validation_error_codes": (
            _sanitize_validation_error_codes(decision_validation_error_codes)
            if isinstance(decision_validation_error_codes, list)
            else None
        ),
        "runtime_exception_class": _sanitize_provider_error_code(runtime_exception_class),
        "runtime_exception_category": _sanitize_provider_error_code(runtime_exception_category),
        "raw_identity_candidate_count": identity_metrics["raw_identity_candidate_count"],
        "normalized_identity_candidate_count": identity_metrics["normalized_identity_candidate_count"],
        "usable_identity_proposal_count": identity_metrics["usable_identity_proposal_count"],
        "rejected_identity_candidate_count": identity_metrics["rejected_identity_candidate_count"],
        "rejection_reasons": identity_metrics["rejection_reasons"],
        "timing_markers": _sanitize_timing_markers(timing_markers),
        "request_metadata": _sanitize_request_metadata(request_metadata),
    }
    try:
        _append_tracking_vlm_attempt_diagnostic_record(payload)
    except Exception:
        return


def _emit_tracking_vlm_observation_timing_diagnostic(
    *,
    attempt_id: str | None,
    llm_correlation_id: str | None,
    timing_markers: dict[str, Any] | None,
    request_metadata: dict[str, Any] | None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "event_type": "tracking_vlm_observation_published",
        "timestamp": time.time(),
        "attempt_id": attempt_id,
        "llm_correlation_id": llm_correlation_id,
        "timing_markers": _sanitize_timing_markers(timing_markers),
        "request_metadata": _sanitize_request_metadata(request_metadata),
    }
    try:
        _append_tracking_vlm_attempt_diagnostic_record(payload)
    except Exception:
        return


@dataclass(slots=True)
class VisualTrackingAgentResult:
    available: bool
    proposals: list[VisualAssociationProposal]
    uncertainty_flags: list[str]
    supporting_evidence: list[str]
    latency_ms: float | None
    error: str | None = None
    llm_correlation_id: str | None = None
    decision: dict[str, Any] | None = None
    attempt_id: str | None = None
    timing_markers: dict[str, float | None] | None = None
    request_metadata: dict[str, Any] | None = None


class VisualTrackingAgentProtocol(Protocol):
    model: str

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        follow_up_context: dict[str, Any] | None = None,
        runtime_timing: dict[str, Any] | None = None,
        request_timeout_seconds: float = 2.0,
    # Implements propose associations behavior for this module.
    ) -> VisualTrackingAgentResult: ...


class LLMVisualTrackingAgent:
    def __init__(
        self,
        model: str = TRACKING_VLM_MODEL,
        openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        system_prompt: str | None = None,
    ) -> None:
        # Implements init behavior for this module.
        load_dotenv()
        self.model = model
        self._openai_base_url = os.getenv("OPENAI_BASE_URL", openai_base_url)
        self._api_key = api_key
        self._client = client
        self._system_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str) and system_prompt.strip()
            else load_tracking_agent_skill()
        )

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        follow_up_context: dict[str, Any] | None = None,
        runtime_timing: dict[str, Any] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult:
        # Implements propose associations behavior for this module.
        request_correlation_id = latency_instrumentation.new_correlation_id("tracking-llm")
        if not detections:
            return VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["no_detections"],
                supporting_evidence=[],
                latency_ms=0.0,
                llm_correlation_id=request_correlation_id,
            )

        if self._client is None:
            load_dotenv()
            api_key = self._api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return VisualTrackingAgentResult(
                    available=False,
                    proposals=[],
                    uncertainty_flags=["vlm_api_key_missing"],
                    supporting_evidence=[],
                    latency_ms=None,
                    error="vlm_api_key_missing",
                    llm_correlation_id=request_correlation_id,
                )
            self._client = AsyncOpenAI(base_url=self._openai_base_url, api_key=api_key)

        identity_evidence_bundle = _gather_tracking_identity_evidence(
            detections=detections,
        )

        payload = {
            "event_reasons": event_reasons or [],
            "detections": [item.to_dict() for item in detections],
            "instructions": {
                "do_not_invent_measurements": True,
                "do_not_override_reliable_tracking_without_evidence": True,
                "return_json_only": True,
            },
        }
        if isinstance(follow_up_context, dict):
            payload["investigation_follow_up"] = dict(follow_up_context)
        if identity_evidence_bundle is not None:
            payload["identity_candidate_evidence"] = identity_evidence_bundle

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    (
                        "Analyze visual continuity and propose conservative track associations. "
                        "Return exactly one compact JSON object with top-level keys: proposals, uncertainty_flags, supporting_evidence. "
                        "You may optionally include a top-level decision object with action, investigation_id, optional correlation_id, "
                        "optional uncertainty/evidence_references/provenance/snapshot_constraints, and optional tool_requests. "
                        "Allowed decision actions are: request_identity_snapshot, request_additional_tracking_evidence, "
                        "propose_identity_association, inconclusive. "
                        "Use request_identity_snapshot when identity continuity is plausible but current identity snapshot/candidate context is insufficient. "
                        "Use request_additional_tracking_evidence when you need more historical evidence for a track before deciding. "
                        "Use inconclusive when current evidence is insufficient or conflicting; include a concise reason and do not guess. "
                        "Use propose_identity_association only when evidence supports a concrete association proposal. "
                        "For request_identity_snapshot/request_additional_tracking_evidence/inconclusive decisions, do not emit actionable association proposals. "
                        "Return no markdown, no commentary, and no prose outside JSON. "
                        "Empty proposals are allowed when no supported association is available. "
                        "For each proposal, include: detection_id, proposed_track_id, association_confidence, occlusion_likelihood, "
                        "id_switch_risk, depth_conflict_flag, position_conflict_flag, uncertainties, supporting_evidence. "
                        "When identity_candidate_evidence is provided and evidence_usable is true with candidate_lookup_state found/none, "
                        "include detection_reference, proposed_app_identity_id (or null), rejected_app_identity_ids, identity_evidence_refs, "
                        "identity_snapshot_generation, identity_snapshot_version, identity_resolution_status. "
                        "If identity evidence is ambiguous, stale, missing, or otherwise unusable, do not force a match; "
                        "set proposed_app_identity_id to null and use concise reason tags in identity_resolution_status and uncertainties. "
                        "Keep uncertainty and evidence values short tag-like strings; avoid repeated descriptions.\n"
                        if not _tracking_compact_decision_mode_enabled()
                        else (
                            "Return one compact JSON object only. Decision-first format is preferred. "
                            "Allowed actions: request_identity_snapshot, request_additional_tracking_evidence, propose_identity_association, inconclusive. "
                            "Semantic-first decision contract: provide action, target selector, concise reason/uncertainty, and optional evidence_bounds. "
                            "For request_identity_snapshot, provide exactly one selector: app_identity_id or detection_reference. "
                            "For request_additional_tracking_evidence, provide track_id. "
                            "For inconclusive, provide a concise reason and no tool request. "
                            "Runtime binds investigation_id, request_id, correlation_id, canonical tool_name, and canonical tool_requests structure. "
                            "If you include explicit tool_name/tool/tool_request, it must exactly match action mapping or the decision is rejected. "
                            "Tool mapping is strict: request_identity_snapshot -> get_identity_snapshot; request_additional_tracking_evidence -> get_track_evidence_history. "
                            "If including provenance, it must be a flat object with non-empty string keys and scalar values only (string, number, boolean, or null); do not use nested objects or arrays. "
                            "evidence_bounds must be an object when present. Supported identity-snapshot bounds: include_history, max_history_entries, max_candidates, min_source_generation, min_source_version, stale_after_seconds. "
                            "Supported track-evidence bounds: max_messages, max_history_records, max_scan_records, now_timestamp. "
                            "For initial decisions, prioritize one evidence request or inconclusive. "
                            "For follow-up decisions, return propose_identity_association or inconclusive. "
                            "Do not fabricate investigation_id, request_id, correlation_id, or evidence references. "
                            "If trusted runtime identifiers are unavailable, you may omit them and runtime will bind IDs. "
                            "Examples: {\"action\":\"request_identity_snapshot\",\"detection_reference\":\"<trusted_ref>\",\"evidence_bounds\":{\"include_history\":true,\"max_candidates\":4}}; {\"action\":\"request_additional_tracking_evidence\",\"track_id\":\"<trusted_track_id>\",\"evidence_bounds\":{\"max_messages\":5}}; {\"action\":\"inconclusive\",\"reason\":\"insufficient_evidence\",\"uncertainty\":[\"needs_more_evidence\"]}. "
                            "Do not emit actionable proposals for evidence-request or inconclusive actions. "
                            "Use short uncertainty tags and compact provenance fields.\n"
                        )
                    )
                    + f"Structured input:\n{json.dumps(payload)}"
                ),
            }
        ]
        if frame_reference:
            user_content.append({"type": "image_url", "image_url": {"url": frame_reference}})

        max_output_tokens = _resolve_tracking_max_output_tokens()
        request_size_estimate_bytes = (
            len(self._system_prompt.encode("utf-8"))
            + len(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            + (len(frame_reference.encode("utf-8")) if isinstance(frame_reference, str) else 0)
        )
        request_metadata = {
            "model_identifier": self.model,
            "image_count": 1 if isinstance(frame_reference, str) and frame_reference else 0,
            "detection_count": len(detections),
            "configured_timeout_seconds": float(request_timeout_seconds),
            "output_token_limit": max_output_tokens,
            "request_size_estimate_bytes": request_size_estimate_bytes,
        }
        if isinstance(follow_up_context, dict):
            follow_up_investigation_id = _nonempty_str(follow_up_context.get("investigation_id"))
            if follow_up_investigation_id is not None:
                request_metadata["follow_up_investigation_id"] = follow_up_investigation_id
            follow_up_request_id = _nonempty_str(follow_up_context.get("request_id"))
            if follow_up_request_id is not None:
                request_metadata["follow_up_request_id"] = follow_up_request_id
            follow_up_result_status = _nonempty_str(follow_up_context.get("result_status"))
            if follow_up_result_status is not None:
                request_metadata["follow_up_result_status"] = follow_up_result_status
            request_metadata["follow_up_mcp_result_present"] = isinstance(follow_up_context.get("mcp_result"), dict)
            request_metadata["follow_up_evidence_refs_present"] = isinstance(
                follow_up_context.get("evidence_references"),
                dict,
            )
        timing_markers: dict[str, float | None] = {
            "runtime_candidate_submitted_at": _coerce_timestamp(
                runtime_timing.get("runtime_candidate_submitted_at")
                if isinstance(runtime_timing, dict)
                else None
            ),
            "runtime_worker_started_at": _coerce_timestamp(
                runtime_timing.get("runtime_worker_started_at")
                if isinstance(runtime_timing, dict)
                else None
            ),
            "provider_request_started_at": None,
            "provider_response_completed_at": None,
            "provider_timeout_at": None,
            "provider_error_at": None,
            "observation_published_at": None,
        }

        try:
            reservation = provider_call_budget.reserve_or_raise(provider_call_budget.ROLE_TRACKING_VLM)
        except provider_call_budget.ProviderCallBudgetExceeded:
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_budget_exhausted"],
                supporting_evidence=[],
                latency_ms=0.0,
                error="vlm_budget_exhausted",
                llm_correlation_id=request_correlation_id,
            )
        attempt_id = getattr(reservation, "attempt_id", None)
        started = time.perf_counter()
        latency_instrumentation.emit_event(
            "llm_request_started",
            stage="tracking_agent_llm_request",
            duration_ms=0.0,
            role="tracking_agent",
            provider="openai_compatible",
            model=self.model,
            timeout_seconds=request_timeout_seconds,
            retry_limit=0,
            attempt_number=1,
            attempts_total=1,
            correlation_id=request_correlation_id,
            detection_count=len(detections),
            frame_reference_present=bool(frame_reference),
        )
        try:
            request_kwargs: dict[str, object] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._system_prompt,
                    },
                    {"role": "user", "content": user_content},
                ],
            }
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            timing_markers["provider_request_started_at"] = time.time()
            response = await asyncio.wait_for(
                self._client.chat.completions.create(**request_kwargs),
                timeout=request_timeout_seconds,
            )
            timing_markers["provider_response_completed_at"] = time.time()
            usage_payload = llm_call.extract_usage_dict(response)
            provider_call_budget.finalize_provider_call_attempt(
                reservation,
                outcome="ok",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage=usage_payload,
            )
            latency_instrumentation.emit_event(
                "llm_request_completed",
                stage="tracking_agent_llm_request",
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                role="tracking_agent",
                provider="openai_compatible",
                model=self.model,
                timeout_seconds=request_timeout_seconds,
                retry_limit=0,
                attempt_number=1,
                attempts_total=1,
                correlation_id=request_correlation_id,
                status="ok",
                usage=usage_payload,
            )
        except asyncio.TimeoutError:
            timing_markers["provider_timeout_at"] = time.time()
            provider_call_budget.finalize_provider_call_attempt(
                reservation,
                outcome="error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error_code="timeout",
            )
            latency_instrumentation.emit_event(
                "llm_request_completed",
                stage="tracking_agent_llm_request",
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                role="tracking_agent",
                provider="openai_compatible",
                model=self.model,
                timeout_seconds=request_timeout_seconds,
                retry_limit=0,
                attempt_number=1,
                attempts_total=1,
                correlation_id=request_correlation_id,
                status="error",
                error_code="timeout",
            )
            _emit_tracking_vlm_attempt_diagnostic(
                attempt_id=attempt_id,
                llm_correlation_id=request_correlation_id,
                provider_outcome="error",
                parse_status="not_attempted",
                parse_error_code=None,
                schema_status="unknown",
                result_available=False,
                result_error_code="vlm_timeout",
                finish_reason=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage=None,
                raw_proposals=None,
                normalized_proposals=None,
                timing_markers=timing_markers,
                request_metadata=request_metadata,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_timeout"],
                supporting_evidence=[],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="vlm_timeout",
                llm_correlation_id=request_correlation_id,
                attempt_id=attempt_id,
                timing_markers=dict(timing_markers),
                request_metadata=dict(request_metadata),
            )
        except Exception as error:
            timing_markers["provider_error_at"] = time.time()
            provider_call_budget.finalize_provider_call_attempt(
                reservation,
                outcome="error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error_code="runtime_error",
            )
            try:
                exception_fields = _extract_exception_telemetry_fields(error)
            except Exception:
                exception_fields = {}
            latency_instrumentation.emit_event(
                "llm_request_completed",
                stage="tracking_agent_llm_request",
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                role="tracking_agent",
                provider="openai_compatible",
                model=self.model,
                timeout_seconds=request_timeout_seconds,
                retry_limit=0,
                attempt_number=1,
                attempts_total=1,
                correlation_id=request_correlation_id,
                status="error",
                error_code="runtime_error",
                **exception_fields,
            )
            _emit_tracking_vlm_attempt_diagnostic(
                attempt_id=attempt_id,
                llm_correlation_id=request_correlation_id,
                provider_outcome="error",
                parse_status="not_attempted",
                parse_error_code=None,
                schema_status="unknown",
                result_available=False,
                result_error_code="vlm_unavailable",
                finish_reason=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage=None,
                raw_proposals=None,
                normalized_proposals=None,
                runtime_exception_class=_sanitize_runtime_exception_class_name(error),
                runtime_exception_category=_classify_runtime_exception_category(error),
                timing_markers=timing_markers,
                request_metadata=request_metadata,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_unavailable"],
                supporting_evidence=[],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="vlm_unavailable",
                llm_correlation_id=request_correlation_id,
                attempt_id=attempt_id,
                timing_markers=dict(timing_markers),
                request_metadata=dict(request_metadata),
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        finish_reason = str(getattr(response.choices[0], "finish_reason", "")) if response.choices else None
        usage_payload = llm_call.extract_usage_dict(response)
        if not response.choices:
            _emit_tracking_vlm_attempt_diagnostic(
                attempt_id=attempt_id,
                llm_correlation_id=request_correlation_id,
                provider_outcome="ok",
                parse_status="parse_failed",
                parse_error_code="vlm_empty_response",
                schema_status="invalid",
                result_available=False,
                result_error_code="vlm_empty_response",
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                usage=usage_payload,
                raw_proposals=None,
                normalized_proposals=None,
                timing_markers=timing_markers,
                request_metadata=request_metadata,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_empty_response"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_empty_response",
                llm_correlation_id=request_correlation_id,
                attempt_id=attempt_id,
                timing_markers=dict(timing_markers),
                request_metadata=dict(request_metadata),
            )

        content = response.choices[0].message.content or ""
        if getattr(response.choices[0], "finish_reason", None) == "length":
            _emit_tracking_vlm_attempt_diagnostic(
                attempt_id=attempt_id,
                llm_correlation_id=request_correlation_id,
                provider_outcome="ok",
                parse_status="parse_failed",
                parse_error_code="vlm_output_truncated",
                schema_status="invalid",
                result_available=False,
                result_error_code="vlm_output_truncated",
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                usage=usage_payload,
                raw_proposals=None,
                normalized_proposals=None,
                timing_markers=timing_markers,
                request_metadata=request_metadata,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_output_truncated"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_output_truncated",
                llm_correlation_id=request_correlation_id,
                attempt_id=attempt_id,
                timing_markers=dict(timing_markers),
                request_metadata=dict(request_metadata),
            )
        parsed = _extract_json_object(content)
        if parsed is None:
            _emit_tracking_vlm_attempt_diagnostic(
                attempt_id=attempt_id,
                llm_correlation_id=request_correlation_id,
                provider_outcome="ok",
                parse_status="parse_failed",
                parse_error_code="vlm_invalid_json_response",
                schema_status="invalid",
                result_available=False,
                result_error_code="vlm_invalid_json_response",
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                usage=usage_payload,
                raw_proposals=None,
                normalized_proposals=None,
                timing_markers=timing_markers,
                request_metadata=request_metadata,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_invalid_json_response"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_invalid_json_response",
                llm_correlation_id=request_correlation_id,
                attempt_id=attempt_id,
                timing_markers=dict(timing_markers),
                request_metadata=dict(request_metadata),
            )

        if _tracking_compact_decision_mode_enabled():
            parsed = _normalize_compact_response_payload(
                parsed,
                request_correlation_id=request_correlation_id,
                follow_up_context=follow_up_context,
                detections=detections,
            )

        raw_proposals = parsed.get("proposals")
        uncertainty_flags = parsed.get("uncertainty_flags")
        supporting_evidence = parsed.get("supporting_evidence")
        decision_payload, decision_errors = _validate_tracking_decision_payload(parsed.get("decision"))

        schema_complete = (
            isinstance(raw_proposals, list)
            and isinstance(uncertainty_flags, list)
            and isinstance(supporting_evidence, list)
        )
        if not isinstance(raw_proposals, list):
            raw_proposals = []
        if not isinstance(uncertainty_flags, list):
            uncertainty_flags = []
        if not isinstance(supporting_evidence, list):
            supporting_evidence = []

        if decision_errors:
            _emit_tracking_vlm_attempt_diagnostic(
                attempt_id=attempt_id,
                llm_correlation_id=request_correlation_id,
                provider_outcome="ok",
                parse_status="parsed",
                parse_error_code="vlm_invalid_decision_contract",
                schema_status="invalid",
                result_available=False,
                result_error_code="vlm_invalid_decision_contract",
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                usage=usage_payload,
                raw_proposals=raw_proposals,
                normalized_proposals=None,
                decision_validation_error_codes=decision_errors,
                timing_markers=timing_markers,
                request_metadata=request_metadata,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_invalid_decision_contract", *decision_errors],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_invalid_decision_contract",
                llm_correlation_id=request_correlation_id,
                decision=None,
                attempt_id=attempt_id,
                timing_markers=dict(timing_markers),
                request_metadata=dict(request_metadata),
            )

        if decision_payload is not None and _decision_disallows_actionable_proposals(decision_payload, raw_proposals):
            _emit_tracking_vlm_attempt_diagnostic(
                attempt_id=attempt_id,
                llm_correlation_id=request_correlation_id,
                provider_outcome="ok",
                parse_status="parsed",
                parse_error_code="vlm_decision_proposal_conflict",
                schema_status="invalid",
                result_available=False,
                result_error_code="vlm_decision_proposal_conflict",
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                usage=usage_payload,
                raw_proposals=raw_proposals,
                normalized_proposals=None,
                timing_markers=timing_markers,
                request_metadata=request_metadata,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_decision_proposal_conflict"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_decision_proposal_conflict",
                llm_correlation_id=request_correlation_id,
                decision=None,
                attempt_id=attempt_id,
                timing_markers=dict(timing_markers),
                request_metadata=dict(request_metadata),
            )

        normalized_proposals: list[VisualAssociationProposal] = []
        detection_by_id = {
            candidate.detection_id: candidate
            for candidate in detections
            if isinstance(candidate.detection_id, str)
        }
        identity_evidence_by_detection_id: dict[str, dict[str, object]] | None = None
        evidence_snapshot_generation: int | None = None
        evidence_snapshot_version: int | None = None
        evidence_snapshot_available: bool | None = None
        eligible_identity_ids: set[str] | None = None
        if isinstance(identity_evidence_bundle, dict):
            per_detection = identity_evidence_bundle.get("per_detection")
            if isinstance(per_detection, list):
                identity_evidence_by_detection_id = {
                    item.get("detection_id"): item
                    for item in per_detection
                    if isinstance(item, dict) and isinstance(item.get("detection_id"), str)
                }
            identity_pool = identity_evidence_bundle.get("identity_pool")
            if isinstance(identity_pool, list):
                eligible_identity_ids = {
                    item.get("app_identity_id")
                    for item in identity_pool
                    if isinstance(item, dict)
                    and isinstance(item.get("app_identity_id"), str)
                    and item.get("app_identity_id").strip()
                }
            snapshot_payload = identity_evidence_bundle.get("snapshot")
            if isinstance(snapshot_payload, dict):
                snapshot_available = snapshot_payload.get("available")
                if isinstance(snapshot_available, bool):
                    evidence_snapshot_available = snapshot_available
                source_generation = snapshot_payload.get("source_generation")
                if isinstance(source_generation, int):
                    evidence_snapshot_generation = source_generation
                source_version = snapshot_payload.get("source_version")
                if isinstance(source_version, int):
                    evidence_snapshot_version = source_version

        for item in raw_proposals:
            if isinstance(item, dict):
                normalized = _normalize_proposal(
                    item,
                    detection_by_id=detection_by_id,
                    identity_evidence_by_detection_id=identity_evidence_by_detection_id,
                    evidence_snapshot_generation=evidence_snapshot_generation,
                    evidence_snapshot_version=evidence_snapshot_version,
                    evidence_snapshot_available=evidence_snapshot_available,
                    eligible_identity_ids=eligible_identity_ids,
                )
                if normalized is not None:
                    normalized_proposals.append(normalized)

        normalized_uncertainty_flags = [str(value) for value in uncertainty_flags]
        if schema_complete:
            normalized_uncertainty_flags.append("schema_complete_response")
        else:
            normalized_uncertainty_flags.append("schema_incomplete_response")

        _emit_tracking_vlm_attempt_diagnostic(
            attempt_id=attempt_id,
            llm_correlation_id=request_correlation_id,
            provider_outcome="ok",
            parse_status="parsed",
            parse_error_code=None,
            schema_status=("complete" if schema_complete else "incomplete"),
            result_available=True,
            result_error_code=None,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            usage=usage_payload,
            raw_proposals=raw_proposals,
            normalized_proposals=normalized_proposals,
            timing_markers=timing_markers,
            request_metadata=request_metadata,
        )

        return VisualTrackingAgentResult(
            available=True,
            proposals=normalized_proposals,
            uncertainty_flags=normalized_uncertainty_flags,
            supporting_evidence=[str(value) for value in supporting_evidence],
            latency_ms=latency_ms,
            llm_correlation_id=request_correlation_id,
            decision=decision_payload,
            attempt_id=attempt_id,
            timing_markers=dict(timing_markers),
            request_metadata=dict(request_metadata),
        )

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return

        close_method = getattr(client, "aclose", None)
        if close_method is None:
            close_method = getattr(client, "close", None)
        if close_method is None or not callable(close_method):
            return

        close_result = close_method()
        if inspect.isawaitable(close_result):
            await close_result


class AsyncVisualTrackingRuntime:
    def __init__(
        self,
        coordinator: AgentCoordinator,
        agent: VisualTrackingAgentProtocol,
        min_invoke_interval_seconds: float = 0.5,
    ) -> None:
        # Implements init behavior for this module.
        self._coordinator = coordinator
        self._agent = agent
        self._min_invoke_interval_seconds = min_invoke_interval_seconds
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future: Future | None = None
        self._last_started_at = 0.0
        self._completed_observations: deque = deque()
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_started = False

    def _consume_completed_future_locked(self) -> None:
        pending_future = self._pending_future
        if pending_future is None or not pending_future.done():
            return

        self._pending_future = None
        try:
            observation = pending_future.result()
        except Exception:
            observation = None

        if observation is not None:
            self._completed_observations.append(observation)

    def submit_if_needed(
        self,
        now: float,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        follow_up_context: dict[str, Any] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> bool:
        # Implements submit if needed behavior for this module.
        with self._lock:
            if self._shutdown_started:
                return False
            self._consume_completed_future_locked()
            if self._pending_future is not None and not self._pending_future.done():
                return False
            has_priority_event_reasons = self._has_priority_event_reasons(event_reasons)
            if (now - self._last_started_at) < self._min_invoke_interval_seconds and not has_priority_event_reasons:
                return False
            self._last_started_at = now
            observation_id = latency_instrumentation.new_correlation_id("tracking-observation")
            self._pending_future = self._executor.submit(
                self._run_once,
                now,
                observation_id,
                frame_reference,
                detections,
                event_reasons or [],
                (dict(follow_up_context) if isinstance(follow_up_context, dict) else None),
                request_timeout_seconds,
                float(now),
            )
            return True

    @staticmethod
    def _has_priority_event_reasons(event_reasons: list[str] | None) -> bool:
        if not event_reasons:
            return False

        for reason in event_reasons:
            if not isinstance(reason, str):
                continue
            normalized = reason.strip()
            if not normalized:
                continue
            if normalized != "periodic_visual_continuity_check":
                return True

        return False

    def poll_latest_observation(self):
        # Implements poll latest observation behavior for this module.
        with self._lock:
            self._consume_completed_future_locked()
            if not self._completed_observations:
                return None
            return self._completed_observations.popleft()

    def is_busy(self) -> bool:
        # Implements is busy behavior for this module.
        with self._lock:
            return self._pending_future is not None and not self._pending_future.done()

    def shutdown(self) -> None:
        # Implements shutdown behavior for this module.
        cleanup_future: Future | None = None
        with self._lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            pending_future = self._pending_future
            self._pending_future = None
            if pending_future is not None and not pending_future.done():
                pending_future.cancel()
            cleanup_future = self._executor.submit(self._shutdown_worker_resources)

        if cleanup_future is not None:
            try:
                cleanup_future.result()
            except Exception:
                pass

        self._executor.shutdown(wait=True, cancel_futures=True)

    def _run_once(
        self,
        now: float,
        observation_id: str,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str],
        follow_up_context: dict[str, Any] | None,
        request_timeout_seconds: float,
        runtime_candidate_submitted_at: float | None,
    ):
        # Runs once workflow steps and returns status outcomes.
        runtime_worker_started_at = time.time()
        call_kwargs: dict[str, Any] = {
            "frame_reference": frame_reference,
            "detections": detections,
            "event_reasons": event_reasons,
            "request_timeout_seconds": request_timeout_seconds,
        }
        try:
            propose_signature = inspect.signature(self._agent.propose_associations)
        except (TypeError, ValueError):
            propose_signature = None

        if isinstance(follow_up_context, dict):
            if propose_signature is not None and "follow_up_context" in propose_signature.parameters:
                call_kwargs["follow_up_context"] = follow_up_context

        if propose_signature is not None and "runtime_timing" in propose_signature.parameters:
            call_kwargs["runtime_timing"] = {
                "runtime_candidate_submitted_at": _coerce_timestamp(runtime_candidate_submitted_at),
                "runtime_worker_started_at": float(runtime_worker_started_at),
            }

        result = self._run_agent_call_on_worker_loop(call_kwargs)

        uncertainties = list(result.uncertainty_flags)
        if not result.available and result.error is not None:
            uncertainties.append(result.error)
        if isinstance(follow_up_context, dict):
            investigation_id = _nonempty_str(follow_up_context.get("investigation_id"))
            request_id = _nonempty_str(follow_up_context.get("request_id"))
            result_key = _nonempty_str(follow_up_context.get("idempotency_key"))
            result_status = _nonempty_str(follow_up_context.get("result_status"))
            if investigation_id is not None:
                uncertainties.append(f"followup_investigation_id:{investigation_id}")
            if request_id is not None:
                uncertainties.append(f"followup_request_id:{request_id}")
            if result_key is not None:
                uncertainties.append(f"followup_result_key:{result_key}")
            if result_status is not None:
                uncertainties.append(f"followup_result_status:{result_status}")

        observation = self._coordinator.build_tracking_agent_observation(
            detections=detections,
            proposals=result.proposals,
            timestamp=now,
            vlm_model=self._agent.model,
            frame_reference=frame_reference,
            proposal_latency_ms=result.latency_ms,
            observation_id=observation_id,
            llm_correlation_id=result.llm_correlation_id,
            additional_uncertainties=uncertainties,
            decision=result.decision,
        )

        merged_timing_markers = (
            dict(result.timing_markers)
            if isinstance(result.timing_markers, dict)
            else {}
        )
        if merged_timing_markers.get("runtime_candidate_submitted_at") is None:
            merged_timing_markers["runtime_candidate_submitted_at"] = _coerce_timestamp(runtime_candidate_submitted_at)
        if merged_timing_markers.get("runtime_worker_started_at") is None:
            merged_timing_markers["runtime_worker_started_at"] = float(runtime_worker_started_at)
        merged_timing_markers["observation_published_at"] = time.time()

        _emit_tracking_vlm_observation_timing_diagnostic(
            attempt_id=result.attempt_id,
            llm_correlation_id=result.llm_correlation_id,
            timing_markers=merged_timing_markers,
            request_metadata=(
                dict(result.request_metadata)
                if isinstance(result.request_metadata, dict)
                else None
            ),
        )

        return observation

    def _get_or_create_worker_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._worker_loop
        if loop is not None and not loop.is_closed():
            return loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._worker_loop = loop
        return loop

    def _run_agent_call_on_worker_loop(self, call_kwargs: dict[str, Any]) -> VisualTrackingAgentResult:
        loop = self._get_or_create_worker_loop()
        return loop.run_until_complete(self._agent.propose_associations(**call_kwargs))

    def _shutdown_worker_resources(self) -> None:
        loop = self._worker_loop
        if loop is None or loop.is_closed():
            return

        async def _close_agent_resources() -> None:
            close_method = getattr(self._agent, "aclose", None)
            if close_method is None or not callable(close_method):
                return
            close_result = close_method()
            if inspect.isawaitable(close_result):
                await close_result

        try:
            loop.run_until_complete(_close_agent_resources())
        except Exception:
            pass
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._worker_loop = None


def encode_frame_as_data_url(frame) -> str:
    # Implements encode frame as data url behavior for this module.
    ok, encoded = __import__("cv2").imencode(".jpg", frame)
    if not ok:
        raise ValueError("unable_to_encode_frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _extract_json_object(raw_text: str) -> dict | None:
    # Extracts json object from structured payloads.
    text = raw_text.strip()
    if not text:
        return None

    if text.startswith("```"):
        parts = text.split("```")
        text = ""
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
        if not text:
            return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def _runtime_fallback_investigation_id(
    *,
    request_correlation_id: str | None,
    follow_up_context: dict[str, Any] | None,
    detections: list[DetectionCandidate] | None,
) -> str | None:
    if isinstance(follow_up_context, dict):
        follow_up_investigation_id = _nonempty_str(follow_up_context.get("investigation_id"))
        if follow_up_investigation_id is not None:
            return follow_up_investigation_id

    if request_correlation_id is None:
        return None

    first_detection_id: str | None = None
    if isinstance(detections, list) and detections:
        detection_id = getattr(detections[0], "detection_id", None)
        if isinstance(detection_id, str):
            first_detection_id = detection_id.strip() or None

    if first_detection_id is not None:
        return f"inv-{request_correlation_id}-{first_detection_id}"
    return f"inv-{request_correlation_id}"


def _runtime_fallback_request_id(
    *,
    request_correlation_id: str | None,
    investigation_id: str | None,
    action: str | None,
    follow_up_context: dict[str, Any] | None,
) -> str | None:
    if isinstance(follow_up_context, dict):
        follow_up_request_id = _nonempty_str(follow_up_context.get("request_id"))
        if follow_up_request_id is not None:
            return follow_up_request_id

    if request_correlation_id is None:
        return None

    normalized_action = action if isinstance(action, str) and action.strip() else "request"
    normalized_investigation = (
        investigation_id if isinstance(investigation_id, str) and investigation_id.strip() else "unknown"
    )
    return f"req-{request_correlation_id}-{normalized_action}-{normalized_investigation}"


def _normalize_uncertainty_for_decision(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return [normalized]
    return value


def _inferred_tool_name_for_action(action: str | None) -> str | None:
    if action == "request_identity_snapshot":
        return "get_identity_snapshot"
    if action == "request_additional_tracking_evidence":
        return "get_track_evidence_history"
    return None


def _collect_semantic_selector(decision: dict[str, Any], key: str) -> Any:
    value = decision.get(key)
    if value is not None:
        return value
    target = decision.get("target")
    if isinstance(target, dict):
        return target.get(key)
    return None


def _collect_semantic_optional_bounds(decision: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}

    evidence_bounds = decision.get("evidence_bounds")
    if isinstance(evidence_bounds, dict):
        for key, value in evidence_bounds.items():
            if isinstance(key, str):
                arguments[key] = value

    for key in allowed_keys:
        if key in decision:
            arguments[key] = decision.get(key)

    return arguments


def _build_semantic_tool_arguments(action: str | None, decision: dict[str, Any]) -> dict[str, Any] | None:
    if action == "request_identity_snapshot":
        arguments: dict[str, Any] = {}

        app_identity_id = _collect_semantic_selector(decision, "app_identity_id")
        detection_reference = _collect_semantic_selector(decision, "detection_reference")
        if app_identity_id is not None:
            arguments["app_identity_id"] = app_identity_id
        if detection_reference is not None:
            arguments["detection_reference"] = detection_reference

        target = decision.get("target")
        if isinstance(target, dict):
            for key, value in target.items():
                if not isinstance(key, str):
                    continue
                if key in {"app_identity_id", "detection_reference"}:
                    continue
                arguments[key] = value

        arguments.update(
            _collect_semantic_optional_bounds(
                decision,
                {
                    "include_history",
                    "max_history_entries",
                    "max_candidates",
                    "min_source_generation",
                    "min_source_version",
                    "stale_after_seconds",
                    "investigation_id",
                    "correlation_id",
                },
            )
        )
        return arguments

    if action == "request_additional_tracking_evidence":
        arguments = {}
        track_id = _collect_semantic_selector(decision, "track_id")
        if track_id is not None:
            arguments["track_id"] = track_id

        target = decision.get("target")
        if isinstance(target, dict):
            for key, value in target.items():
                if not isinstance(key, str):
                    continue
                if key == "track_id":
                    continue
                arguments[key] = value

        arguments.update(
            _collect_semantic_optional_bounds(
                decision,
                {
                    "max_messages",
                    "max_history_records",
                    "max_scan_records",
                    "now_timestamp",
                    "investigation_id",
                    "correlation_id",
                },
            )
        )
        return arguments

    return None


def _normalize_compact_response_payload(
    parsed: dict[str, Any],
    *,
    request_correlation_id: str | None,
    follow_up_context: dict[str, Any] | None,
    detections: list[DetectionCandidate] | None,
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return parsed

    if isinstance(parsed.get("decision"), dict):
        output = dict(parsed)
        decision = dict(parsed.get("decision"))
        trusted_follow_up_investigation_id = (
            _nonempty_str(follow_up_context.get("investigation_id"))
            if isinstance(follow_up_context, dict)
            else None
        )
        trusted_follow_up_request_id = (
            _nonempty_str(follow_up_context.get("request_id"))
            if isinstance(follow_up_context, dict)
            else None
        )

        if trusted_follow_up_investigation_id is not None:
            decision["investigation_id"] = trusted_follow_up_investigation_id
        else:
            action = _nonempty_str(decision.get("action"))
            if _nonempty_str(decision.get("investigation_id")) is None and action in {
                "request_identity_snapshot",
                "request_additional_tracking_evidence",
            }:
                fallback_investigation_id = _runtime_fallback_investigation_id(
                    request_correlation_id=request_correlation_id,
                    follow_up_context=follow_up_context,
                    detections=detections,
                )
                if fallback_investigation_id is not None:
                    decision["investigation_id"] = fallback_investigation_id
        if _nonempty_str(decision.get("correlation_id")) is None and request_correlation_id is not None:
            decision["correlation_id"] = request_correlation_id

        decision_uncertainty = _normalize_uncertainty_for_decision(decision.get("uncertainty"))
        if decision_uncertainty is not None:
            decision["uncertainty"] = decision_uncertainty

        tool_requests = decision.get("tool_requests")
        if isinstance(tool_requests, list) and tool_requests:
            action = _nonempty_str(decision.get("action"))
            inferred_tool_name = _inferred_tool_name_for_action(action)
            runtime_investigation_id = _nonempty_str(decision.get("investigation_id"))
            fallback_request_id = _runtime_fallback_request_id(
                request_correlation_id=request_correlation_id,
                investigation_id=runtime_investigation_id,
                action=action,
                follow_up_context=follow_up_context,
            )
            normalized_requests: list[Any] = []
            for item in tool_requests:
                if not isinstance(item, dict):
                    normalized_requests.append(item)
                    continue

                explicit_tool_name = _nonempty_str(item.get("tool_name"))
                compact_tool_name = _nonempty_str(item.get("tool")) or _nonempty_str(item.get("name"))
                supplied_tool_name = explicit_tool_name or compact_tool_name
                if inferred_tool_name is not None:
                    if supplied_tool_name is not None and supplied_tool_name != inferred_tool_name:
                        tool_name = supplied_tool_name
                    else:
                        tool_name = inferred_tool_name
                else:
                    tool_name = supplied_tool_name

                arguments = item.get("arguments")
                if not isinstance(arguments, dict):
                    arguments = item.get("tool_arguments")
                if not isinstance(arguments, dict):
                    arguments = item.get("args")
                if not isinstance(arguments, dict):
                    arguments = item.get("parameters")

                normalized_requests.append(
                    {
                        "request_id": (
                            _nonempty_str(item.get("request_id"))
                            or _nonempty_str(item.get("id"))
                            or _nonempty_str(item.get("requestId"))
                            or trusted_follow_up_request_id
                            or fallback_request_id
                        ),
                        "tool_name": tool_name,
                        "arguments": arguments,
                    }
                )

            decision["tool_requests"] = normalized_requests
        else:
            action = _nonempty_str(decision.get("action"))
            inferred_tool_name = _inferred_tool_name_for_action(action)
            semantic_arguments = _build_semantic_tool_arguments(action, decision)
            explicit_semantic_tool_name = (
                _nonempty_str(decision.get("tool_name"))
                or _nonempty_str(decision.get("tool"))
                or _nonempty_str(decision.get("name"))
            )

            if action in {"request_identity_snapshot", "request_additional_tracking_evidence"} and (
                isinstance(semantic_arguments, dict) or explicit_semantic_tool_name is not None
            ):
                runtime_investigation_id = _nonempty_str(decision.get("investigation_id"))
                fallback_request_id = _runtime_fallback_request_id(
                    request_correlation_id=request_correlation_id,
                    investigation_id=runtime_investigation_id,
                    action=action,
                    follow_up_context=follow_up_context,
                )
                if inferred_tool_name is not None:
                    if explicit_semantic_tool_name is not None and explicit_semantic_tool_name != inferred_tool_name:
                        normalized_tool_name = explicit_semantic_tool_name
                    else:
                        normalized_tool_name = inferred_tool_name
                else:
                    normalized_tool_name = explicit_semantic_tool_name

                decision["tool_requests"] = [
                    {
                        "request_id": trusted_follow_up_request_id or fallback_request_id,
                        "tool_name": normalized_tool_name,
                        "arguments": semantic_arguments,
                    }
                ]

        output["decision"] = decision
        return output

    action = _nonempty_str(parsed.get("action"))
    if action is None:
        return parsed

    output: dict[str, Any] = {
        "proposals": parsed.get("proposals") if isinstance(parsed.get("proposals"), list) else [],
        "uncertainty_flags": parsed.get("uncertainty_flags") if isinstance(parsed.get("uncertainty_flags"), list) else [],
        "supporting_evidence": parsed.get("supporting_evidence") if isinstance(parsed.get("supporting_evidence"), list) else [],
    }

    if not output["uncertainty_flags"] and isinstance(parsed.get("uncertainty"), list):
        output["uncertainty_flags"] = parsed.get("uncertainty")
    if not output["supporting_evidence"] and isinstance(parsed.get("evidence"), list):
        output["supporting_evidence"] = parsed.get("evidence")

    if not output["proposals"] and isinstance(parsed.get("proposal"), dict):
        output["proposals"] = [parsed.get("proposal")]

    decision: dict[str, Any] = {
        "action": action,
    }

    for key in (
        "investigation_id",
        "correlation_id",
        "reason",
        "uncertainty",
        "evidence_references",
        "provenance",
        "snapshot_constraints",
    ):
        if key in parsed:
            decision[key] = parsed.get(key)

    if _nonempty_str(decision.get("correlation_id")) is None and request_correlation_id is not None:
        decision["correlation_id"] = request_correlation_id

    decision_uncertainty = _normalize_uncertainty_for_decision(decision.get("uncertainty"))
    if decision_uncertainty is not None:
        decision["uncertainty"] = decision_uncertainty

    trusted_follow_up_investigation_id = (
        _nonempty_str(follow_up_context.get("investigation_id"))
        if isinstance(follow_up_context, dict)
        else None
    )
    if trusted_follow_up_investigation_id is not None:
        decision["investigation_id"] = trusted_follow_up_investigation_id
    elif _nonempty_str(decision.get("investigation_id")) is None and action in {
        "request_identity_snapshot",
        "request_additional_tracking_evidence",
    }:
        fallback_investigation_id = _runtime_fallback_investigation_id(
            request_correlation_id=request_correlation_id,
            follow_up_context=follow_up_context,
            detections=detections,
        )
        if fallback_investigation_id is not None:
            decision["investigation_id"] = fallback_investigation_id

    tool_request = parsed.get("tool_request")
    if not isinstance(tool_request, dict):
        maybe_name = parsed.get("tool_name")
        if maybe_name is None:
            maybe_name = parsed.get("tool")
        if maybe_name is None:
            maybe_name = parsed.get("name")
        maybe_args = parsed.get("tool_arguments")
        if maybe_args is None:
            maybe_args = parsed.get("args")
        if maybe_args is None:
            maybe_args = parsed.get("parameters")
        maybe_request_id = parsed.get("request_id")
        if maybe_request_id is None:
            maybe_request_id = parsed.get("id")
        if maybe_request_id is None:
            maybe_request_id = parsed.get("requestId")
        if maybe_name is not None or maybe_args is not None or maybe_request_id is not None:
            tool_request = {
                "request_id": maybe_request_id,
                "tool_name": maybe_name,
                "arguments": maybe_args,
            }

    if isinstance(tool_request, dict):
        tool_name = _nonempty_str(tool_request.get("tool_name"))
        if tool_name is None:
            tool_name = _nonempty_str(tool_request.get("tool")) or _nonempty_str(tool_request.get("name"))
        inferred_tool_name = _inferred_tool_name_for_action(action)
        if inferred_tool_name is not None:
            if tool_name is not None and tool_name != inferred_tool_name:
                pass
            else:
                tool_name = inferred_tool_name
        elif tool_name is None:
            tool_name = inferred_tool_name

        runtime_investigation_id = _nonempty_str(decision.get("investigation_id"))
        fallback_request_id = _runtime_fallback_request_id(
            request_correlation_id=request_correlation_id,
            investigation_id=runtime_investigation_id,
            action=action,
            follow_up_context=follow_up_context,
        )
        trusted_follow_up_request_id = (
            _nonempty_str(follow_up_context.get("request_id"))
            if isinstance(follow_up_context, dict)
            else None
        )

        arguments = tool_request.get("arguments")
        if not isinstance(arguments, dict):
            arguments = tool_request.get("tool_arguments")
        if not isinstance(arguments, dict):
            arguments = tool_request.get("args")
        if not isinstance(arguments, dict):
            arguments = tool_request.get("parameters")
        if not isinstance(arguments, dict):
            semantic_arguments = _build_semantic_tool_arguments(action, parsed)
            if isinstance(semantic_arguments, dict):
                arguments = semantic_arguments

        decision["tool_requests"] = [
            {
                "request_id": (
                    _nonempty_str(tool_request.get("request_id"))
                    or _nonempty_str(tool_request.get("id"))
                    or _nonempty_str(tool_request.get("requestId"))
                    or trusted_follow_up_request_id
                    or fallback_request_id
                ),
                "tool_name": tool_name,
                "arguments": arguments,
            }
        ]
    elif action in {"request_identity_snapshot", "request_additional_tracking_evidence"}:
        semantic_arguments = _build_semantic_tool_arguments(action, parsed)
        explicit_semantic_tool_name = (
            _nonempty_str(parsed.get("tool_name"))
            or _nonempty_str(parsed.get("tool"))
            or _nonempty_str(parsed.get("name"))
        )
        inferred_tool_name = _inferred_tool_name_for_action(action)
        runtime_investigation_id = _nonempty_str(decision.get("investigation_id"))
        fallback_request_id = _runtime_fallback_request_id(
            request_correlation_id=request_correlation_id,
            investigation_id=runtime_investigation_id,
            action=action,
            follow_up_context=follow_up_context,
        )
        trusted_follow_up_request_id = (
            _nonempty_str(follow_up_context.get("request_id"))
            if isinstance(follow_up_context, dict)
            else None
        )

        if inferred_tool_name is not None:
            if explicit_semantic_tool_name is not None and explicit_semantic_tool_name != inferred_tool_name:
                normalized_tool_name = explicit_semantic_tool_name
            else:
                normalized_tool_name = inferred_tool_name
        else:
            normalized_tool_name = explicit_semantic_tool_name

        if isinstance(semantic_arguments, dict) or normalized_tool_name is not None:
            decision["tool_requests"] = [
                {
                    "request_id": trusted_follow_up_request_id or fallback_request_id,
                    "tool_name": normalized_tool_name,
                    "arguments": semantic_arguments,
                }
            ]

    output["decision"] = decision
    return output


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _validate_identity_snapshot_arguments(arguments: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_keys = {
        "app_identity_id",
        "detection_reference",
        "include_history",
        "max_history_entries",
        "max_candidates",
        "min_source_generation",
        "min_source_version",
        "stale_after_seconds",
        "investigation_id",
        "correlation_id",
    }
    unknown = sorted(key for key in arguments.keys() if key not in allowed_keys)
    if unknown:
        errors.append("decision_identity_snapshot_unknown_arguments")

    app_identity_id = _nonempty_str(arguments.get("app_identity_id"))
    detection_reference = _nonempty_str(arguments.get("detection_reference"))
    if int(app_identity_id is not None) + int(detection_reference is not None) != 1:
        errors.append("decision_identity_snapshot_selector_required")

    include_history = arguments.get("include_history")
    if include_history is not None and not isinstance(include_history, bool):
        errors.append("decision_identity_snapshot_include_history_invalid")

    for key in ("max_history_entries", "max_candidates"):
        value = arguments.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value <= 0:
            errors.append(f"decision_identity_snapshot_{key}_invalid")

    for key in ("min_source_generation", "min_source_version"):
        value = arguments.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value < 0:
            errors.append(f"decision_identity_snapshot_{key}_invalid")

    stale_after_seconds = arguments.get("stale_after_seconds")
    if stale_after_seconds is not None:
        if not isinstance(stale_after_seconds, (int, float)) or float(stale_after_seconds) <= 0:
            errors.append("decision_identity_snapshot_stale_after_seconds_invalid")

    for key in ("investigation_id", "correlation_id"):
        value = arguments.get(key)
        if value is None:
            continue
        if _nonempty_str(value) is None:
            errors.append(f"decision_identity_snapshot_{key}_invalid")

    return errors


def _validate_track_evidence_history_arguments(arguments: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    allowed_keys = {
        "track_id",
        "max_messages",
        "max_history_records",
        "max_scan_records",
        "now_timestamp",
        "investigation_id",
        "correlation_id",
    }
    unknown = sorted(key for key in arguments.keys() if key not in allowed_keys)
    if unknown:
        errors.append("decision_track_history_unknown_arguments")

    if _nonempty_str(arguments.get("track_id")) is None:
        errors.append("decision_track_history_track_id_required")

    for key in ("max_messages", "max_history_records", "max_scan_records"):
        value = arguments.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value <= 0:
            errors.append(f"decision_track_history_{key}_invalid")

    now_timestamp = arguments.get("now_timestamp")
    if now_timestamp is not None and not isinstance(now_timestamp, (int, float)):
        errors.append("decision_track_history_now_timestamp_invalid")

    for key in ("investigation_id", "correlation_id"):
        value = arguments.get(key)
        if value is None:
            continue
        if _nonempty_str(value) is None:
            errors.append(f"decision_track_history_{key}_invalid")

    return errors


def _validate_tracking_decision_payload(payload: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if payload is None:
        return None, []
    if not isinstance(payload, dict):
        return None, ["decision_payload_not_object"]

    errors: list[str] = []
    action = _nonempty_str(payload.get("action"))
    if action is None or action not in TRACKING_DECISION_ACTIONS:
        errors.append("decision_action_invalid")

    reason = _nonempty_str(payload.get("reason"))
    if action == "inconclusive" and reason is None:
        errors.append("decision_inconclusive_reason_required")

    investigation_id = _nonempty_str(payload.get("investigation_id"))
    correlation_id = _nonempty_str(payload.get("correlation_id"))
    if payload.get("correlation_id") is not None and correlation_id is None:
        errors.append("decision_correlation_id_invalid")

    uncertainty = payload.get("uncertainty")
    if uncertainty is not None:
        if not isinstance(uncertainty, list) or any(not isinstance(item, str) or not item.strip() for item in uncertainty):
            errors.append("decision_uncertainty_invalid")

    evidence_references = payload.get("evidence_references")
    if evidence_references is not None:
        if not isinstance(evidence_references, dict):
            errors.append("decision_evidence_references_invalid")
        else:
            for key in ("frame_reference", "detection_reference", "tracking_message_id"):
                value = evidence_references.get(key)
                if value is not None and _nonempty_str(value) is None:
                    errors.append(f"decision_{key}_invalid")

    provenance = payload.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            errors.append("decision_provenance_invalid")
        else:
            for key, value in provenance.items():
                if _nonempty_str(key) is None:
                    errors.append("decision_provenance_key_invalid")
                    break
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    errors.append("decision_provenance_value_invalid")
                    break

    snapshot_constraints = payload.get("snapshot_constraints")
    if snapshot_constraints is not None:
        if not isinstance(snapshot_constraints, dict):
            errors.append("decision_snapshot_constraints_invalid")
        else:
            min_generation = snapshot_constraints.get("min_source_generation")
            if min_generation is not None and (not isinstance(min_generation, int) or min_generation < 0):
                errors.append("decision_min_source_generation_invalid")
            min_version = snapshot_constraints.get("min_source_version")
            if min_version is not None and (not isinstance(min_version, int) or min_version < 0):
                errors.append("decision_min_source_version_invalid")

    tool_requests = payload.get("tool_requests")
    if tool_requests is None:
        tool_requests = []
    if not isinstance(tool_requests, list):
        errors.append("decision_tool_requests_invalid")
        tool_requests = []
    if len(tool_requests) > 1:
        errors.append("decision_tool_requests_exceeds_max")

    if action in {"request_identity_snapshot", "request_additional_tracking_evidence"}:
        if len(tool_requests) != 1:
            errors.append("decision_tool_request_required")
        if investigation_id is None:
            errors.append("decision_investigation_id_required")

    for tool_request in tool_requests:
        if not isinstance(tool_request, dict):
            errors.append("decision_tool_request_invalid")
            continue

        if _nonempty_str(tool_request.get("request_id")) is None:
            errors.append("decision_tool_request_id_required")

        tool_name = _nonempty_str(tool_request.get("tool_name"))
        if tool_name is None:
            errors.append("decision_tool_name_required")
            continue
        if tool_name not in TRACKING_DECISION_TOOL_ALLOWLIST:
            errors.append("decision_tool_name_not_allowed")
            continue

        arguments = tool_request.get("arguments")
        if not isinstance(arguments, dict):
            errors.append("decision_tool_arguments_invalid")
            continue

        if action == "request_identity_snapshot" and tool_name != "get_identity_snapshot":
            errors.append("decision_tool_action_mismatch")
        if action == "request_additional_tracking_evidence" and tool_name != "get_track_evidence_history":
            errors.append("decision_tool_action_mismatch")

        if tool_name == "get_identity_snapshot":
            errors.extend(_validate_identity_snapshot_arguments(arguments))
        elif tool_name == "get_track_evidence_history":
            errors.extend(_validate_track_evidence_history_arguments(arguments))

    if action in {"propose_identity_association", "inconclusive"} and len(tool_requests) > 0:
        errors.append("decision_tool_requests_not_allowed_for_action")

    if errors:
        return None, sorted(set(errors))

    return payload, []


def _is_actionable_proposal(item: Any) -> bool:
    if not isinstance(item, dict):
        return False

    proposed_track_id = item.get("proposed_track_id")
    if isinstance(proposed_track_id, str) and proposed_track_id.strip():
        return True

    proposed_app_identity_id = item.get("proposed_app_identity_id")
    if isinstance(proposed_app_identity_id, str) and proposed_app_identity_id.strip():
        return True

    return False


def _decision_disallows_actionable_proposals(decision_payload: dict[str, Any], raw_proposals: list[Any]) -> bool:
    action = _nonempty_str(decision_payload.get("action"))
    if action not in {
        "request_identity_snapshot",
        "request_additional_tracking_evidence",
        "inconclusive",
    }:
        return False
    return any(_is_actionable_proposal(item) for item in raw_proposals)


def _normalize_proposal(
    payload: dict,
    *,
    detection_by_id: dict[str, DetectionCandidate],
    identity_evidence_by_detection_id: dict[str, dict[str, object]] | None = None,
    evidence_snapshot_generation: int | None = None,
    evidence_snapshot_version: int | None = None,
    evidence_snapshot_available: bool | None = None,
    eligible_identity_ids: set[str] | None = None,
) -> VisualAssociationProposal | None:
    # Normalizes proposal into contract-safe values.
    detection_id = payload.get("detection_id")
    if not isinstance(detection_id, str):
        return None

    detection = detection_by_id.get(detection_id)

    proposed_track_id = payload.get("proposed_track_id")
    if proposed_track_id is not None and not isinstance(proposed_track_id, str):
        proposed_track_id = None

    association_confidence = payload.get("association_confidence")
    if not isinstance(association_confidence, (int, float)):
        association_confidence = None

    occlusion_likelihood = payload.get("occlusion_likelihood")
    if not isinstance(occlusion_likelihood, (int, float)):
        occlusion_likelihood = None

    id_switch_risk = payload.get("id_switch_risk")
    if not isinstance(id_switch_risk, (int, float)):
        id_switch_risk = None

    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list):
        uncertainties = []

    supporting_evidence = payload.get("supporting_evidence")
    if not isinstance(supporting_evidence, list):
        supporting_evidence = []

    detection_reference = payload.get("detection_reference")
    if not isinstance(detection_reference, str) or not detection_reference.strip():
        if detection is not None and isinstance(detection.detection_reference, str):
            detection_reference = detection.detection_reference
        else:
            detection_reference = None

    proposed_app_identity_id = payload.get("proposed_app_identity_id")
    if not isinstance(proposed_app_identity_id, str) or not proposed_app_identity_id.strip():
        proposed_app_identity_id = None

    rejected_app_identity_ids = payload.get("rejected_app_identity_ids")
    if not isinstance(rejected_app_identity_ids, list):
        rejected_app_identity_ids = []

    identity_evidence_refs = payload.get("identity_evidence_refs")
    if not isinstance(identity_evidence_refs, list):
        identity_evidence_refs = []

    identity_snapshot_generation = payload.get("identity_snapshot_generation")
    if not isinstance(identity_snapshot_generation, int):
        identity_snapshot_generation = None

    identity_snapshot_version = payload.get("identity_snapshot_version")
    if not isinstance(identity_snapshot_version, int):
        identity_snapshot_version = None

    identity_resolution_status = payload.get("identity_resolution_status")
    if not isinstance(identity_resolution_status, str) or not identity_resolution_status.strip():
        identity_resolution_status = None
    else:
        identity_resolution_status = identity_resolution_status.strip()

    identity_evidence_usable = False
    candidate_lookup_state: str | None = None
    detection_binding_valid = True
    candidate_ids: set[str] = set()

    if identity_evidence_by_detection_id is not None:
        identity_entry = identity_evidence_by_detection_id.get(detection_id)
        if isinstance(identity_entry, dict):
            identity_evidence_usable = bool(identity_entry.get("evidence_usable", False))
        lookup_payload = (
            identity_entry.get("lookup")
            if isinstance(identity_entry, dict)
            else None
        )
        if isinstance(lookup_payload, dict):
            state_value = lookup_payload.get("candidate_lookup_state")
            if isinstance(state_value, str) and state_value.strip():
                candidate_lookup_state = state_value.strip()

            candidates = lookup_payload.get("candidates")
            if isinstance(candidates, list):
                for item in candidates:
                    if not isinstance(item, dict):
                        continue
                    app_identity_id = item.get("app_identity_id")
                    if isinstance(app_identity_id, str) and app_identity_id.strip():
                        candidate_ids.add(app_identity_id)

            resolved_detection_reference = lookup_payload.get("resolved_detection_reference")
            if (
                isinstance(detection_reference, str)
                and detection_reference.strip()
                and isinstance(resolved_detection_reference, str)
                and resolved_detection_reference.strip()
                and detection_reference != resolved_detection_reference
            ):
                uncertainties.append("identity_claim_detection_reference_mismatch")
                identity_resolution_status = "unresolved_detection_reference_mismatch"
                detection_binding_valid = False

        if proposed_app_identity_id is not None:
            if evidence_snapshot_available is False:
                uncertainties.append("identity_claim_without_usable_evidence")
                proposed_app_identity_id = None
                identity_resolution_status = "unresolved_identity_evidence_unusable"
            elif (
                isinstance(eligible_identity_ids, set)
                and eligible_identity_ids
                and proposed_app_identity_id not in eligible_identity_ids
            ):
                uncertainties.append("identity_claim_unknown_app_identity_id")
                proposed_app_identity_id = None
                identity_resolution_status = "unresolved_unknown_app_identity_id"
            if candidate_lookup_state == "found":
                if proposed_app_identity_id not in candidate_ids:
                    uncertainties.append("identity_claim_unknown_app_identity_id")
                    proposed_app_identity_id = None
                    identity_resolution_status = "unresolved_unknown_app_identity_id"
            elif candidate_lookup_state is not None:
                uncertainties.append("identity_claim_lookup_state_invalid")
                proposed_app_identity_id = None
                identity_resolution_status = "unresolved_identity_evidence_unusable"
    elif proposed_app_identity_id is not None:
        uncertainties.append("identity_claim_without_usable_evidence")
        proposed_app_identity_id = None
        identity_resolution_status = "unresolved_identity_evidence_unusable"

    generation_conflict = (
        isinstance(evidence_snapshot_generation, int)
        and isinstance(identity_snapshot_generation, int)
        and identity_snapshot_generation != evidence_snapshot_generation
    )
    version_conflict = (
        isinstance(evidence_snapshot_version, int)
        and isinstance(identity_snapshot_version, int)
        and identity_snapshot_version != evidence_snapshot_version
    )

    if generation_conflict:
        uncertainties.append("identity_claim_snapshot_generation_mismatch")
        identity_resolution_status = "unresolved_snapshot_mismatch"

    if version_conflict:
        uncertainties.append("identity_claim_snapshot_version_mismatch")
        identity_resolution_status = "unresolved_snapshot_mismatch"

    can_backfill_snapshot = (
        identity_evidence_usable
        and detection_binding_valid
        and not generation_conflict
        and not version_conflict
    )
    if can_backfill_snapshot:
        if identity_snapshot_generation is None and isinstance(evidence_snapshot_generation, int):
            identity_snapshot_generation = evidence_snapshot_generation
        if identity_snapshot_version is None and isinstance(evidence_snapshot_version, int):
            identity_snapshot_version = evidence_snapshot_version

    return VisualAssociationProposal(
        detection_id=detection_id,
        proposed_track_id=proposed_track_id,
        association_confidence=None if association_confidence is None else float(association_confidence),
        occlusion_likelihood=None if occlusion_likelihood is None else float(occlusion_likelihood),
        id_switch_risk=None if id_switch_risk is None else float(id_switch_risk),
        depth_conflict_flag=bool(payload.get("depth_conflict_flag", False)),
        position_conflict_flag=bool(payload.get("position_conflict_flag", False)),
        uncertainties=[str(value) for value in uncertainties],
        supporting_evidence=[str(value) for value in supporting_evidence],
        detection_reference=detection_reference,
        proposed_app_identity_id=proposed_app_identity_id,
        rejected_app_identity_ids=[str(value) for value in rejected_app_identity_ids if isinstance(value, str)],
        identity_evidence_refs=[str(value) for value in identity_evidence_refs if isinstance(value, str)],
        identity_snapshot_generation=identity_snapshot_generation,
        identity_snapshot_version=identity_snapshot_version,
        identity_resolution_status=identity_resolution_status,
    )

