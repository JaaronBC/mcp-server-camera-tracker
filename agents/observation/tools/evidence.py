"""Observation MCP tools for tracking-evidence retrieval and visual verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...registry import mcp
from ..agent_observation import (
    ObservationVisualVerificationError,
    ObservationVisualVerificationRequest,
    verify_tracking_evidence_for_observation,
)
from ..evidence_consumption import (
    ObservationEvidenceConsumptionError,
    ObservationEvidenceRequest,
    consume_tracking_evidence_by_message_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT_DIR = PROJECT_ROOT / "captures" / "tracking_evidence"
DEFAULT_VISUAL_VERIFICATION_TIMEOUT_SECONDS = 6.0
DEFAULT_VISUAL_VERIFICATION_RETRY_LIMIT = 0


def _sanitize_details(payload: Any) -> Any:
    if isinstance(payload, dict):
        allowed_keys = {
            "upstream_code",
            "gate_errors",
            "missing_keys",
            "model",
            "attempts",
            "message_id",
            "observation_id",
            "referenced_message_id",
            "validation",
        }
        return {
            str(key): _sanitize_details(value)
            for key, value in payload.items()
            if str(key) in allowed_keys
        }
    if isinstance(payload, list):
        return [_sanitize_details(item) for item in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return str(payload)


def _safe_error_response(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "error_code": code,
        "error": message,
        "details": _sanitize_details(details or {}),
    }


def _sanitize_reference_validation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "reference_results": [],
            "all_references_valid": False,
            "pending_count": 0,
            "expired_count": 0,
            "missing_count": 0,
            "no_substitution_policy": "strict_reference_only",
        }

    raw_results = payload.get("reference_results")
    sanitized_results: list[dict[str, Any]] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            result = item.get("result")
            status = result.get("status") if isinstance(result, dict) else None
            valid = bool(result.get("valid", False)) if isinstance(result, dict) else False
            sanitized_results.append(
                {
                    "reference": item.get("reference"),
                    "kind": item.get("kind"),
                    "result": {
                        "valid": valid,
                        "status": status,
                    },
                }
            )

    return {
        "reference_results": sanitized_results,
        "all_references_valid": bool(payload.get("all_references_valid", False)),
        "pending_count": int(payload.get("pending_count", 0)),
        "expired_count": int(payload.get("expired_count", 0)),
        "missing_count": int(payload.get("missing_count", 0)),
        "no_substitution_policy": str(payload.get("no_substitution_policy", "strict_reference_only")),
    }


@mcp.tool()
def get_tracking_visual_evidence(message_id: str) -> dict[str, Any]:
    try:
        package = consume_tracking_evidence_by_message_id(
            ObservationEvidenceRequest(
                message_id=message_id,
                require_visual_evidence=True,
            ),
            evidence_root_dir=EVIDENCE_ROOT_DIR,
        )
    except ObservationEvidenceConsumptionError as error:
        return _safe_error_response(
            "evidence_unavailable",
            "Unable to retrieve tracking visual evidence",
            details={
                "upstream_code": error.code,
                "gate_errors": (
                    error.details.get("gate_errors")
                    if isinstance(error.details, dict)
                    else None
                ),
                "message_id": message_id,
            },
        )

    image_evidence = package.image_evidence

    return {
        "available": True,
        "message_id": package.message_id,
        "track_id": package.envelope.track_id,
        "timestamp": package.envelope.timestamp,
        "frame_reference": package.envelope.frame_reference,
        "source_references": list(package.source_references),
        "preflight": {
            "preflight_passed": bool(package.preflight.get("preflight_passed", False)),
            "visual_evidence_inspectable": bool(package.preflight.get("visual_evidence_inspectable", False)),
            "gate_errors": list(package.preflight.get("gate_errors", [])),
            "evidence_reference_validation": _sanitize_reference_validation(
                package.preflight.get("evidence_reference_validation", {})
            ),
        },
        "measurement_provenance": package.measurement_provenance,
        "image_evidence": {
            "available": image_evidence is not None,
            "reference": (None if image_evidence is None else image_evidence.reference),
            "timestamp": (None if image_evidence is None else image_evidence.timestamp),
            "width": (None if image_evidence is None else image_evidence.width),
            "height": (None if image_evidence is None else image_evidence.height),
            "status": (None if image_evidence is None else image_evidence.status),
        },
        "bounded_history": [record.to_dict() for record in package.bounded_history],
    }


@mcp.tool()
def verify_tracking_visual_evidence(message_id: str) -> dict[str, Any]:
    try:
        result = verify_tracking_evidence_for_observation(
            ObservationVisualVerificationRequest(message_id=message_id),
            evidence_root_dir=EVIDENCE_ROOT_DIR,
            timeout_seconds=DEFAULT_VISUAL_VERIFICATION_TIMEOUT_SECONDS,
            retry_limit=DEFAULT_VISUAL_VERIFICATION_RETRY_LIMIT,
        )
    except ObservationVisualVerificationError as error:
        return _safe_error_response(
            error.code,
            "Visual verification failed",
            details=(error.details if isinstance(error.details, dict) else {}),
        )

    return {
        "available": True,
        "message_id": message_id,
        "envelope": result.envelope.to_dict(),
        "validation": result.validation,
        "llm_meta": {
            "status": result.llm_meta.get("status"),
            "model": result.llm_meta.get("model"),
            "latency_ms": result.llm_meta.get("latency_ms"),
            "error_code": result.llm_meta.get("error_code"),
            "attempts": result.llm_meta.get("attempts"),
            "retry_count": result.llm_meta.get("retry_count"),
        },
        "preflight": {
            "preflight_passed": bool(result.preflight.get("preflight_passed", False)),
            "visual_evidence_inspectable": bool(result.preflight.get("visual_evidence_inspectable", False)),
            "gate_errors": list(result.preflight.get("gate_errors", [])),
        },
        "measurement_provenance": result.measurement_provenance,
    }
