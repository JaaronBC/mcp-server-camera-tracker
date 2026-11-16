"""Observation MCP tool for bounded read-only track evidence history retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...registry import mcp
from ..evidence_consumption import ObservationEvidenceConsumptionError, load_validated_envelope_by_message_id
from ...tracking.evidence_store import TrackingEvidenceResolver

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT_DIR = PROJECT_ROOT / "captures" / "tracking_evidence"

DEFAULT_MAX_MESSAGES = 5
MAX_MESSAGES_LIMIT = 20
DEFAULT_MAX_HISTORY_RECORDS = 10
MAX_HISTORY_RECORDS_LIMIT = 50
DEFAULT_MAX_SCAN_RECORDS = 120
MAX_SCAN_RECORDS_LIMIT = 500


def _bounded_int(value: int, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _resolve_reference_statuses(
    *,
    envelope: Any,
    resolver: TrackingEvidenceResolver,
    now_timestamp: float | None,
) -> dict[str, Any]:
    statuses: list[dict[str, Any]] = []

    frame_reference = envelope.frame_reference if isinstance(envelope.frame_reference, str) else None
    if isinstance(frame_reference, str) and frame_reference.strip():
        result = resolver.resolve_reference(frame_reference, now_timestamp=now_timestamp)
        statuses.append(
            {
                "reference": frame_reference,
                "kind": "frame",
                "status": result.get("status"),
                "valid": bool(result.get("valid", False)),
            }
        )

    source_refs = list(envelope.source_evidence_refs) if isinstance(envelope.source_evidence_refs, list) else []
    for source_ref in source_refs:
        if not isinstance(source_ref, str) or not source_ref.strip():
            continue
        if source_ref == frame_reference:
            continue
        result = resolver.resolve_reference(source_ref, now_timestamp=now_timestamp)
        statuses.append(
            {
                "reference": source_ref,
                "kind": "detection" if source_ref.startswith("evidence://detection/") else "other",
                "status": result.get("status"),
                "valid": bool(result.get("valid", False)),
            }
        )

    expired_count = sum(1 for item in statuses if item.get("status") == "expired_reference")
    missing_count = sum(1 for item in statuses if item.get("status") in {"missing_reference", "missing_frame_payload"})
    pending_count = sum(1 for item in statuses if item.get("status") == "pending_reference")
    ok_count = sum(1 for item in statuses if item.get("status") == "ok")

    aggregate_state = "ok"
    if statuses and ok_count == 0 and expired_count > 0 and missing_count == 0:
        aggregate_state = "expired"
    elif statuses and ok_count == 0 and missing_count > 0 and expired_count == 0:
        aggregate_state = "missing"
    elif statuses and ok_count == 0 and (expired_count > 0 or missing_count > 0):
        aggregate_state = "expired_or_missing"
    elif statuses and pending_count > 0 and ok_count == 0:
        aggregate_state = "pending"

    return {
        "statuses": statuses,
        "summary": {
            "ok_count": ok_count,
            "expired_count": expired_count,
            "missing_count": missing_count,
            "pending_count": pending_count,
            "aggregate_state": aggregate_state,
        },
    }


@mcp.tool()
def get_track_evidence_history(
    track_id: str,
    max_messages: int = DEFAULT_MAX_MESSAGES,
    max_history_records: int = DEFAULT_MAX_HISTORY_RECORDS,
    max_scan_records: int = DEFAULT_MAX_SCAN_RECORDS,
    now_timestamp: float | None = None,
    investigation_id: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    normalized_track_id = track_id.strip() if isinstance(track_id, str) else ""
    if not normalized_track_id:
        return {
            "available": False,
            "state": "invalid_request",
            "error": "track_id_required",
            "mutation_authorized": False,
            "mutation_authorization_reason": "read_only_track_evidence_history_tool",
        }

    selected_max_messages = _bounded_int(
        max_messages,
        default=DEFAULT_MAX_MESSAGES,
        minimum=1,
        maximum=MAX_MESSAGES_LIMIT,
    )
    selected_max_history_records = _bounded_int(
        max_history_records,
        default=DEFAULT_MAX_HISTORY_RECORDS,
        minimum=1,
        maximum=MAX_HISTORY_RECORDS_LIMIT,
    )
    selected_max_scan_records = _bounded_int(
        max_scan_records,
        default=DEFAULT_MAX_SCAN_RECORDS,
        minimum=1,
        maximum=MAX_SCAN_RECORDS_LIMIT,
    )

    envelopes_dir = EVIDENCE_ROOT_DIR / "envelopes"
    if not EVIDENCE_ROOT_DIR.exists() or not envelopes_dir.exists() or not envelopes_dir.is_dir():
        return {
            "available": False,
            "state": "storage_unavailable",
            "error": "evidence_storage_unavailable",
            "track_id": normalized_track_id,
            "evidence_root_dir": str(EVIDENCE_ROOT_DIR),
            "envelopes_dir": str(envelopes_dir),
            "mutation_authorized": False,
            "mutation_authorization_reason": "read_only_track_evidence_history_tool",
        }

    try:
        envelope_paths = [path for path in envelopes_dir.glob("*.json") if path.is_file()]
    except OSError:
        return {
            "available": False,
            "state": "storage_unavailable",
            "error": "evidence_storage_unreadable",
            "track_id": normalized_track_id,
            "evidence_root_dir": str(EVIDENCE_ROOT_DIR),
            "envelopes_dir": str(envelopes_dir),
            "mutation_authorized": False,
            "mutation_authorization_reason": "read_only_track_evidence_history_tool",
        }

    envelope_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    resolver = TrackingEvidenceResolver(EVIDENCE_ROOT_DIR)
    scanned_records = 0
    skipped_invalid_records = 0
    entries: list[dict[str, Any]] = []

    for path in envelope_paths:
        if scanned_records >= selected_max_scan_records:
            break
        scanned_records += 1
        message_id = path.stem
        try:
            envelope_record = load_validated_envelope_by_message_id(
                message_id,
                evidence_root_dir=EVIDENCE_ROOT_DIR,
            )
        except ObservationEvidenceConsumptionError:
            skipped_invalid_records += 1
            continue

        envelope = envelope_record.envelope
        if envelope.track_id != normalized_track_id:
            continue

        reference_statuses = _resolve_reference_statuses(
            envelope=envelope,
            resolver=resolver,
            now_timestamp=now_timestamp,
        )

        history_records = [record.to_dict() for record in envelope.track_history[-selected_max_history_records:]]
        entries.append(
            {
                "message_id": envelope.message_id,
                "track_id": envelope.track_id,
                "timestamp": envelope.timestamp,
                "frame_reference": envelope.frame_reference,
                "source_references": list(envelope.source_evidence_refs),
                "history_record_count": len(history_records),
                "bounded_history": history_records,
                "reference_statuses": reference_statuses,
                "provenance": {
                    "source": "tracking_evidence_envelope",
                    "validation_mode": "validated_envelope_only",
                    "retrieved_via": "bounded_envelope_scan",
                },
            }
        )
        if len(entries) >= selected_max_messages:
            break

    entries.sort(key=lambda item: float(item.get("timestamp", 0.0)), reverse=True)

    evidence_state = "recorded_observation"
    if not entries:
        evidence_state = "no_recorded_observation"
    elif all(
        isinstance(item.get("reference_statuses"), dict)
        and isinstance(item["reference_statuses"].get("summary"), dict)
        and item["reference_statuses"]["summary"].get("aggregate_state") == "expired"
        for item in entries
    ):
        evidence_state = "expired_evidence"
    elif all(
        isinstance(item.get("reference_statuses"), dict)
        and isinstance(item["reference_statuses"].get("summary"), dict)
        and item["reference_statuses"]["summary"].get("aggregate_state") == "missing"
        for item in entries
    ):
        evidence_state = "missing_evidence"

    response: dict[str, Any] = {
        "available": True,
        "state": evidence_state,
        "track_id": normalized_track_id,
        "messages": entries,
        "message_count": len(entries),
        "work": {
            "scanned_records": scanned_records,
            "scan_limit": selected_max_scan_records,
            "scan_limit_reached": scanned_records >= selected_max_scan_records,
            "skipped_invalid_records": skipped_invalid_records,
        },
        "bounds": {
            "max_messages": selected_max_messages,
            "max_history_records": selected_max_history_records,
            "max_scan_records": selected_max_scan_records,
        },
        "provenance": {
            "source": "tracking_evidence_storage",
            "retrieval_mode": "bounded_scan_no_track_index",
            "indexed_track_lookup_available": False,
            "occlusion_inference_performed": False,
            "note": "Missing/expired evidence is not interpreted as occlusion proof.",
        },
        "evidence_root_dir": str(EVIDENCE_ROOT_DIR),
        "envelopes_dir": str(envelopes_dir),
        "mutation_authorized": False,
        "mutation_authorization_reason": "read_only_track_evidence_history_tool",
    }
    if isinstance(investigation_id, str) and investigation_id.strip():
        response["investigation_id"] = investigation_id.strip()
    if isinstance(correlation_id, str) and correlation_id.strip():
        response["correlation_id"] = correlation_id.strip()
    return response
