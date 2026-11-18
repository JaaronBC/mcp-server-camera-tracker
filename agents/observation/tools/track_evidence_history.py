"""Observation MCP tool for bounded read-only track evidence history retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ...registry import mcp
from ..evidence_consumption import ObservationEvidenceConsumptionError, load_validated_envelope_by_message_id
from ...tracking.evidence_store import TrackingEvidenceResolver

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT_DIR = PROJECT_ROOT / "captures" / "tracking_evidence"
ADVISORY_STORE_PATH = PROJECT_ROOT / "captures" / "latest_tracking_prediction_detection_advisories.json"

DEFAULT_MAX_MESSAGES = 5
MAX_MESSAGES_LIMIT = 20
DEFAULT_MAX_HISTORY_RECORDS = 10
MAX_HISTORY_RECORDS_LIMIT = 50
DEFAULT_MAX_SCAN_RECORDS = 120
MAX_SCAN_RECORDS_LIMIT = 500
DEFAULT_MAX_ADVISORY_EVIDENCE = 5
MAX_ADVISORY_EVIDENCE_LIMIT = 20
ADVISORY_STALE_SECONDS = 3.0


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


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _advisory_store_payload() -> dict[str, Any] | None:
    if not ADVISORY_STORE_PATH.exists() or not ADVISORY_STORE_PATH.is_file():
        return None
    try:
        payload = json.loads(ADVISORY_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _extract_track_identity_ids(entries: list[dict[str, Any]]) -> set[str]:
    identity_ids: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        identity_claim = item.get("identity_claim")
        if isinstance(identity_claim, dict):
            identity_id = _nonempty_str(identity_claim.get("app_identity_id"))
            if identity_id is not None:
                identity_ids.add(identity_id)
    return identity_ids


def _observation_motion_consistency_assessment(
    *,
    entries: list[dict[str, Any]],
    advisory_evidence: dict[str, Any],
) -> dict[str, Any]:
    supporting_observation_ids: list[str] = []
    supporting_message_ids: list[str] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        message_id = _nonempty_str(entry.get("message_id"))
        if message_id is not None:
            supporting_message_ids.append(message_id)
        bounded_history = entry.get("bounded_history")
        if not isinstance(bounded_history, list):
            continue
        for record in bounded_history:
            if not isinstance(record, dict):
                continue
            observation_id = _nonempty_str(record.get("observation_id"))
            if observation_id is not None:
                supporting_observation_ids.append(observation_id)

    unique_observation_ids = sorted(set(supporting_observation_ids))
    unique_message_ids = sorted(set(supporting_message_ids))
    advisory_ambiguous_count = int(advisory_evidence.get("ambiguous_count", 0)) if isinstance(advisory_evidence, dict) else 0
    advisory_matched_count = int(advisory_evidence.get("matched_count", 0)) if isinstance(advisory_evidence, dict) else 0

    uncertainty: list[str] = []
    if len(unique_observation_ids) < 2:
        uncertainty.append("insufficient_history")

    if advisory_ambiguous_count > 0:
        uncertainty.append("advisory_ambiguity")
        return {
            "available": True,
            "state": "advisory_ambiguous",
            "motion_consistency": "ambiguous",
            "supporting_observation_ids": unique_observation_ids,
            "supporting_tracking_message_ids": unique_message_ids,
            "matched_advisory_count": advisory_matched_count,
            "ambiguous_advisory_count": advisory_ambiguous_count,
            "uncertainty": uncertainty,
            "provenance": {
                "source": "observation_track_evidence_history",
                "method": "bounded_history_and_advisory_consistency",
                "advisory_evidence_used": True,
                "identity_verification_authorized": False,
                "identity_mutation_authorized": False,
            },
        }

    if len(unique_observation_ids) < 2:
        return {
            "available": False,
            "state": "insufficient_history",
            "motion_consistency": "insufficient_evidence",
            "supporting_observation_ids": unique_observation_ids,
            "supporting_tracking_message_ids": unique_message_ids,
            "matched_advisory_count": advisory_matched_count,
            "ambiguous_advisory_count": advisory_ambiguous_count,
            "uncertainty": uncertainty,
            "provenance": {
                "source": "observation_track_evidence_history",
                "method": "bounded_history_and_advisory_consistency",
                "advisory_evidence_used": advisory_matched_count > 0,
                "identity_verification_authorized": False,
                "identity_mutation_authorized": False,
            },
        }

    state = "history_consistent"
    if advisory_matched_count > 0:
        state = "history_consistent_with_advisory_match"

    return {
        "available": True,
        "state": state,
        "motion_consistency": "consistent",
        "supporting_observation_ids": unique_observation_ids,
        "supporting_tracking_message_ids": unique_message_ids,
        "matched_advisory_count": advisory_matched_count,
        "ambiguous_advisory_count": advisory_ambiguous_count,
        "uncertainty": uncertainty,
        "provenance": {
            "source": "observation_track_evidence_history",
            "method": "bounded_history_and_advisory_consistency",
            "advisory_evidence_used": advisory_matched_count > 0,
            "identity_verification_authorized": False,
            "identity_mutation_authorized": False,
        },
    }


def _advisory_evidence_for_track(
    *,
    track_id: str,
    track_identity_ids: set[str],
    now_timestamp: float | None,
    max_records: int,
) -> dict[str, Any]:
    payload = _advisory_store_payload()
    if payload is None:
        return {
            "available": False,
            "state": "advisory_store_unavailable",
            "advisory_count": 0,
            "matched_count": 0,
            "ambiguous_count": 0,
            "rejected_count": 0,
            "summary": {},
            "advisories": [],
            "provenance": {
                "source": "tracking_prediction_detection_advisory_store",
                "retrieval_mode": "bounded_read_only_filter",
                "identity_verification_authorized": False,
                "identity_mutation_authorized": False,
            },
        }

    advisories_payload = payload.get("advisories")
    advisories = list(advisories_payload) if isinstance(advisories_payload, list) else []
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    updated_at = payload.get("updated_at") if isinstance(payload.get("updated_at"), (int, float)) else None

    selected: list[dict[str, Any]] = []
    rejected_count = 0

    for advisory in advisories:
        if not isinstance(advisory, dict):
            rejected_count += 1
            continue

        decision = _nonempty_str(advisory.get("decision"))
        app_identity_id = _nonempty_str(advisory.get("app_identity_id"))
        candidate_yolo_track_id = _nonempty_str(advisory.get("candidate_yolo_track_id"))
        source_yolo_track_id = _nonempty_str(advisory.get("source_yolo_track_id"))
        advisory_created_at = advisory.get("advisory_created_at")
        expires_at = advisory.get("expires_at")
        predicted_for_timestamp = advisory.get("predicted_for_timestamp")

        if (
            decision not in {"matched", "ambiguous"}
            or app_identity_id is None
            or candidate_yolo_track_id is None
            or source_yolo_track_id is None
            or not isinstance(advisory_created_at, (int, float))
            or not isinstance(expires_at, (int, float))
            or not isinstance(predicted_for_timestamp, (int, float))
        ):
            rejected_count += 1
            continue

        if isinstance(now_timestamp, (int, float)):
            if float(expires_at) <= float(now_timestamp):
                rejected_count += 1
                continue
            if abs(float(now_timestamp) - float(advisory_created_at)) > ADVISORY_STALE_SECONDS:
                rejected_count += 1
                continue

        matches_track = candidate_yolo_track_id == track_id or source_yolo_track_id == track_id
        matches_identity = app_identity_id in track_identity_ids
        if not matches_track and not matches_identity:
            rejected_count += 1
            continue

        selected.append(
            {
                "decision": decision,
                "reason": advisory.get("reason"),
                "classification": advisory.get("classification"),
                "app_identity_id": app_identity_id,
                "candidate_yolo_track_id": candidate_yolo_track_id,
                "source_yolo_track_id": source_yolo_track_id,
                "match_metric": advisory.get("match_metric"),
                "match_distance": advisory.get("match_distance"),
                "match_threshold": advisory.get("match_threshold"),
                "normalized_score": advisory.get("normalized_score"),
                "predicted_for_timestamp": float(predicted_for_timestamp),
                "expires_at": float(expires_at),
                "advisory_created_at": float(advisory_created_at),
                "source_identity_registry_generation": advisory.get("source_identity_registry_generation"),
                "source_identity_registry_version": advisory.get("source_identity_registry_version"),
                "source_tracking_message_ids": advisory.get("source_tracking_message_ids"),
                "source_verification_observation_id": advisory.get("source_verification_observation_id"),
                "source_verification_message_id": advisory.get("source_verification_message_id"),
            }
        )

    selected.sort(
        key=lambda item: (
            0 if item.get("decision") == "matched" else 1,
            float(item.get("normalized_score", 1.0) or 1.0),
        )
    )
    bounded = selected[:max_records]

    matched_count = sum(1 for item in bounded if item.get("decision") == "matched")
    ambiguous_count = sum(1 for item in bounded if item.get("decision") == "ambiguous")
    if not bounded:
        state = "no_current_advisory"
    elif ambiguous_count > 0 and matched_count == 0:
        state = "advisory_ambiguous"
    elif ambiguous_count > 0:
        state = "advisory_matched_and_ambiguous"
    else:
        state = "advisory_matched"

    return {
        "available": bool(bounded),
        "state": state,
        "advisory_count": len(bounded),
        "matched_count": matched_count,
        "ambiguous_count": ambiguous_count,
        "rejected_count": rejected_count,
        "summary": dict(summary),
        "advisories": bounded,
        "updated_at": updated_at,
        "provenance": {
            "source": "tracking_prediction_detection_advisory_store",
            "retrieval_mode": "bounded_read_only_filter",
            "classification_preserved": True,
            "identity_verification_authorized": False,
            "identity_mutation_authorized": False,
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
    selected_max_advisories = _bounded_int(
        selected_max_messages,
        default=DEFAULT_MAX_ADVISORY_EVIDENCE,
        minimum=1,
        maximum=MAX_ADVISORY_EVIDENCE_LIMIT,
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
                "identity_claim": (
                    dict(envelope.tracking_claims.identity_claim)
                    if isinstance(envelope.tracking_claims.identity_claim, dict)
                    else None
                ),
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

    track_identity_ids = _extract_track_identity_ids(entries)
    advisory_evidence = _advisory_evidence_for_track(
        track_id=normalized_track_id,
        track_identity_ids=track_identity_ids,
        now_timestamp=now_timestamp,
        max_records=selected_max_advisories,
    )
    motion_consistency_assessment = _observation_motion_consistency_assessment(
        entries=entries,
        advisory_evidence=advisory_evidence,
    )

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
            "max_advisory_records": selected_max_advisories,
        },
        "identity_context": {
            "app_identity_ids": sorted(track_identity_ids),
            "count": len(track_identity_ids),
        },
        "prediction_detection_advisory_evidence": advisory_evidence,
        "observation_motion_consistency_assessment": motion_consistency_assessment,
        "provenance": {
            "source": "tracking_evidence_storage",
            "retrieval_mode": "bounded_scan_no_track_index",
            "indexed_track_lookup_available": False,
            "occlusion_inference_performed": False,
            "note": "Missing/expired evidence is not interpreted as occlusion proof.",
            "advisory_evidence_authorizes_identity_verification": False,
            "advisory_evidence_authorizes_identity_mutation": False,
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
