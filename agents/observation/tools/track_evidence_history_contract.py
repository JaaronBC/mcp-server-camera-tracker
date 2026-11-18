from __future__ import annotations

from typing import Any

TRACK_EVIDENCE_HISTORY_MODEL_SELECTABLE_ARGUMENTS = {
    "max_messages": "positive_integer",
    "max_history_records": "positive_integer",
    "max_scan_records": "positive_integer",
}

TRACK_EVIDENCE_HISTORY_TRUSTED_METADATA_ARGUMENTS = {
    "track_id": "authoritative_context_string",
    "now_timestamp": "number",
    "investigation_id": "nonempty_string",
    "correlation_id": "nonempty_string",
}

TRACK_EVIDENCE_HISTORY_SUPPORTED_ARGUMENTS = frozenset(
    {
        *TRACK_EVIDENCE_HISTORY_MODEL_SELECTABLE_ARGUMENTS.keys(),
        *TRACK_EVIDENCE_HISTORY_TRUSTED_METADATA_ARGUMENTS.keys(),
    }
)


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def validate_track_evidence_history_arguments(
    arguments: Any,
    *,
    require_track_id: bool = True,
    validate_track_id_value: bool = True,
) -> dict[str, Any]:
    requested_arguments = dict(arguments) if isinstance(arguments, dict) else {}
    errors: list[str] = []

    if not isinstance(arguments, dict):
        errors.append("track_evidence_history_arguments_invalid")

    unknown_arguments = sorted(
        key for key in requested_arguments.keys() if key not in TRACK_EVIDENCE_HISTORY_SUPPORTED_ARGUMENTS
    )
    if unknown_arguments:
        errors.append("track_evidence_history_unknown_arguments")

    if require_track_id:
        track_id_value = requested_arguments.get("track_id")
        if track_id_value is None:
            errors.append("track_evidence_history_track_id_required")
        elif validate_track_id_value and _nonempty_str(track_id_value) is None:
            errors.append("track_evidence_history_track_id_required")

    for key in TRACK_EVIDENCE_HISTORY_MODEL_SELECTABLE_ARGUMENTS.keys():
        value = requested_arguments.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value <= 0:
            errors.append(f"track_evidence_history_{key}_invalid")

    now_timestamp = requested_arguments.get("now_timestamp")
    if now_timestamp is not None and not isinstance(now_timestamp, (int, float)):
        errors.append("track_evidence_history_now_timestamp_invalid")

    for key in ("investigation_id", "correlation_id"):
        value = requested_arguments.get(key)
        if value is None:
            continue
        if _nonempty_str(value) is None:
            errors.append(f"track_evidence_history_{key}_invalid")

    deduped_errors = sorted(set(errors))
    return {
        "valid": not deduped_errors,
        "errors": deduped_errors,
        "error_code": None if not deduped_errors else "invalid_tool_arguments",
        "unknown_arguments": unknown_arguments,
        "original_requested_arguments": requested_arguments,
    }


def build_track_evidence_history_argument_diagnostics(
    *,
    validation: dict[str, Any],
    request_id: str | None,
    investigation_id: str | None,
    correlation_id: str | None,
) -> dict[str, Any]:
    return {
        "error_code": "invalid_tool_arguments",
        "validation_errors": list(validation.get("errors", [])),
        "unknown_arguments": list(validation.get("unknown_arguments", [])),
        "original_requested_arguments": dict(validation.get("original_requested_arguments", {})),
        "request_id": request_id,
        "investigation_id": investigation_id,
        "correlation_id": correlation_id,
    }
