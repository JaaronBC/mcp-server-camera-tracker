"""Read-only orchestration MCP tool for identity snapshot lookups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...registry import mcp
from ..runtime.identity_snapshot_service import (
    DEFAULT_EVIDENCE_ROOT_DIR,
    DEFAULT_SNAPSHOT_PATH,
    _bounded_positive_int,
    find_identity_candidates,
    get_identity_history,
    get_identity_status,
)


def _optional_nonnegative_int(value: int | None, *, field: str) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None, f"{field}_must_be_integer"
    if parsed < 0:
        return None, f"{field}_must_be_nonnegative"
    return parsed, None


def _with_metadata(result: dict[str, Any], *, investigation_id: str | None, correlation_id: str | None) -> dict[str, Any]:
    output = dict(result)
    if isinstance(investigation_id, str) and investigation_id.strip():
        output["investigation_id"] = investigation_id.strip()
    if isinstance(correlation_id, str) and correlation_id.strip():
        output["correlation_id"] = correlation_id.strip()
    output["mutation_authorized"] = False
    output["mutation_authorization_reason"] = "read_only_identity_snapshot_tool"
    return output


def _compatibility_report(
    *,
    source_generation: Any,
    source_version: Any,
    min_source_generation: int | None,
    min_source_version: int | None,
) -> dict[str, Any]:
    incompatibilities: list[str] = []

    generation_value = int(source_generation) if isinstance(source_generation, int) else None
    version_value = int(source_version) if isinstance(source_version, int) else None

    if min_source_generation is not None:
        if generation_value is None:
            incompatibilities.append("snapshot_generation_missing")
        elif generation_value < min_source_generation:
            incompatibilities.append("snapshot_generation_too_old")

    if min_source_version is not None:
        if version_value is None:
            incompatibilities.append("snapshot_version_missing")
        elif version_value < min_source_version:
            incompatibilities.append("snapshot_version_too_old")

    return {
        "compatible": len(incompatibilities) == 0,
        "requirements": {
            "min_source_generation": min_source_generation,
            "min_source_version": min_source_version,
        },
        "snapshot": {
            "source_generation": generation_value,
            "source_version": version_value,
        },
        "incompatibilities": incompatibilities,
    }


@mcp.tool()
def get_identity_snapshot(
    app_identity_id: str | None = None,
    detection_reference: str | None = None,
    include_history: bool = False,
    max_history_entries: int = 20,
    max_candidates: int = 8,
    min_source_generation: int | None = None,
    min_source_version: int | None = None,
    stale_after_seconds: float | None = None,
    investigation_id: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    normalized_identity_id = app_identity_id.strip() if isinstance(app_identity_id, str) else ""
    normalized_detection_reference = detection_reference.strip() if isinstance(detection_reference, str) else ""

    lookup_mode_count = int(bool(normalized_identity_id)) + int(bool(normalized_detection_reference))
    if lookup_mode_count != 1:
        return _with_metadata(
            {
                "available": False,
                "state": "invalid_request",
                "error": "exactly_one_lookup_selector_required",
                "details": {
                    "required": "Provide exactly one of app_identity_id or detection_reference",
                },
                "provenance": {
                    "source": "identity_snapshot_service",
                    "resolution_mode": "read_only_snapshot_lookup",
                },
            },
            investigation_id=investigation_id,
            correlation_id=correlation_id,
        )

    selected_max_history_entries = _bounded_positive_int(max_history_entries, default=20, minimum=1, maximum=256)
    selected_max_candidates = _bounded_positive_int(max_candidates, default=8, minimum=1, maximum=128)

    parsed_min_generation, min_generation_error = _optional_nonnegative_int(
        min_source_generation,
        field="min_source_generation",
    )
    if min_generation_error is not None:
        return _with_metadata(
            {
                "available": False,
                "state": "invalid_request",
                "error": min_generation_error,
            },
            investigation_id=investigation_id,
            correlation_id=correlation_id,
        )

    parsed_min_version, min_version_error = _optional_nonnegative_int(
        min_source_version,
        field="min_source_version",
    )
    if min_version_error is not None:
        return _with_metadata(
            {
                "available": False,
                "state": "invalid_request",
                "error": min_version_error,
            },
            investigation_id=investigation_id,
            correlation_id=correlation_id,
        )

    base_provenance = {
        "source": "identity_snapshot_service",
        "lookup_mode": "app_identity_id" if normalized_identity_id else "detection_reference",
        "selection_authorized": False,
        "mutation_authorized": False,
        "substitution_policy": "no_older_snapshot_substitution",
    }

    if normalized_identity_id:
        if include_history:
            result = get_identity_history(
                app_identity_id=normalized_identity_id,
                max_entries=selected_max_history_entries,
                path=Path(DEFAULT_SNAPSHOT_PATH),
                enabled=True,
                stale_after_seconds=stale_after_seconds,
            )
        else:
            result = get_identity_status(
                app_identity_id=normalized_identity_id,
                path=Path(DEFAULT_SNAPSHOT_PATH),
                enabled=True,
                stale_after_seconds=stale_after_seconds,
            )
    else:
        result = find_identity_candidates(
            detection_reference=normalized_detection_reference,
            max_candidates=selected_max_candidates,
            path=Path(DEFAULT_SNAPSHOT_PATH),
            evidence_root_dir=Path(DEFAULT_EVIDENCE_ROOT_DIR),
            enabled=True,
            stale_after_seconds=stale_after_seconds,
        )

    compatibility = _compatibility_report(
        source_generation=result.get("source_generation"),
        source_version=result.get("source_version"),
        min_source_generation=parsed_min_generation,
        min_source_version=parsed_min_version,
    )

    stale_snapshot = result.get("state") == "stale" or bool(result.get("fresh") is False)
    response = {
        **result,
        "stale_snapshot": stale_snapshot,
        "compatibility": compatibility,
        "provenance": {
            **base_provenance,
            "operation": result.get("operation"),
            "resolution_mode": result.get("resolution_mode"),
        },
    }

    if not compatibility.get("compatible", False):
        response["available"] = False
        response["state"] = "incompatible_snapshot"
        response["error"] = "snapshot_incompatible_with_minimum_requirements"
        if normalized_identity_id:
            response["identity_lookup_state"] = "incompatible_snapshot"
            response["identity"] = None
            response["history"] = []
            response["history_count"] = 0
            response["history_truncated"] = False
        else:
            response["candidate_lookup_state"] = "incompatible_snapshot"
            response["candidates"] = []
            response["candidate_count"] = 0
            response["identity_match_decision"] = "not_decided"

    return _with_metadata(
        response,
        investigation_id=investigation_id,
        correlation_id=correlation_id,
    )
