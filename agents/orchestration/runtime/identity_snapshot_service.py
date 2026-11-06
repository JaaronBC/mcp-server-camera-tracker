from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "captures" / "latest_tracking_identity_snapshot.json"
DEFAULT_EVIDENCE_ROOT_DIR = PROJECT_ROOT / "captures" / "tracking_evidence"
DEFAULT_STALE_SECONDS = 3.0


def _flag_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _default_enabled() -> bool:
    return _flag_enabled("IDENTITY_SNAPSHOT_BRIDGE_ENABLED", False)


def _default_stale_seconds() -> float:
    return _positive_float_env("IDENTITY_SNAPSHOT_FRESHNESS_SECONDS", DEFAULT_STALE_SECONDS)


def load_identity_snapshot(
    *,
    path: Path = DEFAULT_SNAPSHOT_PATH,
    enabled: bool | None = None,
    now: float | None = None,
    stale_after_seconds: float | None = None,
) -> dict[str, Any]:
    selected_enabled = _default_enabled() if enabled is None else bool(enabled)
    selected_now = time.time() if now is None else float(now)
    selected_stale_seconds = (
        _default_stale_seconds() if stale_after_seconds is None else max(0.1, float(stale_after_seconds))
    )

    result: dict[str, Any] = {
        "available": False,
        "state": "disabled" if not selected_enabled else "missing",
        "snapshot_path": str(path),
        "snapshot_created_at": None,
        "snapshot_age_seconds": None,
        "freshness_window_seconds": selected_stale_seconds,
        "fresh": False,
        "source_generation": None,
        "source_version": None,
        "identity_count": 0,
        "snapshot": None,
        "error": None,
    }

    if not selected_enabled:
        return result

    if not path.is_file():
        result["state"] = "missing"
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        result["state"] = "unavailable"
        result["error"] = "snapshot_unreadable"
        return result
    except json.JSONDecodeError:
        result["state"] = "invalid"
        result["error"] = "snapshot_invalid_json"
        return result

    if not isinstance(payload, dict):
        result["state"] = "invalid"
        result["error"] = "snapshot_not_object"
        return result

    snapshot_created_at = payload.get("snapshot_created_at")
    if isinstance(snapshot_created_at, (int, float)):
        created_at_value = float(snapshot_created_at)
    else:
        created_at_value = None

    freshness_window = payload.get("freshness_window_seconds")
    freshness_window_value = (
        max(0.1, float(freshness_window))
        if isinstance(freshness_window, (int, float))
        else selected_stale_seconds
    )

    snapshot_age = (
        max(0.0, selected_now - created_at_value)
        if created_at_value is not None
        else None
    )
    fresh = bool(snapshot_age is not None and snapshot_age <= freshness_window_value)

    payload_state = payload.get("state")
    state_value = payload_state if isinstance(payload_state, str) else "available"
    if state_value == "available" and not fresh:
        state_value = "stale"

    identity_count = payload.get("identity_count")
    if not isinstance(identity_count, int):
        identities = payload.get("identities")
        identity_count = len(identities) if isinstance(identities, list) else 0

    result.update(
        {
            "available": state_value == "available" and fresh,
            "state": state_value,
            "snapshot_created_at": created_at_value,
            "snapshot_age_seconds": snapshot_age,
            "freshness_window_seconds": freshness_window_value,
            "fresh": fresh,
            "source_generation": payload.get("source_generation"),
            "source_version": payload.get("source_version"),
            "identity_count": identity_count,
            "snapshot": payload,
        }
    )

    return result


def _bounded_positive_int(value: int, *, default: int, minimum: int = 1, maximum: int = 512) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(minimum, min(maximum, parsed))


def _identity_state_for_record(record: dict[str, Any]) -> str:
    status = record.get("status") if isinstance(record.get("status"), str) else "unknown"
    if status == "retired":
        return "retired"
    if bool(record.get("provisional", False)):
        return "provisional"
    if status in {"visible", "temporarily_lost"}:
        return "confirmed"
    return "unknown"


def _base_operation_result(
    *,
    operation: str,
    snapshot_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "operation": operation,
        "available": bool(snapshot_result.get("available", False)),
        "state": snapshot_result.get("state"),
        "snapshot_path": snapshot_result.get("snapshot_path"),
        "snapshot_created_at": snapshot_result.get("snapshot_created_at"),
        "snapshot_age_seconds": snapshot_result.get("snapshot_age_seconds"),
        "freshness_window_seconds": snapshot_result.get("freshness_window_seconds"),
        "fresh": snapshot_result.get("fresh"),
        "source_generation": snapshot_result.get("source_generation"),
        "source_version": snapshot_result.get("source_version"),
        "identity_count": snapshot_result.get("identity_count"),
        "mutation_authorized": False,
        "mutation_authorization_reason": "read_only_snapshot_service",
        "error": snapshot_result.get("error"),
    }


def _normalize_detection_reference_for_lookup(
    *,
    detection_reference: str,
    evidence_root_dir: Path,
) -> dict[str, Any]:
    normalized_reference = detection_reference.strip()
    if not normalized_reference:
        return {
            "ok": False,
            "error": "detection_reference_required",
            "state": "invalid_request",
            "reference_kind": "missing",
            "resolved_reference": None,
        }

    if normalized_reference.startswith("fdet-") or normalized_reference.startswith("frame:"):
        return {
            "ok": True,
            "error": None,
            "state": "ok",
            "reference_kind": (
                "tracking_detection_reference"
                if normalized_reference.startswith("fdet-")
                else "legacy_registry_detection_reference"
            ),
            "resolved_reference": normalized_reference,
        }

    prefix = "evidence://detection/"
    if not normalized_reference.startswith(prefix):
        return {
            "ok": False,
            "error": "unsupported_detection_reference_format",
            "state": "invalid_request",
            "reference_kind": "unknown",
            "resolved_reference": None,
        }

    detection_ref_id = normalized_reference[len(prefix):].strip()
    if not detection_ref_id:
        return {
            "ok": False,
            "error": "invalid_evidence_detection_reference",
            "state": "invalid_request",
            "reference_kind": "evidence_detection_reference",
            "resolved_reference": None,
        }

    metadata_path = evidence_root_dir / "refs" / f"{detection_ref_id}.json"
    if not metadata_path.is_file():
        return {
            "ok": False,
            "error": "evidence_detection_reference_missing",
            "state": "invalid_reference",
            "reference_kind": "evidence_detection_reference",
            "resolved_reference": None,
        }

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "ok": False,
            "error": "evidence_detection_reference_invalid",
            "state": "invalid_reference",
            "reference_kind": "evidence_detection_reference",
            "resolved_reference": None,
        }

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "evidence_detection_reference_invalid",
            "state": "invalid_reference",
            "reference_kind": "evidence_detection_reference",
            "resolved_reference": None,
        }

    aliases_payload = payload.get("tracking_detection_reference_aliases")
    aliases: list[str] = []
    primary = payload.get("tracking_detection_reference")
    if isinstance(primary, str) and primary.strip():
        aliases.append(primary.strip())
    if isinstance(aliases_payload, list):
        for item in aliases_payload:
            if isinstance(item, str) and item.strip():
                aliases.append(item.strip())

    unique_aliases: list[str] = []
    for item in aliases:
        if item not in unique_aliases:
            unique_aliases.append(item)

    if not unique_aliases:
        return {
            "ok": False,
            "error": "evidence_detection_reference_mapping_missing",
            "state": "invalid_reference",
            "reference_kind": "evidence_detection_reference",
            "resolved_reference": None,
        }

    if len(unique_aliases) > 1:
        return {
            "ok": False,
            "error": "evidence_detection_reference_mapping_conflict",
            "state": "conflicting_reference",
            "reference_kind": "evidence_detection_reference",
            "resolved_reference": None,
        }

    resolved_reference = unique_aliases[0]
    if not (resolved_reference.startswith("fdet-") or resolved_reference.startswith("frame:")):
        return {
            "ok": False,
            "error": "evidence_detection_reference_mapping_invalid",
            "state": "invalid_reference",
            "reference_kind": "evidence_detection_reference",
            "resolved_reference": None,
        }

    return {
        "ok": True,
        "error": None,
        "state": "ok",
        "reference_kind": "evidence_detection_reference",
        "resolved_reference": resolved_reference,
    }


def get_identity_status(
    *,
    app_identity_id: str,
    path: Path = DEFAULT_SNAPSHOT_PATH,
    enabled: bool | None = None,
    now: float | None = None,
    stale_after_seconds: float | None = None,
) -> dict[str, Any]:
    snapshot_result = load_identity_snapshot(
        path=path,
        enabled=enabled,
        now=now,
        stale_after_seconds=stale_after_seconds,
    )
    result = _base_operation_result(operation="get_identity_status", snapshot_result=snapshot_result)
    result["app_identity_id"] = app_identity_id

    if not snapshot_result.get("available", False):
        result.update(
            {
                "identity_lookup_state": "unavailable",
                "identity": None,
            }
        )
        return result

    snapshot = snapshot_result.get("snapshot")
    identities = snapshot.get("identities") if isinstance(snapshot, dict) else None
    identity_items = identities if isinstance(identities, list) else []

    selected: dict[str, Any] | None = None
    for item in identity_items:
        if not isinstance(item, dict):
            continue
        if item.get("app_identity_id") == app_identity_id:
            selected = item
            break

    if selected is None:
        result.update(
            {
                "identity_lookup_state": "missing",
                "identity": None,
            }
        )
        return result

    identity_state = _identity_state_for_record(selected)
    result.update(
        {
            "identity_lookup_state": "found",
            "identity": {
                "app_identity_id": selected.get("app_identity_id"),
                "lifecycle_status": selected.get("status"),
                "identity_state": identity_state,
                "provisional": bool(selected.get("provisional", False)),
                "generation": selected.get("generation"),
                "identity_version": selected.get("identity_version"),
                "created_at": selected.get("created_at"),
                "updated_at": selected.get("updated_at"),
                "detection_id": selected.get("detection_id"),
                "detection_reference": selected.get("detection_reference"),
                "yolo_track_id": selected.get("yolo_track_id"),
                "loss_started_at": selected.get("loss_started_at"),
                "historical_yolo_track_ids": list(selected.get("historical_yolo_track_ids", [])),
            },
        }
    )
    return result


def get_identity_history(
    *,
    app_identity_id: str,
    max_entries: int = 20,
    path: Path = DEFAULT_SNAPSHOT_PATH,
    enabled: bool | None = None,
    now: float | None = None,
    stale_after_seconds: float | None = None,
) -> dict[str, Any]:
    selected_max_entries = _bounded_positive_int(max_entries, default=20, minimum=1, maximum=256)
    status_result = get_identity_status(
        app_identity_id=app_identity_id,
        path=path,
        enabled=enabled,
        now=now,
        stale_after_seconds=stale_after_seconds,
    )
    result = {
        **status_result,
        "operation": "get_identity_history",
        "max_entries": selected_max_entries,
        "history": [],
        "history_count": 0,
        "history_truncated": False,
    }

    if status_result.get("identity_lookup_state") != "found":
        return result

    snapshot_result = load_identity_snapshot(
        path=path,
        enabled=enabled,
        now=now,
        stale_after_seconds=stale_after_seconds,
    )
    snapshot = snapshot_result.get("snapshot")
    identities = snapshot.get("identities") if isinstance(snapshot, dict) else None
    identity_items = identities if isinstance(identities, list) else []

    selected: dict[str, Any] | None = None
    for item in identity_items:
        if not isinstance(item, dict):
            continue
        if item.get("app_identity_id") == app_identity_id:
            selected = item
            break

    if selected is None:
        result["identity_lookup_state"] = "missing"
        result["identity"] = None
        return result

    history_payload = selected.get("observation_history")
    history_items = list(history_payload) if isinstance(history_payload, list) else []
    history_truncated = len(history_items) > selected_max_entries
    bounded_history = history_items[-selected_max_entries:]

    result.update(
        {
            "history": bounded_history,
            "history_count": len(bounded_history),
            "history_truncated": history_truncated,
            "provenance": {
                "source": "identity_snapshot",
                "resolution_mode": "app_identity_id_exact_match",
                "includes_detection_reference": True,
            },
        }
    )
    return result


def find_identity_candidates(
    *,
    detection_reference: str,
    max_candidates: int = 8,
    path: Path = DEFAULT_SNAPSHOT_PATH,
    evidence_root_dir: Path = DEFAULT_EVIDENCE_ROOT_DIR,
    enabled: bool | None = None,
    now: float | None = None,
    stale_after_seconds: float | None = None,
) -> dict[str, Any]:
    selected_max_candidates = _bounded_positive_int(max_candidates, default=8, minimum=1, maximum=128)
    snapshot_result = load_identity_snapshot(
        path=path,
        enabled=enabled,
        now=now,
        stale_after_seconds=stale_after_seconds,
    )
    result = _base_operation_result(operation="find_identity_candidates", snapshot_result=snapshot_result)
    result["detection_reference"] = detection_reference
    result["max_candidates"] = selected_max_candidates
    result["resolution_mode"] = "exact_detection_reference_match_only"
    result["candidates"] = []
    result["candidate_count"] = 0
    result["candidate_lookup_state"] = "unavailable"
    result["reference_kind"] = None
    result["resolved_detection_reference"] = None
    result["ambiguity"] = {
        "ambiguous": False,
        "reason": None,
    }
    result["identity_match_decision"] = "not_decided"

    normalized_reference = str(detection_reference).strip() if detection_reference is not None else ""
    normalized_reference_result = _normalize_detection_reference_for_lookup(
        detection_reference=normalized_reference,
        evidence_root_dir=evidence_root_dir,
    )
    result["reference_kind"] = normalized_reference_result.get("reference_kind")
    result["resolved_detection_reference"] = normalized_reference_result.get("resolved_reference")

    if not bool(normalized_reference_result.get("ok", False)):
        result.update(
            {
                "state": normalized_reference_result.get("state", "invalid_request"),
                "available": False,
                "candidate_lookup_state": normalized_reference_result.get("state", "invalid_request"),
                "error": normalized_reference_result.get("error", "detection_reference_required"),
            }
        )
        return result

    resolved_reference = normalized_reference_result.get("resolved_reference")
    if not isinstance(resolved_reference, str) or not resolved_reference:
        result.update(
            {
                "state": "invalid_reference",
                "available": False,
                "candidate_lookup_state": "invalid_reference",
                "error": "detection_reference_resolution_failed",
            }
        )
        return result

    if not snapshot_result.get("available", False):
        return result

    snapshot = snapshot_result.get("snapshot")
    identities = snapshot.get("identities") if isinstance(snapshot, dict) else None
    identity_items = identities if isinstance(identities, list) else []

    matched_candidates: list[dict[str, Any]] = []
    for item in identity_items:
        if not isinstance(item, dict):
            continue

        match_sources: list[dict[str, Any]] = []
        current_reference = item.get("detection_reference")
        if isinstance(current_reference, str) and current_reference == resolved_reference:
            match_sources.append(
                {
                    "source": "current_detection_reference",
                    "detection_reference": current_reference,
                }
            )

        history_payload = item.get("observation_history")
        history_items = history_payload if isinstance(history_payload, list) else []
        for history_item in history_items:
            if not isinstance(history_item, dict):
                continue
            history_reference = history_item.get("detection_reference")
            if not isinstance(history_reference, str) or history_reference != resolved_reference:
                continue
            match_sources.append(
                {
                    "source": "observation_history",
                    "detection_reference": history_reference,
                    "timestamp": history_item.get("timestamp"),
                    "generation": history_item.get("generation"),
                    "event": history_item.get("event"),
                    "detection_id": history_item.get("detection_id"),
                    "yolo_track_id": history_item.get("yolo_track_id"),
                }
            )

        if not match_sources:
            continue

        matched_candidates.append(
            {
                "app_identity_id": item.get("app_identity_id"),
                "lifecycle_status": item.get("status"),
                "identity_state": _identity_state_for_record(item),
                "provisional": bool(item.get("provisional", False)),
                "generation": item.get("generation"),
                "identity_version": item.get("identity_version"),
                "updated_at": item.get("updated_at"),
                "detection_id": item.get("detection_id"),
                "detection_reference": item.get("detection_reference"),
                "yolo_track_id": item.get("yolo_track_id"),
                "match_sources": match_sources,
            }
        )

    matched_candidates.sort(
        key=lambda item: (
            -float(item.get("updated_at", 0.0) or 0.0),
            str(item.get("app_identity_id") or ""),
        )
    )
    bounded_candidates = matched_candidates[:selected_max_candidates]
    ambiguity = len(bounded_candidates) > 1

    result.update(
        {
            "candidate_lookup_state": "found" if bounded_candidates else "none",
            "candidates": bounded_candidates,
            "candidate_count": len(bounded_candidates),
            "ambiguity": {
                "ambiguous": ambiguity,
                "reason": "multiple_reference_matches" if ambiguity else None,
            },
            "provenance": {
                "source": "identity_snapshot",
                "resolution_mode": "exact_detection_reference_match_only",
                "note": (
                    "No identity inference from identifier similarity is performed; matching uses "
                    "only exact detection_reference equality (direct or explicitly resolved alias)."
                ),
            },
        }
    )
    return result
