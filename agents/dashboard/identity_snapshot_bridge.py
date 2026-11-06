from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from agents.tracking.identity_registry import PersistentIdentityRegistry


def build_identity_snapshot_payload(
    *,
    registry: PersistentIdentityRegistry,
    now: float,
    freshness_window_seconds: float,
    max_identities: int,
    max_history_per_identity: int,
    max_historical_track_ids: int,
    max_payload_bytes: int,
) -> dict[str, Any]:
    selected_freshness_window = max(0.1, float(freshness_window_seconds))
    selected_max_payload_bytes = max(1024, int(max_payload_bytes))

    registry_snapshot = registry.capture_read_only_snapshot(
        max_identities=max_identities,
        max_history_per_identity=max_history_per_identity,
        max_historical_track_ids=max_historical_track_ids,
    )

    records = registry_snapshot.get("identities")
    identities = list(records) if isinstance(records, list) else []
    identities_truncated_by_size = False

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "state": "available",
        "snapshot_created_at": float(now),
        "freshness_window_seconds": selected_freshness_window,
        "fresh_until": float(now) + selected_freshness_window,
        "source_generation": int(registry_snapshot.get("generation", 0)),
        "source_version": int(registry_snapshot.get("version", 0)),
        "source_total_records": int(registry_snapshot.get("total_records", 0)),
        "identities": identities,
        "identity_count": len(identities),
        "truncation": {
            "identities_truncated_by_count": bool(registry_snapshot.get("identities_truncated_by_count", False)),
            "identities_truncated_by_size": False,
            "history_truncated": bool(registry_snapshot.get("history_truncated", False)),
            "historical_track_ids_truncated": bool(registry_snapshot.get("historical_track_ids_truncated", False)),
        },
        "payload_bytes": 0,
    }

    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    while len(encoded) > selected_max_payload_bytes and payload["identities"]:
        payload["identities"].pop()
        identities_truncated_by_size = True
        payload["identity_count"] = len(payload["identities"])
        payload["truncation"]["identities_truncated_by_size"] = True
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    payload["payload_bytes"] = len(encoded)
    if identities_truncated_by_size:
        payload["truncation"]["identities_truncated_by_size"] = True
    return payload


def build_identity_snapshot_unavailable_payload(
    *,
    now: float,
    freshness_window_seconds: float,
    state: str,
    reason: str,
) -> dict[str, Any]:
    selected_freshness_window = max(0.1, float(freshness_window_seconds))
    return {
        "schema_version": "1.0",
        "state": state,
        "reason": reason,
        "snapshot_created_at": float(now),
        "freshness_window_seconds": selected_freshness_window,
        "fresh_until": float(now) + selected_freshness_window,
        "source_generation": None,
        "source_version": None,
        "source_total_records": 0,
        "identities": [],
        "identity_count": 0,
        "truncation": {
            "identities_truncated_by_count": False,
            "identities_truncated_by_size": False,
            "history_truncated": False,
            "historical_track_ids_truncated": False,
        },
        "payload_bytes": 0,
    }


def publish_identity_snapshot(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    temp_path = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    try:
        temp_path.write_text(serialized, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
