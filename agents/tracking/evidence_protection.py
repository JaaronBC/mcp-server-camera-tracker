from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import fcntl


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _frame_id_from_reference(reference: str) -> str | None:
    if not isinstance(reference, str):
        return None
    prefix = "evidence://frame/"
    if not reference.startswith(prefix):
        return None
    frame_id = reference[len(prefix):].strip()
    if not frame_id:
        return None
    if "/" in frame_id or "\\" in frame_id:
        return None
    if frame_id in {".", ".."}:
        return None
    return frame_id


class EvidenceProtectionLock:
    _inproc_guard = threading.Lock()
    _inproc_held_paths: set[str] = set()

    def __init__(self, evidence_root_dir: Path) -> None:
        self._root_dir = evidence_root_dir
        self._lock_path = self._root_dir / "protection" / "lease_cleanup.lock"
        self._handle: Any | None = None

    def acquire(self, *, timeout_seconds: float = 0.0, poll_seconds: float = 0.005) -> bool:
        timeout = max(0.0, float(timeout_seconds))
        deadline = time.monotonic() + timeout
        lock_key = str(self._lock_path)
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)

        if self._handle is not None:
            return True

        try:
            handle = self._lock_path.open("a+", encoding="utf-8")
        except Exception:
            return False

        while True:
            try:
                with self._inproc_guard:
                    if lock_key in self._inproc_held_paths:
                        raise BlockingIOError()
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self._inproc_guard:
                    self._inproc_held_paths.add(lock_key)
                try:
                    handle.seek(0)
                    handle.truncate(0)
                    handle.write(f"pid={os.getpid()} acquired_at={time.time()}\n")
                    handle.flush()
                except Exception:
                    pass
                self._handle = handle
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    try:
                        handle.close()
                    except Exception:
                        pass
                    return False
                time.sleep(max(0.0, min(float(poll_seconds), deadline - time.monotonic())))
            except Exception:
                try:
                    handle.close()
                except Exception:
                    pass
                return False

    def release(self) -> None:
        lock_key = str(self._lock_path)
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass
        with self._inproc_guard:
            self._inproc_held_paths.discard(lock_key)


def write_lease_atomic(
    *,
    evidence_root_dir: Path,
    request_id: str,
    payload: dict[str, Any],
) -> Path:
    leases_dir = evidence_root_dir / "protection" / "leases"
    leases_dir.mkdir(parents=True, exist_ok=True)
    destination = leases_dir / f"{request_id}.json"
    temp_path = leases_dir / f".{request_id}.{uuid.uuid4().hex}.tmp"
    temp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temp_path.replace(destination)
    return destination


def delete_lease(*, evidence_root_dir: Path, request_id: str) -> None:
    lease_path = evidence_root_dir / "protection" / "leases" / f"{request_id}.json"
    _safe_unlink(lease_path)


def _safe_json_load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _resolve_frame_reference_from_envelope_payload(envelope_payload: dict[str, Any]) -> str | None:
    frame_reference = envelope_payload.get("frame_reference")
    if isinstance(frame_reference, str) and _frame_id_from_reference(frame_reference) is not None:
        return frame_reference

    source_refs = envelope_payload.get("source_evidence_refs")
    if isinstance(source_refs, list):
        for item in source_refs:
            if isinstance(item, str) and _frame_id_from_reference(item) is not None:
                return item
    return None


def _iter_active_lease_candidates(
    *,
    evidence_root_dir: Path,
    now_timestamp: float,
) -> tuple[list[tuple[float, Path, str, dict[str, Any]]], dict[str, int]]:
    leases_dir = evidence_root_dir / "protection" / "leases"
    refs_dir = evidence_root_dir / "refs"

    stats = {
        "leases_active": 0,
        "leases_pruned_expired": 0,
        "leases_pruned_malformed": 0,
        "leases_pruned_stale": 0,
        "leases_pruned_overflow": 0,
        "leases_over_budget": 0,
    }

    if not leases_dir.is_dir():
        return [], stats

    candidates: list[tuple[float, Path, str, dict[str, Any]]] = []
    for lease_path in leases_dir.glob("*.json"):
        payload = _safe_json_load(lease_path)
        if payload is None:
            stats["leases_pruned_malformed"] += 1
            _safe_unlink(lease_path)
            continue

        request_id = payload.get("request_id")
        frame_reference = payload.get("frame_reference")
        expires_at = payload.get("expires_at")
        created_at = payload.get("created_at")

        if not isinstance(request_id, str) or not request_id.strip():
            stats["leases_pruned_malformed"] += 1
            _safe_unlink(lease_path)
            continue
        request_id = request_id.strip()
        if request_id != lease_path.stem:
            stats["leases_pruned_malformed"] += 1
            _safe_unlink(lease_path)
            continue
        if not isinstance(expires_at, (int, float)):
            stats["leases_pruned_malformed"] += 1
            _safe_unlink(lease_path)
            continue
        if not isinstance(created_at, (int, float)):
            created_at = float(lease_path.stat().st_mtime)

        if float(now_timestamp) > float(expires_at):
            stats["leases_pruned_expired"] += 1
            _safe_unlink(lease_path)
            continue

        frame_id = _frame_id_from_reference(frame_reference)
        if frame_id is None:
            stats["leases_pruned_malformed"] += 1
            _safe_unlink(lease_path)
            continue

        if not (refs_dir / f"{frame_id}.json").is_file():
            stats["leases_pruned_stale"] += 1
            _safe_unlink(lease_path)
            continue

        candidates.append((float(created_at), lease_path, frame_id, payload))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates, stats


def acquire_verification_evidence_lease(
    *,
    evidence_root_dir: Path,
    request_id: str,
    message_id: str,
    track_id: str,
    now_timestamp: float,
    lease_ttl_seconds: float,
    max_protected_frames: int,
    max_active_leases: int,
    lock_timeout_seconds: float = 0.0,
) -> dict[str, Any]:
    normalized_request_id = request_id.strip() if isinstance(request_id, str) else ""
    normalized_message_id = message_id.strip() if isinstance(message_id, str) else ""
    normalized_track_id = track_id.strip() if isinstance(track_id, str) else ""
    if not normalized_request_id:
        return {"acquired": False, "status": "invalid_request_id"}
    if not normalized_message_id:
        return {"acquired": False, "status": "invalid_message_id"}
    if not normalized_track_id:
        return {"acquired": False, "status": "invalid_track_id"}

    now = float(now_timestamp)
    ttl_seconds = max(1.0, float(lease_ttl_seconds))
    budget = max(0, int(max_protected_frames))
    active_limit = max(1, int(max_active_leases))

    protection_lock = EvidenceProtectionLock(evidence_root_dir)
    if not protection_lock.acquire(timeout_seconds=max(0.0, float(lock_timeout_seconds))):
        return {"acquired": False, "status": "lease_lock_unavailable"}

    try:
        envelope_path = evidence_root_dir / "envelopes" / f"{normalized_message_id}.json"
        envelope_record = _safe_json_load(envelope_path)
        if envelope_record is None:
            return {"acquired": False, "status": "envelope_unavailable"}

        if envelope_record.get("message_id") != normalized_message_id:
            return {"acquired": False, "status": "envelope_message_id_mismatch"}

        envelope_payload = envelope_record.get("envelope")
        if not isinstance(envelope_payload, dict):
            return {"acquired": False, "status": "envelope_payload_missing"}

        envelope_track_id = envelope_payload.get("track_id")
        if not isinstance(envelope_track_id, str) or envelope_track_id.strip() != normalized_track_id:
            return {"acquired": False, "status": "envelope_track_mismatch"}

        frame_reference = _resolve_frame_reference_from_envelope_payload(envelope_payload)
        frame_id = _frame_id_from_reference(frame_reference)
        if frame_id is None:
            return {"acquired": False, "status": "envelope_frame_reference_missing"}

        frame_metadata_path = evidence_root_dir / "refs" / f"{frame_id}.json"
        frame_metadata = _safe_json_load(frame_metadata_path)
        if frame_metadata is None:
            return {"acquired": False, "status": "frame_metadata_unavailable"}

        if frame_metadata.get("kind") != "frame":
            return {"acquired": False, "status": "frame_metadata_invalid_kind"}
        if frame_metadata.get("id") != frame_id:
            return {"acquired": False, "status": "frame_metadata_id_mismatch"}

        frame_status = frame_metadata.get("status")
        if frame_status != "available":
            return {"acquired": False, "status": "frame_not_available"}

        expires_at = frame_metadata.get("expires_at")
        if not isinstance(expires_at, (int, float)) or now > float(expires_at):
            return {"acquired": False, "status": "frame_expired"}

        frame_path = frame_metadata.get("frame_path")
        if not isinstance(frame_path, str) or not Path(frame_path).is_file():
            return {"acquired": False, "status": "frame_payload_missing"}

        active_candidates, stats = _iter_active_lease_candidates(
            evidence_root_dir=evidence_root_dir,
            now_timestamp=now,
        )

        existing_payload: dict[str, Any] | None = None
        active_unique_frame_ids: set[str] = set()
        active_lease_count = 0
        for _, lease_path, existing_frame_id, payload in active_candidates:
            if lease_path.stem == normalized_request_id:
                existing_payload = payload
                continue
            active_lease_count += 1
            active_unique_frame_ids.add(existing_frame_id)

        if active_lease_count >= active_limit:
            return {
                "acquired": False,
                "status": "max_active_leases_exceeded",
                "details": {"active_lease_count": active_lease_count, "max_active_leases": active_limit},
            }

        if frame_id not in active_unique_frame_ids and len(active_unique_frame_ids) >= budget:
            return {
                "acquired": False,
                "status": "max_protected_frames_exceeded",
                "details": {"active_protected_frames": len(active_unique_frame_ids), "max_protected_frames": budget},
            }

        if isinstance(existing_payload, dict):
            if (
                existing_payload.get("message_id") != normalized_message_id
                or existing_payload.get("track_id") != normalized_track_id
                or existing_payload.get("frame_reference") != frame_reference
            ):
                return {"acquired": False, "status": "request_lease_association_mismatch"}

        lease_payload = {
            "request_id": normalized_request_id,
            "message_id": normalized_message_id,
            "track_id": normalized_track_id,
            "frame_reference": frame_reference,
            "created_at": now,
            "expires_at": min(float(expires_at), now + ttl_seconds),
        }
        write_lease_atomic(
            evidence_root_dir=evidence_root_dir,
            request_id=normalized_request_id,
            payload=lease_payload,
        )

        return {
            "acquired": True,
            "status": "acquired",
            "lease": lease_payload,
            "lease_stats": stats,
        }
    finally:
        protection_lock.release()


def release_verification_evidence_lease(
    *,
    evidence_root_dir: Path,
    request_id: str,
    lock_timeout_seconds: float = 0.0,
) -> dict[str, Any]:
    normalized_request_id = request_id.strip() if isinstance(request_id, str) else ""
    if not normalized_request_id:
        return {"released": False, "status": "invalid_request_id"}

    protection_lock = EvidenceProtectionLock(evidence_root_dir)
    if not protection_lock.acquire(timeout_seconds=max(0.0, float(lock_timeout_seconds))):
        return {"released": False, "status": "lease_lock_unavailable"}

    try:
        delete_lease(evidence_root_dir=evidence_root_dir, request_id=normalized_request_id)
        return {"released": True, "status": "released"}
    finally:
        protection_lock.release()


def load_active_protected_frame_ids(
    *,
    evidence_root_dir: Path,
    now_timestamp: float,
    max_protected_frames: int,
) -> tuple[set[str], dict[str, int]]:
    budget = max(0, int(max_protected_frames))

    candidates, stats = _iter_active_lease_candidates(
        evidence_root_dir=evidence_root_dir,
        now_timestamp=float(now_timestamp),
    )

    protected_frame_ids: set[str] = set()
    for _, _, frame_id, _ in candidates:
        if frame_id in protected_frame_ids:
            stats["leases_active"] += 1
            continue
        protected_frame_ids.add(frame_id)
        stats["leases_active"] += 1

    if len(protected_frame_ids) > budget:
        stats["leases_over_budget"] = len(protected_frame_ids) - budget

    return protected_frame_ids, stats
