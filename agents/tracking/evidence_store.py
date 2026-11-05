from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_protection import EvidenceProtectionLock, load_active_protected_frame_ids


def _safe_value(value: str | None, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        normalized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
        if normalized:
            return normalized
    return fallback


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


@dataclass(slots=True)
class FrameEvidenceRecord:
    frame_reference: str
    frame_id: str
    timestamp: float
    expires_at: float
    status: str


class TrackingEvidenceStore:
    def __init__(
        self,
        root_dir: Path,
        *,
        max_age_seconds: float = 30.0,
        max_frame_records: int = 120,
        max_protected_frames: int = 32,
        max_detection_records: int = 2000,
        max_envelope_records: int = 4000,
        cleanup_lock_timeout_seconds: float = 0.0,
        process_nonce: str | None = None,
        clock: Any | None = None,
    ) -> None:
        self._root_dir = root_dir
        self._frames_dir = root_dir / "frames"
        self._refs_dir = root_dir / "refs"
        self._envelopes_dir = root_dir / "envelopes"
        self._max_age_seconds = max(1.0, float(max_age_seconds))
        self._max_frame_records = max(1, int(max_frame_records))
        self._max_protected_frames = min(self._max_frame_records, max(0, int(max_protected_frames)))
        self._max_detection_records = max(1, int(max_detection_records))
        self._max_envelope_records = max(1, int(max_envelope_records))
        self._cleanup_lock_timeout_seconds = max(0.0, float(cleanup_lock_timeout_seconds))
        self._clock = clock or time.time

        self._process_nonce = _safe_value(process_nonce, "") if process_nonce else uuid.uuid4().hex[:12]
        self._counter = 0
        self._lock = threading.Lock()
        self._pending_frames: set[str] = set()
        self._cleanup_tick = 0
        self._admission_capacity_evictions = 0

        self._frame_ids = deque(maxlen=self._max_frame_records)
        self._detection_ids = deque(maxlen=self._max_detection_records)
        self._envelope_ids = deque(maxlen=self._max_envelope_records)

        self._tasks: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    @property
    def process_nonce(self) -> str:
        return self._process_nonce

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._tasks.put_nowait(("shutdown", {}))
        except queue.Full:
            pass
        self._worker.join(timeout=1.0)

    def _next_id(self, prefix: str, timestamp: float, *, track_id: str | None = None, detection_id: str | None = None) -> str:
        with self._lock:
            self._counter += 1
            index = self._counter
        ts_us = int(float(timestamp) * 1_000_000)
        track_token = _safe_value(track_id, "track") if track_id is not None else "none"
        detection_token = _safe_value(detection_id, "det") if detection_id is not None else "none"
        return f"{prefix}-{self._process_nonce}-{ts_us}-{index}-{track_token}-{detection_token}"

    def register_frame(self, *, frame: Any, timestamp: float, width: int, height: int) -> FrameEvidenceRecord:
        frame_id = self._next_id("frame", timestamp)
        frame_reference = f"evidence://frame/{frame_id}"
        now = float(self._clock())
        expires_at = now + self._max_age_seconds
        frame_path = self._frames_dir / f"{frame_id}.jpg"
        metadata_path = self._refs_dir / f"{frame_id}.json"

        metadata = {
            "kind": "frame",
            "id": frame_id,
            "reference": frame_reference,
            "timestamp": float(timestamp),
            "created_at": now,
            "expires_at": expires_at,
            "status": "pending",
            "frame_path": str(frame_path),
            "width": int(width),
            "height": int(height),
            "process_nonce": self._process_nonce,
        }

        admission = self._admit_frame_slot(
            now_timestamp=now,
            new_frame_id=frame_id,
            metadata=metadata,
            metadata_path=metadata_path,
        )
        if not admission.get("admitted"):
            status = str(admission.get("status") or "dropped_storage_error")
            return FrameEvidenceRecord(
                frame_reference=frame_reference,
                frame_id=frame_id,
                timestamp=float(timestamp),
                expires_at=expires_at,
                status=status,
            )

        queued = True
        try:
            self._tasks.put_nowait(
                (
                    "write_frame",
                    {
                        "frame_id": frame_id,
                        "frame": frame,
                        "frame_path": frame_path,
                        "metadata_path": metadata_path,
                    },
                )
            )
        except queue.Full:
            queued = False

        if not queued:
            metadata["status"] = "dropped_queue_full"
            _json_write(metadata_path, metadata)
            with self._lock:
                self._pending_frames.discard(frame_id)

        self._cleanup_maybe(now)
        return FrameEvidenceRecord(
            frame_reference=frame_reference,
            frame_id=frame_id,
            timestamp=float(timestamp),
            expires_at=expires_at,
            status=metadata["status"],
        )

    def register_detection_reference(
        self,
        *,
        track_id: str,
        detection_id: str,
        timestamp: float,
        frame_reference: str | None,
        pixel_bbox: dict[str, float] | None,
    ) -> str:
        detection_ref_id = self._next_id(
            "detref",
            timestamp,
            track_id=track_id,
            detection_id=detection_id,
        )
        reference = f"evidence://detection/{detection_ref_id}"
        now = float(self._clock())
        payload = {
            "kind": "detection",
            "id": detection_ref_id,
            "reference": reference,
            "timestamp": float(timestamp),
            "created_at": now,
            "expires_at": now + self._max_age_seconds,
            "track_id": track_id,
            "detection_id": detection_id,
            "frame_reference": frame_reference,
            "pixel_bbox": pixel_bbox,
            "process_nonce": self._process_nonce,
        }
        _json_write(self._refs_dir / f"{detection_ref_id}.json", payload)
        with self._lock:
            self._detection_ids.append(detection_ref_id)
        self._cleanup_maybe(now)
        return reference

    def store_validated_envelope(
        self,
        *,
        message_id: str,
        envelope_payload: dict[str, Any],
        validation_payload: dict[str, Any],
    ) -> str:
        record_path = self._envelopes_dir / f"{message_id}.json"
        payload = {
            "message_id": message_id,
            "stored_at": float(self._clock()),
            "validation": validation_payload,
            "envelope": envelope_payload,
        }
        _json_write(record_path, payload)
        with self._lock:
            self._envelope_ids.append(message_id)
        self._cleanup_maybe(float(self._clock()))
        return f"evidence://envelope/{message_id}"

    def resolve_reference(
        self,
        reference: str,
        *,
        expected_track_id: str | None = None,
        expected_detection_id: str | None = None,
        expected_frame_reference: str | None = None,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = float(self._clock()) if now_timestamp is None else float(now_timestamp)
        kind, ref_id = self._parse_reference(reference)
        if kind is None or ref_id is None:
            return {
                "valid": False,
                "status": "invalid_reference_format",
                "reference": reference,
            }

        metadata = self._load_metadata(ref_id)
        if metadata is None:
            return {
                "valid": False,
                "status": "missing_reference",
                "reference": reference,
            }

        expires_at = metadata.get("expires_at")
        if isinstance(expires_at, (int, float)) and now > float(expires_at):
            return {
                "valid": False,
                "status": "expired_reference",
                "reference": reference,
                "metadata": metadata,
            }

        if kind == "frame":
            frame_path = metadata.get("frame_path")
            if not isinstance(frame_path, str):
                return {
                    "valid": False,
                    "status": "invalid_frame_metadata",
                    "reference": reference,
                    "metadata": metadata,
                }

            if metadata.get("status") != "available":
                if metadata.get("status") == "pending":
                    return {
                        "valid": False,
                        "status": "pending_reference",
                        "reference": reference,
                        "metadata": metadata,
                    }
                return {
                    "valid": False,
                    "status": "frame_unavailable",
                    "reference": reference,
                    "metadata": metadata,
                }

            if not Path(frame_path).is_file():
                return {
                    "valid": False,
                    "status": "missing_frame_payload",
                    "reference": reference,
                    "metadata": metadata,
                }

            return {
                "valid": True,
                "status": "ok",
                "reference": reference,
                "metadata": metadata,
            }

        if kind == "detection":
            if expected_track_id is not None and metadata.get("track_id") != expected_track_id:
                return {
                    "valid": False,
                    "status": "track_linkage_mismatch",
                    "reference": reference,
                    "metadata": metadata,
                }
            if expected_detection_id is not None and metadata.get("detection_id") != expected_detection_id:
                return {
                    "valid": False,
                    "status": "detection_linkage_mismatch",
                    "reference": reference,
                    "metadata": metadata,
                }
            frame_reference = metadata.get("frame_reference")
            if expected_frame_reference is not None and frame_reference != expected_frame_reference:
                return {
                    "valid": False,
                    "status": "frame_linkage_mismatch",
                    "reference": reference,
                    "metadata": metadata,
                }
            if isinstance(frame_reference, str):
                frame_resolution = self.resolve_reference(frame_reference, now_timestamp=now)
                if not frame_resolution.get("valid"):
                    return {
                        "valid": False,
                        "status": "linked_frame_unavailable",
                        "reference": reference,
                        "metadata": metadata,
                        "linked_frame": frame_resolution,
                    }

            return {
                "valid": True,
                "status": "ok",
                "reference": reference,
                "metadata": metadata,
            }

        return {
            "valid": False,
            "status": "unsupported_reference_kind",
            "reference": reference,
            "metadata": metadata,
        }

    def _parse_reference(self, reference: str) -> tuple[str | None, str | None]:
        if not isinstance(reference, str):
            return None, None
        prefix = "evidence://"
        if not reference.startswith(prefix):
            return None, None
        remaining = reference[len(prefix):]
        parts = remaining.split("/", maxsplit=1)
        if len(parts) != 2:
            return None, None
        kind, ref_id = parts
        if kind not in {"frame", "detection", "envelope"}:
            return None, None
        if not ref_id.strip():
            return None, None
        return kind, ref_id.strip()

    def _load_metadata(self, ref_id: str) -> dict[str, Any] | None:
        metadata_path = self._refs_dir / f"{ref_id}.json"
        if not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    def _cleanup_maybe(self, now: float) -> None:
        self._cleanup_tick += 1
        if self._cleanup_tick % 20 != 0:
            return
        self.cleanup(now_timestamp=now)

    def _admit_frame_slot(
        self,
        *,
        now_timestamp: float,
        new_frame_id: str,
        metadata: dict[str, Any],
        metadata_path: Path,
    ) -> dict[str, Any]:
        protection_lock = EvidenceProtectionLock(self._root_dir)
        if not protection_lock.acquire(timeout_seconds=self._cleanup_lock_timeout_seconds):
            return {"admitted": False, "status": "dropped_storage_lock_unavailable"}

        try:
            protected_frame_ids, _ = load_active_protected_frame_ids(
                evidence_root_dir=self._root_dir,
                now_timestamp=float(now_timestamp),
                max_protected_frames=self._max_protected_frames,
            )

            frame_records = self._sorted_records(kind="frame")
            while len(frame_records) >= self._max_frame_records:
                evict_index: int | None = None
                for index, (_, payload, _) in enumerate(frame_records):
                    frame_id = payload.get("id")
                    if isinstance(frame_id, str) and frame_id in protected_frame_ids:
                        continue
                    evict_index = index
                    break

                if evict_index is None:
                    return {"admitted": False, "status": "dropped_storage_full_protected"}

                _, payload, metadata_file = frame_records.pop(evict_index)
                frame_id = payload.get("id")
                frame_path = payload.get("frame_path")
                if isinstance(frame_path, str):
                    try:
                        Path(frame_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                metadata_file.unlink(missing_ok=True)
                if isinstance(frame_id, str):
                    with self._lock:
                        self._pending_frames.discard(frame_id)
                with self._lock:
                    self._admission_capacity_evictions += 1

            _json_write(metadata_path, metadata)
            with self._lock:
                self._pending_frames.add(new_frame_id)
                self._frame_ids.append(new_frame_id)

            return {"admitted": True, "status": "ok"}
        except Exception:
            return {"admitted": False, "status": "dropped_storage_error"}
        finally:
            protection_lock.release()

    def cleanup(self, *, now_timestamp: float | None = None) -> dict[str, int]:
        now = float(self._clock()) if now_timestamp is None else float(now_timestamp)
        removed_expired = 0
        removed_capacity = 0
        capacity_blocked_all_protected = 0

        protection_lock = EvidenceProtectionLock(self._root_dir)
        if not protection_lock.acquire(timeout_seconds=self._cleanup_lock_timeout_seconds):
            return {
                "removed_expired": 0,
                "removed_capacity": 0,
                "capacity_blocked_all_protected": 0,
                "lock_skipped": 1,
                "leases_pruned_expired": 0,
                "leases_pruned_malformed": 0,
                "leases_pruned_stale": 0,
                "leases_pruned_overflow": 0,
                "leases_over_budget": 0,
            }

        lease_stats = {
            "leases_pruned_expired": 0,
            "leases_pruned_malformed": 0,
            "leases_pruned_stale": 0,
            "leases_pruned_overflow": 0,
            "leases_over_budget": 0,
        }

        try:
            with self._lock:
                removed_capacity += self._admission_capacity_evictions
                self._admission_capacity_evictions = 0

            protected_frame_ids, lease_stats = load_active_protected_frame_ids(
                evidence_root_dir=self._root_dir,
                now_timestamp=now,
                max_protected_frames=self._max_protected_frames,
            )

            for metadata_file in self._refs_dir.glob("*.json"):
                try:
                    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue

                frame_id = payload.get("id") if payload.get("kind") == "frame" else None
                if isinstance(frame_id, str) and frame_id in protected_frame_ids:
                    continue

                expires_at = payload.get("expires_at")
                if isinstance(expires_at, (int, float)) and now > float(expires_at):
                    frame_path = payload.get("frame_path")
                    if isinstance(frame_path, str):
                        try:
                            Path(frame_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                    metadata_file.unlink(missing_ok=True)
                    if isinstance(frame_id, str):
                        with self._lock:
                            self._pending_frames.discard(frame_id)
                    removed_expired += 1

            frame_records = self._sorted_records(kind="frame")
            while len(frame_records) > self._max_frame_records:
                evict_index: int | None = None
                for index, (_, payload, _) in enumerate(frame_records):
                    frame_id = payload.get("id")
                    if isinstance(frame_id, str) and frame_id in protected_frame_ids:
                        continue
                    evict_index = index
                    break

                if evict_index is None:
                    capacity_blocked_all_protected = max(0, len(frame_records) - self._max_frame_records)
                    break

                _, payload, metadata_file = frame_records.pop(evict_index)
                frame_id = payload.get("id")
                frame_path = payload.get("frame_path")
                if isinstance(frame_path, str):
                    try:
                        Path(frame_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                metadata_file.unlink(missing_ok=True)
                if isinstance(frame_id, str):
                    with self._lock:
                        self._pending_frames.discard(frame_id)
                removed_capacity += 1

            detection_records = self._sorted_records(kind="detection")
            while len(detection_records) > self._max_detection_records:
                _, _, metadata_file = detection_records.pop(0)
                metadata_file.unlink(missing_ok=True)
                removed_capacity += 1

            envelope_records = sorted(
                [
                    (path.stat().st_mtime, path)
                    for path in self._envelopes_dir.glob("*.json")
                    if path.is_file()
                ],
                key=lambda item: item[0],
            )
            while len(envelope_records) > self._max_envelope_records:
                _, path = envelope_records.pop(0)
                path.unlink(missing_ok=True)
                removed_capacity += 1
        finally:
            protection_lock.release()

        return {
            "removed_expired": removed_expired,
            "removed_capacity": removed_capacity,
            "capacity_blocked_all_protected": capacity_blocked_all_protected,
            "lock_skipped": 0,
            "leases_pruned_expired": lease_stats.get("leases_pruned_expired", 0),
            "leases_pruned_malformed": lease_stats.get("leases_pruned_malformed", 0),
            "leases_pruned_stale": lease_stats.get("leases_pruned_stale", 0),
            "leases_pruned_overflow": lease_stats.get("leases_pruned_overflow", 0),
            "leases_over_budget": lease_stats.get("leases_over_budget", 0),
        }

    def _sorted_records(self, *, kind: str) -> list[tuple[float, dict[str, Any], Path]]:
        records: list[tuple[float, dict[str, Any], Path]] = []
        for metadata_file in self._refs_dir.glob("*.json"):
            try:
                payload = json.loads(metadata_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("kind") != kind:
                continue
            created_at = payload.get("created_at")
            if not isinstance(created_at, (int, float)):
                created_at = metadata_file.stat().st_mtime
            records.append((float(created_at), payload, metadata_file))
        records.sort(key=lambda item: item[0])
        return records

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                task_type, payload = self._tasks.get(timeout=0.2)
            except queue.Empty:
                continue

            if task_type == "shutdown":
                self._tasks.task_done()
                break

            if task_type == "write_frame":
                self._write_frame_payload(payload)

            self._tasks.task_done()

    def _write_frame_payload(self, payload: dict[str, Any]) -> None:
        frame_id = payload.get("frame_id")
        frame = payload.get("frame")
        frame_path = payload.get("frame_path")
        metadata_path = payload.get("metadata_path")
        if not isinstance(frame_id, str):
            return
        if not isinstance(frame_path, Path):
            return
        if not isinstance(metadata_path, Path):
            return

        expected_frame_path = str(frame_path)

        if not metadata_path.is_file():
            try:
                frame_path.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                self._pending_frames.discard(frame_id)
            return

        metadata_before_write = self._load_metadata(frame_id)
        if (
            not isinstance(metadata_before_write, dict)
            or metadata_before_write.get("status") != "pending"
            or metadata_before_write.get("frame_path") != expected_frame_path
        ):
            try:
                frame_path.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                self._pending_frames.discard(frame_id)
            return

        status = "available"
        try:
            import cv2

            frame_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(frame_path), frame)
            if not ok:
                status = "write_failed"
        except Exception:
            status = "write_failed"

        protection_lock = EvidenceProtectionLock(self._root_dir)
        if not protection_lock.acquire(timeout_seconds=self._cleanup_lock_timeout_seconds):
            try:
                frame_path.unlink(missing_ok=True)
            except Exception:
                pass
            with self._lock:
                self._pending_frames.discard(frame_id)
            return

        try:
            metadata_after_write = self._load_metadata(frame_id)
            if (
                not isinstance(metadata_after_write, dict)
                or metadata_after_write.get("status") != "pending"
                or metadata_after_write.get("frame_path") != expected_frame_path
            ):
                try:
                    frame_path.unlink(missing_ok=True)
                except Exception:
                    pass
                with self._lock:
                    self._pending_frames.discard(frame_id)
                return

            metadata_after_write["status"] = status
            try:
                _json_write(metadata_path, metadata_after_write)
            except Exception:
                status = "write_failed"
                fallback_metadata = self._load_metadata(frame_id)
                if (
                    isinstance(fallback_metadata, dict)
                    and fallback_metadata.get("status") == "pending"
                    and fallback_metadata.get("frame_path") == expected_frame_path
                ):
                    fallback_metadata["status"] = "write_failed"
                    try:
                        _json_write(metadata_path, fallback_metadata)
                    except Exception:
                        pass

            if status != "available":
                try:
                    frame_path.unlink(missing_ok=True)
                except Exception:
                    pass
        finally:
            protection_lock.release()

        with self._lock:
            self._pending_frames.discard(frame_id)


class TrackingEvidenceResolver:
    def __init__(self, root_dir: Path, *, clock: Any | None = None) -> None:
        self._root_dir = root_dir
        self._refs_dir = root_dir / "refs"
        self._clock = clock or time.time

    def _parse_reference(self, reference: str) -> tuple[str | None, str | None]:
        if not isinstance(reference, str):
            return None, None
        prefix = "evidence://"
        if not reference.startswith(prefix):
            return None, None
        remaining = reference[len(prefix):]
        parts = remaining.split("/", maxsplit=1)
        if len(parts) != 2:
            return None, None
        kind, ref_id = parts
        if kind not in {"frame", "detection", "envelope"}:
            return None, None
        if not ref_id.strip():
            return None, None
        return kind, ref_id.strip()

    def _load_metadata(self, ref_id: str) -> dict[str, Any] | None:
        metadata_path = self._refs_dir / f"{ref_id}.json"
        if not metadata_path.is_file():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict):
            return payload
        return None

    def resolve_reference(
        self,
        reference: str,
        *,
        expected_track_id: str | None = None,
        expected_detection_id: str | None = None,
        expected_frame_reference: str | None = None,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = float(self._clock()) if now_timestamp is None else float(now_timestamp)
        kind, ref_id = self._parse_reference(reference)
        if kind is None or ref_id is None:
            return {
                "valid": False,
                "status": "invalid_reference_format",
                "reference": reference,
            }

        metadata = self._load_metadata(ref_id)
        if metadata is None:
            return {
                "valid": False,
                "status": "missing_reference",
                "reference": reference,
            }

        expires_at = metadata.get("expires_at")
        if isinstance(expires_at, (int, float)) and now > float(expires_at):
            return {
                "valid": False,
                "status": "expired_reference",
                "reference": reference,
                "metadata": metadata,
            }

        if kind == "frame":
            frame_path = metadata.get("frame_path")
            if not isinstance(frame_path, str):
                return {
                    "valid": False,
                    "status": "invalid_frame_metadata",
                    "reference": reference,
                    "metadata": metadata,
                }
            if metadata.get("status") != "available":
                if metadata.get("status") == "pending":
                    return {
                        "valid": False,
                        "status": "pending_reference",
                        "reference": reference,
                        "metadata": metadata,
                    }
                return {
                    "valid": False,
                    "status": "frame_unavailable",
                    "reference": reference,
                    "metadata": metadata,
                }
            if not Path(frame_path).is_file():
                return {
                    "valid": False,
                    "status": "missing_frame_payload",
                    "reference": reference,
                    "metadata": metadata,
                }
            return {
                "valid": True,
                "status": "ok",
                "reference": reference,
                "metadata": metadata,
            }

        if kind == "detection":
            if expected_track_id is not None and metadata.get("track_id") != expected_track_id:
                return {
                    "valid": False,
                    "status": "track_linkage_mismatch",
                    "reference": reference,
                    "metadata": metadata,
                }
            if expected_detection_id is not None and metadata.get("detection_id") != expected_detection_id:
                return {
                    "valid": False,
                    "status": "detection_linkage_mismatch",
                    "reference": reference,
                    "metadata": metadata,
                }
            frame_reference = metadata.get("frame_reference")
            if expected_frame_reference is not None and frame_reference != expected_frame_reference:
                return {
                    "valid": False,
                    "status": "frame_linkage_mismatch",
                    "reference": reference,
                    "metadata": metadata,
                }
            if isinstance(frame_reference, str):
                frame_resolution = self.resolve_reference(frame_reference, now_timestamp=now)
                if not frame_resolution.get("valid"):
                    return {
                        "valid": False,
                        "status": "linked_frame_unavailable",
                        "reference": reference,
                        "metadata": metadata,
                        "linked_frame": frame_resolution,
                    }

            return {
                "valid": True,
                "status": "ok",
                "reference": reference,
                "metadata": metadata,
            }

        return {
            "valid": False,
            "status": "unsupported_reference_kind",
            "reference": reference,
            "metadata": metadata,
        }

    def shutdown(self) -> None:
        return None
