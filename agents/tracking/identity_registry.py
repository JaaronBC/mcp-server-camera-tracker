from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

IdentityLifecycleStatus = Literal["visible", "temporarily_lost", "retired"]


@dataclass(slots=True)
class IdentityHistoryEntry:
    generation: int
    timestamp: float
    event: str
    detection_id: str | None
    detection_reference: str | None
    yolo_track_id: str | None


@dataclass(slots=True)
class PersistentIdentityRecord:
    app_identity_id: str
    provisional: bool
    status: IdentityLifecycleStatus
    generation: int
    version: int
    created_at: float
    updated_at: float
    detection_id: str | None
    detection_reference: str | None
    yolo_track_id: str | None
    loss_started_at: float | None
    history: deque[IdentityHistoryEntry]


@dataclass(slots=True)
class IdentityMutationResult:
    accepted: bool
    reason: str | None
    app_identity_id: str
    generation: int
    version: int


class PersistentIdentityRegistry:
    def __init__(
        self,
        *,
        max_history_per_identity: int = 20,
        temporary_loss_seconds: float = 1.5,
        retirement_seconds: float = 8.0,
        max_retired_records: int = 256,
    ) -> None:
        self._max_history_per_identity = max(1, int(max_history_per_identity))
        self._temporary_loss_seconds = max(0.1, float(temporary_loss_seconds))
        self._retirement_seconds = max(self._temporary_loss_seconds, float(retirement_seconds))
        self._max_retired_records = max(1, int(max_retired_records))

        self._lock = Lock()
        self._generation = 0
        self._version = 0
        self._identity_counter = 0

        self._records: dict[str, PersistentIdentityRecord] = {}
        self._active_yolo_to_identity: dict[str, str] = {}
        self._retired_ids: deque[str] = deque()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def process_detections(
        self,
        *,
        now: float,
        detections: list[Any],
    ) -> dict[str, Any]:
        with self._lock:
            self._generation += 1
            generation = self._generation

            visible_yolo_ids: set[str] = set()
            visible_identity_ids: list[str] = []
            created_count = 0

            for index, detection in enumerate(detections):
                detection_id = self._as_non_empty_string(getattr(detection, "detection_id", None))
                yolo_track_id = self._as_non_empty_string(getattr(detection, "yolo_track_id", None))
                detection_reference = self._resolve_detection_reference(
                    detection=detection,
                    generation=generation,
                    detection_index=index,
                    detection_id=detection_id,
                    yolo_track_id=yolo_track_id,
                )

                if yolo_track_id is None:
                    continue

                visible_yolo_ids.add(yolo_track_id)

                app_identity_id = self._active_yolo_to_identity.get(yolo_track_id)
                if app_identity_id is None:
                    app_identity_id = self._create_identity(
                        now=now,
                        generation=generation,
                        detection_id=detection_id,
                        detection_reference=detection_reference,
                        yolo_track_id=yolo_track_id,
                    )
                    self._active_yolo_to_identity[yolo_track_id] = app_identity_id
                    created_count += 1

                record = self._records.get(app_identity_id)
                if record is None:
                    continue

                record.status = "visible"
                record.generation = generation
                record.version += 1
                record.updated_at = float(now)
                record.loss_started_at = None
                record.detection_id = detection_id
                record.detection_reference = detection_reference
                record.yolo_track_id = yolo_track_id
                self._append_history(
                    record,
                    generation=generation,
                    timestamp=now,
                    event="visible",
                    detection_id=detection_id,
                    detection_reference=detection_reference,
                    yolo_track_id=yolo_track_id,
                )

                if hasattr(detection, "app_identity_id"):
                    try:
                        setattr(detection, "app_identity_id", app_identity_id)
                    except Exception:
                        pass

                visible_identity_ids.append(app_identity_id)

            lost_count = 0
            retired_count = 0
            for yolo_track_id, app_identity_id in list(self._active_yolo_to_identity.items()):
                if yolo_track_id in visible_yolo_ids:
                    continue

                record = self._records.get(app_identity_id)
                if record is None:
                    self._active_yolo_to_identity.pop(yolo_track_id, None)
                    continue

                if record.status == "retired":
                    self._active_yolo_to_identity.pop(yolo_track_id, None)
                    continue

                if record.loss_started_at is None:
                    record.loss_started_at = float(now)

                elapsed = float(now) - float(record.loss_started_at)
                if elapsed >= self._retirement_seconds:
                    self._retire_identity(record, generation=generation, timestamp=now)
                    self._active_yolo_to_identity.pop(yolo_track_id, None)
                    retired_count += 1
                    continue

                record.status = "temporarily_lost"
                record.generation = generation
                record.version += 1
                record.updated_at = float(now)
                self._append_history(
                    record,
                    generation=generation,
                    timestamp=now,
                    event="temporarily_lost",
                    detection_id=record.detection_id,
                    detection_reference=record.detection_reference,
                    yolo_track_id=record.yolo_track_id,
                )
                lost_count += 1

            self._version += 1
            self._enforce_retirement_bounds()

            return {
                "enabled": True,
                "generation": generation,
                "version": self._version,
                "created_count": created_count,
                "visible_count": len(visible_identity_ids),
                "temporarily_lost_count": lost_count,
                "retired_count": retired_count,
                "active_assignment_count": len(self._active_yolo_to_identity),
                "temporary_loss_seconds": self._temporary_loss_seconds,
            }

    def apply_identity_mutation(
        self,
        *,
        app_identity_id: str,
        new_yolo_track_id: str,
        now: float,
        expected_generation_at_least: int | None = None,
        expected_version: int | None = None,
    ) -> IdentityMutationResult:
        with self._lock:
            record = self._records.get(app_identity_id)
            if record is None:
                return IdentityMutationResult(
                    accepted=False,
                    reason="unknown_identity",
                    app_identity_id=app_identity_id,
                    generation=self._generation,
                    version=self._version,
                )

            if record.status == "retired":
                return IdentityMutationResult(
                    accepted=False,
                    reason="identity_retired",
                    app_identity_id=app_identity_id,
                    generation=self._generation,
                    version=self._version,
                )

            if (
                expected_generation_at_least is not None
                and int(expected_generation_at_least) < int(record.generation)
            ):
                return IdentityMutationResult(
                    accepted=False,
                    reason="stale_generation",
                    app_identity_id=app_identity_id,
                    generation=self._generation,
                    version=self._version,
                )

            if expected_version is not None and int(expected_version) != int(self._version):
                return IdentityMutationResult(
                    accepted=False,
                    reason="version_conflict",
                    app_identity_id=app_identity_id,
                    generation=self._generation,
                    version=self._version,
                )

            normalized_track_id = self._as_non_empty_string(new_yolo_track_id)
            if normalized_track_id is None:
                return IdentityMutationResult(
                    accepted=False,
                    reason="invalid_track_id",
                    app_identity_id=app_identity_id,
                    generation=self._generation,
                    version=self._version,
                )

            existing_owner = self._active_yolo_to_identity.get(normalized_track_id)
            if existing_owner is not None and existing_owner != app_identity_id:
                return IdentityMutationResult(
                    accepted=False,
                    reason="track_id_already_assigned",
                    app_identity_id=app_identity_id,
                    generation=self._generation,
                    version=self._version,
                )

            old_track_id = self._as_non_empty_string(record.yolo_track_id)
            if old_track_id is not None and old_track_id != normalized_track_id:
                self._active_yolo_to_identity.pop(old_track_id, None)

            self._active_yolo_to_identity[normalized_track_id] = app_identity_id

            record.generation = self._generation
            record.version += 1
            record.updated_at = float(now)
            record.status = "visible"
            record.yolo_track_id = normalized_track_id
            record.loss_started_at = None

            self._append_history(
                record,
                generation=self._generation,
                timestamp=now,
                event="mutation_applied",
                detection_id=record.detection_id,
                detection_reference=record.detection_reference,
                yolo_track_id=record.yolo_track_id,
            )

            self._version += 1
            return IdentityMutationResult(
                accepted=True,
                reason=None,
                app_identity_id=app_identity_id,
                generation=self._generation,
                version=self._version,
            )

    def get_record(self, app_identity_id: str) -> PersistentIdentityRecord | None:
        with self._lock:
            record = self._records.get(app_identity_id)
            if record is None:
                return None

            return PersistentIdentityRecord(
                app_identity_id=record.app_identity_id,
                provisional=record.provisional,
                status=record.status,
                generation=record.generation,
                version=record.version,
                created_at=record.created_at,
                updated_at=record.updated_at,
                detection_id=record.detection_id,
                detection_reference=record.detection_reference,
                yolo_track_id=record.yolo_track_id,
                loss_started_at=record.loss_started_at,
                history=deque(record.history, maxlen=self._max_history_per_identity),
            )

    def get_identity_for_yolo_track(self, yolo_track_id: str) -> str | None:
        with self._lock:
            return self._active_yolo_to_identity.get(str(yolo_track_id))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            visible = 0
            lost = 0
            retired = 0
            for record in self._records.values():
                if record.status == "visible":
                    visible += 1
                elif record.status == "temporarily_lost":
                    lost += 1
                elif record.status == "retired":
                    retired += 1

            return {
                "generation": self._generation,
                "version": self._version,
                "total_records": len(self._records),
                "visible_count": visible,
                "temporarily_lost_count": lost,
                "retired_count": retired,
                "active_assignment_count": len(self._active_yolo_to_identity),
            }

    def capture_read_only_snapshot(
        self,
        *,
        max_identities: int,
        max_history_per_identity: int,
        max_historical_track_ids: int,
    ) -> dict[str, Any]:
        selected_max_identities = max(1, int(max_identities))
        selected_max_history = max(1, int(max_history_per_identity))
        selected_max_historical_track_ids = max(1, int(max_historical_track_ids))

        with self._lock:
            generation = self._generation
            version = self._version
            records_total = len(self._records)
            copied_records: list[dict[str, Any]] = []

            for record in self._records.values():
                copied_history = [
                    {
                        "generation": int(item.generation),
                        "timestamp": float(item.timestamp),
                        "event": str(item.event),
                        "detection_id": item.detection_id,
                        "detection_reference": item.detection_reference,
                        "yolo_track_id": item.yolo_track_id,
                    }
                    for item in record.history
                ]
                copied_records.append(
                    {
                        "app_identity_id": record.app_identity_id,
                        "provisional": bool(record.provisional),
                        "status": record.status,
                        "generation": int(record.generation),
                        "identity_version": int(record.version),
                        "created_at": float(record.created_at),
                        "updated_at": float(record.updated_at),
                        "detection_id": record.detection_id,
                        "detection_reference": record.detection_reference,
                        "yolo_track_id": record.yolo_track_id,
                        "loss_started_at": record.loss_started_at,
                        "history": copied_history,
                    }
                )

        copied_records.sort(key=lambda item: float(item.get("updated_at", 0.0)), reverse=True)
        selected_records = copied_records[:selected_max_identities]
        identities_truncated_by_count = len(copied_records) > len(selected_records)

        history_truncated = False
        historical_track_ids_truncated = False
        identities: list[dict[str, Any]] = []

        for record in selected_records:
            history = record.get("history")
            history_items = list(history) if isinstance(history, list) else []
            if len(history_items) > selected_max_history:
                history_items = history_items[-selected_max_history:]
                history_truncated = True

            historical_track_ids: list[str] = []
            seen_track_ids: set[str] = set()
            for entry in history_items:
                yolo_track_id = entry.get("yolo_track_id") if isinstance(entry, dict) else None
                if not isinstance(yolo_track_id, str) or not yolo_track_id:
                    continue
                if yolo_track_id in seen_track_ids:
                    continue
                seen_track_ids.add(yolo_track_id)
                historical_track_ids.append(yolo_track_id)

            if len(historical_track_ids) > selected_max_historical_track_ids:
                historical_track_ids = historical_track_ids[-selected_max_historical_track_ids:]
                historical_track_ids_truncated = True

            identities.append(
                {
                    "app_identity_id": record.get("app_identity_id"),
                    "provisional": bool(record.get("provisional", False)),
                    "status": record.get("status"),
                    "generation": int(record.get("generation", 0)),
                    "identity_version": int(record.get("identity_version", 0)),
                    "created_at": float(record.get("created_at", 0.0)),
                    "updated_at": float(record.get("updated_at", 0.0)),
                    "detection_id": record.get("detection_id"),
                    "detection_reference": record.get("detection_reference"),
                    "yolo_track_id": record.get("yolo_track_id"),
                    "loss_started_at": record.get("loss_started_at"),
                    "historical_yolo_track_ids": historical_track_ids,
                    "observation_history": history_items,
                }
            )

        return {
            "generation": generation,
            "version": version,
            "total_records": records_total,
            "identities": identities,
            "identities_truncated_by_count": identities_truncated_by_count,
            "history_truncated": history_truncated,
            "historical_track_ids_truncated": historical_track_ids_truncated,
        }

    def _create_identity(
        self,
        *,
        now: float,
        generation: int,
        detection_id: str | None,
        detection_reference: str,
        yolo_track_id: str,
    ) -> str:
        self._identity_counter += 1
        app_identity_id = f"app-{self._identity_counter}"

        record = PersistentIdentityRecord(
            app_identity_id=app_identity_id,
            provisional=True,
            status="visible",
            generation=generation,
            version=1,
            created_at=float(now),
            updated_at=float(now),
            detection_id=detection_id,
            detection_reference=detection_reference,
            yolo_track_id=yolo_track_id,
            loss_started_at=None,
            history=deque(maxlen=self._max_history_per_identity),
        )
        self._append_history(
            record,
            generation=generation,
            timestamp=now,
            event="created",
            detection_id=detection_id,
            detection_reference=detection_reference,
            yolo_track_id=yolo_track_id,
        )
        self._records[app_identity_id] = record
        return app_identity_id

    def _retire_identity(
        self,
        record: PersistentIdentityRecord,
        *,
        generation: int,
        timestamp: float,
    ) -> None:
        record.status = "retired"
        record.generation = generation
        record.version += 1
        record.updated_at = float(timestamp)
        self._append_history(
            record,
            generation=generation,
            timestamp=timestamp,
            event="retired",
            detection_id=record.detection_id,
            detection_reference=record.detection_reference,
            yolo_track_id=record.yolo_track_id,
        )
        self._retired_ids.append(record.app_identity_id)

    def _enforce_retirement_bounds(self) -> None:
        if len(self._retired_ids) <= self._max_retired_records:
            return

        while len(self._retired_ids) > self._max_retired_records:
            stale_identity_id = self._retired_ids.popleft()
            self._records.pop(stale_identity_id, None)

    def _append_history(
        self,
        record: PersistentIdentityRecord,
        *,
        generation: int,
        timestamp: float,
        event: str,
        detection_id: str | None,
        detection_reference: str | None,
        yolo_track_id: str | None,
    ) -> None:
        record.history.append(
            IdentityHistoryEntry(
                generation=int(generation),
                timestamp=float(timestamp),
                event=str(event),
                detection_id=detection_id,
                detection_reference=detection_reference,
                yolo_track_id=yolo_track_id,
            )
        )

    @staticmethod
    def _resolve_detection_reference(
        *,
        detection: Any,
        generation: int,
        detection_index: int,
        detection_id: str | None,
        yolo_track_id: str | None,
    ) -> str:
        candidate_reference = PersistentIdentityRegistry._as_non_empty_string(
            getattr(detection, "detection_reference", None)
        )
        if candidate_reference is not None:
            return candidate_reference

        return PersistentIdentityRegistry._build_detection_reference(
            generation=generation,
            detection_index=detection_index,
            detection_id=detection_id,
            yolo_track_id=yolo_track_id,
        )

    @staticmethod
    def _build_detection_reference(
        *,
        generation: int,
        detection_index: int,
        detection_id: str | None,
        yolo_track_id: str | None,
    ) -> str:
        detection_part = detection_id or "unknown"
        yolo_part = yolo_track_id or "none"
        return f"frame:{generation}:idx:{detection_index}:det:{detection_part}:yolo:{yolo_part}"

    @staticmethod
    def _as_non_empty_string(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None
