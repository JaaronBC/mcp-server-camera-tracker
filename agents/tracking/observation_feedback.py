from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

from agents.contracts import (
    TrackingObservationFeedbackArtifact,
    TrackingObservationFeedbackRecord,
)

TRACKING_OBSERVATION_FEEDBACK_SCHEMA_VERSION = "1.0"


class TrackingObservationFeedbackConsumer:
    def __init__(
        self,
        *,
        artifact_path: Path,
        max_seen_feedback_ids: int = 1024,
        max_artifact_bytes: int = 512_000,
    ) -> None:
        self._artifact_path = artifact_path
        self._max_artifact_bytes = max(4_096, int(max_artifact_bytes))
        self._seen_feedback_ids: set[str] = set()
        self._seen_feedback_order: deque[str] = deque(maxlen=max(32, int(max_seen_feedback_ids)))

    def poll(
        self,
        *,
        current_track_ids: set[str],
        temporarily_lost_track_ids: set[str] | None = None,
        latest_message_id_by_track: dict[str, str],
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now_timestamp is None else float(now_timestamp)
        payload = self._load_payload()
        if payload is None:
            return {
                "accepted": [],
                "accepted_count": 0,
                "rejected_count": 0,
                "error": None,
                "evaluated_records": [],
                "consumer_outcome": {
                    "status": "ok",
                    "evaluated_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "duplicate_count": 0,
                },
            }

        artifact = TrackingObservationFeedbackArtifact.from_dict(payload)
        if artifact.schema_version != TRACKING_OBSERVATION_FEEDBACK_SCHEMA_VERSION:
            return {
                "accepted": [],
                "accepted_count": 0,
                "rejected_count": 0,
                "error": "schema_mismatch",
                "evaluated_records": [],
                "consumer_outcome": {
                    "status": "schema_mismatch",
                    "evaluated_count": 0,
                    "accepted_count": 0,
                    "rejected_count": 0,
                    "duplicate_count": 0,
                },
            }

        accepted: list[dict[str, Any]] = []
        evaluated_records: list[dict[str, Any]] = []
        rejected_count = 0
        duplicate_count = 0

        for record in artifact.records:
            record_view = {
                "feedback_id": record.feedback_id,
                "verification_request_id": record.verification_request_id,
                "track_id": record.track_id,
                "evidence_message_id": record.evidence_message_id,
                "verification_observation_id": record.verification_observation_id,
            }

            valid, rejection_reason = self._validate_record(
                record,
                now_timestamp=now,
                current_track_ids=current_track_ids,
                temporarily_lost_track_ids=temporarily_lost_track_ids,
                latest_message_id_by_track=latest_message_id_by_track,
            )
            if not valid:
                if rejection_reason is not None:
                    rejected_count += 1
                evaluated_records.append(
                    {
                        **record_view,
                        "decision": "rejected",
                        "reason": rejection_reason,
                    }
                )
                continue

            if record.feedback_id in self._seen_feedback_ids:
                duplicate_count += 1
                evaluated_records.append(
                    {
                        **record_view,
                        "decision": "duplicate",
                        "reason": "already_seen",
                    }
                )
                continue

            self._track_seen_feedback(record.feedback_id)
            accepted.append(record.to_dict())
            evaluated_records.append(
                {
                    **record_view,
                    "decision": "accepted",
                    "reason": None,
                }
            )

        accepted_count = len(accepted)
        return {
            "accepted": accepted,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "error": None,
            "evaluated_records": evaluated_records,
            "consumer_outcome": {
                "status": "ok",
                "evaluated_count": len(evaluated_records),
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "duplicate_count": duplicate_count,
            },
        }

    def _load_payload(self) -> dict[str, Any] | None:
        if not self._artifact_path.is_file():
            return None
        try:
            stat = self._artifact_path.stat()
        except OSError:
            return None
        if stat.st_size <= 0:
            return None
        if stat.st_size > self._max_artifact_bytes:
            return None

        try:
            text = self._artifact_path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _validate_record(
        self,
        record: TrackingObservationFeedbackRecord,
        *,
        now_timestamp: float,
        current_track_ids: set[str],
        temporarily_lost_track_ids: set[str] | None,
        latest_message_id_by_track: dict[str, str],
    ) -> tuple[bool, str | None]:
        required_strings = [
            record.feedback_id,
            record.verification_request_id,
            record.track_id,
            record.evidence_message_id,
            record.verification_observation_id,
            record.verification_message_id,
            record.source,
        ]
        if any(not isinstance(value, str) or not value.strip() for value in required_strings):
            return False, "malformed"

        if record.current_applicability not in {"current", "historical"}:
            return False, "malformed"

        if not math.isfinite(float(record.verification_timestamp)):
            return False, "malformed"

        if not record.coordinator_valid or record.validation_outcome != "accepted":
            return False, "unvalidated"

        if record.expires_at <= now_timestamp:
            return False, "stale"

        if record.current_applicability == "current":
            eligible_current_track_ids = set(current_track_ids)
            if isinstance(temporarily_lost_track_ids, set):
                eligible_current_track_ids.update(temporarily_lost_track_ids)
            if record.stale_for_current_decisions:
                return False, "stale_for_current_decisions"
            if record.track_id not in eligible_current_track_ids:
                return False, "mismatched_track"
            latest_message_id = latest_message_id_by_track.get(record.track_id)
            if not isinstance(latest_message_id, str) or not latest_message_id:
                return False, "missing_track_message"
            if latest_message_id != record.evidence_message_id:
                return False, "mismatched_evidence_reference"

        return True, None

    def _track_seen_feedback(self, feedback_id: str) -> None:
        if feedback_id in self._seen_feedback_ids:
            return
        if len(self._seen_feedback_order) == self._seen_feedback_order.maxlen:
            oldest = self._seen_feedback_order[0]
            self._seen_feedback_ids.discard(oldest)
        self._seen_feedback_order.append(feedback_id)
        self._seen_feedback_ids.add(feedback_id)
