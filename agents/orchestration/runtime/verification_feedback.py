from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from agents.contracts import (
    TrackingObservationFeedbackArtifact,
    TrackingObservationFeedbackRecord,
)

TRACKING_OBSERVATION_FEEDBACK_SCHEMA_VERSION = "1.0"
MAX_FINDING_REFS = 64
MAX_FINDING_LIST = 64
MAX_FINDING_ITEM_LENGTH = 256
VALID_VERIFICATION_STATUSES = {"supported", "contradicted", "inconclusive"}
VALID_IDENTITY_CONSISTENCY = {"consistent", "inconsistent", "unknown"}
VALID_VISIBILITY_STATUSES = {
    "visible",
    "partially_occluded",
    "occluded",
    "not_visible",
    "unknown",
}
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


class VerificationFeedbackPublisher:
    def __init__(
        self,
        *,
        artifact_path: Path,
        max_records: int = 256,
        record_ttl_seconds: float = 45.0,
    ) -> None:
        self._artifact_path = artifact_path
        self._max_records = max(1, int(max_records))
        self._record_ttl_seconds = max(1.0, float(record_ttl_seconds))

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    def publish_from_results(
        self,
        results: list[dict[str, Any]],
        *,
        now_timestamp: float | None = None,
    ) -> dict[str, int]:
        now = time.time() if now_timestamp is None else float(now_timestamp)
        existing = self._load_existing_records(now)
        existing_by_id = {item.feedback_id: item for item in existing}

        published = 0
        malformed = 0
        skipped = 0

        for item in results:
            candidate = self._build_feedback_record(item, now_timestamp=now)
            if candidate is None:
                malformed += 1
                continue
            if candidate.feedback_id in existing_by_id:
                skipped += 1
                continue
            existing_by_id[candidate.feedback_id] = candidate
            published += 1

        merged = list(existing_by_id.values())
        merged.sort(key=lambda item: item.finished_at if isinstance(item.finished_at, (int, float)) else item.published_at)
        if len(merged) > self._max_records:
            merged = merged[-self._max_records :]

        artifact = TrackingObservationFeedbackArtifact(
            schema_version=TRACKING_OBSERVATION_FEEDBACK_SCHEMA_VERSION,
            generated_at=now,
            records=merged,
        )
        self._atomic_write(artifact)

        return {
            "published": published,
            "malformed": malformed,
            "skipped": skipped,
            "retained": len(merged),
        }

    def _load_existing_records(self, now_timestamp: float) -> list[TrackingObservationFeedbackRecord]:
        if not self._artifact_path.is_file():
            return []

        try:
            payload = json.loads(self._artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, dict):
            return []

        artifact = TrackingObservationFeedbackArtifact.from_dict(payload)
        if artifact.schema_version != TRACKING_OBSERVATION_FEEDBACK_SCHEMA_VERSION:
            return []

        retained: list[TrackingObservationFeedbackRecord] = []
        for item in artifact.records:
            if not self._record_has_required_fields(item):
                continue
            if item.expires_at <= now_timestamp:
                continue
            retained.append(item)
        return retained

    def _record_has_required_fields(self, item: TrackingObservationFeedbackRecord) -> bool:
        required_strings = [
            item.feedback_id,
            item.verification_request_id,
            item.track_id,
            item.evidence_message_id,
            item.verification_observation_id,
            item.verification_message_id,
            item.source,
        ]
        if any(not isinstance(value, str) or not value.strip() for value in required_strings):
            return False
        if item.current_applicability not in {"current", "historical"}:
            return False
        if not isinstance(item.coordinator_valid, bool):
            return False
        if not isinstance(item.stale_for_current_decisions, bool):
            return False
        if not isinstance(item.historical_observation, bool):
            return False
        if not math.isfinite(float(item.verification_timestamp)):
            return False
        if not math.isfinite(float(item.published_at)):
            return False
        if not math.isfinite(float(item.expires_at)):
            return False
        return True

    def _build_feedback_record(
        self,
        result: dict[str, Any],
        *,
        now_timestamp: float,
    ) -> TrackingObservationFeedbackRecord | None:
        if not isinstance(result, dict):
            return None

        if result.get("status") != "completed":
            return None
        if result.get("validation_outcome") != "accepted":
            return None
        if not bool(result.get("coordinator_valid", False)):
            return None

        request_id = result.get("request_id")
        track_id = result.get("track_id")
        message_id = result.get("evidence_message_id")
        source = result.get("source")
        if not isinstance(request_id, str) or not request_id.strip():
            return None
        if not isinstance(track_id, str) or not track_id.strip():
            return None
        if not isinstance(message_id, str) or not message_id.strip():
            return None
        if not isinstance(source, str) or not source.strip():
            return None

        details = result.get("details")
        if not isinstance(details, dict):
            return None

        verification_observation_id = details.get("verification_observation_id")
        verification_message_id = details.get("verification_referenced_message_id")
        verification_timestamp = details.get("verification_timestamp")
        if not isinstance(verification_observation_id, str) or not verification_observation_id.strip():
            return None
        if not isinstance(verification_message_id, str) or not verification_message_id.strip():
            return None
        if not isinstance(verification_timestamp, (int, float)):
            return None

        findings = self._build_validated_findings(details)
        if findings is None:
            return None

        applicability = result.get("current_applicability", "historical")
        if applicability not in {"current", "historical"}:
            return None

        stale_for_current_decisions = bool(result.get("stale_for_current_decisions", False))
        historical_observation = applicability == "historical" or stale_for_current_decisions

        submitted_at = result.get("submitted_at")
        started_at = result.get("started_at")
        finished_at = result.get("finished_at")

        return TrackingObservationFeedbackRecord(
            schema_version=TRACKING_OBSERVATION_FEEDBACK_SCHEMA_VERSION,
            feedback_id=f"obsfeedback-{request_id}",
            verification_request_id=request_id,
            track_id=track_id,
            evidence_message_id=message_id,
            verification_observation_id=verification_observation_id,
            verification_message_id=verification_message_id,
            verification_timestamp=float(verification_timestamp),
            source=source,
            current_applicability=applicability,
            historical_observation=historical_observation,
            stale_for_current_decisions=stale_for_current_decisions,
            coordinator_valid=True,
            validation_outcome="accepted",
            submitted_at=(float(submitted_at) if isinstance(submitted_at, (int, float)) else None),
            started_at=(float(started_at) if isinstance(started_at, (int, float)) else None),
            finished_at=(float(finished_at) if isinstance(finished_at, (int, float)) else None),
            published_at=now_timestamp,
            expires_at=now_timestamp + self._record_ttl_seconds,
            provenance={
                "publisher": "verification_queue",
                "source": source,
            },
            details=findings,
        )

    def _build_validated_findings(self, details: dict[str, Any]) -> dict[str, Any] | None:
        verification_status = details.get("verification_status")
        identity_consistency = details.get("identity_consistency")
        visibility_status = details.get("visibility_status")
        possible_id_switch = details.get("possible_id_switch")
        confidence = details.get("confidence")
        visual_content_inspected = details.get("visual_content_inspected")
        if verification_status not in VALID_VERIFICATION_STATUSES:
            return None
        if identity_consistency not in VALID_IDENTITY_CONSISTENCY:
            return None
        if visibility_status not in VALID_VISIBILITY_STATUSES:
            return None
        if not isinstance(possible_id_switch, bool):
            return None
        if visual_content_inspected is not None and not isinstance(visual_content_inspected, bool):
            return None
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                return None
            confidence_value = float(confidence)
            if not math.isfinite(confidence_value):
                return None
            if confidence_value < MIN_CONFIDENCE or confidence_value > MAX_CONFIDENCE:
                return None
            confidence = confidence_value

        source_evidence_refs = self._sanitize_string_list(
            details.get("source_evidence_refs"),
            max_items=MAX_FINDING_REFS,
        )
        supporting_evidence_refs = self._sanitize_string_list(
            details.get("supporting_evidence_refs"),
            max_items=MAX_FINDING_REFS,
        )
        contradicting_evidence_refs = self._sanitize_string_list(
            details.get("contradicting_evidence_refs"),
            max_items=MAX_FINDING_REFS,
        )
        uncertainty = self._sanitize_string_list(
            details.get("uncertainty"),
            max_items=MAX_FINDING_LIST,
        )
        limitations = self._sanitize_string_list(
            details.get("limitations"),
            max_items=MAX_FINDING_LIST,
        )

        if source_evidence_refs is None:
            return None
        if supporting_evidence_refs is None:
            return None
        if contradicting_evidence_refs is None:
            return None
        if uncertainty is None:
            return None
        if limitations is None:
            return None

        return {
            "verification_status": verification_status,
            "identity_consistency": identity_consistency,
            "visibility_status": visibility_status,
            "possible_id_switch": possible_id_switch,
            "confidence": confidence,
            "visual_content_inspected": visual_content_inspected,
            "source_evidence_refs": source_evidence_refs,
            "supporting_evidence_refs": supporting_evidence_refs,
            "contradicting_evidence_refs": contradicting_evidence_refs,
            "uncertainty": uncertainty,
            "limitations": limitations,
        }

    def _sanitize_string_list(self, value: Any, *, max_items: int) -> list[str] | None:
        if value is None:
            return []
        if not isinstance(value, list):
            return None

        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return None
            normalized = item.strip()
            if not normalized:
                continue
            if len(normalized) > MAX_FINDING_ITEM_LENGTH:
                normalized = normalized[:MAX_FINDING_ITEM_LENGTH]
            items.append(normalized)

        if len(items) > max_items:
            return None
        return items

    def _atomic_write(self, artifact: TrackingObservationFeedbackArtifact) -> None:
        payload = artifact.to_dict()
        self._artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._artifact_path.with_name(f"{self._artifact_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temp_path, self._artifact_path)
