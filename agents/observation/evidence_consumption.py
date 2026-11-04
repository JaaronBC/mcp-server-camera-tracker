from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.contracts import TrackingEvidenceEnvelope, TrackingObservationRecord
from agents.coordinator import AgentCoordinator
from agents.observation.evidence_preflight import preflight_tracking_evidence_for_observation
from agents.tracking.evidence_store import TrackingEvidenceResolver


class ObservationEvidenceConsumptionError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = {} if details is None else details


@dataclass(slots=True)
class ObservationEvidenceRequest:
    message_id: str
    now_timestamp: float | None = None
    require_visual_evidence: bool = True
    max_history_records: int = 20


@dataclass(slots=True)
class ObservationImageEvidence:
    reference: str
    frame_path: str
    image_bytes: bytes
    timestamp: float | None
    width: int | None
    height: int | None
    status: str


@dataclass(slots=True)
class ObservationEvidencePackage:
    message_id: str
    envelope: TrackingEvidenceEnvelope
    stored_validation: dict[str, Any]
    preflight: dict[str, Any]
    measurement_provenance: dict[str, dict[str, Any]]
    image_evidence: ObservationImageEvidence | None
    bounded_history: list[TrackingObservationRecord]
    source_references: list[str]


@dataclass(slots=True)
class TrackingEvidenceEnvelopeRecord:
    message_id: str
    stored_at: float | None
    validation: dict[str, Any]
    envelope: TrackingEvidenceEnvelope


def _ensure_nonempty_message_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationEvidenceConsumptionError(
            "invalid_message_id",
            "message_id must be a non-empty string",
            details={"message_id": value},
        )
    return value.strip()


def load_validated_envelope_by_message_id(
    message_id: str,
    *,
    evidence_root_dir: Path,
) -> TrackingEvidenceEnvelopeRecord:
    normalized_message_id = _ensure_nonempty_message_id(message_id)
    envelope_path = evidence_root_dir / "envelopes" / f"{normalized_message_id}.json"

    if not envelope_path.is_file():
        raise ObservationEvidenceConsumptionError(
            "envelope_not_found",
            f"No validated envelope record found for message_id={normalized_message_id}",
            details={"message_id": normalized_message_id, "envelope_path": str(envelope_path)},
        )

    try:
        payload = json.loads(envelope_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ObservationEvidenceConsumptionError(
            "envelope_record_invalid_json",
            "Envelope record is not valid JSON",
            details={"message_id": normalized_message_id, "error": str(error)},
        ) from error

    if not isinstance(payload, dict):
        raise ObservationEvidenceConsumptionError(
            "envelope_record_invalid_format",
            "Envelope record root payload must be a JSON object",
            details={"message_id": normalized_message_id},
        )

    recorded_message_id = payload.get("message_id")
    if recorded_message_id != normalized_message_id:
        raise ObservationEvidenceConsumptionError(
            "envelope_message_id_mismatch",
            "Envelope record message ID does not match requested message ID",
            details={
                "requested_message_id": normalized_message_id,
                "recorded_message_id": recorded_message_id,
            },
        )

    validation_payload = payload.get("validation")
    if not isinstance(validation_payload, dict):
        raise ObservationEvidenceConsumptionError(
            "envelope_validation_missing",
            "Envelope record is missing validation payload",
            details={"message_id": normalized_message_id},
        )

    if not bool(validation_payload.get("valid", False)):
        raise ObservationEvidenceConsumptionError(
            "envelope_validation_failed",
            "Envelope record exists but is not marked valid",
            details={"message_id": normalized_message_id, "validation": validation_payload},
        )

    envelope_payload = payload.get("envelope")
    if not isinstance(envelope_payload, dict):
        raise ObservationEvidenceConsumptionError(
            "envelope_payload_missing",
            "Envelope record is missing envelope payload",
            details={"message_id": normalized_message_id},
        )

    envelope = TrackingEvidenceEnvelope.from_dict(envelope_payload)
    if envelope.message_id != normalized_message_id:
        raise ObservationEvidenceConsumptionError(
            "envelope_payload_message_id_mismatch",
            "Envelope payload message ID does not match requested ID",
            details={
                "requested_message_id": normalized_message_id,
                "payload_message_id": envelope.message_id,
            },
        )

    stored_at = payload.get("stored_at")
    stored_at_value = float(stored_at) if isinstance(stored_at, (int, float)) else None

    return TrackingEvidenceEnvelopeRecord(
        message_id=normalized_message_id,
        stored_at=stored_at_value,
        validation=validation_payload,
        envelope=envelope,
    )


def _safe_load_frame_bytes(
    *,
    frame_reference: str,
    resolver: TrackingEvidenceResolver,
    evidence_root_dir: Path,
    now_timestamp: float | None,
) -> ObservationImageEvidence:
    frame_resolution = resolver.resolve_reference(
        frame_reference,
        now_timestamp=now_timestamp,
    )
    if not bool(frame_resolution.get("valid", False)):
        raise ObservationEvidenceConsumptionError(
            "frame_reference_unresolvable",
            "Frame reference did not resolve to retrievable evidence",
            details={"reference": frame_reference, "result": frame_resolution},
        )

    metadata = frame_resolution.get("metadata")
    if not isinstance(metadata, dict):
        raise ObservationEvidenceConsumptionError(
            "frame_metadata_missing",
            "Resolved frame reference did not include metadata",
            details={"reference": frame_reference},
        )

    frame_path_value = metadata.get("frame_path")
    if not isinstance(frame_path_value, str) or not frame_path_value.strip():
        raise ObservationEvidenceConsumptionError(
            "frame_path_missing",
            "Resolved frame metadata did not include frame path",
            details={"reference": frame_reference, "metadata": metadata},
        )

    frame_path = Path(frame_path_value).resolve()
    frames_root = (evidence_root_dir / "frames").resolve()
    try:
        frame_path.relative_to(frames_root)
    except ValueError as error:
        raise ObservationEvidenceConsumptionError(
            "unsafe_frame_path",
            "Frame path resolves outside evidence frames directory",
            details={
                "reference": frame_reference,
                "frame_path": str(frame_path),
                "frames_root": str(frames_root),
            },
        ) from error

    if not frame_path.is_file():
        raise ObservationEvidenceConsumptionError(
            "frame_file_missing",
            "Resolved frame path does not exist",
            details={"reference": frame_reference, "frame_path": str(frame_path)},
        )

    image_bytes = frame_path.read_bytes()

    timestamp_value = metadata.get("timestamp")
    timestamp = float(timestamp_value) if isinstance(timestamp_value, (int, float)) else None

    width_value = metadata.get("width")
    width = int(width_value) if isinstance(width_value, (int, float)) else None

    height_value = metadata.get("height")
    height = int(height_value) if isinstance(height_value, (int, float)) else None

    return ObservationImageEvidence(
        reference=frame_reference,
        frame_path=str(frame_path),
        image_bytes=image_bytes,
        timestamp=timestamp,
        width=width,
        height=height,
        status=str(frame_resolution.get("status", "ok")),
    )


def consume_tracking_evidence_by_message_id(
    request: ObservationEvidenceRequest,
    *,
    evidence_root_dir: Path,
    resolver: TrackingEvidenceResolver | None = None,
    coordinator: AgentCoordinator | None = None,
) -> ObservationEvidencePackage:
    normalized_message_id = _ensure_nonempty_message_id(request.message_id)
    max_history_records = max(1, int(request.max_history_records))

    envelope_record = load_validated_envelope_by_message_id(
        normalized_message_id,
        evidence_root_dir=evidence_root_dir,
    )

    selected_resolver = resolver or TrackingEvidenceResolver(evidence_root_dir)
    selected_coordinator = coordinator or AgentCoordinator()

    preflight = preflight_tracking_evidence_for_observation(
        envelope_record.envelope,
        resolver=selected_resolver,
        coordinator=selected_coordinator,
        now_timestamp=request.now_timestamp,
        require_visual_evidence=request.require_visual_evidence,
    )

    if not bool(preflight.get("preflight_passed", False)):
        raise ObservationEvidenceConsumptionError(
            "observation_evidence_preflight_failed",
            "Observation evidence preflight failed",
            details={
                "message_id": normalized_message_id,
                "gate_errors": preflight.get("gate_errors", []),
                "preflight": preflight,
            },
        )

    image_evidence: ObservationImageEvidence | None = None
    frame_reference = envelope_record.envelope.frame_reference
    if request.require_visual_evidence:
        if not isinstance(frame_reference, str) or not frame_reference.strip():
            raise ObservationEvidenceConsumptionError(
                "frame_reference_required",
                "Visual evidence is required but frame reference is missing",
                details={"message_id": normalized_message_id},
            )
        image_evidence = _safe_load_frame_bytes(
            frame_reference=frame_reference,
            resolver=selected_resolver,
            evidence_root_dir=evidence_root_dir,
            now_timestamp=request.now_timestamp,
        )

    bounded_history = list(envelope_record.envelope.track_history[-max_history_records:])

    return ObservationEvidencePackage(
        message_id=normalized_message_id,
        envelope=envelope_record.envelope,
        stored_validation=envelope_record.validation,
        preflight=preflight,
        measurement_provenance=preflight.get("measurement_provenance", {}),
        image_evidence=image_evidence,
        bounded_history=bounded_history,
        source_references=list(envelope_record.envelope.source_evidence_refs),
    )
