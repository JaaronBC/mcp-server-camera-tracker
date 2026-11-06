from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from typing import Any

from agents.contracts import (
    GroundPoint,
    TrackingClaims,
    TrackingEvidenceEnvelope,
    TrackingObservationRecord,
)
from agents.coordinator import AgentCoordinator

from .evidence_store import FrameEvidenceRecord, TrackingEvidenceStore
from .tracking_types import PixelPosition


def _to_ground_point(candidate: Any) -> GroundPoint | None:
    if not isinstance(getattr(candidate, "ground_x", None), (int, float)):
        return None
    if not isinstance(getattr(candidate, "ground_z", None), (int, float)):
        return None
    unit = getattr(candidate, "ground_coordinate_unit", None)
    if not isinstance(unit, str) or not unit.strip():
        return None
    return GroundPoint(
        x=float(candidate.ground_x),
        z=float(candidate.ground_z),
        unit=unit,
    )


@dataclass(slots=True)
class PackagedEvidenceResult:
    envelope: TrackingEvidenceEnvelope
    validation: dict[str, Any]
    envelope_reference: str | None


class TrackingEvidencePackager:
    def __init__(
        self,
        *,
        coordinator: AgentCoordinator,
        evidence_store: TrackingEvidenceStore,
        max_history_per_track: int = 20,
        schema_version: str = "1.0",
    ) -> None:
        self._coordinator = coordinator
        self._evidence_store = evidence_store
        self._schema_version = schema_version
        self._seen_message_ids: set[str] = set()
        self._history: dict[str, deque[TrackingObservationRecord]] = defaultdict(
            lambda: deque(maxlen=max_history_per_track)
        )
        self._message_counter = 0

    def package_detection(
        self,
        *,
        detection: Any,
        timestamp: float,
        frame_evidence: FrameEvidenceRecord | None,
        update_payload: dict[str, Any] | None,
    ) -> PackagedEvidenceResult:
        self._message_counter += 1

        track_id = self._resolve_track_id(detection, update_payload)
        detection_id = str(getattr(detection, "detection_id", "unknown"))

        message_id = (
            f"trkmsg-{self._evidence_store.process_nonce}-"
            f"{int(float(timestamp) * 1_000_000)}-"
            f"{self._message_counter}-{track_id}-{detection_id}"
        )

        uncertainty: list[str] = []
        quality_flags: list[str] = []

        if getattr(detection, "relative_depth", None) is None:
            uncertainty.append("relative_depth_unavailable")

        position_kind = str(getattr(detection, "position_kind", "unknown"))
        quality_flags.append(f"position_kind:{position_kind}")
        if position_kind != "calibrated_ground_plane":
            uncertainty.append("ground_position_relative_or_estimated")

        tracking_claims = self._build_claims(
            detection=detection,
            update_payload=update_payload,
            timestamp=timestamp,
        )

        frame_reference = None
        source_refs: list[str] = []
        if frame_evidence is not None:
            frame_reference = frame_evidence.frame_reference
            if frame_evidence.status == "available":
                source_refs.append(frame_reference)
            else:
                uncertainty.append("frame_evidence_pending_or_unavailable")
        else:
            uncertainty.append("frame_evidence_unavailable")

        pixel_bbox = getattr(detection, "pixel_bbox", None)
        pixel_center = PixelPosition(
            x=float(getattr(detection, "pixel_center_x", 0.0)),
            y=float(getattr(detection, "pixel_center_y", 0.0)),
        )
        ground_point = _to_ground_point(detection)

        observation_record = TrackingObservationRecord(
            observation_id=f"obs-{track_id}-{int(float(timestamp) * 1_000_000)}-{detection_id}",
            track_id=track_id,
            timestamp=float(timestamp),
            source="tracking_frame",
            pixel_x=pixel_center.x,
            pixel_y=pixel_center.y,
            ground_point=ground_point,
            confidence=(
                float(getattr(detection, "detection_confidence"))
                if isinstance(getattr(detection, "detection_confidence", None), (int, float))
                else None
            ),
            quality_flags=[
                flag
                for flag in [position_kind, *uncertainty]
                if isinstance(flag, str) and flag
            ],
        )
        self._history[track_id].append(observation_record)
        history_records = list(self._history[track_id])

        detection_ref = self._evidence_store.register_detection_reference(
            track_id=track_id,
            detection_id=detection_id,
            timestamp=float(timestamp),
            frame_reference=frame_reference,
            pixel_bbox=(None if pixel_bbox is None else pixel_bbox.to_dict()),
            tracking_detection_reference=(
                getattr(detection, "detection_reference", None)
                if isinstance(getattr(detection, "detection_reference", None), str)
                else None
            ),
        )
        source_refs.append(detection_ref)

        envelope = TrackingEvidenceEnvelope(
            schema_version=self._schema_version,
            message_id=message_id,
            track_id=track_id,
            timestamp=float(timestamp),
            tracking_claims=tracking_claims,
            uncertainty=sorted(set(uncertainty)),
            frame_id=(None if frame_evidence is None else frame_evidence.frame_id),
            detection_id=detection_id,
            pixel_bbox=pixel_bbox,
            pixel_center=pixel_center,
            relative_depth=(
                float(getattr(detection, "relative_depth"))
                if isinstance(getattr(detection, "relative_depth", None), (int, float))
                else None
            ),
            ground_position=ground_point,
            tracking_confidence=None,
            quality_flags=quality_flags,
            track_history=history_records,
            frame_reference=frame_reference,
            source_evidence_refs=source_refs,
        )

        validation = self._coordinator.validate_tracking_evidence_envelope(
            envelope,
            known_message_ids=self._seen_message_ids,
        )

        envelope_reference = None
        if validation.get("valid", False):
            self._seen_message_ids.add(message_id)
            envelope_reference = self._evidence_store.store_validated_envelope(
                message_id=message_id,
                envelope_payload=envelope.to_dict(),
                validation_payload=validation,
            )

        return PackagedEvidenceResult(
            envelope=envelope,
            validation=validation,
            envelope_reference=envelope_reference,
        )

    def _resolve_track_id(self, detection: Any, update_payload: dict[str, Any] | None) -> str:
        if isinstance(update_payload, dict):
            resolved_track_id = update_payload.get("resolved_track_id")
            if isinstance(resolved_track_id, str) and resolved_track_id.strip():
                return resolved_track_id

        yolo_track_id = getattr(detection, "yolo_track_id", None)
        if isinstance(yolo_track_id, str) and yolo_track_id.strip():
            return yolo_track_id

        detection_id = getattr(detection, "detection_id", "unknown")
        return f"candidate:{detection_id}"

    def _build_claims(self, *, detection: Any, update_payload: dict[str, Any] | None, timestamp: float) -> TrackingClaims:
        if not isinstance(update_payload, dict):
            yolo_track_id = getattr(detection, "yolo_track_id", None)
            return TrackingClaims(
                claimed_resolved_track_id=(yolo_track_id if isinstance(yolo_track_id, str) else None),
                association_status="fallback",
                possible_id_switch=False,
                visibility_status="unknown",
                claim_reasons=["no_tracking_agent_update"],
            )

        accepted = bool(update_payload.get("accepted", False))
        resolved_track_id = update_payload.get("resolved_track_id")
        rejection_reasons = update_payload.get("rejection_reasons")
        claim_reasons = [
            str(item)
            for item in rejection_reasons
            if isinstance(item, str)
        ] if isinstance(rejection_reasons, list) else []

        if not claim_reasons:
            claim_reasons = ["vlm_association_accepted" if accepted else "vlm_association_rejected"]

        possible_id_switch = any("id_switch" in reason for reason in claim_reasons)
        yolo_track_id = getattr(detection, "yolo_track_id", None)
        if accepted and isinstance(yolo_track_id, str) and isinstance(resolved_track_id, str):
            if resolved_track_id != yolo_track_id:
                possible_id_switch = True

        identity_claim: dict[str, Any] | None = None
        has_identity_fields = any(
            key in update_payload
            for key in (
                "proposed_app_identity_id",
                "identity_evidence_refs",
                "identity_snapshot_generation",
                "identity_snapshot_version",
                "identity_resolution_status",
                "rejected_app_identity_ids",
            )
        )
        if has_identity_fields:
            proposed_app_identity_id = update_payload.get("proposed_app_identity_id")
            if not isinstance(proposed_app_identity_id, str) or not proposed_app_identity_id.strip():
                proposed_app_identity_id = None

            detection_reference = update_payload.get("detection_reference")
            if not isinstance(detection_reference, str) or not detection_reference.strip():
                candidate_detection_reference = getattr(detection, "detection_reference", None)
                detection_reference = (
                    candidate_detection_reference
                    if isinstance(candidate_detection_reference, str) and candidate_detection_reference.strip()
                    else None
                )

            identity_evidence_refs = update_payload.get("identity_evidence_refs")
            if not isinstance(identity_evidence_refs, list):
                identity_evidence_refs = []
            normalized_identity_evidence_refs = [
                str(item)
                for item in identity_evidence_refs
                if isinstance(item, str) and item.strip()
            ]

            rejected_app_identity_ids = update_payload.get("rejected_app_identity_ids")
            if not isinstance(rejected_app_identity_ids, list):
                rejected_app_identity_ids = []
            alternatives = [
                str(item)
                for item in rejected_app_identity_ids
                if isinstance(item, str) and item.strip()
            ]

            snapshot_generation = update_payload.get("identity_snapshot_generation")
            if not isinstance(snapshot_generation, int):
                snapshot_generation = None

            snapshot_version = update_payload.get("identity_snapshot_version")
            if not isinstance(snapshot_version, int):
                snapshot_version = None

            identity_resolution_status = update_payload.get("identity_resolution_status")
            if not isinstance(identity_resolution_status, str) or not identity_resolution_status.strip():
                identity_resolution_status = None

            unusable_reasons: list[str] = []
            if not accepted:
                unusable_reasons.append("association_rejected")
            if proposed_app_identity_id is None:
                unusable_reasons.append("missing_proposed_app_identity_id")
            if detection_reference is None:
                unusable_reasons.append("missing_detection_reference")
            if snapshot_generation is None:
                unusable_reasons.append("missing_identity_snapshot_generation")
            if snapshot_version is None:
                unusable_reasons.append("missing_identity_snapshot_version")
            if isinstance(identity_resolution_status, str) and identity_resolution_status.startswith("unresolved_"):
                unusable_reasons.append(identity_resolution_status)

            claim_status = "usable" if not unusable_reasons else "unusable"
            claim_fingerprint = "|".join(
                [
                    str(getattr(detection, "detection_id", "")),
                    str(resolved_track_id),
                    str(detection_reference or ""),
                    str(proposed_app_identity_id or ""),
                    str(snapshot_generation if snapshot_generation is not None else ""),
                    str(snapshot_version if snapshot_version is not None else ""),
                    str(int(float(timestamp) * 1_000_000)),
                ]
            )
            claim_id = f"idclaim-{hashlib.sha1(claim_fingerprint.encode('utf-8')).hexdigest()[:16]}"

            identity_claim = {
                "claim_id": claim_id,
                "claim_kind": "tracking_identity_proposal",
                "claim_status": claim_status,
                "claim_usable": claim_status == "usable",
                "claim_unusable_reasons": unusable_reasons,
                "detection_id": str(getattr(detection, "detection_id", "")),
                "detection_reference": detection_reference,
                "proposed_app_identity_id": proposed_app_identity_id,
                "alternative_app_identity_ids": alternatives,
                "identity_evidence_refs": normalized_identity_evidence_refs,
                "identity_snapshot_generation": snapshot_generation,
                "identity_snapshot_version": snapshot_version,
                "identity_resolution_status": identity_resolution_status,
                "provenance": {
                    "source_agent": "visual_tracking",
                    "source_contract": "VisualAssociationUpdate",
                    "mutation_authorized": False,
                },
            }

        return TrackingClaims(
            claimed_resolved_track_id=(resolved_track_id if isinstance(resolved_track_id, str) else None),
            association_status=("accepted" if accepted else "rejected"),
            possible_id_switch=possible_id_switch,
            visibility_status="unknown",
            claim_reasons=claim_reasons,
            identity_claim=identity_claim,
        )
