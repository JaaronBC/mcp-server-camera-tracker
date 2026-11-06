from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.tracking.evidence_store import TrackingEvidenceResolver


INTENT_ARTIFACT_SCHEMA_VERSION = "1.1"
INTENT_ACK_LEDGER_SCHEMA_VERSION = "1.0"
TERMINAL_ACK_OUTCOMES = {"applied", "rejected", "expired"}
EXPIRY_REASONS = {"intent_expired", "intent_previously_expired"}


def _as_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _sanitize_string_list(value: Any, *, max_items: int = 64, max_length: int = 256) -> list[str] | None:
    if not isinstance(value, list):
        return None
    sanitized: list[str] = []
    for item in value:
        normalized = _as_non_empty_string(item)
        if normalized is None:
            return None
        if len(normalized) > max_length:
            normalized = normalized[:max_length]
        sanitized.append(normalized)
        if len(sanitized) > max_items:
            return None
    return sanitized


def _sanitize_provenance_dict(
    value: Any,
    *,
    max_keys: int = 16,
    max_key_length: int = 64,
    max_value_length: int = 128,
) -> dict[str, Any] | None:
    if value is None:
        return {}
    if not isinstance(value, dict):
        return None

    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if len(sanitized) >= max_keys:
            return None
        normalized_key = _as_non_empty_string(key)
        if normalized_key is None:
            return None
        if len(normalized_key) > max_key_length:
            normalized_key = normalized_key[:max_key_length]

        if isinstance(item, str):
            sanitized[normalized_key] = item[:max_value_length]
        elif isinstance(item, (int, float, bool)) or item is None:
            sanitized[normalized_key] = item
        else:
            return None

    return sanitized


@dataclass(slots=True)
class IdentityReassociationMutationIntent:
    schema_version: str
    intent_id: str
    generated_at: float
    expires_at: float
    verification_request_id: str
    source: str
    evidence_message_id: str
    target_yolo_track_id: str
    claim_id: str
    app_identity_id: str
    detection_reference: str
    evidence_detection_reference: str
    frame_reference: str
    capture_context: dict[str, Any]
    source_evidence_refs: list[str]
    identity_evidence_refs: list[str]
    identity_snapshot_generation: int
    identity_snapshot_version: int
    verification_provenance: dict[str, Any]
    identity_claim_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "generated_at": self.generated_at,
            "expires_at": self.expires_at,
            "verification_request_id": self.verification_request_id,
            "source": self.source,
            "evidence_message_id": self.evidence_message_id,
            "target_yolo_track_id": self.target_yolo_track_id,
            "claim_id": self.claim_id,
            "app_identity_id": self.app_identity_id,
            "detection_reference": self.detection_reference,
            "evidence_detection_reference": self.evidence_detection_reference,
            "frame_reference": self.frame_reference,
            "capture_context": dict(self.capture_context),
            "source_evidence_refs": list(self.source_evidence_refs),
            "identity_evidence_refs": list(self.identity_evidence_refs),
            "identity_snapshot_generation": self.identity_snapshot_generation,
            "identity_snapshot_version": self.identity_snapshot_version,
            "verification_provenance": dict(self.verification_provenance),
            "identity_claim_provenance": dict(self.identity_claim_provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdentityReassociationMutationIntent | None":
        if not isinstance(payload, dict):
            return None

        schema_version = _as_non_empty_string(payload.get("schema_version"))
        intent_id = _as_non_empty_string(payload.get("intent_id"))
        verification_request_id = _as_non_empty_string(payload.get("verification_request_id"))
        source = _as_non_empty_string(payload.get("source"))
        evidence_message_id = _as_non_empty_string(payload.get("evidence_message_id"))
        target_yolo_track_id = _as_non_empty_string(payload.get("target_yolo_track_id"))
        claim_id = _as_non_empty_string(payload.get("claim_id"))
        app_identity_id = _as_non_empty_string(payload.get("app_identity_id"))
        detection_reference = _as_non_empty_string(payload.get("detection_reference"))
        evidence_detection_reference = _as_non_empty_string(payload.get("evidence_detection_reference"))
        frame_reference = _as_non_empty_string(payload.get("frame_reference"))

        if any(
            value is None
            for value in (
                schema_version,
                intent_id,
                verification_request_id,
                source,
                evidence_message_id,
                target_yolo_track_id,
                claim_id,
                app_identity_id,
                detection_reference,
                evidence_detection_reference,
                frame_reference,
            )
        ):
            return None

        generated_at_raw = payload.get("generated_at")
        expires_at_raw = payload.get("expires_at")
        if not isinstance(generated_at_raw, (int, float)):
            return None
        if not isinstance(expires_at_raw, (int, float)):
            return None

        capture_context = _sanitize_provenance_dict(payload.get("capture_context"))
        if capture_context is None or len(capture_context) == 0:
            return None

        source_evidence_refs = _sanitize_string_list(payload.get("source_evidence_refs"))
        identity_evidence_refs = _sanitize_string_list(payload.get("identity_evidence_refs"))
        verification_provenance = _sanitize_provenance_dict(payload.get("verification_provenance"))
        identity_claim_provenance = _sanitize_provenance_dict(payload.get("identity_claim_provenance"))

        generation = payload.get("identity_snapshot_generation")
        version = payload.get("identity_snapshot_version")
        if not isinstance(generation, int) or not isinstance(version, int):
            return None
        if source_evidence_refs is None or identity_evidence_refs is None:
            return None
        if verification_provenance is None or identity_claim_provenance is None:
            return None

        return cls(
            schema_version=schema_version,
            intent_id=intent_id,
            generated_at=float(generated_at_raw),
            expires_at=float(expires_at_raw),
            verification_request_id=verification_request_id,
            source=source,
            evidence_message_id=evidence_message_id,
            target_yolo_track_id=target_yolo_track_id,
            claim_id=claim_id,
            app_identity_id=app_identity_id,
            detection_reference=detection_reference,
            evidence_detection_reference=evidence_detection_reference,
            frame_reference=frame_reference,
            capture_context=capture_context,
            source_evidence_refs=source_evidence_refs,
            identity_evidence_refs=identity_evidence_refs,
            identity_snapshot_generation=int(generation),
            identity_snapshot_version=int(version),
            verification_provenance=verification_provenance,
            identity_claim_provenance=identity_claim_provenance,
        )


class IdentityReassociationIntentPublisher:
    def __init__(
        self,
        *,
        artifact_path: Path,
        ack_ledger_path: Path | None = None,
        evidence_root_dir: Path | None = None,
        max_intents: int = 256,
        max_decisions: int = 1024,
        intent_ttl_seconds: float = 45.0,
    ) -> None:
        self._artifact_path = artifact_path
        self._backup_path = artifact_path.with_name(f"{artifact_path.name}.bak")
        self._ack_ledger_path = (
            artifact_path.with_name(f"{artifact_path.stem}_ack_ledger.json")
            if ack_ledger_path is None
            else ack_ledger_path
        )
        self._ack_backup_path = self._ack_ledger_path.with_name(f"{self._ack_ledger_path.name}.bak")
        self._evidence_root_dir = evidence_root_dir or (artifact_path.parent / "tracking_evidence")
        self._resolver = TrackingEvidenceResolver(self._evidence_root_dir)
        self._max_intents = max(1, int(max_intents))
        self._max_decisions = max(1, int(max_decisions))
        self._intent_ttl_seconds = max(1.0, float(intent_ttl_seconds))

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def ack_ledger_path(self) -> Path:
        return self._ack_ledger_path

    def publish_from_results(
        self,
        results: list[dict[str, Any]],
        *,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now_timestamp is None else float(now_timestamp)
        existing_payload = self._load_existing_artifact()
        if not bool(existing_payload.get("load_ok", False)):
            return {
                "accepted": 0,
                "rejected": 0,
                "duplicates": 0,
                "conflicts": 0,
                "overflow": 0,
                "expired_pruned": 0,
                "retained_pending_intents": 0,
                "retained_decisions": 0,
                "load_failed": True,
                "load_failure_reason": existing_payload.get("load_failure_reason", "intent_artifact_unrecoverable"),
            }

        ack_payload = self._load_ack_ledger(now_timestamp=now)
        if not bool(ack_payload.get("load_ok", False)):
            return {
                "accepted": 0,
                "rejected": 0,
                "duplicates": 0,
                "conflicts": 0,
                "overflow": 0,
                "expired_pruned": 0,
                "retained_pending_intents": 0,
                "retained_decisions": 0,
                "load_failed": True,
                "load_failure_reason": ack_payload.get("load_failure_reason", "ack_ledger_unrecoverable"),
            }

        existing_intents: list[IdentityReassociationMutationIntent] = list(existing_payload.get("intents", []))
        existing_decisions: list[dict[str, Any]] = list(existing_payload.get("decisions", []))
        expired_tombstones: set[str] = set(existing_payload.get("expired_tombstones", set()))

        terminal_outcomes_by_intent: dict[str, str] = dict(ack_payload.get("terminal_outcomes", {}))
        pending_existing_intents: list[IdentityReassociationMutationIntent] = []
        expired_pruned = 0
        for existing_intent in existing_intents:
            if existing_intent.expires_at <= now:
                expired_pruned += 1
                expired_tombstones.add(existing_intent.intent_id)
                existing_decisions.append(
                    {
                        "schema_version": INTENT_ARTIFACT_SCHEMA_VERSION,
                        "generated_at": now,
                        "verification_request_id": existing_intent.verification_request_id,
                        "source": existing_intent.source,
                        "target_yolo_track_id": existing_intent.target_yolo_track_id,
                        "evidence_message_id": existing_intent.evidence_message_id,
                        "claim_id": existing_intent.claim_id,
                        "intent_id": existing_intent.intent_id,
                        "accepted": False,
                        "reason": "intent_expired",
                    }
                )
                continue
            terminal_outcome = terminal_outcomes_by_intent.get(existing_intent.intent_id)
            if terminal_outcome in TERMINAL_ACK_OUTCOMES:
                continue
            pending_existing_intents.append(existing_intent)

        self._ensure_ack_ledger_exists(now_timestamp=now)

        intent_by_id = {intent.intent_id: intent for intent in pending_existing_intents}

        claim_binding_by_id: dict[str, tuple[str, str, str, int, int, str]] = {}
        for existing in pending_existing_intents:
            claim_binding_by_id[existing.claim_id] = (
                existing.app_identity_id,
                existing.detection_reference,
                existing.target_yolo_track_id,
                existing.identity_snapshot_generation,
                existing.identity_snapshot_version,
                existing.evidence_message_id,
            )

        accepted = 0
        rejected = 0
        duplicates = 0
        conflicts = 0
        overflow = 0
        decisions: list[dict[str, Any]] = []

        for result in results:
            intent, decision = self._build_intent_from_result(result, now_timestamp=now)
            if intent is None:
                rejected += 1
                decisions.append(decision)
                continue

            if intent.intent_id in expired_tombstones:
                rejected += 1
                decisions.append(
                    {
                        **decision,
                        "accepted": False,
                        "reason": "intent_previously_expired",
                    }
                )
                continue

            terminal_outcome = terminal_outcomes_by_intent.get(intent.intent_id)
            if terminal_outcome in TERMINAL_ACK_OUTCOMES:
                rejected += 1
                decisions.append(
                    {
                        **decision,
                        "accepted": False,
                        "reason": "intent_terminal_outcome",
                        "terminal_outcome": terminal_outcome,
                    }
                )
                continue

            existing_binding = claim_binding_by_id.get(intent.claim_id)
            new_binding = (
                intent.app_identity_id,
                intent.detection_reference,
                intent.target_yolo_track_id,
                intent.identity_snapshot_generation,
                intent.identity_snapshot_version,
                intent.evidence_message_id,
            )
            if existing_binding is not None and existing_binding != new_binding:
                conflicts += 1
                rejected += 1
                decisions.append(
                    {
                        **decision,
                        "accepted": False,
                        "reason": "conflicting_claim_binding",
                    }
                )
                continue

            claim_binding_by_id[intent.claim_id] = new_binding

            if intent.intent_id in intent_by_id:
                duplicates += 1
                decisions.append(
                    {
                        **decision,
                        "accepted": False,
                        "reason": "duplicate_intent",
                    }
                )
                continue

            if len(intent_by_id) >= self._max_intents:
                overflow += 1
                rejected += 1
                decisions.append(
                    {
                        **decision,
                        "accepted": False,
                        "reason": "queue_capacity_reached",
                    }
                )
                continue

            intent_by_id[intent.intent_id] = intent
            accepted += 1
            decisions.append({**decision, "accepted": True, "reason": None})

        merged_intents = list(intent_by_id.values())
        merged_intents.sort(key=lambda item: item.generated_at)

        merged_decisions = [*existing_decisions, *decisions]
        if len(merged_decisions) > self._max_decisions:
            merged_decisions = merged_decisions[-self._max_decisions :]

        self._atomic_write(
            {
                "schema_version": INTENT_ARTIFACT_SCHEMA_VERSION,
                "generated_at": now,
                "intent_count": len(merged_intents),
                "decision_count": len(merged_decisions),
                "intent_ttl_seconds": self._intent_ttl_seconds,
                "intents": [item.to_dict() for item in merged_intents],
                "decisions": merged_decisions,
            }
        )

        return {
            "accepted": accepted,
            "rejected": rejected,
            "duplicates": duplicates,
            "conflicts": conflicts,
            "overflow": overflow,
            "expired_pruned": expired_pruned,
            "retained_pending_intents": len(merged_intents),
            "retained_decisions": len(merged_decisions),
            "recovered_from_backup": bool(existing_payload.get("recovered_from_backup", False)),
            "load_failed": False,
        }

    def _build_intent_from_result(
        self,
        result: dict[str, Any],
        *,
        now_timestamp: float,
    ) -> tuple[IdentityReassociationMutationIntent | None, dict[str, Any]]:
        request_id = _as_non_empty_string(result.get("request_id"))
        source = _as_non_empty_string(result.get("source"))
        target_track_id = _as_non_empty_string(result.get("track_id"))
        evidence_message_id = _as_non_empty_string(result.get("evidence_message_id"))
        details = result.get("details")

        decision_base = {
            "schema_version": INTENT_ARTIFACT_SCHEMA_VERSION,
            "generated_at": now_timestamp,
            "verification_request_id": request_id,
            "source": source,
            "target_yolo_track_id": target_track_id,
            "evidence_message_id": evidence_message_id,
            "claim_id": None,
            "intent_id": None,
            "accepted": False,
            "reason": "invalid_result_shape",
        }

        if request_id is None or source is None or target_track_id is None or evidence_message_id is None:
            return None, decision_base
        if not isinstance(details, dict):
            return None, {**decision_base, "reason": "missing_details"}

        if result.get("status") != "completed":
            return None, {**decision_base, "reason": "verification_not_completed"}
        if result.get("validation_outcome") != "accepted" or not bool(result.get("coordinator_valid", False)):
            return None, {**decision_base, "reason": "verification_not_validated"}
        if result.get("current_applicability") != "current":
            return None, {**decision_base, "reason": "verification_not_current"}
        if bool(result.get("stale_for_current_decisions", False)):
            return None, {**decision_base, "reason": "verification_stale_for_current"}

        verification_status = _as_non_empty_string(details.get("verification_status"))
        identity_consistency = _as_non_empty_string(details.get("identity_consistency"))
        if verification_status != "supported":
            return None, {**decision_base, "reason": "verification_not_supported"}
        if identity_consistency != "consistent":
            return None, {**decision_base, "reason": "identity_not_consistent"}

        claim_id = _as_non_empty_string(details.get("identity_claim_id"))
        claim_status = _as_non_empty_string(details.get("identity_claim_status"))
        app_identity_id = _as_non_empty_string(details.get("referenced_proposed_app_identity_id"))
        evidence_detection_reference = _as_non_empty_string(details.get("referenced_detection_reference"))
        snapshot_generation = details.get("referenced_identity_snapshot_generation")
        snapshot_version = details.get("referenced_identity_snapshot_version")

        decision_base["claim_id"] = claim_id

        if claim_id is None:
            return None, {**decision_base, "reason": "missing_claim_id"}
        if claim_status != "usable":
            return None, {**decision_base, "reason": "claim_not_usable"}
        if app_identity_id is None:
            return None, {**decision_base, "reason": "missing_app_identity_id"}
        if evidence_detection_reference is None:
            return None, {**decision_base, "reason": "missing_detection_reference"}
        if not isinstance(snapshot_generation, int) or not isinstance(snapshot_version, int):
            return None, {**decision_base, "reason": "missing_snapshot_binding"}

        resolved_binding = self._resolve_current_detection_binding(
            detection_reference=evidence_detection_reference,
            expected_track_id=target_track_id,
            now_timestamp=now_timestamp,
        )
        if resolved_binding is None:
            return None, {**decision_base, "reason": "detection_binding_invalid"}

        canonical_detection_reference = _as_non_empty_string(
            resolved_binding.get("canonical_detection_reference")
        )
        frame_reference = _as_non_empty_string(resolved_binding.get("frame_reference"))
        capture_context = _sanitize_provenance_dict(resolved_binding.get("capture_context"))
        if canonical_detection_reference is None:
            return None, {**decision_base, "reason": "canonical_detection_reference_missing"}
        if frame_reference is None:
            return None, {**decision_base, "reason": "frame_reference_missing"}
        if capture_context is None or len(capture_context) == 0:
            return None, {**decision_base, "reason": "capture_context_missing"}

        source_evidence_refs = _sanitize_string_list(details.get("source_evidence_refs"))
        identity_evidence_refs = _sanitize_string_list(details.get("identity_claim_evidence_refs"))
        if source_evidence_refs is None or len(source_evidence_refs) == 0:
            return None, {**decision_base, "reason": "invalid_source_evidence_refs"}
        if identity_evidence_refs is None or len(identity_evidence_refs) == 0:
            return None, {**decision_base, "reason": "invalid_identity_evidence_refs"}
        if frame_reference not in source_evidence_refs:
            return None, {**decision_base, "reason": "frame_reference_not_in_source_evidence"}

        identity_claim_provenance = _sanitize_provenance_dict(details.get("identity_claim_provenance"))
        if identity_claim_provenance is None:
            return None, {**decision_base, "reason": "invalid_identity_claim_provenance"}

        verification_observation_id = _as_non_empty_string(details.get("verification_observation_id"))
        verification_message_id = _as_non_empty_string(details.get("verification_referenced_message_id"))
        verification_timestamp = details.get("verification_timestamp")
        if verification_observation_id is None or verification_message_id is None:
            return None, {**decision_base, "reason": "missing_verification_binding"}
        if not isinstance(verification_timestamp, (int, float)):
            return None, {**decision_base, "reason": "invalid_verification_timestamp"}

        verification_provenance = {
            "verification_observation_id": verification_observation_id,
            "verification_referenced_message_id": verification_message_id,
            "verification_timestamp": float(verification_timestamp),
            "verification_status": verification_status,
            "identity_consistency": identity_consistency,
            "visual_content_inspected": bool(details.get("visual_content_inspected", False)),
        }

        fingerprint = "|".join(
            [
                claim_id,
                app_identity_id,
                canonical_detection_reference,
                target_track_id,
                str(snapshot_generation),
                str(snapshot_version),
                evidence_message_id,
            ]
        )
        intent_id = f"reassoc-{hashlib.sha1(fingerprint.encode('utf-8')).hexdigest()[:20]}"
        expires_at = float(now_timestamp) + self._intent_ttl_seconds
        if expires_at <= float(now_timestamp):
            return None, {**decision_base, "reason": "intent_expiry_invalid"}

        intent = IdentityReassociationMutationIntent(
            schema_version=INTENT_ARTIFACT_SCHEMA_VERSION,
            intent_id=intent_id,
            generated_at=float(now_timestamp),
            expires_at=expires_at,
            verification_request_id=request_id,
            source=source,
            evidence_message_id=evidence_message_id,
            target_yolo_track_id=target_track_id,
            claim_id=claim_id,
            app_identity_id=app_identity_id,
            detection_reference=canonical_detection_reference,
            evidence_detection_reference=evidence_detection_reference,
            frame_reference=frame_reference,
            capture_context=capture_context,
            source_evidence_refs=source_evidence_refs,
            identity_evidence_refs=identity_evidence_refs,
            identity_snapshot_generation=int(snapshot_generation),
            identity_snapshot_version=int(snapshot_version),
            verification_provenance=verification_provenance,
            identity_claim_provenance=identity_claim_provenance,
        )

        return intent, {
            **decision_base,
            "claim_id": claim_id,
            "intent_id": intent_id,
            "reason": None,
        }

    def _resolve_current_detection_binding(
        self,
        *,
        detection_reference: str,
        expected_track_id: str,
        now_timestamp: float,
    ) -> dict[str, Any] | None:
        if not detection_reference.startswith("evidence://detection/"):
            return None

        resolution = self._resolver.resolve_reference(
            detection_reference,
            expected_track_id=expected_track_id,
            now_timestamp=now_timestamp,
        )
        if not bool(resolution.get("valid", False)):
            return None

        metadata = resolution.get("metadata")
        if not isinstance(metadata, dict):
            return None

        canonical_reference = _as_non_empty_string(metadata.get("tracking_detection_reference"))
        frame_reference = _as_non_empty_string(metadata.get("frame_reference"))
        process_nonce = _as_non_empty_string(metadata.get("process_nonce"))
        detection_ref_id = _as_non_empty_string(metadata.get("id"))

        if canonical_reference is None:
            return None
        if frame_reference is None:
            return None
        if process_nonce is None:
            return None
        if detection_ref_id is None:
            return None

        return {
            "canonical_detection_reference": canonical_reference,
            "frame_reference": frame_reference,
            "capture_context": {
                "process_nonce": process_nonce,
                "evidence_detection_ref_id": detection_ref_id,
            },
        }

    def _load_existing_artifact(self) -> dict[str, Any]:
        if not self._artifact_path.is_file():
            return {
                "load_ok": True,
                "recovered_from_backup": False,
                "intents": [],
                "decisions": [],
                "expired_tombstones": set(),
            }

        primary = self._parse_intent_artifact(self._artifact_path)
        if bool(primary.get("ok", False)):
            return {
                "load_ok": True,
                "recovered_from_backup": False,
                "intents": primary.get("intents", []),
                "decisions": primary.get("decisions", []),
                "expired_tombstones": self._collect_expired_tombstones(primary.get("decisions", [])),
            }

        backup = self._parse_intent_artifact(self._backup_path)
        if bool(backup.get("ok", False)):
            return {
                "load_ok": True,
                "recovered_from_backup": True,
                "intents": backup.get("intents", []),
                "decisions": backup.get("decisions", []),
                "expired_tombstones": self._collect_expired_tombstones(backup.get("decisions", [])),
            }

        return {
            "load_ok": False,
            "load_failure_reason": "intent_artifact_unrecoverable",
            "recovered_from_backup": False,
            "intents": [],
            "decisions": [],
            "expired_tombstones": set(),
        }

    def _collect_expired_tombstones(self, decisions: list[dict[str, Any]]) -> set[str]:
        tombstones: set[str] = set()
        for item in decisions:
            if not isinstance(item, dict):
                continue
            reason = item.get("reason")
            intent_id = _as_non_empty_string(item.get("intent_id"))
            if reason in EXPIRY_REASONS and intent_id is not None:
                tombstones.add(intent_id)
        return tombstones

    def _parse_intent_artifact(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"ok": False}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"ok": False}
        if not isinstance(payload, dict):
            return {"ok": False}
        if payload.get("schema_version") != INTENT_ARTIFACT_SCHEMA_VERSION:
            return {"ok": False}

        intents: list[IdentityReassociationMutationIntent] = []
        raw_intents = payload.get("intents")
        if isinstance(raw_intents, list):
            for item in raw_intents:
                if not isinstance(item, dict):
                    continue
                parsed = IdentityReassociationMutationIntent.from_dict(item)
                if parsed is not None:
                    intents.append(parsed)

        decisions: list[dict[str, Any]] = []
        raw_decisions = payload.get("decisions")
        if isinstance(raw_decisions, list):
            for item in raw_decisions:
                if isinstance(item, dict):
                    decisions.append(dict(item))

        return {
            "ok": True,
            "intents": intents,
            "decisions": decisions,
        }

    def _default_ack_ledger_payload(self, *, now_timestamp: float) -> dict[str, Any]:
        return {
            "schema_version": INTENT_ACK_LEDGER_SCHEMA_VERSION,
            "generated_at": float(now_timestamp),
            "acknowledgments": [],
        }

    def _load_ack_ledger(self, *, now_timestamp: float) -> dict[str, Any]:
        if not self._ack_ledger_path.is_file():
            self._atomic_write_json(
                path=self._ack_ledger_path,
                backup_path=self._ack_backup_path,
                payload=self._default_ack_ledger_payload(now_timestamp=now_timestamp),
            )

        primary = self._parse_ack_ledger(self._ack_ledger_path)
        if bool(primary.get("ok", False)):
            return {
                "load_ok": True,
                "recovered_from_backup": False,
                "terminal_outcomes": primary.get("terminal_outcomes", {}),
            }

        backup = self._parse_ack_ledger(self._ack_backup_path)
        if bool(backup.get("ok", False)):
            return {
                "load_ok": True,
                "recovered_from_backup": True,
                "terminal_outcomes": backup.get("terminal_outcomes", {}),
            }

        return {
            "load_ok": False,
            "load_failure_reason": "ack_ledger_unrecoverable",
            "terminal_outcomes": {},
        }

    def _parse_ack_ledger(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"ok": False}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"ok": False}
        if not isinstance(payload, dict):
            return {"ok": False}
        if payload.get("schema_version") != INTENT_ACK_LEDGER_SCHEMA_VERSION:
            return {"ok": False}

        terminal_outcomes: dict[str, str] = {}
        raw_acks = payload.get("acknowledgments")
        if isinstance(raw_acks, list):
            for item in raw_acks:
                if not isinstance(item, dict):
                    continue
                intent_id = _as_non_empty_string(item.get("intent_id"))
                outcome = _as_non_empty_string(item.get("outcome"))
                if intent_id is None or outcome is None:
                    continue
                if outcome not in TERMINAL_ACK_OUTCOMES:
                    continue
                terminal_outcomes[intent_id] = outcome

        return {
            "ok": True,
            "terminal_outcomes": terminal_outcomes,
        }

    def _ensure_ack_ledger_exists(self, *, now_timestamp: float) -> None:
        if self._ack_ledger_path.is_file():
            return
        self._atomic_write_json(
            path=self._ack_ledger_path,
            backup_path=self._ack_backup_path,
            payload=self._default_ack_ledger_payload(now_timestamp=now_timestamp),
        )

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self._atomic_write_json(
            path=self._artifact_path,
            backup_path=self._backup_path,
            payload=payload,
        )

    def _atomic_write_json(self, *, path: Path, backup_path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, separators=(",", ":"))
        temp_primary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
        temp_backup = backup_path.with_name(f"{backup_path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
        try:
            temp_primary.write_text(serialized, encoding="utf-8")
            os.replace(temp_primary, path)

            temp_backup.write_text(serialized, encoding="utf-8")
            os.replace(temp_backup, backup_path)
        finally:
            try:
                if temp_primary.exists():
                    temp_primary.unlink()
            except OSError:
                pass
            try:
                if temp_backup.exists():
                    temp_backup.unlink()
            except OSError:
                pass
