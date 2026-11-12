from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from agents.orchestration.runtime.identity_reassociation_intents import (
    INTENT_ACK_LEDGER_SCHEMA_VERSION,
    INTENT_ARTIFACT_SCHEMA_VERSION,
    IdentityReassociationMutationIntent,
    TERMINAL_ACK_OUTCOMES,
)
from agents.tracking.evidence_store import TrackingEvidenceResolver
from agents.tracking.identity_registry import PersistentIdentityRegistry


def _as_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


class IdentityReassociationIntentConsumer:
    def __init__(
        self,
        *,
        intents_path: Path,
        ack_ledger_path: Path,
        evidence_root_dir: Path,
        enabled: bool = False,
        max_intents_per_poll: int = 8,
        max_ack_entries: int = 2048,
        min_poll_interval_seconds: float = 0.25,
    ) -> None:
        self._intents_path = intents_path
        self._intents_backup_path = intents_path.with_name(f"{intents_path.name}.bak")
        self._ack_ledger_path = ack_ledger_path
        self._ack_backup_path = ack_ledger_path.with_name(f"{ack_ledger_path.name}.bak")
        self._resolver = TrackingEvidenceResolver(evidence_root_dir)
        self._enabled = bool(enabled)
        self._max_intents_per_poll = max(1, int(max_intents_per_poll))
        self._max_ack_entries = max(1, int(max_ack_entries))
        self._min_poll_interval_seconds = max(0.05, float(min_poll_interval_seconds))
        self._last_poll_at = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def poll_and_apply(
        self,
        *,
        registry: PersistentIdentityRegistry | None,
        tracking_candidates: list[Any],
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now_timestamp is None else float(now_timestamp)
        if not self._enabled:
            return {
                "enabled": False,
                "skipped": True,
                "reason": "feature_disabled",
            }
        if registry is None:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "registry_unavailable",
            }
        if now - self._last_poll_at < self._min_poll_interval_seconds:
            return {
                "enabled": True,
                "skipped": True,
                "reason": "poll_interval",
                "next_poll_after_seconds": self._min_poll_interval_seconds,
            }
        self._last_poll_at = now

        intents_payload = self._load_intent_artifact()
        if not bool(intents_payload.get("load_ok", False)):
            return {
                "enabled": True,
                "skipped": True,
                "reason": "intent_artifact_unrecoverable",
                "load_failed": True,
            }

        ack_payload = self._load_ack_ledger(now_timestamp=now)
        if not bool(ack_payload.get("load_ok", False)):
            return {
                "enabled": True,
                "skipped": True,
                "reason": "ack_ledger_unrecoverable",
                "load_failed": True,
            }

        intents: list[IdentityReassociationMutationIntent] = list(intents_payload.get("intents", []))
        terminal_outcomes: dict[str, str] = dict(ack_payload.get("terminal_outcomes", {}))
        existing_acks: list[dict[str, Any]] = list(ack_payload.get("acknowledgments", []))

        pending_intents = [
            item
            for item in intents
            if item.intent_id not in terminal_outcomes
        ]
        pending_intents.sort(key=lambda item: (float(item.generated_at), item.intent_id))

        latest_detection_reference_by_track: dict[str, str] = {}
        for candidate in tracking_candidates:
            track_id = _as_non_empty_string(getattr(candidate, "yolo_track_id", None))
            detection_reference = _as_non_empty_string(getattr(candidate, "detection_reference", None))
            if track_id is None or detection_reference is None:
                continue
            latest_detection_reference_by_track[track_id] = detection_reference

        applied = 0
        rejected = 0
        expired = 0
        already_terminal = len(intents) - len(pending_intents)
        processed = 0
        ack_appends: list[dict[str, Any]] = []

        reserved_target_tracks: set[str] = set()
        reserved_identity_ids: set[str] = set()

        for intent in pending_intents:
            if processed >= self._max_intents_per_poll:
                break
            processed += 1

            if intent.intent_id in terminal_outcomes:
                continue

            if intent.expires_at <= now:
                expired += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="expired", reason="intent_expired", now=now))
                terminal_outcomes[intent.intent_id] = "expired"
                continue

            validation_error = self._validate_intent_binding(
                intent=intent,
                now_timestamp=now,
                latest_detection_reference_by_track=latest_detection_reference_by_track,
            )
            if validation_error is not None:
                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason=validation_error, now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue

            if intent.target_yolo_track_id in reserved_target_tracks:
                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason="conflicting_pending_target_track", now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue
            if intent.app_identity_id in reserved_identity_ids:
                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason="conflicting_pending_app_identity", now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue

            record = registry.get_record(intent.app_identity_id)
            if record is None:
                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason="unknown_identity", now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue
            if record.status == "retired":
                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason="identity_retired", now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue

            snapshot_before = registry.snapshot()
            current_generation = snapshot_before.get("generation")
            current_version = snapshot_before.get("version")
            if isinstance(current_generation, int) and intent.identity_snapshot_generation < current_generation:
                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason="stale_snapshot_generation", now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue
            if isinstance(current_version, int) and intent.identity_snapshot_version < current_version:
                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason="stale_snapshot_version", now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue

            current_owner = registry.get_identity_for_yolo_track(intent.target_yolo_track_id)
            if isinstance(current_owner, str) and current_owner and current_owner != intent.app_identity_id:
                owner_record = registry.get_record(current_owner)
                transfer_eligible = (
                    owner_record is not None
                    and owner_record.provisional
                    and record.status == "temporarily_lost"
                    and self._intent_supports_provisional_transfer(
                        intent=intent,
                        expected_source_identity_id=intent.app_identity_id,
                    )
                )
                if transfer_eligible:
                    transfer_result = registry.apply_verified_provisional_owner_transfer(
                        app_identity_id=intent.app_identity_id,
                        provisional_owner_identity_id=current_owner,
                        target_yolo_track_id=intent.target_yolo_track_id,
                        now=now,
                        expected_generation_at_least=intent.identity_snapshot_generation,
                        expected_version=intent.identity_snapshot_version,
                    )
                    if transfer_result.accepted:
                        applied += 1
                        reserved_target_tracks.add(intent.target_yolo_track_id)
                        reserved_identity_ids.add(intent.app_identity_id)
                        ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="applied", reason="provisional_transfer_applied", now=now))
                        terminal_outcomes[intent.intent_id] = "applied"
                    else:
                        rejected += 1
                        reason = _as_non_empty_string(transfer_result.reason) or "mutation_rejected"
                        ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason=reason, now=now))
                        terminal_outcomes[intent.intent_id] = "rejected"
                    continue

                rejected += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason="target_track_owned_by_different_identity", now=now))
                terminal_outcomes[intent.intent_id] = "rejected"
                continue
            if current_owner == intent.app_identity_id:
                applied += 1
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="applied", reason="already_applied", now=now))
                terminal_outcomes[intent.intent_id] = "applied"
                continue

            mutation = registry.apply_identity_mutation(
                app_identity_id=intent.app_identity_id,
                new_yolo_track_id=intent.target_yolo_track_id,
                now=now,
                expected_generation_at_least=intent.identity_snapshot_generation,
                expected_version=intent.identity_snapshot_version,
            )
            if mutation.accepted:
                applied += 1
                reserved_target_tracks.add(intent.target_yolo_track_id)
                reserved_identity_ids.add(intent.app_identity_id)
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="applied", reason="mutation_applied", now=now))
                terminal_outcomes[intent.intent_id] = "applied"
            else:
                rejected += 1
                reason = _as_non_empty_string(mutation.reason) or "mutation_rejected"
                ack_appends.append(self._build_ack(intent_id=intent.intent_id, outcome="rejected", reason=reason, now=now))
                terminal_outcomes[intent.intent_id] = "rejected"

        if ack_appends:
            merged_acks = [*existing_acks, *ack_appends]
            if len(merged_acks) > self._max_ack_entries:
                merged_acks = merged_acks[-self._max_ack_entries :]
            self._write_ack_ledger(
                {
                    "schema_version": INTENT_ACK_LEDGER_SCHEMA_VERSION,
                    "generated_at": now,
                    "acknowledgments": merged_acks,
                }
            )

        return {
            "enabled": True,
            "skipped": False,
            "processed": processed,
            "applied": applied,
            "rejected": rejected,
            "expired": expired,
            "already_terminal": already_terminal,
            "ack_written": len(ack_appends),
            "pending_total": len(pending_intents),
        }

    def _validate_intent_binding(
        self,
        *,
        intent: IdentityReassociationMutationIntent,
        now_timestamp: float,
        latest_detection_reference_by_track: dict[str, str],
    ) -> str | None:
        if not intent.detection_reference.startswith("fdet-"):
            return "invalid_canonical_detection_reference"
        if not intent.evidence_detection_reference.startswith("evidence://detection/"):
            return "invalid_evidence_detection_reference"
        if not intent.frame_reference.startswith("evidence://frame/"):
            return "invalid_frame_reference"

        process_nonce = _as_non_empty_string(intent.capture_context.get("process_nonce"))
        evidence_detection_ref_id = _as_non_empty_string(intent.capture_context.get("evidence_detection_ref_id"))
        if process_nonce is None or evidence_detection_ref_id is None:
            return "invalid_capture_context"

        resolution = self._resolver.resolve_reference(
            intent.evidence_detection_reference,
            expected_track_id=intent.target_yolo_track_id,
            expected_frame_reference=intent.frame_reference,
            expected_tracking_detection_reference=intent.detection_reference,
            now_timestamp=now_timestamp,
        )
        if not bool(resolution.get("valid", False)):
            status = _as_non_empty_string(resolution.get("status")) or "detection_binding_invalid"
            return status

        metadata = resolution.get("metadata")
        if not isinstance(metadata, dict):
            return "detection_metadata_missing"
        metadata_process_nonce = _as_non_empty_string(metadata.get("process_nonce"))
        metadata_detection_ref_id = _as_non_empty_string(metadata.get("id"))
        if metadata_process_nonce != process_nonce:
            return "capture_context_process_nonce_mismatch"
        if metadata_detection_ref_id != evidence_detection_ref_id:
            return "capture_context_detection_ref_id_mismatch"

        current_reference = latest_detection_reference_by_track.get(intent.target_yolo_track_id)
        if current_reference is None:
            return "target_track_not_currently_visible"
        if current_reference != intent.detection_reference:
            return "target_track_detection_reference_mismatch"

        return None

    def _intent_supports_provisional_transfer(
        self,
        *,
        intent: IdentityReassociationMutationIntent,
        expected_source_identity_id: str,
    ) -> bool:
        verification_provenance = intent.verification_provenance
        if not isinstance(verification_provenance, dict):
            return False
        if verification_provenance.get("verification_status") != "supported":
            return False
        if verification_provenance.get("identity_consistency") != "consistent":
            return False
        if verification_provenance.get("verification_referenced_message_id") != intent.evidence_message_id:
            return False

        identity_claim_provenance = intent.identity_claim_provenance
        if not isinstance(identity_claim_provenance, dict):
            return False
        if identity_claim_provenance.get("source_agent") != "visual_tracking":
            return False
        if identity_claim_provenance.get("source_contract") != "VisualAssociationUpdate":
            return False
        if identity_claim_provenance.get("mutation_authorized") is not False:
            return False

        if intent.app_identity_id != expected_source_identity_id:
            return False
        return True

    def _load_intent_artifact(self) -> dict[str, Any]:
        if not self._intents_path.is_file():
            return {
                "load_ok": True,
                "intents": [],
            }

        primary = self._parse_intent_file(self._intents_path)
        if bool(primary.get("ok", False)):
            return {
                "load_ok": True,
                "intents": primary.get("intents", []),
            }

        backup = self._parse_intent_file(self._intents_backup_path)
        if bool(backup.get("ok", False)):
            return {
                "load_ok": True,
                "intents": backup.get("intents", []),
            }

        return {
            "load_ok": False,
        }

    def _parse_intent_file(self, path: Path) -> dict[str, Any]:
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

        if len(intents) != payload.get("intent_count"):
            return {"ok": False}

        return {
            "ok": True,
            "intents": intents,
        }

    def _default_ack_payload(self, *, now_timestamp: float) -> dict[str, Any]:
        return {
            "schema_version": INTENT_ACK_LEDGER_SCHEMA_VERSION,
            "generated_at": float(now_timestamp),
            "acknowledgments": [],
        }

    def _load_ack_ledger(self, *, now_timestamp: float) -> dict[str, Any]:
        if not self._ack_ledger_path.is_file():
            self._write_ack_ledger(self._default_ack_payload(now_timestamp=now_timestamp))

        primary = self._parse_ack_file(self._ack_ledger_path)
        if bool(primary.get("ok", False)):
            return {
                "load_ok": True,
                "terminal_outcomes": primary.get("terminal_outcomes", {}),
                "acknowledgments": primary.get("acknowledgments", []),
            }

        backup = self._parse_ack_file(self._ack_backup_path)
        if bool(backup.get("ok", False)):
            return {
                "load_ok": True,
                "terminal_outcomes": backup.get("terminal_outcomes", {}),
                "acknowledgments": backup.get("acknowledgments", []),
            }

        return {
            "load_ok": False,
        }

    def _parse_ack_file(self, path: Path) -> dict[str, Any]:
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
        acknowledgments: list[dict[str, Any]] = []
        raw_acks = payload.get("acknowledgments")
        if isinstance(raw_acks, list):
            for item in raw_acks:
                if not isinstance(item, dict):
                    continue
                intent_id = _as_non_empty_string(item.get("intent_id"))
                outcome = _as_non_empty_string(item.get("outcome"))
                timestamp_value = item.get("timestamp")
                reason = _as_non_empty_string(item.get("reason"))
                if intent_id is None or outcome is None:
                    continue
                if outcome not in TERMINAL_ACK_OUTCOMES:
                    continue
                if not isinstance(timestamp_value, (int, float)):
                    continue
                acknowledgments.append(
                    {
                        "intent_id": intent_id,
                        "outcome": outcome,
                        "timestamp": float(timestamp_value),
                        "reason": reason,
                    }
                )
                terminal_outcomes[intent_id] = outcome

        return {
            "ok": True,
            "terminal_outcomes": terminal_outcomes,
            "acknowledgments": acknowledgments,
        }

    def _build_ack(self, *, intent_id: str, outcome: str, reason: str, now: float) -> dict[str, Any]:
        return {
            "intent_id": intent_id,
            "outcome": outcome,
            "timestamp": float(now),
            "reason": reason,
        }

    def _write_ack_ledger(self, payload: dict[str, Any]) -> None:
        self._atomic_write_json(
            path=self._ack_ledger_path,
            backup_path=self._ack_backup_path,
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
