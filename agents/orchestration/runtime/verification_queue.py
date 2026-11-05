from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mcp import ClientSession

from agents.contracts import VisualVerificationEnvelope
from agents.coordinator import AgentCoordinator
from agents.observation.evidence_consumption import load_validated_envelope_by_message_id
from agents.tracking.evidence_protection import acquire_verification_evidence_lease, release_verification_evidence_lease


@dataclass(slots=True)
class VerificationQueueRequest:
    request_id: str
    track_id: str
    message_id: str | None
    source: str
    requested_at: float
    trigger_event_id: str | None = None
    trigger_timestamp: float | None = None


@dataclass(slots=True)
class VerificationSubmissionDecision:
    event_origin: str | None
    trigger_event_id: str | None
    track_id: str | None
    evidence_requested: bool
    evidence_message_id: str | None
    queue_request: VerificationQueueRequest | None
    submission_eligibility: bool
    skip_reason: str | None


@dataclass(slots=True)
class VerificationQueueRecord:
    request_id: str
    source: str
    track_id: str
    evidence_message_id: str
    status: str
    validation_outcome: str
    submitted_at: float
    started_at: float | None
    finished_at: float | None
    current_applicability: str
    stale_for_current_decisions: bool
    coordinator_valid: bool
    error_code: str | None
    error: str | None
    details: dict[str, Any]
    run_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source": self.source,
            "track_id": self.track_id,
            "evidence_message_id": self.evidence_message_id,
            "status": self.status,
            "validation_outcome": self.validation_outcome,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_applicability": self.current_applicability,
            "stale_for_current_decisions": self.stale_for_current_decisions,
            "coordinator_valid": self.coordinator_valid,
            "error_code": self.error_code,
            "error": self.error,
            "details": self.details,
            "run_id": self.run_id,
        }


@dataclass(slots=True)
class VerificationSubmissionRecord:
    request_id: str
    source: str
    track_id: str
    message_id: str | None
    submitted_at: float
    accepted: bool
    status: str
    reason: str | None
    run_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source": self.source,
            "track_id": self.track_id,
            "message_id": self.message_id,
            "submitted_at": self.submitted_at,
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "run_id": self.run_id,
        }


class VerificationQueueManager:
    def __init__(
        self,
        *,
        evidence_root_dir: Path,
        run_id: str | None = None,
        queue_capacity: int = 32,
        max_concurrency: int = 2,
        request_timeout_seconds: float = 8.0,
        result_registry_capacity: int = 512,
        verify_tool_name: str = "verify_tracking_visual_evidence",
        evidence_tool_name: str = "get_tracking_visual_evidence",
        max_protected_frames: int = 32,
        max_active_leases: int = 128,
        lease_ttl_seconds: float | None = None,
        lease_lock_timeout_seconds: float = 0.1,
        coordinator: AgentCoordinator | None = None,
    ) -> None:
        self._evidence_root_dir = evidence_root_dir
        self._run_id = run_id
        self._queue_capacity = max(1, int(queue_capacity))
        self._max_concurrency = max(1, int(max_concurrency))
        self._request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self._result_registry_capacity = max(1, int(result_registry_capacity))
        self._verify_tool_name = verify_tool_name
        self._evidence_tool_name = evidence_tool_name
        self._max_protected_frames = max(0, int(max_protected_frames))
        self._max_active_leases = max(1, int(max_active_leases))
        self._lease_ttl_seconds = (
            max(1.0, float(lease_ttl_seconds))
            if lease_ttl_seconds is not None
            else max(5.0, min(120.0, self._request_timeout_seconds * 4.0))
        )
        self._lease_lock_timeout_seconds = max(0.0, float(lease_lock_timeout_seconds))
        self._coordinator = coordinator or AgentCoordinator()

        self._queue: asyncio.Queue[VerificationQueueRequest] = asyncio.Queue(maxsize=self._queue_capacity)
        self._workers: list[asyncio.Task[None]] = []
        self._shutdown = False

        self._queued_message_ids: set[str] = set()
        self._inflight_message_ids: set[str] = set()
        self._completed_message_ids: set[str] = set()

        self._records_by_request_id: dict[str, VerificationQueueRecord] = {}
        self._record_order: deque[str] = deque(maxlen=self._result_registry_capacity)
        self._submission_records_by_request_id: dict[str, VerificationSubmissionRecord] = {}
        self._submission_order: deque[str] = deque(maxlen=self._result_registry_capacity)
        self._leases_by_request_id: dict[str, dict[str, Any]] = {}
        self._lease_fallback_status_by_request_id: dict[str, str] = {}
        self._pending_lease_release_request_ids: set[str] = set()
        self._lease_release_retry_attempts_by_request_id: dict[str, int] = {}
        self._max_lease_release_retries = 2

    async def start(self, *, session: ClientSession) -> None:
        if self._workers:
            return
        self._shutdown = False
        for index in range(self._max_concurrency):
            self._workers.append(asyncio.create_task(self._worker_loop(session, index)))

    async def stop(self) -> None:
        self._shutdown = True
        while True:
            try:
                pending_request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queued_message_ids.discard(pending_request.message_id or "")
            self._release_request_lease(pending_request.request_id)
            self._store_record(
                VerificationQueueRecord(
                    request_id=pending_request.request_id,
                    source=pending_request.source,
                    track_id=pending_request.track_id,
                    evidence_message_id=pending_request.message_id or "",
                    status="cancelled",
                    validation_outcome="not_executed",
                    submitted_at=pending_request.requested_at,
                    started_at=None,
                    finished_at=time.time(),
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="shutdown_cancelled",
                    error="Request cancelled during shutdown",
                    run_id=self._run_id,
                    details={},
                )
            )
            self._queue.task_done()

        for task in self._workers:
            task.cancel()
        for task in self._workers:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._workers.clear()
        self._inflight_message_ids.clear()
        self._retry_pending_lease_releases(max_passes=self._max_lease_release_retries)

    def submit_request(self, request: VerificationQueueRequest) -> dict[str, Any]:
        self._retry_pending_lease_releases(max_passes=1)

        if self._shutdown:
            self._store_submission(
                VerificationSubmissionRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=request.track_id,
                    message_id=request.message_id,
                    submitted_at=time.time(),
                    accepted=False,
                    status="manager_shutdown",
                    reason="manager_shutdown",
                    run_id=self._run_id,
                )
            )
            return {"accepted": False, "status": "manager_shutdown", "request_id": request.request_id}

        now = time.time()
        message_id = request.message_id
        track_id = request.track_id.strip() if isinstance(request.track_id, str) else ""
        if not track_id:
            self._store_submission(
                VerificationSubmissionRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id="",
                    message_id=message_id,
                    submitted_at=now,
                    accepted=False,
                    status="invalid_track_id",
                    reason="invalid_track_id",
                    run_id=self._run_id,
                )
            )
            return {"accepted": False, "status": "invalid_track_id", "request_id": request.request_id}

        if not isinstance(message_id, str) or not message_id.strip():
            resolved = self._resolve_latest_message_id_for_track(track_id)
            if resolved is None:
                self._store_record(
                    VerificationQueueRecord(
                        request_id=request.request_id,
                        source=request.source,
                        track_id=track_id,
                        evidence_message_id="",
                        status="rejected",
                        validation_outcome="resolve_failed",
                        submitted_at=now,
                        started_at=None,
                        finished_at=now,
                        current_applicability="unknown",
                        stale_for_current_decisions=False,
                        coordinator_valid=False,
                        error_code="missing_or_invalid_evidence",
                        error="Unable to resolve validated evidence message_id for track",
                        run_id=self._run_id,
                        details={},
                    )
                )
                return {"accepted": False, "status": "missing_or_invalid_evidence"}
            message_id = resolved

        assert isinstance(message_id, str)
        message_id = message_id.strip()
        if not message_id:
            return {"accepted": False, "status": "invalid_message_id"}

        if message_id in self._queued_message_ids or message_id in self._inflight_message_ids:
            return {"accepted": False, "status": "duplicate_inflight", "message_id": message_id}

        if self._queue.full():
            self._store_record(
                VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    evidence_message_id=message_id,
                    status="rejected",
                    validation_outcome="queue_saturated",
                    submitted_at=now,
                    started_at=None,
                    finished_at=now,
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="queue_saturated",
                    error="Verification queue capacity reached",
                    run_id=self._run_id,
                    details={"queue_capacity": self._queue_capacity},
                )
            )
            return {"accepted": False, "status": "queue_saturated", "message_id": message_id}

        lease_result = acquire_verification_evidence_lease(
            evidence_root_dir=self._evidence_root_dir,
            request_id=request.request_id,
            message_id=message_id,
            track_id=track_id,
            now_timestamp=now,
            lease_ttl_seconds=self._lease_ttl_seconds,
            max_protected_frames=self._max_protected_frames,
            max_active_leases=self._max_active_leases,
            lock_timeout_seconds=self._lease_lock_timeout_seconds,
        )
        lease_payload: dict[str, Any] | None = None
        if not bool(lease_result.get("acquired", False)):
            status = str(lease_result.get("status", "lease_acquire_failed"))
            if status == "frame_metadata_unavailable":
                self._lease_fallback_status_by_request_id[request.request_id] = status
                lease_payload = {
                    "request_id": request.request_id,
                    "message_id": message_id,
                    "track_id": track_id,
                    "frame_reference": None,
                    "created_at": now,
                    "expires_at": now + self._lease_ttl_seconds,
                }
            else:
                self._store_submission(
                    VerificationSubmissionRecord(
                        request_id=request.request_id,
                        source=request.source,
                        track_id=track_id,
                        message_id=message_id,
                        submitted_at=now,
                        accepted=False,
                        status=status,
                        reason=status,
                        run_id=self._run_id,
                    )
                )
                self._store_record(
                    VerificationQueueRecord(
                        request_id=request.request_id,
                        source=request.source,
                        track_id=track_id,
                        evidence_message_id=message_id,
                        status="rejected",
                        validation_outcome="lease_acquire_failed",
                        submitted_at=now,
                        started_at=None,
                        finished_at=now,
                        current_applicability="unknown",
                        stale_for_current_decisions=False,
                        coordinator_valid=False,
                        error_code=status,
                        error="Unable to acquire evidence protection lease",
                        run_id=self._run_id,
                        details=self._sanitize_details(lease_result.get("details")),
                    )
                )
                return {"accepted": False, "status": status, "message_id": message_id, "request_id": request.request_id}

        if lease_payload is None:
            lease_payload = lease_result.get("lease")
        if not isinstance(lease_payload, dict):
            status = "lease_payload_invalid"
            self._leases_by_request_id.setdefault(request.request_id, {"request_id": request.request_id})
            self._release_request_lease(request.request_id)
            self._store_submission(
                VerificationSubmissionRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    message_id=message_id,
                    submitted_at=now,
                    accepted=False,
                    status=status,
                    reason=status,
                    run_id=self._run_id,
                )
            )
            self._store_record(
                VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    evidence_message_id=message_id,
                    status="rejected",
                    validation_outcome="lease_acquire_failed",
                    submitted_at=now,
                    started_at=None,
                    finished_at=now,
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code=status,
                    error="Lease acquisition returned invalid payload",
                    run_id=self._run_id,
                    details={},
                )
            )
            return {"accepted": False, "status": status, "message_id": message_id, "request_id": request.request_id}

        if (
            lease_payload.get("request_id") != request.request_id
            or lease_payload.get("message_id") != message_id
            or lease_payload.get("track_id") != track_id
        ):
            self._lease_fallback_status_by_request_id.pop(request.request_id, None)
            self._release_request_lease(request.request_id)
            status = "lease_association_mismatch"
            self._store_submission(
                VerificationSubmissionRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    message_id=message_id,
                    submitted_at=now,
                    accepted=False,
                    status=status,
                    reason=status,
                    run_id=self._run_id,
                )
            )
            return {"accepted": False, "status": status, "message_id": message_id, "request_id": request.request_id}

        self._leases_by_request_id[request.request_id] = lease_payload

        queue_request = VerificationQueueRequest(
            request_id=request.request_id,
            track_id=track_id,
            message_id=message_id,
            source=request.source,
            requested_at=now,
        )
        try:
            self._queue.put_nowait(queue_request)
        except Exception:
            self._release_request_lease(request.request_id)
            self._store_submission(
                VerificationSubmissionRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    message_id=message_id,
                    submitted_at=now,
                    accepted=False,
                    status="queue_enqueue_failed",
                    reason="queue_enqueue_failed",
                    run_id=self._run_id,
                )
            )
            return {"accepted": False, "status": "queue_enqueue_failed", "message_id": message_id, "request_id": request.request_id}
        self._queued_message_ids.add(message_id)
        self._store_submission(
            VerificationSubmissionRecord(
                request_id=request.request_id,
                source=request.source,
                track_id=track_id,
                message_id=message_id,
                submitted_at=now,
                accepted=True,
                status="queued",
                reason=None,
                run_id=self._run_id,
            )
        )
        return {"accepted": True, "status": "queued", "message_id": message_id, "request_id": request.request_id}

    def get_recent_results(self, limit: int = 50) -> list[dict[str, Any]]:
        limited = max(1, int(limit))
        request_ids = list(self._record_order)[-limited:]
        return [self._records_by_request_id[item].to_dict() for item in request_ids if item in self._records_by_request_id]

    def get_recent_submissions(self, limit: int = 50) -> list[dict[str, Any]]:
        limited = max(1, int(limit))
        request_ids = list(self._submission_order)[-limited:]
        return [
            self._submission_records_by_request_id[item].to_dict()
            for item in request_ids
            if item in self._submission_records_by_request_id
        ]

    def _resolve_latest_message_id_for_track(self, track_id: str) -> str | None:
        envelopes_dir = self._evidence_root_dir / "envelopes"
        if not envelopes_dir.is_dir():
            return None

        latest_message_id: str | None = None
        latest_timestamp: float | None = None
        for path in envelopes_dir.glob("*.json"):
            try:
                message_id = path.stem
                record = load_validated_envelope_by_message_id(
                    message_id,
                    evidence_root_dir=self._evidence_root_dir,
                )
            except Exception:
                continue

            if record.envelope.track_id != track_id:
                continue
            envelope_timestamp = float(record.envelope.timestamp)
            if latest_timestamp is None or envelope_timestamp > latest_timestamp:
                latest_timestamp = envelope_timestamp
                latest_message_id = record.envelope.message_id

        return latest_message_id

    async def _worker_loop(self, session: ClientSession, worker_index: int) -> None:
        _ = worker_index
        while True:
            request = await self._queue.get()
            message_id = request.message_id or ""
            self._queued_message_ids.discard(message_id)
            self._inflight_message_ids.add(message_id)
            started_at = time.time()
            try:
                record = await self._execute_request(session, request, started_at)
            except asyncio.CancelledError:
                finished_at = time.time()
                record = VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=request.track_id,
                    evidence_message_id=message_id,
                    status="cancelled",
                    validation_outcome="cancelled",
                    submitted_at=request.requested_at,
                    started_at=started_at,
                    finished_at=finished_at,
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="cancelled",
                    error="Verification worker cancelled",
                    run_id=self._run_id,
                    details={},
                )
                self._store_record(record)
                self._inflight_message_ids.discard(message_id)
                self._release_request_lease(request.request_id)
                self._queue.task_done()
                raise
            except Exception as error:
                finished_at = time.time()
                record = VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=request.track_id,
                    evidence_message_id=message_id,
                    status="failed",
                    validation_outcome="runtime_error",
                    submitted_at=request.requested_at,
                    started_at=started_at,
                    finished_at=finished_at,
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="runtime_error",
                    error=f"{type(error).__name__}:{error}",
                    run_id=self._run_id,
                    details={},
                )
            self._store_record(record)
            self._completed_message_ids.add(message_id)
            self._inflight_message_ids.discard(message_id)
            self._release_request_lease(request.request_id)
            self._queue.task_done()

    async def _execute_request(
        self,
        session: ClientSession,
        request: VerificationQueueRequest,
        started_at: float,
    ) -> VerificationQueueRecord:
        message_id = request.message_id or ""
        track_id = request.track_id

        precheck_deadline = min(
            started_at + self._request_timeout_seconds,
            time.time() + min(0.5, self._request_timeout_seconds),
        )
        precheck_attempts = 0
        evidence_payload: dict[str, Any] | None = None
        latest_pending_details: dict[str, Any] | None = None
        while True:
            precheck_attempts += 1
            remaining = precheck_deadline - time.time()
            if remaining <= 0:
                if latest_pending_details is not None:
                    latest_pending_details["attempts"] = precheck_attempts
                    return VerificationQueueRecord(
                        request_id=request.request_id,
                        source=request.source,
                        track_id=track_id,
                        evidence_message_id=message_id,
                        status="rejected",
                        validation_outcome="evidence_unavailable",
                        submitted_at=request.requested_at,
                        started_at=started_at,
                        finished_at=time.time(),
                        current_applicability="unknown",
                        stale_for_current_decisions=False,
                        coordinator_valid=False,
                        error_code="missing_or_expired_evidence",
                        error="Evidence unavailable prior to verification",
                        run_id=self._run_id,
                        details=latest_pending_details,
                    )
                return VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    evidence_message_id=message_id,
                    status="timed_out",
                    validation_outcome="operator_timeout",
                    submitted_at=request.requested_at,
                    started_at=started_at,
                    finished_at=time.time(),
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="operator_timeout",
                    error="Operator request timeout exceeded",
                    run_id=self._run_id,
                    details={"timeout_seconds": self._request_timeout_seconds},
                )

            evidence_precheck = await asyncio.wait_for(
                session.call_tool(self._evidence_tool_name, arguments={"message_id": message_id}),
                timeout=remaining,
            )
            evidence_payload = self._extract_payload(evidence_precheck)
            if not isinstance(evidence_payload, dict):
                return VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    evidence_message_id=message_id,
                    status="rejected",
                    validation_outcome="evidence_unavailable",
                    submitted_at=request.requested_at,
                    started_at=started_at,
                    finished_at=time.time(),
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="precheck_response_malformed",
                    error="Malformed precheck response prior to verification",
                    run_id=self._run_id,
                    details=self._build_precheck_malformed_details(
                        precheck_response=evidence_precheck,
                        extracted_payload=evidence_payload,
                        attempts=precheck_attempts,
                    ),
                )
            if evidence_payload.get("available", False):
                break

            details = self._sanitize_details(evidence_payload)
            nested_details = evidence_payload.get("details") if isinstance(evidence_payload, dict) else None
            if isinstance(nested_details, dict):
                for key, value in self._sanitize_details(nested_details).items():
                    details.setdefault(key, value)
                for key, value in self._derive_precheck_reference_diagnostics(nested_details).items():
                    details.setdefault(key, value)

            if not self._is_pending_visual_evidence(details):
                return VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    evidence_message_id=message_id,
                    status="rejected",
                    validation_outcome="evidence_unavailable",
                    submitted_at=request.requested_at,
                    started_at=started_at,
                    finished_at=time.time(),
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="missing_or_expired_evidence",
                    error="Evidence unavailable prior to verification",
                    run_id=self._run_id,
                    details=details,
                )

            latest_pending_details = details

            sleep_seconds = min(0.05, max(0.0, precheck_deadline - time.time()))
            if sleep_seconds <= 0:
                details["attempts"] = precheck_attempts
                return VerificationQueueRecord(
                    request_id=request.request_id,
                    source=request.source,
                    track_id=track_id,
                    evidence_message_id=message_id,
                    status="rejected",
                    validation_outcome="evidence_unavailable",
                    submitted_at=request.requested_at,
                    started_at=started_at,
                    finished_at=time.time(),
                    current_applicability="unknown",
                    stale_for_current_decisions=False,
                    coordinator_valid=False,
                    error_code="missing_or_expired_evidence",
                    error="Evidence unavailable prior to verification",
                    run_id=self._run_id,
                    details=details,
                )
            await asyncio.sleep(sleep_seconds)

        if str(evidence_payload.get("track_id", "")) != track_id:
            return VerificationQueueRecord(
                request_id=request.request_id,
                source=request.source,
                track_id=track_id,
                evidence_message_id=message_id,
                status="rejected",
                validation_outcome="track_mismatch",
                submitted_at=request.requested_at,
                started_at=started_at,
                finished_at=time.time(),
                current_applicability="unknown",
                stale_for_current_decisions=False,
                coordinator_valid=False,
                error_code="track_id_mismatch",
                error="Resolved evidence track_id does not match request track_id",
                run_id=self._run_id,
                details={},
            )

        try:
            verify_result = await asyncio.wait_for(
                session.call_tool(self._verify_tool_name, arguments={"message_id": message_id}),
                timeout=self._request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return VerificationQueueRecord(
                request_id=request.request_id,
                source=request.source,
                track_id=track_id,
                evidence_message_id=message_id,
                status="timed_out",
                validation_outcome="operator_timeout",
                submitted_at=request.requested_at,
                started_at=started_at,
                finished_at=time.time(),
                current_applicability="unknown",
                stale_for_current_decisions=False,
                coordinator_valid=False,
                error_code="operator_timeout",
                error="Operator request timeout exceeded",
                run_id=self._run_id,
                details={"timeout_seconds": self._request_timeout_seconds},
            )

        verify_payload = self._extract_payload(verify_result)
        if not isinstance(verify_payload, dict):
            return self._rejected_record(request, message_id, started_at, "malformed_response", "Malformed verification payload")

        if not bool(verify_payload.get("available", False)):
            return VerificationQueueRecord(
                request_id=request.request_id,
                source=request.source,
                track_id=track_id,
                evidence_message_id=message_id,
                status="failed",
                validation_outcome="provider_failure",
                submitted_at=request.requested_at,
                started_at=started_at,
                finished_at=time.time(),
                current_applicability="unknown",
                stale_for_current_decisions=False,
                coordinator_valid=False,
                error_code=str(verify_payload.get("error_code", "provider_failure")),
                error=str(verify_payload.get("error", "Visual verification failed")),
                run_id=self._run_id,
                details=self._sanitize_details(verify_payload.get("details")),
            )

        envelope_payload = verify_payload.get("envelope")
        if not isinstance(envelope_payload, dict):
            return self._rejected_record(request, message_id, started_at, "malformed_envelope", "Missing verification envelope")

        try:
            envelope = VisualVerificationEnvelope.from_dict(envelope_payload)
        except Exception:
            return self._rejected_record(request, message_id, started_at, "malformed_envelope", "Unable to parse verification envelope")

        if envelope.referenced_message_id != message_id:
            return self._rejected_record(request, message_id, started_at, "message_id_mismatch", "Verification referenced unexpected message_id")
        if envelope.referenced_track_id != track_id:
            return self._rejected_record(request, message_id, started_at, "track_id_mismatch", "Verification referenced unexpected track_id")

        try:
            evidence_record = load_validated_envelope_by_message_id(message_id, evidence_root_dir=self._evidence_root_dir)
        except Exception:
            return self._rejected_record(request, message_id, started_at, "missing_evidence_envelope", "Referenced evidence envelope not found")

        validation = self._coordinator.validate_visual_verification_envelope(
            envelope,
            evidence_by_message_id={message_id: evidence_record.envelope},
        )
        if not bool(validation.get("valid", False)):
            return VerificationQueueRecord(
                request_id=request.request_id,
                source=request.source,
                track_id=track_id,
                evidence_message_id=message_id,
                status="rejected",
                validation_outcome="coordinator_validation_failed",
                submitted_at=request.requested_at,
                started_at=started_at,
                finished_at=time.time(),
                current_applicability="unknown",
                stale_for_current_decisions=False,
                coordinator_valid=False,
                error_code="coordinator_validation_failed",
                error="Coordinator rejected verification envelope",
                run_id=self._run_id,
                details=self._sanitize_details(validation),
            )

        latest_message_id = self._resolve_latest_message_id_for_track(track_id)
        is_historical = latest_message_id is not None and latest_message_id != message_id

        return VerificationQueueRecord(
            request_id=request.request_id,
            source=request.source,
            track_id=track_id,
            evidence_message_id=message_id,
            status="completed",
            validation_outcome="accepted",
            submitted_at=request.requested_at,
            started_at=started_at,
            finished_at=time.time(),
            current_applicability=("historical" if is_historical else "current"),
            stale_for_current_decisions=is_historical,
            coordinator_valid=True,
            error_code=None,
            error=None,
            run_id=self._run_id,
            details={
                "verification_observation_id": envelope.observation_id,
                "verification_referenced_message_id": envelope.referenced_message_id,
                "verification_timestamp": envelope.timestamp,
                "verification_status": envelope.verification_status,
                "identity_consistency": envelope.identity_consistency,
                "visibility_status": envelope.visibility_status,
                "possible_id_switch": envelope.possible_id_switch,
                "confidence": envelope.confidence,
                "visual_content_inspected": envelope.visual_content_inspected,
                "source_evidence_refs": list(envelope.source_evidence_refs),
                "supporting_evidence_refs": list(envelope.supporting_evidence_refs),
                "contradicting_evidence_refs": list(envelope.contradicting_evidence_refs),
                "uncertainty": list(envelope.uncertainty),
                "limitations": list(envelope.limitations),
            },
        )

    def _extract_payload(self, result: Any) -> dict[str, Any] | None:
        content = getattr(result, "content", None)
        if not isinstance(content, list):
            return None
        for item in content:
            for key in ("json", "data", "value"):
                value = getattr(item, key, None)
                if isinstance(value, dict):
                    return value
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                continue
            raw_text = text.strip()
            if not raw_text:
                continue
            try:
                decoded = json.loads(raw_text)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                return decoded
        return None

    def _rejected_record(
        self,
        request: VerificationQueueRequest,
        message_id: str,
        started_at: float,
        code: str,
        error: str,
    ) -> VerificationQueueRecord:
        return VerificationQueueRecord(
            request_id=request.request_id,
            source=request.source,
            track_id=request.track_id,
            evidence_message_id=message_id,
            status="rejected",
            validation_outcome=code,
            submitted_at=request.requested_at,
            started_at=started_at,
            finished_at=time.time(),
            current_applicability="unknown",
            stale_for_current_decisions=False,
            coordinator_valid=False,
            error_code=code,
            error=error,
            run_id=self._run_id,
            details={},
        )

    def _sanitize_details(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        allowed = {
            "error_code",
            "upstream_code",
            "gate_errors",
            "attempts",
            "model",
            "message_id",
            "precheck_failure_kind",
            "precheck_response_shape",
            "precheck_payload_type",
            "precheck_content_item_count",
            "failing_reference",
            "failing_reference_kind",
            "failing_reference_status",
        }
        return {
            str(key): value
            for key, value in payload.items()
            if str(key) in allowed and isinstance(value, (str, int, float, bool, list, dict, type(None)))
        }

    def _build_precheck_malformed_details(
        self,
        *,
        precheck_response: Any,
        extracted_payload: Any,
        attempts: int,
    ) -> dict[str, Any]:
        details: dict[str, Any] = {
            "precheck_failure_kind": "precheck_response_malformed",
            "attempts": int(attempts),
        }
        if extracted_payload is not None:
            details["precheck_payload_type"] = type(extracted_payload).__name__

        content = getattr(precheck_response, "content", None)
        if isinstance(content, list):
            details["precheck_content_item_count"] = len(content)
            if content:
                details["precheck_response_shape"] = type(content[0]).__name__
            else:
                details["precheck_response_shape"] = "empty_content"
        else:
            details["precheck_response_shape"] = type(precheck_response).__name__
        return details

    def _derive_precheck_reference_diagnostics(self, details: dict[str, Any]) -> dict[str, Any]:
        reference_results = details.get("reference_results")
        if not isinstance(reference_results, list):
            return {}

        for item in reference_results:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if not isinstance(status, str) or not status:
                continue
            if status not in {
                "missing_reference",
                "expired_reference",
                "pending_reference",
                "reference_mismatch",
                "invalid_reference",
            }:
                continue

            diagnostics: dict[str, Any] = {
                "precheck_failure_kind": "reference_validation_failed",
                "failing_reference_status": status,
            }
            reference = item.get("reference")
            if isinstance(reference, str) and reference:
                diagnostics["failing_reference"] = reference
            kind = item.get("kind")
            if isinstance(kind, str) and kind:
                diagnostics["failing_reference_kind"] = kind
            return diagnostics
        return {}

    def _is_pending_visual_evidence(self, details: dict[str, Any]) -> bool:
        if str(details.get("upstream_code", "")) != "observation_evidence_preflight_failed":
            return False
        gate_errors = details.get("gate_errors")
        if not isinstance(gate_errors, list):
            return False
        return any(isinstance(item, str) and item == "pending_visual_evidence" for item in gate_errors)

    def _store_record(self, record: VerificationQueueRecord) -> None:
        request_id = record.request_id
        fallback_status = self._lease_fallback_status_by_request_id.get(request_id)
        if isinstance(fallback_status, str) and fallback_status:
            details = record.details if isinstance(record.details, dict) else {}
            enriched_details = dict(details)
            enriched_details["evidence_lease_fallback_used"] = True
            enriched_details["evidence_lease_fallback_status"] = fallback_status
            record.details = enriched_details

        if request_id not in self._records_by_request_id and len(self._record_order) == self._result_registry_capacity:
            oldest = self._record_order[0]
            self._records_by_request_id.pop(oldest, None)
        if request_id not in self._records_by_request_id:
            self._record_order.append(request_id)
        self._records_by_request_id[request_id] = record
        self._lease_fallback_status_by_request_id.pop(request_id, None)

    def _store_submission(self, record: VerificationSubmissionRecord) -> None:
        request_id = record.request_id
        if request_id not in self._submission_records_by_request_id and len(self._submission_order) == self._result_registry_capacity:
            oldest = self._submission_order[0]
            self._submission_records_by_request_id.pop(oldest, None)
        if request_id not in self._submission_records_by_request_id:
            self._submission_order.append(request_id)
        self._submission_records_by_request_id[request_id] = record

    def _release_request_lease(self, request_id: str) -> bool:
        normalized_request_id = request_id.strip() if isinstance(request_id, str) else ""
        if not normalized_request_id:
            return True

        release_result = release_verification_evidence_lease(
            evidence_root_dir=self._evidence_root_dir,
            request_id=normalized_request_id,
            lock_timeout_seconds=self._lease_lock_timeout_seconds,
        )
        released = bool(release_result.get("released", False))
        status = str(release_result.get("status", "unknown"))

        if released or status == "invalid_request_id":
            self._leases_by_request_id.pop(normalized_request_id, None)
            self._pending_lease_release_request_ids.discard(normalized_request_id)
            self._lease_release_retry_attempts_by_request_id.pop(normalized_request_id, None)
            return True

        if status == "lease_lock_unavailable":
            attempts = self._lease_release_retry_attempts_by_request_id.get(normalized_request_id, 0) + 1
            self._lease_release_retry_attempts_by_request_id[normalized_request_id] = attempts
            self._pending_lease_release_request_ids.add(normalized_request_id)
            self._leases_by_request_id.setdefault(normalized_request_id, {"request_id": normalized_request_id})
            return False

        self._leases_by_request_id.pop(normalized_request_id, None)
        self._pending_lease_release_request_ids.discard(normalized_request_id)
        self._lease_release_retry_attempts_by_request_id.pop(normalized_request_id, None)
        return False

    def _retry_pending_lease_releases(self, *, max_passes: int) -> None:
        total_passes = max(1, int(max_passes))
        for _ in range(total_passes):
            pending_request_ids = list(self._pending_lease_release_request_ids)
            if not pending_request_ids:
                return

            for pending_request_id in pending_request_ids:
                self._release_request_lease(pending_request_id)


def build_request_from_track_id(track_id: str, *, source: str) -> VerificationQueueRequest:
    now = time.time()
    return VerificationQueueRequest(
        request_id=f"verifyreq-{uuid.uuid4().hex}",
        track_id=track_id,
        message_id=None,
        source=source,
        requested_at=now,
    )


def build_request_from_message_id(track_id: str, message_id: str, *, source: str) -> VerificationQueueRequest:
    now = time.time()
    return VerificationQueueRequest(
        request_id=f"verifyreq-{uuid.uuid4().hex}",
        track_id=track_id,
        message_id=message_id,
        source=source,
        requested_at=now,
    )


def extract_submission_decisions_from_cycle_payload(payload: dict[str, Any]) -> list[VerificationSubmissionDecision]:
    events = payload.get("events")
    if not isinstance(events, list):
        return [
            VerificationSubmissionDecision(
                event_origin=None,
                trigger_event_id=None,
                track_id=None,
                evidence_requested=False,
                evidence_message_id=None,
                queue_request=None,
                submission_eligibility=False,
                skip_reason="events_missing_or_invalid",
            )
        ]

    decisions: list[VerificationSubmissionDecision] = []
    for event in events:
        if not isinstance(event, dict):
            decisions.append(
                VerificationSubmissionDecision(
                    event_origin=None,
                    trigger_event_id=None,
                    track_id=None,
                    evidence_requested=False,
                    evidence_message_id=None,
                    queue_request=None,
                    submission_eligibility=False,
                    skip_reason="event_not_dict",
                )
            )
            continue

        trigger_event_id = event.get("event_id")
        if not isinstance(trigger_event_id, str) or not trigger_event_id.strip():
            trigger_event_id = event.get("task_id")
        if not isinstance(trigger_event_id, str) or not trigger_event_id.strip():
            trigger_event_id = None

        evidence_requested = bool(event.get("evidence_requested", False))
        track_id_value = event.get("track_id")
        track_id = track_id_value.strip() if isinstance(track_id_value, str) else None

        evidence_message_id_value = event.get("evidence_message_id")
        evidence_message_id = (
            evidence_message_id_value.strip()
            if isinstance(evidence_message_id_value, str) and evidence_message_id_value.strip()
            else None
        )
        event_origin_value = event.get("event_origin")
        event_origin: str | None
        invalid_event_origin = False
        if isinstance(event_origin_value, str):
            normalized_event_origin = event_origin_value.strip()
            if normalized_event_origin in {"startup_activation", "production_monitor"}:
                event_origin = normalized_event_origin
            elif normalized_event_origin:
                event_origin = None
                invalid_event_origin = True
            else:
                event_origin = "production_monitor"
        elif event_origin_value is None:
            event_origin = "production_monitor"
        else:
            event_origin = None
            invalid_event_origin = True

        if not evidence_requested:
            decisions.append(
                VerificationSubmissionDecision(
                    event_origin=event_origin,
                    trigger_event_id=trigger_event_id,
                    track_id=track_id,
                    evidence_requested=False,
                    evidence_message_id=evidence_message_id,
                    queue_request=None,
                    submission_eligibility=False,
                    skip_reason="evidence_not_requested",
                )
            )
            continue

        if event_origin != "production_monitor":
            decisions.append(
                VerificationSubmissionDecision(
                    event_origin=event_origin,
                    trigger_event_id=trigger_event_id,
                    track_id=track_id,
                    evidence_requested=True,
                    evidence_message_id=evidence_message_id,
                    queue_request=None,
                    submission_eligibility=False,
                    skip_reason=(
                        "startup_activation_event"
                        if event_origin == "startup_activation"
                        else (
                            "invalid_event_origin"
                            if invalid_event_origin
                            else "invalid_or_missing_event_origin"
                        )
                    ),
                )
            )
            continue

        if bool(event.get("skip_verification", False)):
            skip_reason = event.get("skip_reason")
            normalized_skip_reason = skip_reason.strip() if isinstance(skip_reason, str) and skip_reason.strip() else "verification_skipped"
            decisions.append(
                VerificationSubmissionDecision(
                    event_origin=event_origin,
                    trigger_event_id=trigger_event_id,
                    track_id=track_id,
                    evidence_requested=True,
                    evidence_message_id=evidence_message_id,
                    queue_request=None,
                    submission_eligibility=False,
                    skip_reason=normalized_skip_reason,
                )
            )
            continue

        if not isinstance(track_id, str) or not track_id.strip():
            decisions.append(
                VerificationSubmissionDecision(
                    event_origin=event_origin,
                    trigger_event_id=trigger_event_id,
                    track_id=None,
                    evidence_requested=True,
                    evidence_message_id=evidence_message_id,
                    queue_request=None,
                    submission_eligibility=False,
                    skip_reason="invalid_track_id",
                )
            )
            continue

        if evidence_message_id is not None:
            request = build_request_from_message_id(track_id, evidence_message_id, source="production_monitor")
        else:
            request = build_request_from_track_id(track_id, source="production_monitor")

        request.trigger_event_id = trigger_event_id
        event_timestamp = event.get("timestamp")
        if isinstance(event_timestamp, (int, float)):
            request.trigger_timestamp = float(event_timestamp)

        decisions.append(
            VerificationSubmissionDecision(
                event_origin=event_origin,
                trigger_event_id=trigger_event_id,
                track_id=track_id,
                evidence_requested=True,
                evidence_message_id=evidence_message_id,
                queue_request=request,
                submission_eligibility=True,
                skip_reason=None,
            )
        )

    return decisions


def extract_track_requests_from_cycle_payload(payload: dict[str, Any]) -> Iterable[VerificationQueueRequest]:
    decisions = extract_submission_decisions_from_cycle_payload(payload)
    requests: list[VerificationQueueRequest] = []
    for decision in decisions:
        request = decision.queue_request
        if request is None:
            continue
        requests.append(
            VerificationQueueRequest(
                request_id=request.request_id,
                track_id=request.track_id,
                message_id=request.message_id,
                source="reasoning_cycle",
                requested_at=request.requested_at,
                trigger_event_id=request.trigger_event_id,
                trigger_timestamp=request.trigger_timestamp,
            )
        )
    return requests
