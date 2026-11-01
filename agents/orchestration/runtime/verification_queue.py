from __future__ import annotations

import asyncio
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


@dataclass(slots=True)
class VerificationQueueRequest:
    request_id: str
    track_id: str
    message_id: str | None
    source: str
    requested_at: float


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
        }


class VerificationQueueManager:
    def __init__(
        self,
        *,
        evidence_root_dir: Path,
        queue_capacity: int = 32,
        max_concurrency: int = 2,
        request_timeout_seconds: float = 8.0,
        result_registry_capacity: int = 512,
        verify_tool_name: str = "verify_tracking_visual_evidence",
        evidence_tool_name: str = "get_tracking_visual_evidence",
        coordinator: AgentCoordinator | None = None,
    ) -> None:
        self._evidence_root_dir = evidence_root_dir
        self._queue_capacity = max(1, int(queue_capacity))
        self._max_concurrency = max(1, int(max_concurrency))
        self._request_timeout_seconds = max(0.1, float(request_timeout_seconds))
        self._result_registry_capacity = max(1, int(result_registry_capacity))
        self._verify_tool_name = verify_tool_name
        self._evidence_tool_name = evidence_tool_name
        self._coordinator = coordinator or AgentCoordinator()

        self._queue: asyncio.Queue[VerificationQueueRequest] = asyncio.Queue(maxsize=self._queue_capacity)
        self._workers: list[asyncio.Task[None]] = []
        self._shutdown = False

        self._queued_message_ids: set[str] = set()
        self._inflight_message_ids: set[str] = set()
        self._completed_message_ids: set[str] = set()

        self._records_by_request_id: dict[str, VerificationQueueRecord] = {}
        self._record_order: deque[str] = deque(maxlen=self._result_registry_capacity)

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

    def submit_request(self, request: VerificationQueueRequest) -> dict[str, Any]:
        if self._shutdown:
            return {"accepted": False, "status": "manager_shutdown"}

        now = time.time()
        message_id = request.message_id
        track_id = request.track_id.strip() if isinstance(request.track_id, str) else ""
        if not track_id:
            return {"accepted": False, "status": "invalid_track_id"}

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
                    details={"queue_capacity": self._queue_capacity},
                )
            )
            return {"accepted": False, "status": "queue_saturated", "message_id": message_id}

        queue_request = VerificationQueueRequest(
            request_id=request.request_id,
            track_id=track_id,
            message_id=message_id,
            source=request.source,
            requested_at=now,
        )
        self._queue.put_nowait(queue_request)
        self._queued_message_ids.add(message_id)
        return {"accepted": True, "status": "queued", "message_id": message_id}

    def get_recent_results(self, limit: int = 50) -> list[dict[str, Any]]:
        limited = max(1, int(limit))
        request_ids = list(self._record_order)[-limited:]
        return [self._records_by_request_id[item].to_dict() for item in request_ids if item in self._records_by_request_id]

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
                    details={},
                )
                self._store_record(record)
                self._inflight_message_ids.discard(message_id)
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
                    details={},
                )
            self._store_record(record)
            self._completed_message_ids.add(message_id)
            self._inflight_message_ids.discard(message_id)
            self._queue.task_done()

    async def _execute_request(
        self,
        session: ClientSession,
        request: VerificationQueueRequest,
        started_at: float,
    ) -> VerificationQueueRecord:
        message_id = request.message_id or ""
        track_id = request.track_id

        evidence_precheck = await asyncio.wait_for(
            session.call_tool(self._evidence_tool_name, arguments={"message_id": message_id}),
            timeout=self._request_timeout_seconds,
        )
        evidence_payload = self._extract_payload(evidence_precheck)
        if not isinstance(evidence_payload, dict) or not evidence_payload.get("available", False):
            error_code = "missing_or_expired_evidence"
            details = self._sanitize_details(evidence_payload)
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
                error_code=error_code,
                error="Evidence unavailable prior to verification",
                details=details,
            )

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
            details={},
        )

    def _sanitize_details(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        allowed = {"error_code", "upstream_code", "gate_errors", "attempts", "model", "message_id"}
        return {
            str(key): value
            for key, value in payload.items()
            if str(key) in allowed and isinstance(value, (str, int, float, bool, list, dict, type(None)))
        }

    def _store_record(self, record: VerificationQueueRecord) -> None:
        request_id = record.request_id
        if request_id not in self._records_by_request_id and len(self._record_order) == self._result_registry_capacity:
            oldest = self._record_order[0]
            self._records_by_request_id.pop(oldest, None)
        if request_id not in self._records_by_request_id:
            self._record_order.append(request_id)
        self._records_by_request_id[request_id] = record


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


def extract_track_requests_from_cycle_payload(payload: dict[str, Any]) -> Iterable[VerificationQueueRequest]:
    events = payload.get("events")
    if not isinstance(events, list):
        return []

    requests: list[VerificationQueueRequest] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if not bool(event.get("evidence_requested", False)):
            continue
        track_id = event.get("track_id")
        if not isinstance(track_id, str) or not track_id.strip():
            continue
        requests.append(build_request_from_track_id(track_id.strip(), source="reasoning_cycle"))
    return requests
