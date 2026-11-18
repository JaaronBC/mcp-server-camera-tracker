from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

REQUEST_SCHEMA_VERSION = "1.0"
RESULT_SCHEMA_VERSION = "1.0"
REQUEST_EVENT_TYPE = "decision_tool_request"
RESULT_EVENT_TYPE = "decision_tool_result"

REQUEST_ACTIONS = {
    "request_identity_snapshot",
    "request_additional_tracking_evidence",
}
REQUEST_ACTION_TO_ALLOWED_TOOLS = {
    "request_identity_snapshot": frozenset({"get_identity_snapshot"}),
    "request_additional_tracking_evidence": frozenset({"get_track_evidence_history"}),
}
ALLOWLISTED_TOOLS = {
    "get_identity_snapshot",
    "get_track_evidence_history",
}
VALID_RESULT_STATUSES = {
    "ok",
    "timeout",
    "unavailable",
    "rejected",
    "expired",
    "duplicate",
    "terminal",
    "incompatible",
    "stale",
}
TERMINAL_INVESTIGATION_STATES = {"resolved", "closed"}
VALID_WRITER_ROLES = {"shared", "dashboard", "orchestration"}
TOOL_BUDGET_EXHAUSTED_STATES = {"exhausted"}
OBSERVATION_FOLLOW_UP_HANDOFF_MAX_RECORDS = 512
OBSERVATION_FOLLOW_UP_HANDOFF_MAX_AGE_SECONDS = 120.0
OBSERVATION_FOLLOW_UP_ASSESSMENT_DELIVERY_SUFFIX = ":observation_follow_up_assessment"


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REQUESTS_PATH = PROJECT_ROOT / "captures" / "tracking_decision_tool_requests.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "captures" / "tracking_decision_tool_results.jsonl"
DEFAULT_REQUEST_INDEX_PATH = PROJECT_ROOT / "captures" / "latest_tracking_decision_request_index.json"
DEFAULT_RESULT_INDEX_PATH = PROJECT_ROOT / "captures" / "latest_tracking_decision_result_index.json"


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _event_from_jsonl_line(line: str) -> dict[str, Any] | None:
    raw = line.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _request_key(*, investigation_id: str, request_id: str) -> str:
    return f"{investigation_id}:{request_id}"


def _result_key(*, investigation_id: str, request_id: str, tool_name: str) -> str:
    return f"{investigation_id}:{request_id}:{tool_name}"


def _observation_assessment_delivery_key(result_key: str) -> str:
    return f"{result_key}{OBSERVATION_FOLLOW_UP_ASSESSMENT_DELIVERY_SUFFIX}"


def _fsync_append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, backup_path: Path, payload: dict[str, Any]) -> None:
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


class TrackingDecisionIpcTransport:
    def __init__(
        self,
        *,
        requests_path: Path = DEFAULT_REQUESTS_PATH,
        results_path: Path = DEFAULT_RESULTS_PATH,
        request_index_path: Path = DEFAULT_REQUEST_INDEX_PATH,
        result_index_path: Path = DEFAULT_RESULT_INDEX_PATH,
        max_request_records: int = 10_000,
        max_result_records: int = 10_000,
        max_terminal_investigations: int = 2_000,
        request_ttl_seconds: float = 45.0,
        max_dashboard_results_per_poll: int = 8,
        writer_role: str = "shared",
    ) -> None:
        self._requests_path = requests_path
        self._results_path = results_path
        self._request_index_path = request_index_path
        self._request_index_backup = request_index_path.with_name(f"{request_index_path.name}.bak")
        self._result_index_path = result_index_path
        self._result_index_backup = result_index_path.with_name(f"{result_index_path.name}.bak")

        self._max_request_records = max(1, int(max_request_records))
        self._max_result_records = max(1, int(max_result_records))
        self._max_terminal_investigations = max(16, int(max_terminal_investigations))
        self._request_ttl_seconds = max(1.0, float(request_ttl_seconds))
        self._max_dashboard_results_per_poll = max(1, int(max_dashboard_results_per_poll))
        normalized_role = _nonempty_str(writer_role)
        if normalized_role not in VALID_WRITER_ROLES:
            normalized_role = "shared"
        self._writer_role = str(normalized_role)

    @property
    def writer_role(self) -> str:
        return self._writer_role

    def _writer_role_rejected_response(self, *, operation: str) -> dict[str, Any]:
        return {
            "accepted": False,
            "duplicate": False,
            "rejected": True,
            "reason": "writer_role_violation",
            "operation": operation,
            "writer_role": self._writer_role,
        }

    def _result_writer_role_rejected_response(self, *, operation: str) -> dict[str, Any]:
        return {
            "published": False,
            "duplicate": False,
            "rejected": True,
            "reason": "writer_role_violation",
            "operation": operation,
            "writer_role": self._writer_role,
        }

    def _is_writer_role_allowed(self, *, allowed_roles: set[str]) -> bool:
        return self._writer_role in allowed_roles

    @property
    def requests_path(self) -> Path:
        return self._requests_path

    @property
    def results_path(self) -> Path:
        return self._results_path

    def dashboard_submit_decision_request(
        self,
        *,
        decision: dict[str, Any],
        llm_correlation_id: str | None = None,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        if not self._is_writer_role_allowed(allowed_roles={"shared", "dashboard"}):
            return self._writer_role_rejected_response(operation="dashboard_submit_decision_request")
        now = time.time() if now_timestamp is None else float(now_timestamp)
        payload, errors = self._build_request_event_from_decision(
            decision=decision,
            llm_correlation_id=llm_correlation_id,
            now_timestamp=now,
        )
        if payload is None:
            return {
                "accepted": False,
                "duplicate": False,
                "rejected": True,
                "reason": "invalid_decision_request",
                "errors": errors,
            }
        return self._append_request_event(payload, now_timestamp=now)

    def dashboard_poll_results(self, *, now_timestamp: float | None = None) -> dict[str, Any]:
        now = time.time() if now_timestamp is None else float(now_timestamp)
        state = self._load_state(now_timestamp=now)

        delivered_keys = state["delivered_result_keys"]
        terminal_result_keys = state["terminal_result_keys"]
        request_by_dedupe = state["request_by_dedupe"]
        delivered_assessment_keys = state["delivered_observation_assessment_keys"]

        emitted: list[dict[str, Any]] = []
        for result in state["results"]:
            result_key = _nonempty_str(result.get("idempotency_key"))
            if result_key is None:
                continue
            if result_key in delivered_keys:
                continue
            if result.get("status") == "duplicate":
                delivered_keys.add(result_key)
                continue

            request_id = _nonempty_str(result.get("request_id"))
            investigation_id = _nonempty_str(result.get("investigation_id"))
            tool_name = _nonempty_str(result.get("tool_name"))
            iteration_index = result.get("iteration_index")
            if (
                request_id is None
                or investigation_id is None
                or tool_name is None
                or not isinstance(iteration_index, int)
            ):
                delivered_keys.add(result_key)
                continue

            dedupe_key = _request_key(investigation_id=investigation_id, request_id=request_id)
            request_event = request_by_dedupe.get(dedupe_key)
            if not isinstance(request_event, dict):
                delivered_keys.add(result_key)
                continue
            if request_event.get("tool_name") != tool_name:
                delivered_keys.add(result_key)
                continue
            if request_event.get("iteration_index") != iteration_index:
                delivered_keys.add(result_key)
                continue

            emitted_result = self._attach_observation_follow_up_handoff(
                state=state,
                result=result,
                now_timestamp=now,
            )
            emitted_result["delivery_kind"] = "mcp_result"
            emitted.append(emitted_result)
            delivered_keys.add(result_key)
            if len(emitted) >= self._max_dashboard_results_per_poll:
                break

        if len(emitted) < self._max_dashboard_results_per_poll:
            for handoff in self._observation_handoffs_ready_for_delivery(state=state, now_timestamp=now):
                handoff_delivery_key = _observation_assessment_delivery_key(
                    _nonempty_str(handoff.get("idempotency_key")) or ""
                )
                if handoff_delivery_key in delivered_assessment_keys:
                    continue
                assessment_event = self._build_observation_assessment_delivery_event(
                    state=state,
                    handoff=handoff,
                    now_timestamp=now,
                )
                if not isinstance(assessment_event, dict):
                    continue
                emitted.append(assessment_event)
                delivered_assessment_keys.add(handoff_delivery_key)
                if len(emitted) >= self._max_dashboard_results_per_poll:
                    break

        self._persist_indexes(state=state, now_timestamp=now)
        return {
            "available": True,
            "result_count": len(emitted),
            "results": emitted,
            "retained_terminal_results": len(terminal_result_keys),
        }

    def dashboard_register_observation_follow_up_assessment(
        self,
        *,
        investigation_id: str,
        request_id: str,
        tool_name: str,
        idempotency_key: str,
        track_id: str | None,
        assessment: dict[str, Any],
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        if not self._is_writer_role_allowed(allowed_roles={"shared", "dashboard"}):
            return self._writer_role_rejected_response(
                operation="dashboard_register_observation_follow_up_assessment"
            )

        normalized_investigation_id = _nonempty_str(investigation_id)
        normalized_request_id = _nonempty_str(request_id)
        normalized_tool_name = _nonempty_str(tool_name)
        normalized_result_key = _nonempty_str(idempotency_key)
        normalized_track_id = _nonempty_str(track_id)
        if (
            normalized_investigation_id is None
            or normalized_request_id is None
            or normalized_tool_name is None
            or normalized_result_key is None
            or not isinstance(assessment, dict)
        ):
            return {
                "published": False,
                "duplicate": False,
                "rejected": True,
                "reason": "invalid_handoff_shape",
            }

        now = time.time() if now_timestamp is None else float(now_timestamp)
        state = self._load_state(now_timestamp=now)

        matched_result: dict[str, Any] | None = None
        for result in state["results"]:
            if _nonempty_str(result.get("idempotency_key")) != normalized_result_key:
                continue
            if _nonempty_str(result.get("investigation_id")) != normalized_investigation_id:
                continue
            if _nonempty_str(result.get("request_id")) != normalized_request_id:
                continue
            if _nonempty_str(result.get("tool_name")) != normalized_tool_name:
                continue
            matched_result = dict(result)
            break

        if matched_result is None:
            return {
                "published": False,
                "duplicate": False,
                "rejected": True,
                "reason": "correlation_mismatch",
            }

        finished_at = matched_result.get("finished_at")
        if isinstance(finished_at, (int, float)):
            if (float(now) - float(finished_at)) > OBSERVATION_FOLLOW_UP_HANDOFF_MAX_AGE_SECONDS:
                return {
                    "published": False,
                    "duplicate": False,
                    "rejected": True,
                    "reason": "expired_assessment",
                }

        handoffs = state["observation_follow_up_handoffs"]
        existing = handoffs.get(normalized_result_key)
        normalized_assessment = dict(assessment)
        if isinstance(existing, dict):
            if existing.get("assessment") == normalized_assessment:
                return {
                    "published": False,
                    "duplicate": True,
                    "rejected": False,
                    "reason": "duplicate_handoff",
                }
            return {
                "published": False,
                "duplicate": False,
                "rejected": True,
                "reason": "superseded_assessment_rejected",
            }

        handoffs[normalized_result_key] = {
            "idempotency_key": normalized_result_key,
            "investigation_id": normalized_investigation_id,
            "request_id": normalized_request_id,
            "tool_name": normalized_tool_name,
            "track_id": normalized_track_id,
            "assessment": normalized_assessment,
            "generated_at": now,
        }
        self._bound_observation_follow_up_handoffs(handoffs)
        self._persist_indexes(state=state, now_timestamp=now)
        return {
            "published": True,
            "duplicate": False,
            "rejected": False,
            "idempotency_key": normalized_result_key,
        }

    def orchestration_poll_pending_requests(
        self,
        *,
        max_requests: int = 8,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now_timestamp is None else float(now_timestamp)
        bounded_max = max(1, int(max_requests))
        state = self._load_state(now_timestamp=now)

        terminal_result_keys = state["terminal_result_keys"]
        terminal_investigations = state["terminal_investigations"]
        exhausted_investigations = state["tool_budget_exhausted_investigations"]

        pending: list[dict[str, Any]] = []
        expired_count = 0
        terminal_rejections = 0
        budget_exhausted_rejections = 0

        for request in state["requests"]:
            request_id = _nonempty_str(request.get("request_id"))
            investigation_id = _nonempty_str(request.get("investigation_id"))
            tool_name = _nonempty_str(request.get("tool_name"))
            if request_id is None or investigation_id is None or tool_name is None:
                continue

            result_key = _result_key(
                investigation_id=investigation_id,
                request_id=request_id,
                tool_name=tool_name,
            )
            if result_key in terminal_result_keys:
                continue

            if investigation_id in terminal_investigations:
                terminal_rejections += 1
                self._append_result_event(
                    payload=self._build_result_event(
                        request=request,
                        status="terminal",
                        result_payload={},
                        error_code="terminal_investigation",
                        error="investigation_is_terminal",
                        now_timestamp=now,
                    ),
                    state=state,
                    now_timestamp=now,
                )
                continue

            if investigation_id in exhausted_investigations:
                budget_exhausted_rejections += 1
                self._append_result_event(
                    payload=self._build_result_event(
                        request=request,
                        status="rejected",
                        result_payload={},
                        error_code="per_investigation_call_limit",
                        error="tool_budget_exhausted_for_investigation",
                        now_timestamp=now,
                    ),
                    state=state,
                    now_timestamp=now,
                )
                continue

            expires_at = request.get("expires_at")
            if isinstance(expires_at, (int, float)) and float(expires_at) <= now:
                expired_count += 1
                self._append_result_event(
                    payload=self._build_result_event(
                        request=request,
                        status="expired",
                        result_payload={},
                        error_code="request_expired",
                        error="request_expired_before_dispatch",
                        now_timestamp=now,
                    ),
                    state=state,
                    now_timestamp=now,
                )
                continue

            pending.append(dict(request))
            if len(pending) >= bounded_max:
                break

        self._persist_indexes(state=state, now_timestamp=now)
        return {
            "available": True,
            "pending_count": len(pending),
            "pending_requests": pending,
            "expired_count": expired_count,
            "terminal_rejections": terminal_rejections,
            "budget_exhausted_rejections": budget_exhausted_rejections,
        }

    def lookup_results_for_request(
        self,
        *,
        investigation_id: str,
        request_id: str,
        tool_name: str,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if now_timestamp is None else float(now_timestamp)
        normalized_investigation_id = _nonempty_str(investigation_id)
        normalized_request_id = _nonempty_str(request_id)
        normalized_tool_name = _nonempty_str(tool_name)
        if (
            normalized_investigation_id is None
            or normalized_request_id is None
            or normalized_tool_name is None
        ):
            return {
                "available": False,
                "result_count": 0,
                "results": [],
                "error": "invalid_lookup_selector",
            }

        state = self._load_state(now_timestamp=now)
        matched: list[dict[str, Any]] = []
        for result in state["results"]:
            if _nonempty_str(result.get("investigation_id")) != normalized_investigation_id:
                continue
            if _nonempty_str(result.get("request_id")) != normalized_request_id:
                continue
            if _nonempty_str(result.get("tool_name")) != normalized_tool_name:
                continue
            matched.append(dict(result))

        matched.sort(
            key=lambda item: float(item.get("finished_at", item.get("started_at", 0.0)) or 0.0),
            reverse=True,
        )
        latest = matched[0] if matched else None
        return {
            "available": True,
            "result_count": len(matched),
            "results": matched,
            "latest": latest,
        }

    def orchestration_publish_fixture_result(
        self,
        *,
        request_event: dict[str, Any],
        status: str = "ok",
        result_payload: dict[str, Any] | None = None,
        error_code: str | None = None,
        error: str | None = None,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        if not self._is_writer_role_allowed(allowed_roles={"shared", "orchestration"}):
            return self._result_writer_role_rejected_response(operation="orchestration_publish_fixture_result")
        now = time.time() if now_timestamp is None else float(now_timestamp)
        state = self._load_state(now_timestamp=now)
        if status not in VALID_RESULT_STATUSES:
            status = "rejected"
            error_code = error_code or "invalid_status"
            error = error or "invalid_result_status"
        payload = self._build_result_event(
            request=request_event,
            status=status,
            result_payload={} if result_payload is None else dict(result_payload),
            error_code=error_code,
            error=error,
            now_timestamp=now,
        )
        return self._append_result_event(payload=payload, state=state, now_timestamp=now)

    def orchestration_publish_result_event(
        self,
        *,
        result_event: dict[str, Any],
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        if not self._is_writer_role_allowed(allowed_roles={"shared", "orchestration"}):
            return self._result_writer_role_rejected_response(operation="orchestration_publish_result_event")
        now = time.time() if now_timestamp is None else float(now_timestamp)
        state = self._load_state(now_timestamp=now)
        return self._append_result_event(payload=dict(result_event), state=state, now_timestamp=now)

    def mark_terminal_investigation(
        self,
        *,
        investigation_id: str,
        state: str = "closed",
        reason: str | None = None,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        if not self._is_writer_role_allowed(allowed_roles={"shared", "orchestration"}):
            return {
                "updated": False,
                "reason": "writer_role_violation",
                "operation": "mark_terminal_investigation",
                "writer_role": self._writer_role,
            }
        now = time.time() if now_timestamp is None else float(now_timestamp)
        normalized_investigation_id = _nonempty_str(investigation_id)
        if normalized_investigation_id is None:
            return {
                "updated": False,
                "reason": "investigation_id_required",
            }

        normalized_state = _nonempty_str(state)
        if normalized_state not in TERMINAL_INVESTIGATION_STATES:
            normalized_state = "closed"

        loaded = self._load_state(now_timestamp=now)
        terminal_investigations = loaded["terminal_investigations"]
        terminal_investigations[normalized_investigation_id] = {
            "investigation_id": normalized_investigation_id,
            "investigation_state": normalized_state,
            "investigation_termination_reason": _nonempty_str(reason),
            "generated_at": now,
        }
        self._bound_terminal_investigations(terminal_investigations)
        self._persist_indexes(state=loaded, now_timestamp=now)
        return {
            "updated": True,
            "investigation_id": normalized_investigation_id,
            "investigation_state": normalized_state,
        }

    def mark_tool_budget_exhausted_investigation(
        self,
        *,
        investigation_id: str,
        reason: str | None = None,
        now_timestamp: float | None = None,
    ) -> dict[str, Any]:
        if not self._is_writer_role_allowed(allowed_roles={"shared", "orchestration"}):
            return {
                "updated": False,
                "reason": "writer_role_violation",
                "operation": "mark_tool_budget_exhausted_investigation",
                "writer_role": self._writer_role,
            }
        now = time.time() if now_timestamp is None else float(now_timestamp)
        normalized_investigation_id = _nonempty_str(investigation_id)
        if normalized_investigation_id is None:
            return {
                "updated": False,
                "reason": "investigation_id_required",
            }

        loaded = self._load_state(now_timestamp=now)
        exhausted = loaded["tool_budget_exhausted_investigations"]
        exhausted[normalized_investigation_id] = {
            "investigation_id": normalized_investigation_id,
            "tool_budget_state": "exhausted",
            "tool_budget_exhausted_reason": _nonempty_str(reason),
            "generated_at": now,
        }
        self._bound_tool_budget_exhausted_investigations(exhausted)
        self._persist_indexes(state=loaded, now_timestamp=now)
        return {
            "updated": True,
            "investigation_id": normalized_investigation_id,
            "tool_budget_state": "exhausted",
        }

    def run_fixture_orchestration_cycle(
        self,
        *,
        now_timestamp: float | None = None,
        max_requests: int = 8,
    ) -> dict[str, Any]:
        if not self._is_writer_role_allowed(allowed_roles={"shared", "orchestration"}):
            return {
                "pending_count": 0,
                "published_count": 0,
                "duplicate_count": 0,
                "expired_count": 0,
                "terminal_rejections": 0,
                "budget_exhausted_rejections": 0,
                "writer_role_violation": True,
                "writer_role": self._writer_role,
            }
        now = time.time() if now_timestamp is None else float(now_timestamp)
        pending = self.orchestration_poll_pending_requests(
            max_requests=max_requests,
            now_timestamp=now,
        )
        requests = pending.get("pending_requests")
        if not isinstance(requests, list):
            requests = []

        published = 0
        duplicates = 0
        for request in requests:
            response = self.orchestration_publish_fixture_result(
                request_event=request,
                status="ok",
                result_payload={
                    "fixture": True,
                    "consumed_by": "orchestration_ipc_transport",
                    "tool_name": request.get("tool_name"),
                    "investigation_id": request.get("investigation_id"),
                },
                now_timestamp=now,
            )
            if bool(response.get("published", False)):
                published += 1
            if bool(response.get("duplicate", False)):
                duplicates += 1

        return {
            "pending_count": int(pending.get("pending_count", 0)),
            "published_count": published,
            "duplicate_count": duplicates,
            "expired_count": int(pending.get("expired_count", 0)),
            "terminal_rejections": int(pending.get("terminal_rejections", 0)),
            "budget_exhausted_rejections": int(pending.get("budget_exhausted_rejections", 0)),
        }

    def validate_pending_request_for_dispatch(self, request_event: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        if not isinstance(request_event, dict):
            return {
                "valid": False,
                "errors": ["request_not_object"],
                "request": None,
            }

        normalized = dict(request_event)
        request_id = _nonempty_str(normalized.get("request_id"))
        investigation_id = _nonempty_str(normalized.get("investigation_id"))
        tool_name = _nonempty_str(normalized.get("tool_name"))
        iteration_index = normalized.get("iteration_index")
        created_at = normalized.get("created_at")
        expires_at = normalized.get("expires_at")
        arguments = normalized.get("arguments")

        if request_id is None:
            errors.append("request_id_required")
        if investigation_id is None:
            errors.append("investigation_id_required")
        if tool_name not in ALLOWLISTED_TOOLS:
            errors.append("tool_name_not_allowed")
        if not isinstance(iteration_index, int) or iteration_index < 1 or iteration_index > 2:
            errors.append("iteration_index_invalid")
        if not isinstance(created_at, (int, float)):
            errors.append("created_at_invalid")
        if not isinstance(expires_at, (int, float)):
            errors.append("expires_at_invalid")
        if not isinstance(arguments, dict):
            errors.append("arguments_invalid")

        dedupe_key = _nonempty_str(normalized.get("dedupe_key"))
        if request_id is not None and investigation_id is not None:
            expected_dedupe_key = _request_key(
                investigation_id=investigation_id,
                request_id=request_id,
            )
            if dedupe_key != expected_dedupe_key:
                errors.append("dedupe_key_mismatch")

        if errors:
            return {
                "valid": False,
                "errors": sorted(set(errors)),
                "request": None,
            }

        normalized["request_id"] = request_id
        normalized["investigation_id"] = investigation_id
        normalized["tool_name"] = tool_name
        normalized["iteration_index"] = int(iteration_index)
        normalized["created_at"] = float(created_at)
        normalized["expires_at"] = float(expires_at)
        normalized["arguments"] = dict(arguments)

        return {
            "valid": True,
            "errors": [],
            "request": normalized,
        }

    def _append_request_event(self, payload: dict[str, Any], *, now_timestamp: float) -> dict[str, Any]:
        state = self._load_state(now_timestamp=now_timestamp)
        dedupe_key = _nonempty_str(payload.get("dedupe_key"))
        investigation_id = _nonempty_str(payload.get("investigation_id"))
        if dedupe_key is None or investigation_id is None:
            return {
                "accepted": False,
                "duplicate": False,
                "rejected": True,
                "reason": "invalid_request_shape",
            }

        if investigation_id in state["terminal_investigations"]:
            return {
                "accepted": False,
                "duplicate": False,
                "rejected": True,
                "reason": "terminal_investigation",
            }

        if investigation_id in state["tool_budget_exhausted_investigations"]:
            return {
                "accepted": False,
                "duplicate": False,
                "rejected": True,
                "reason": "tool_budget_exhausted",
            }

        if dedupe_key in state["request_by_dedupe"]:
            return {
                "accepted": False,
                "duplicate": True,
                "rejected": False,
                "reason": "duplicate_request",
            }

        _fsync_append_jsonl(self._requests_path, payload)
        state["requests"].append(dict(payload))
        state["request_by_dedupe"][dedupe_key] = dict(payload)
        state["request_order"].append(dedupe_key)

        if len(state["request_order"]) > self._max_request_records:
            self._compact_requests_file(state=state)

        self._persist_indexes(state=state, now_timestamp=now_timestamp)
        return {
            "accepted": True,
            "duplicate": False,
            "rejected": False,
            "dedupe_key": dedupe_key,
        }

    def _append_result_event(
        self,
        *,
        payload: dict[str, Any],
        state: dict[str, Any],
        now_timestamp: float,
    ) -> dict[str, Any]:
        validation = self._validate_result_event(payload)
        if not bool(validation.get("valid", False)):
            return {
                "published": False,
                "duplicate": False,
                "rejected": True,
                "reason": "invalid_result_shape",
                "errors": validation.get("errors", []),
            }

        normalized = dict(validation["event"])
        result_key = _nonempty_str(normalized.get("idempotency_key"))
        if result_key is None:
            return {
                "published": False,
                "duplicate": False,
                "rejected": True,
                "reason": "invalid_result_shape",
            }

        request_id = _nonempty_str(normalized.get("request_id"))
        investigation_id = _nonempty_str(normalized.get("investigation_id"))
        tool_name = _nonempty_str(normalized.get("tool_name"))
        iteration_index = normalized.get("iteration_index")
        if (
            request_id is None
            or investigation_id is None
            or tool_name is None
            or not isinstance(iteration_index, int)
        ):
            return {
                "published": False,
                "duplicate": False,
                "rejected": True,
                "reason": "invalid_result_shape",
            }

        request_dedupe = _request_key(investigation_id=investigation_id, request_id=request_id)
        request_event = state["request_by_dedupe"].get(request_dedupe)
        if not isinstance(request_event, dict):
            normalized["status"] = "rejected"
            normalized["error_code"] = "correlation_mismatch"
            normalized["error"] = "unknown_request"
        else:
            if (
                request_event.get("tool_name") != tool_name
                or request_event.get("iteration_index") != iteration_index
            ):
                normalized["status"] = "rejected"
                normalized["error_code"] = "correlation_mismatch"
                normalized["error"] = "request_result_shape_mismatch"

        if investigation_id in state["terminal_investigations"] and normalized.get("status") != "terminal":
            normalized["status"] = "terminal"
            normalized["error_code"] = "terminal_investigation"
            normalized["error"] = "investigation_is_terminal"

        if result_key in state["terminal_result_keys"]:
            duplicate_event = dict(normalized)
            duplicate_event["status"] = "duplicate"
            duplicate_event["error_code"] = "duplicate_result"
            duplicate_event["error"] = "result_already_persisted"
            duplicate_event["finished_at"] = now_timestamp
            _fsync_append_jsonl(self._results_path, duplicate_event)
            state["results"].append(duplicate_event)
            if len(state["results"]) > self._max_result_records:
                self._compact_results_file(state=state)
            self._persist_indexes(state=state, now_timestamp=now_timestamp)
            return {
                "published": False,
                "duplicate": True,
                "rejected": False,
                "idempotency_key": result_key,
            }

        _fsync_append_jsonl(self._results_path, normalized)
        state["results"].append(dict(normalized))
        state["terminal_result_keys"].add(result_key)

        status = _nonempty_str(normalized.get("status")) or "rejected"
        if status in {"terminal", "expired", "rejected", "ok", "timeout", "unavailable", "incompatible", "stale"}:
            if investigation_id in state["terminal_investigations"]:
                pass

        if len(state["results"]) > self._max_result_records:
            self._compact_results_file(state=state)

        self._persist_indexes(state=state, now_timestamp=now_timestamp)
        return {
            "published": True,
            "duplicate": False,
            "rejected": False,
            "idempotency_key": result_key,
            "status": normalized.get("status"),
        }

    def _build_request_event_from_decision(
        self,
        *,
        decision: dict[str, Any],
        llm_correlation_id: str | None,
        now_timestamp: float,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        errors: list[str] = []
        if not isinstance(decision, dict):
            return None, ["decision_not_object"]

        action = _nonempty_str(decision.get("action"))
        if action not in REQUEST_ACTIONS:
            errors.append("decision_action_not_request_type")

        investigation_id = _nonempty_str(decision.get("investigation_id"))
        if investigation_id is None:
            errors.append("decision_investigation_id_required")

        tool_requests = decision.get("tool_requests")
        if not isinstance(tool_requests, list):
            errors.append("decision_tool_requests_invalid")
            tool_requests = []
        if len(tool_requests) != 1:
            errors.append("decision_tool_request_required")

        tool_request = tool_requests[0] if len(tool_requests) == 1 and isinstance(tool_requests[0], dict) else None
        if tool_request is None:
            errors.append("decision_tool_request_invalid")

        request_id = None
        tool_name = None
        arguments: dict[str, Any] = {}
        if isinstance(tool_request, dict):
            request_id = _nonempty_str(tool_request.get("request_id"))
            if request_id is None:
                errors.append("decision_tool_request_id_required")
            tool_name = _nonempty_str(tool_request.get("tool_name"))
            if tool_name not in ALLOWLISTED_TOOLS:
                errors.append("decision_tool_name_not_allowed")
            if action in REQUEST_ACTION_TO_ALLOWED_TOOLS and tool_name not in REQUEST_ACTION_TO_ALLOWED_TOOLS[action]:
                errors.append("decision_tool_name_mismatch_for_action")
            raw_arguments = tool_request.get("arguments")
            if not isinstance(raw_arguments, dict):
                errors.append("decision_tool_arguments_invalid")
            else:
                arguments = dict(raw_arguments)

        iteration_index = decision.get("iteration_index")
        if iteration_index is None:
            iteration_index = 1
        if not isinstance(iteration_index, int) or iteration_index < 1 or iteration_index > 2:
            errors.append("decision_iteration_index_invalid")

        if errors:
            return None, sorted(set(errors))

        assert investigation_id is not None
        assert request_id is not None
        assert tool_name is not None

        created_at = float(now_timestamp)
        expires_at = created_at + self._request_ttl_seconds
        dedupe_key = _request_key(investigation_id=investigation_id, request_id=request_id)
        correlation_id = _nonempty_str(decision.get("correlation_id"))

        request_event: dict[str, Any] = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "event_type": REQUEST_EVENT_TYPE,
            "request_id": request_id,
            "investigation_id": investigation_id,
            "iteration_index": int(iteration_index),
            "llm_correlation_id": _nonempty_str(llm_correlation_id),
            "decision_action": action,
            "tool_name": tool_name,
            "arguments": arguments,
            "snapshot_constraints": (
                dict(decision.get("snapshot_constraints"))
                if isinstance(decision.get("snapshot_constraints"), dict)
                else {}
            ),
            "correlation_id": correlation_id,
            "created_at": created_at,
            "expires_at": expires_at,
            "dedupe_key": dedupe_key,
        }
        return request_event, []

    def _build_result_event(
        self,
        *,
        request: dict[str, Any],
        status: str,
        result_payload: dict[str, Any],
        error_code: str | None,
        error: str | None,
        now_timestamp: float,
    ) -> dict[str, Any]:
        request_id = _nonempty_str(request.get("request_id"))
        investigation_id = _nonempty_str(request.get("investigation_id"))
        tool_name = _nonempty_str(request.get("tool_name"))
        iteration_index = request.get("iteration_index")
        if not isinstance(iteration_index, int):
            iteration_index = 1

        if request_id is None:
            request_id = "unknown-request"
        if investigation_id is None:
            investigation_id = "unknown-investigation"
        if tool_name is None:
            tool_name = "unknown-tool"

        idempotency_key = _result_key(
            investigation_id=investigation_id,
            request_id=request_id,
            tool_name=tool_name,
        )

        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "event_type": RESULT_EVENT_TYPE,
            "request_id": request_id,
            "investigation_id": investigation_id,
            "iteration_index": int(iteration_index),
            "tool_name": tool_name,
            "status": status if status in VALID_RESULT_STATUSES else "rejected",
            "result_payload": dict(result_payload),
            "error_code": _nonempty_str(error_code),
            "error": _nonempty_str(error),
            "started_at": float(now_timestamp),
            "finished_at": float(now_timestamp),
            "idempotency_key": idempotency_key,
        }

    def _validate_result_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        if not isinstance(payload, dict):
            return {
                "valid": False,
                "errors": ["result_not_object"],
            }

        event = dict(payload)
        if _nonempty_str(event.get("schema_version")) is None:
            event["schema_version"] = RESULT_SCHEMA_VERSION
        if _nonempty_str(event.get("event_type")) is None:
            event["event_type"] = RESULT_EVENT_TYPE

        request_id = _nonempty_str(event.get("request_id"))
        investigation_id = _nonempty_str(event.get("investigation_id"))
        tool_name = _nonempty_str(event.get("tool_name"))
        if request_id is None:
            errors.append("request_id_required")
        if investigation_id is None:
            errors.append("investigation_id_required")
        if tool_name is None:
            errors.append("tool_name_required")

        iteration_index = event.get("iteration_index")
        if not isinstance(iteration_index, int) or iteration_index < 1 or iteration_index > 2:
            errors.append("iteration_index_invalid")

        status = _nonempty_str(event.get("status"))
        if status not in VALID_RESULT_STATUSES:
            errors.append("status_invalid")
        else:
            event["status"] = status

        result_payload = event.get("result_payload")
        if result_payload is None:
            result_payload = {}
        if not isinstance(result_payload, dict):
            errors.append("result_payload_invalid")
            result_payload = {}
        event["result_payload"] = dict(result_payload)

        for key in ("started_at", "finished_at"):
            value = event.get(key)
            if not isinstance(value, (int, float)):
                event[key] = time.time()
            else:
                event[key] = float(value)

        if request_id is not None and investigation_id is not None and tool_name is not None:
            event["idempotency_key"] = _result_key(
                investigation_id=investigation_id,
                request_id=request_id,
                tool_name=tool_name,
            )
        elif _nonempty_str(event.get("idempotency_key")) is None:
            errors.append("idempotency_key_missing")

        event["error_code"] = _nonempty_str(event.get("error_code"))
        event["error"] = _nonempty_str(event.get("error"))

        return {
            "valid": len(errors) == 0,
            "errors": sorted(set(errors)),
            "event": event,
        }

    def _load_state(self, *, now_timestamp: float) -> dict[str, Any]:
        requests = self._load_request_events()
        results = self._load_result_events()
        request_index = self._load_index_json(
            primary_path=self._request_index_path,
            backup_path=self._request_index_backup,
        )
        result_index = self._load_index_json(
            primary_path=self._result_index_path,
            backup_path=self._result_index_backup,
        )

        request_by_dedupe: dict[str, dict[str, Any]] = {}
        request_order: list[str] = []
        for request in requests:
            dedupe = _nonempty_str(request.get("dedupe_key"))
            if dedupe is None:
                continue
            request_by_dedupe[dedupe] = dict(request)
            request_order.append(dedupe)

        terminal_result_keys: set[str] = set()
        for result in results:
            status = _nonempty_str(result.get("status"))
            result_key = _nonempty_str(result.get("idempotency_key"))
            if result_key is None:
                continue
            if status in VALID_RESULT_STATUSES:
                terminal_result_keys.add(result_key)

        delivered_result_keys: set[str] = set()
        if isinstance(result_index.get("delivered_result_keys"), list):
            for item in result_index["delivered_result_keys"]:
                key = _nonempty_str(item)
                if key is not None:
                    delivered_result_keys.add(key)

        terminal_investigations: dict[str, dict[str, Any]] = {}
        raw_terminal = result_index.get("terminal_investigations")
        if isinstance(raw_terminal, list):
            for item in raw_terminal:
                if not isinstance(item, dict):
                    continue
                investigation_id = _nonempty_str(item.get("investigation_id"))
                state = _nonempty_str(item.get("investigation_state"))
                if investigation_id is None or state not in TERMINAL_INVESTIGATION_STATES:
                    continue
                terminal_investigations[investigation_id] = {
                    "investigation_id": investigation_id,
                    "investigation_state": state,
                    "investigation_termination_reason": _nonempty_str(item.get("investigation_termination_reason")),
                    "generated_at": float(item.get("generated_at")) if isinstance(item.get("generated_at"), (int, float)) else float(now_timestamp),
                }

        tool_budget_exhausted_investigations: dict[str, dict[str, Any]] = {}
        raw_exhausted = result_index.get("tool_budget_exhausted_investigations")
        if isinstance(raw_exhausted, list):
            for item in raw_exhausted:
                if not isinstance(item, dict):
                    continue
                investigation_id = _nonempty_str(item.get("investigation_id"))
                state = _nonempty_str(item.get("tool_budget_state"))
                if investigation_id is None or state not in TOOL_BUDGET_EXHAUSTED_STATES:
                    continue
                tool_budget_exhausted_investigations[investigation_id] = {
                    "investigation_id": investigation_id,
                    "tool_budget_state": state,
                    "tool_budget_exhausted_reason": _nonempty_str(item.get("tool_budget_exhausted_reason")),
                    "generated_at": float(item.get("generated_at")) if isinstance(item.get("generated_at"), (int, float)) else float(now_timestamp),
                }

        delivered_observation_assessment_keys: set[str] = set()
        if isinstance(result_index.get("delivered_observation_assessment_keys"), list):
            for item in result_index["delivered_observation_assessment_keys"]:
                key = _nonempty_str(item)
                if key is not None:
                    delivered_observation_assessment_keys.add(key)

        self._bound_terminal_investigations(terminal_investigations)
        self._bound_tool_budget_exhausted_investigations(tool_budget_exhausted_investigations)

        observation_follow_up_handoffs: dict[str, dict[str, Any]] = {}
        raw_handoffs = result_index.get("observation_follow_up_handoffs")
        if isinstance(raw_handoffs, list):
            for item in raw_handoffs:
                if not isinstance(item, dict):
                    continue
                result_key = _nonempty_str(item.get("idempotency_key"))
                investigation_id = _nonempty_str(item.get("investigation_id"))
                request_id = _nonempty_str(item.get("request_id"))
                tool_name = _nonempty_str(item.get("tool_name"))
                assessment = item.get("assessment")
                if (
                    result_key is None
                    or investigation_id is None
                    or request_id is None
                    or tool_name is None
                    or not isinstance(assessment, dict)
                ):
                    continue
                generated_at = item.get("generated_at")
                observation_follow_up_handoffs[result_key] = {
                    "idempotency_key": result_key,
                    "investigation_id": investigation_id,
                    "request_id": request_id,
                    "tool_name": tool_name,
                    "track_id": _nonempty_str(item.get("track_id")),
                    "assessment": dict(assessment),
                    "generated_at": (
                        float(generated_at)
                        if isinstance(generated_at, (int, float))
                        else float(now_timestamp)
                    ),
                }

        self._bound_observation_follow_up_handoffs(observation_follow_up_handoffs)

        return {
            "requests": requests,
            "results": results,
            "request_by_dedupe": request_by_dedupe,
            "request_order": request_order,
            "terminal_result_keys": terminal_result_keys,
            "delivered_result_keys": delivered_result_keys,
            "terminal_investigations": terminal_investigations,
            "tool_budget_exhausted_investigations": tool_budget_exhausted_investigations,
            "observation_follow_up_handoffs": observation_follow_up_handoffs,
            "delivered_observation_assessment_keys": delivered_observation_assessment_keys,
            "request_index_raw": request_index,
            "result_index_raw": result_index,
        }

    def _load_request_events(self) -> list[dict[str, Any]]:
        if not self._requests_path.is_file():
            return []

        parsed: list[dict[str, Any]] = []
        try:
            lines = self._requests_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        for line in lines:
            payload = _event_from_jsonl_line(line)
            if payload is None:
                continue
            if payload.get("event_type") != REQUEST_EVENT_TYPE:
                continue
            if _nonempty_str(payload.get("request_id")) is None:
                continue
            if _nonempty_str(payload.get("investigation_id")) is None:
                continue
            if _nonempty_str(payload.get("tool_name")) is None:
                continue
            if not isinstance(payload.get("iteration_index"), int):
                continue
            dedupe = _nonempty_str(payload.get("dedupe_key"))
            if dedupe is None:
                continue
            parsed.append(dict(payload))

        if len(parsed) > self._max_request_records:
            parsed = parsed[-self._max_request_records :]
        return parsed

    def _load_result_events(self) -> list[dict[str, Any]]:
        if not self._results_path.is_file():
            return []

        parsed: list[dict[str, Any]] = []
        try:
            lines = self._results_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        for line in lines:
            payload = _event_from_jsonl_line(line)
            if payload is None:
                continue
            validation = self._validate_result_event(payload)
            if not bool(validation.get("valid", False)):
                continue
            parsed.append(dict(validation["event"]))

        if len(parsed) > self._max_result_records:
            parsed = parsed[-self._max_result_records :]
        return parsed

    def _compact_requests_file(self, *, state: dict[str, Any]) -> None:
        retained = state["requests"][-self._max_request_records :]
        self._rewrite_jsonl(self._requests_path, retained)
        state["requests"] = list(retained)
        state["request_by_dedupe"] = {
            _nonempty_str(item.get("dedupe_key")) or "": dict(item)
            for item in retained
            if _nonempty_str(item.get("dedupe_key")) is not None
        }
        state["request_order"] = [
            _nonempty_str(item.get("dedupe_key"))
            for item in retained
            if _nonempty_str(item.get("dedupe_key")) is not None
        ]

    def _compact_results_file(self, *, state: dict[str, Any]) -> None:
        retained = state["results"][-self._max_result_records :]
        self._rewrite_jsonl(self._results_path, retained)
        state["results"] = list(retained)
        state["terminal_result_keys"] = {
            _nonempty_str(item.get("idempotency_key"))
            for item in retained
            if _nonempty_str(item.get("idempotency_key")) is not None
        }
        state["delivered_result_keys"] = {
            key
            for key in state["delivered_result_keys"]
            if key in state["terminal_result_keys"]
        }
        state["delivered_observation_assessment_keys"] = {
            key
            for key in state.get("delivered_observation_assessment_keys", set())
            if isinstance(key, str) and key.strip()
        }

    def _rewrite_jsonl(self, path: Path, events: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
        try:
            with temp.open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass

    def _bound_terminal_investigations(self, terminal_investigations: dict[str, dict[str, Any]]) -> None:
        if len(terminal_investigations) <= self._max_terminal_investigations:
            return
        items = list(terminal_investigations.values())
        items.sort(key=lambda item: float(item.get("generated_at", 0.0)))
        retained = items[-self._max_terminal_investigations :]
        terminal_investigations.clear()
        for item in retained:
            investigation_id = _nonempty_str(item.get("investigation_id"))
            if investigation_id is None:
                continue
            terminal_investigations[investigation_id] = dict(item)

    def _bound_tool_budget_exhausted_investigations(
        self,
        tool_budget_exhausted_investigations: dict[str, dict[str, Any]],
    ) -> None:
        if len(tool_budget_exhausted_investigations) <= self._max_terminal_investigations:
            return
        items = list(tool_budget_exhausted_investigations.values())
        items.sort(key=lambda item: float(item.get("generated_at", 0.0)))
        retained = items[-self._max_terminal_investigations :]
        tool_budget_exhausted_investigations.clear()
        for item in retained:
            investigation_id = _nonempty_str(item.get("investigation_id"))
            if investigation_id is None:
                continue
            tool_budget_exhausted_investigations[investigation_id] = dict(item)

    def _bound_observation_follow_up_handoffs(
        self,
        handoffs: dict[str, dict[str, Any]],
    ) -> None:
        if len(handoffs) <= OBSERVATION_FOLLOW_UP_HANDOFF_MAX_RECORDS:
            return
        items = list(handoffs.values())
        items.sort(key=lambda item: float(item.get("generated_at", 0.0)))
        retained = items[-OBSERVATION_FOLLOW_UP_HANDOFF_MAX_RECORDS :]
        handoffs.clear()
        for item in retained:
            result_key = _nonempty_str(item.get("idempotency_key"))
            if result_key is None:
                continue
            handoffs[result_key] = dict(item)

    def _attach_observation_follow_up_handoff(
        self,
        *,
        state: dict[str, Any],
        result: dict[str, Any],
        now_timestamp: float,
    ) -> dict[str, Any]:
        enriched = dict(result)
        result_key = _nonempty_str(enriched.get("idempotency_key"))
        if result_key is None:
            return enriched

        handoffs = state.get("observation_follow_up_handoffs")
        if not isinstance(handoffs, dict):
            return enriched

        handoff = handoffs.get(result_key)
        if not isinstance(handoff, dict):
            return enriched

        if _nonempty_str(handoff.get("investigation_id")) != _nonempty_str(enriched.get("investigation_id")):
            handoffs.pop(result_key, None)
            return enriched
        if _nonempty_str(handoff.get("request_id")) != _nonempty_str(enriched.get("request_id")):
            handoffs.pop(result_key, None)
            return enriched
        if _nonempty_str(handoff.get("tool_name")) != _nonempty_str(enriched.get("tool_name")):
            handoffs.pop(result_key, None)
            return enriched

        generated_at = handoff.get("generated_at")
        if not isinstance(generated_at, (int, float)):
            handoffs.pop(result_key, None)
            return enriched
        if (float(now_timestamp) - float(generated_at)) > OBSERVATION_FOLLOW_UP_HANDOFF_MAX_AGE_SECONDS:
            handoffs.pop(result_key, None)
            return enriched

        assessment = handoff.get("assessment")
        if not isinstance(assessment, dict):
            handoffs.pop(result_key, None)
            return enriched

        payload = enriched.get("result_payload")
        if not isinstance(payload, dict):
            payload = {}
        merged_payload = dict(payload)
        merged_payload["observation_llm_follow_up_assessment"] = dict(assessment)
        merged_payload["observation_llm_follow_up_assessment_provenance"] = {
            "source": "observation_follow_up_llm",
            "track_id": _nonempty_str(handoff.get("track_id")),
            "idempotency_key": result_key,
            "identity_verification_authorized": False,
            "identity_mutation_authorized": False,
        }
        enriched["result_payload"] = merged_payload
        return enriched

    def _observation_handoffs_ready_for_delivery(
        self,
        *,
        state: dict[str, Any],
        now_timestamp: float,
    ) -> list[dict[str, Any]]:
        handoffs = state.get("observation_follow_up_handoffs")
        if not isinstance(handoffs, dict):
            return []
        delivered_assessment_keys = state.get("delivered_observation_assessment_keys")
        if not isinstance(delivered_assessment_keys, set):
            delivered_assessment_keys = set()

        ordered = sorted(
            [item for item in handoffs.values() if isinstance(item, dict)],
            key=lambda item: float(item.get("generated_at", 0.0)),
        )
        ready: list[dict[str, Any]] = []
        for handoff in ordered:
            result_key = _nonempty_str(handoff.get("idempotency_key"))
            if result_key is None:
                continue
            delivery_key = _observation_assessment_delivery_key(result_key)
            if delivery_key in delivered_assessment_keys:
                continue

            generated_at = handoff.get("generated_at")
            if not isinstance(generated_at, (int, float)):
                continue
            if (float(now_timestamp) - float(generated_at)) > OBSERVATION_FOLLOW_UP_HANDOFF_MAX_AGE_SECONDS:
                delivered_assessment_keys.add(delivery_key)
                continue
            ready.append(dict(handoff))
        return ready

    def _build_observation_assessment_delivery_event(
        self,
        *,
        state: dict[str, Any],
        handoff: dict[str, Any],
        now_timestamp: float,
    ) -> dict[str, Any] | None:
        result_key = _nonempty_str(handoff.get("idempotency_key"))
        investigation_id = _nonempty_str(handoff.get("investigation_id"))
        request_id = _nonempty_str(handoff.get("request_id"))
        tool_name = _nonempty_str(handoff.get("tool_name"))
        assessment = handoff.get("assessment")
        if (
            result_key is None
            or investigation_id is None
            or request_id is None
            or tool_name is None
            or not isinstance(assessment, dict)
        ):
            return None

        matched_result: dict[str, Any] | None = None
        for result in state.get("results", []):
            if not isinstance(result, dict):
                continue
            if _nonempty_str(result.get("idempotency_key")) != result_key:
                continue
            if _nonempty_str(result.get("investigation_id")) != investigation_id:
                continue
            if _nonempty_str(result.get("request_id")) != request_id:
                continue
            if _nonempty_str(result.get("tool_name")) != tool_name:
                continue
            matched_result = result
            break
        if not isinstance(matched_result, dict):
            return None

        generated_at = handoff.get("generated_at")
        finished_at = float(generated_at) if isinstance(generated_at, (int, float)) else float(now_timestamp)
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "event_type": RESULT_EVENT_TYPE,
            "delivery_kind": "observation_follow_up_assessment",
            "request_id": request_id,
            "investigation_id": investigation_id,
            "iteration_index": matched_result.get("iteration_index", 1),
            "tool_name": tool_name,
            "status": "ok",
            "result_payload": {
                "observation_llm_follow_up_assessment": dict(assessment),
                "observation_llm_follow_up_assessment_provenance": {
                    "source": "observation_follow_up_llm",
                    "track_id": _nonempty_str(handoff.get("track_id")),
                    "source_result_idempotency_key": result_key,
                    "identity_verification_authorized": False,
                    "identity_mutation_authorized": False,
                },
            },
            "error_code": None,
            "error": None,
            "started_at": finished_at,
            "finished_at": finished_at,
            "idempotency_key": _observation_assessment_delivery_key(result_key),
        }

    def _persist_indexes(self, *, state: dict[str, Any], now_timestamp: float) -> None:
        request_index_payload = {
            "schema_version": "1.0",
            "generated_at": float(now_timestamp),
            "request_count": len(state["request_by_dedupe"]),
            "request_keys": list(state["request_by_dedupe"].keys()),
        }
        result_index_payload = {
            "schema_version": "1.0",
            "generated_at": float(now_timestamp),
            "result_count": len(state["terminal_result_keys"]),
            "terminal_result_keys": list(state["terminal_result_keys"]),
            "delivered_result_keys": list(state["delivered_result_keys"]),
            "delivered_observation_assessment_keys": list(state["delivered_observation_assessment_keys"]),
            "terminal_investigations": list(state["terminal_investigations"].values()),
            "tool_budget_exhausted_investigations": list(state["tool_budget_exhausted_investigations"].values()),
            "observation_follow_up_handoffs": list(state["observation_follow_up_handoffs"].values()),
        }

        _atomic_write_json(self._request_index_path, self._request_index_backup, request_index_payload)
        _atomic_write_json(self._result_index_path, self._result_index_backup, result_index_payload)

    def _load_index_json(self, *, primary_path: Path, backup_path: Path) -> dict[str, Any]:
        primary = self._parse_index_json(primary_path)
        if bool(primary.get("ok", False)):
            return primary["payload"]
        backup = self._parse_index_json(backup_path)
        if bool(backup.get("ok", False)):
            return backup["payload"]
        return {}

    def _parse_index_json(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"ok": False}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"ok": False}
        if not isinstance(payload, dict):
            return {"ok": False}
        return {"ok": True, "payload": payload}
