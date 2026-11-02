from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fcntl

ROLE_TRACKING_VLM = "tracking_vlm"
ROLE_OBSERVATION_TEXT = "observation_text"
ROLE_REASONING_TEXT = "reasoning_text"
ROLE_OBSERVATION_VISUAL = "observation_visual"

_ROLE_TO_ENV_LIMIT = {
    ROLE_TRACKING_VLM: "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT",
    ROLE_OBSERVATION_TEXT: "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT",
    ROLE_REASONING_TEXT: "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT",
    ROLE_OBSERVATION_VISUAL: "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT",
}

_DEFAULT_STATE_FILE = "provider_call_budget_state.json"
_DEFAULT_LOCK_FILE = "provider_call_budget_state.lock"
_DEFAULT_LOG_FILE = "provider_call_budget_events.jsonl"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CAPTURES_DIR = _PROJECT_ROOT / "captures"

_THREAD_LOCK = threading.Lock()
_BUDGET_ERROR_INVALID_LIMITS = "invalid_limits_configuration"
_BUDGET_ERROR_STATE_UNREADABLE = "state_unreadable"
_BUDGET_ERROR_STATE_CORRUPTED = "state_corrupted"
_BUDGET_ERROR_RESERVATION_FAILED = "reservation_failed"


class ProviderCallBudgetExceeded(RuntimeError):
    def __init__(self, role: str, reason: str) -> None:
        super().__init__(f"provider_call_budget_exhausted:{role}:{reason}")
        self.role = role
        self.reason = reason


@dataclass(slots=True)
class ProviderCallReservation:
    enabled: bool
    allowed: bool
    role: str
    attempt_id: str
    reserved_at: float
    reason: str | None = None


def _is_enabled() -> bool:
    value = os.getenv("PROVIDER_CALL_BUDGET_ENABLED", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_non_negative_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _require_non_negative_int(value: str | None, *, env_name: str) -> int:
    parsed = _parse_non_negative_int(value)
    if parsed is None:
        raise ValueError(f"{env_name} must be a non-negative integer")
    return parsed


def _resolve_limits(*, strict: bool = False) -> dict[str, Any]:
    per_role: dict[str, int] = {}
    for role, env_name in _ROLE_TO_ENV_LIMIT.items():
        raw_value = os.getenv(env_name)
        if strict:
            per_role[role] = _require_non_negative_int(raw_value, env_name=env_name)
        else:
            parsed = _parse_non_negative_int(raw_value)
            per_role[role] = 0 if parsed is None else parsed

    raw_total_limit = os.getenv("PROVIDER_CALL_BUDGET_TOTAL_LIMIT")
    if strict:
        total_limit = _require_non_negative_int(
            raw_total_limit,
            env_name="PROVIDER_CALL_BUDGET_TOTAL_LIMIT",
        )
    else:
        parsed_total_limit = _parse_non_negative_int(raw_total_limit)
        total_limit = 0 if parsed_total_limit is None else parsed_total_limit

    return {
        "total_limit": total_limit,
        "per_role_limits": per_role,
    }


def _resolve_paths() -> tuple[Path, Path, Path]:
    state_path = os.getenv("PROVIDER_CALL_BUDGET_STATE_PATH")
    lock_path = os.getenv("PROVIDER_CALL_BUDGET_LOCK_PATH")
    log_path = os.getenv("PROVIDER_CALL_BUDGET_LOG_PATH")

    resolved_state = Path(state_path) if state_path else (_CAPTURES_DIR / _DEFAULT_STATE_FILE)
    resolved_lock = Path(lock_path) if lock_path else (_CAPTURES_DIR / _DEFAULT_LOCK_FILE)
    resolved_log = Path(log_path) if log_path else (_CAPTURES_DIR / _DEFAULT_LOG_FILE)
    return resolved_state, resolved_lock, resolved_log


def _load_state(path: Path, limits: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "1.0",
            "updated_at": time.time(),
            "limits": limits,
            "totals": {
                "attempted": 0,
                "per_role_attempted": {role: 0 for role in _ROLE_TO_ENV_LIMIT},
            },
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(_BUDGET_ERROR_STATE_UNREADABLE) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(_BUDGET_ERROR_STATE_CORRUPTED) from error

    if not isinstance(payload, dict):
        raise RuntimeError(_BUDGET_ERROR_STATE_CORRUPTED)

    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else None
    if totals is None:
        raise RuntimeError(_BUDGET_ERROR_STATE_CORRUPTED)

    per_role_attempted = (
        totals.get("per_role_attempted")
        if isinstance(totals.get("per_role_attempted"), dict)
        else None
    )
    if per_role_attempted is None:
        raise RuntimeError(_BUDGET_ERROR_STATE_CORRUPTED)

    normalized_per_role: dict[str, int] = {}
    for role in _ROLE_TO_ENV_LIMIT:
        value = per_role_attempted.get(role)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(_BUDGET_ERROR_STATE_CORRUPTED)
        normalized_per_role[role] = value

    attempted = totals.get("attempted")
    if not isinstance(attempted, int) or attempted < 0:
        raise RuntimeError(_BUDGET_ERROR_STATE_CORRUPTED)

    per_role_sum = sum(normalized_per_role.values())
    if attempted < per_role_sum:
        raise RuntimeError(_BUDGET_ERROR_STATE_CORRUPTED)

    return {
        "schema_version": "1.0",
        "updated_at": payload.get("updated_at", time.time()),
        "limits": limits,
        "totals": {
            "attempted": attempted,
            "per_role_attempted": normalized_per_role,
        },
    }


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    temp_path.write_text(json.dumps(payload), encoding="utf-8")
    temp_path.replace(path)


def _append_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(event)}\n")


def _extract_usage_fields(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    output: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        if isinstance(value, int):
            output[field] = value
    if output:
        return output
    return None


def _log_event(
    *,
    event_type: str,
    role: str,
    attempt_id: str,
    outcome: str,
    reserved_at: float,
    latency_ms: float | None,
    usage: dict[str, Any] | None,
    reason: str | None = None,
    error_code: str | None = None,
) -> None:
    _, _, log_path = _resolve_paths()
    event: dict[str, Any] = {
        "event_type": event_type,
        "timestamp": time.time(),
        "attempt_id": attempt_id,
        "role": role,
        "outcome": outcome,
        "reserved_at": reserved_at,
        "latency_ms": latency_ms,
        "usage": _extract_usage_fields(usage),
        "reason": reason,
        "error_code": error_code,
        "pid": os.getpid(),
    }
    _append_log(log_path, event)


def reserve_provider_call_attempt(role: str) -> ProviderCallReservation:
    if not _is_enabled():
        return ProviderCallReservation(
            enabled=False,
            allowed=True,
            role=role,
            attempt_id=f"attempt-{uuid.uuid4().hex}",
            reserved_at=time.time(),
        )

    if role not in _ROLE_TO_ENV_LIMIT:
        return ProviderCallReservation(
            enabled=True,
            allowed=False,
            role=role,
            attempt_id=f"attempt-{uuid.uuid4().hex}",
            reserved_at=time.time(),
            reason="unknown_role",
        )

    try:
        limits = _resolve_limits(strict=True)
    except Exception:
        reservation = ProviderCallReservation(
            enabled=True,
            allowed=False,
            role=role,
            attempt_id=f"attempt-{uuid.uuid4().hex}",
            reserved_at=time.time(),
            reason=_BUDGET_ERROR_INVALID_LIMITS,
        )
        try:
            _log_event(
                event_type="provider_call_blocked",
                role=reservation.role,
                attempt_id=reservation.attempt_id,
                outcome="blocked",
                reserved_at=reservation.reserved_at,
                latency_ms=0.0,
                usage=None,
                reason=reservation.reason,
                error_code="budget_exhausted",
            )
        except Exception:
            pass
        return reservation

    state_path, lock_path, _ = _resolve_paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    reserved_at = time.time()
    attempt_id = f"attempt-{uuid.uuid4().hex}"

    try:
        with _THREAD_LOCK:
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    state = _load_state(state_path, limits)
                    totals = state["totals"]
                    per_role = totals["per_role_attempted"]

                    total_limit = limits["total_limit"]
                    role_limit = limits["per_role_limits"][role]

                    if totals["attempted"] >= total_limit:
                        reservation = ProviderCallReservation(
                            enabled=True,
                            allowed=False,
                            role=role,
                            attempt_id=attempt_id,
                            reserved_at=reserved_at,
                            reason="global_limit_exhausted",
                        )
                    elif per_role.get(role, 0) >= role_limit:
                        reservation = ProviderCallReservation(
                            enabled=True,
                            allowed=False,
                            role=role,
                            attempt_id=attempt_id,
                            reserved_at=reserved_at,
                            reason="role_limit_exhausted",
                        )
                    else:
                        totals["attempted"] += 1
                        per_role[role] = per_role.get(role, 0) + 1
                        state["updated_at"] = time.time()
                        _write_state(state_path, state)
                        reservation = ProviderCallReservation(
                            enabled=True,
                            allowed=True,
                            role=role,
                            attempt_id=attempt_id,
                            reserved_at=reserved_at,
                        )
                finally:
                    try:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
    except RuntimeError as error:
        reservation = ProviderCallReservation(
            enabled=True,
            allowed=False,
            role=role,
            attempt_id=attempt_id,
            reserved_at=reserved_at,
            reason=str(error) or _BUDGET_ERROR_RESERVATION_FAILED,
        )
    except Exception:
        reservation = ProviderCallReservation(
            enabled=True,
            allowed=False,
            role=role,
            attempt_id=attempt_id,
            reserved_at=reserved_at,
            reason=_BUDGET_ERROR_RESERVATION_FAILED,
        )

    if not reservation.allowed:
        try:
            _log_event(
                event_type="provider_call_blocked",
                role=reservation.role,
                attempt_id=reservation.attempt_id,
                outcome="blocked",
                reserved_at=reservation.reserved_at,
                latency_ms=0.0,
                usage=None,
                reason=reservation.reason,
                error_code="budget_exhausted",
            )
        except Exception:
            pass

    return reservation


def finalize_provider_call_attempt(
    reservation: ProviderCallReservation,
    *,
    outcome: str,
    latency_ms: float | None,
    usage: dict[str, Any] | None = None,
    error_code: str | None = None,
) -> None:
    if not reservation.enabled or not reservation.allowed:
        return

    _log_event(
        event_type="provider_call_attempt",
        role=reservation.role,
        attempt_id=reservation.attempt_id,
        outcome=outcome,
        reserved_at=reservation.reserved_at,
        latency_ms=latency_ms,
        usage=usage,
        error_code=error_code,
    )


def reserve_or_raise(role: str) -> ProviderCallReservation:
    reservation = reserve_provider_call_attempt(role)
    if reservation.allowed:
        return reservation
    raise ProviderCallBudgetExceeded(role=role, reason=reservation.reason or "budget_exhausted")


def read_budget_totals() -> dict[str, Any]:
    limits = _resolve_limits(strict=_is_enabled())
    state_path, _, _ = _resolve_paths()
    state = _load_state(state_path, limits)
    return {
        "enabled": _is_enabled(),
        "limits": state.get("limits", {}),
        "totals": state.get("totals", {}),
    }
