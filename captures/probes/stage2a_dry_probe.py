from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
SERVER = REPO / "server.py"
ENVELOPES = REPO / "captures" / "tracking_evidence" / "envelopes"
REFS = REPO / "captures" / "tracking_evidence" / "refs"
PROBE_RUNS = REPO / "captures" / "probes" / "runs"
DEFAULT_OBSERVATION_VISUAL_MODEL = "qwen-3.8-27b"
DEFAULT_PROVIDER_ENDPOINT = "https://128.171.10.83:9443/v1"
DEFAULT_STAGE2A_VISUAL_VERIFY_TIMEOUT_SECONDS = "30"


class ProbeAbort(RuntimeError):
    pass


def build_probe_paths() -> dict[str, Path]:
    probe_id = f"stage2a_{int(time.time())}_{uuid4().hex[:8]}"
    probe_dir = PROBE_RUNS / probe_id
    probe_dir.mkdir(parents=True, exist_ok=True)
    return {
        "probe_dir": probe_dir,
        "state_path": probe_dir / "provider_call_budget_state.json",
        "lock_path": probe_dir / "provider_call_budget_state.lock",
        "log_path": probe_dir / "provider_call_budget_events.jsonl",
        "report_path": probe_dir / "stage2a_probe_report.json",
    }


def build_server_env(mode: str, paths: dict[str, Path]) -> tuple[dict[str, str], str, str]:
    configured_visual_model = os.getenv("OBSERVATION_VISUAL_AGENT_MODEL", "").strip() or DEFAULT_OBSERVATION_VISUAL_MODEL
    configured_endpoint = os.getenv("OPENAI_BASE_URL", "").strip() or DEFAULT_PROVIDER_ENDPOINT
    configured_verify_timeout_seconds = (
        os.getenv("STAGE2A_VISUAL_VERIFY_TIMEOUT_SECONDS", "").strip()
        or DEFAULT_STAGE2A_VISUAL_VERIFY_TIMEOUT_SECONDS
    )
    server_env = dict(os.environ)
    server_env.update(
        {
            "PROVIDER_CALL_BUDGET_ENABLED": "1",
            "PROVIDER_CALL_BUDGET_TOTAL_LIMIT": "1",
            "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT": "0",
            "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT": "0",
            "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT": "0",
            "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT": "1",
            "PROVIDER_CALL_BUDGET_STATE_PATH": str(paths["state_path"]),
            "PROVIDER_CALL_BUDGET_LOCK_PATH": str(paths["lock_path"]),
            "PROVIDER_CALL_BUDGET_LOG_PATH": str(paths["log_path"]),
            "OBSERVATION_VISUAL_VERIFICATION_TIMEOUT_SECONDS": configured_verify_timeout_seconds,
            "OBSERVATION_VISUAL_AGENT_MODEL": configured_visual_model,
            "OPENAI_BASE_URL": configured_endpoint,
        }
    )
    if mode == "dry":
        server_env["OPENROUTER_API_KEY"] = ""
    elif mode == "real":
        inherited_key = os.getenv("OPENROUTER_API_KEY", "")
        if not inherited_key:
            raise ProbeAbort("OPENROUTER_API_KEY missing for MODE=real")
        server_env["OPENROUTER_API_KEY"] = inherited_key
    else:
        raise ProbeAbort("MODE must be 'dry' or 'real'")
    return server_env, configured_visual_model, configured_endpoint


def run_env_inheritance_preflight(
    server_env: dict[str, str],
    *,
    expected_visual_model: str,
    expected_endpoint: str,
    expected_api_key_state: str,
) -> dict[str, Any]:
    keys = [
        "PROVIDER_CALL_BUDGET_ENABLED",
        "PROVIDER_CALL_BUDGET_TOTAL_LIMIT",
        "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT",
        "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT",
        "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT",
        "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT",
        "PROVIDER_CALL_BUDGET_STATE_PATH",
        "PROVIDER_CALL_BUDGET_LOCK_PATH",
        "PROVIDER_CALL_BUDGET_LOG_PATH",
        "OBSERVATION_VISUAL_VERIFICATION_TIMEOUT_SECONDS",
        "OBSERVATION_VISUAL_AGENT_MODEL",
        "OPENAI_BASE_URL",
    ]
    expected = {key: server_env.get(key, "") for key in keys}
    script = """
import json
import os

keys = [
    "PROVIDER_CALL_BUDGET_ENABLED",
    "PROVIDER_CALL_BUDGET_TOTAL_LIMIT",
    "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT",
    "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT",
    "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT",
    "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT",
    "PROVIDER_CALL_BUDGET_STATE_PATH",
    "PROVIDER_CALL_BUDGET_LOCK_PATH",
    "PROVIDER_CALL_BUDGET_LOG_PATH",
    "OBSERVATION_VISUAL_VERIFICATION_TIMEOUT_SECONDS",
    "OBSERVATION_VISUAL_AGENT_MODEL",
    "OPENAI_BASE_URL",
]
payload = {key: os.getenv(key, "") for key in keys}
api_key = os.getenv("OPENROUTER_API_KEY", "")
payload["OPENROUTER_API_KEY_STATE"] = "empty" if api_key == "" else "set"
print(json.dumps(payload))
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=server_env,
    )

    output: dict[str, Any] = {
        "returncode": child.returncode,
        "passed": False,
        "api_key_state": "unknown",
        "expected_visual_model": expected_visual_model,
        "expected_provider_endpoint": expected_endpoint,
        "expected_api_key_state": expected_api_key_state,
        "mismatches": [],
    }
    if child.returncode != 0:
        output["stderr_preview"] = (child.stderr or "")[:240]
        return output

    try:
        inherited = json.loads(child.stdout)
    except Exception as exc:
        output["parse_error"] = f"{type(exc).__name__}:{exc}"
        output["stdout_preview"] = (child.stdout or "")[:240]
        return output

    mismatches: list[dict[str, str]] = []
    for key, expected_value in expected.items():
        got_value = inherited.get(key)
        if got_value != expected_value:
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "got": "" if got_value is None else str(got_value),
                }
            )

    api_key_state = inherited.get("OPENROUTER_API_KEY_STATE")
    output["api_key_state"] = api_key_state
    output["mismatches"] = mismatches
    output["passed"] = (api_key_state == expected_api_key_state) and (len(mismatches) == 0)
    return output


def parse_tool_result(result: Any) -> dict[str, Any]:
    for item in (result.content or []):
        for attr in ("json", "data", "value"):
            value = getattr(item, attr, None)
            if isinstance(value, dict):
                return value
        text = getattr(item, "text", None)
        if isinstance(text, str):
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return {"_raw_text_present": True}


async def call_tool(session: ClientSession, name: str, args: dict[str, Any] | None = None) -> tuple[dict[str, Any], bool]:
    result = await session.call_tool(name, arguments=(args or {}))
    return parse_tool_result(result), bool(getattr(result, "is_error", False))


def reference_meta(reference: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "reference": reference,
        "valid_format": False,
        "meta_exists": False,
    }
    if not isinstance(reference, str) or not reference.startswith("evidence://"):
        return output

    suffix = reference[len("evidence://") :]
    if "/" not in suffix:
        return output

    output["valid_format"] = True
    _, ref_id = suffix.split("/", 1)
    ref_path = REFS / f"{ref_id}.json"
    if not ref_path.is_file():
        return output

    output["meta_exists"] = True
    try:
        payload = json.loads(ref_path.read_text(encoding="utf-8"))
    except Exception as exc:
        output["meta_error"] = f"{type(exc).__name__}:{exc}"
        return output

    output["kind"] = payload.get("kind")
    output["status"] = payload.get("status")
    output["expires_at"] = payload.get("expires_at")
    return output


def load_envelope(message_id: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    path = ENVELOPES / f"{message_id}.json"
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return None, info

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}:{exc}"
        return None, info

    envelope = payload.get("envelope") if isinstance(payload, dict) else None
    info["record_message_id"] = payload.get("message_id") if isinstance(payload, dict) else None
    info["envelope_message_id"] = envelope.get("message_id") if isinstance(envelope, dict) else None
    info["envelope_timestamp"] = envelope.get("timestamp") if isinstance(envelope, dict) else None
    info["frame_reference"] = envelope.get("frame_reference") if isinstance(envelope, dict) else None
    if isinstance(envelope, dict) and isinstance(envelope.get("source_evidence_refs"), list):
        info["source_ref_count"] = len(envelope["source_evidence_refs"])
    else:
        info["source_ref_count"] = None
    return (payload if isinstance(payload, dict) else None), info


def process_check_live_tracking_display() -> list[str]:
    result = subprocess.run(
        ["pgrep", "-fl", "agents.dashboard.live_tracking_display"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines


def safe_write_report(path: Path, report: dict[str, Any]) -> None:
    report["sanitization"] = {
        "credentials_logged": False,
        "image_payload_logged": False,
        "base64_logged": False,
        "full_visual_prompt_logged": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


async def run_probe() -> dict[str, Any]:
    started_at = time.time()

    mode = os.getenv("MODE", "").strip().lower()
    if mode not in {"dry", "real"}:
        raise ProbeAbort("MODE must be 'dry' or 'real'")

    paths = build_probe_paths()
    state_path = paths["state_path"]
    lock_path = paths["lock_path"]
    log_path = paths["log_path"]
    report_path = paths["report_path"]
    probe_dir = paths["probe_dir"]
    server_env, configured_visual_model, configured_endpoint = build_server_env(mode, paths)
    expected_api_key_state = "empty" if mode == "dry" else "set"
    preflight = run_env_inheritance_preflight(
        server_env,
        expected_visual_model=configured_visual_model,
        expected_endpoint=configured_endpoint,
        expected_api_key_state=expected_api_key_state,
    )

    report: dict[str, Any] = {
        "probe_id": probe_dir.name,
        "mode": mode,
        "started_at": started_at,
        "report_path": str(report_path),
        "verification_outcome": "verification_skipped",
        "error": None,
        "configured_child_env": {
            "budget_enabled": server_env.get("PROVIDER_CALL_BUDGET_ENABLED"),
            "budget_total_limit": server_env.get("PROVIDER_CALL_BUDGET_TOTAL_LIMIT"),
            "budget_tracking_vlm_limit": server_env.get("PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT"),
            "budget_observation_text_limit": server_env.get("PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT"),
            "budget_reasoning_text_limit": server_env.get("PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT"),
            "budget_observation_visual_limit": server_env.get("PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT"),
            "budget_state_path": str(state_path),
            "budget_lock_path": str(lock_path),
            "budget_log_path": str(log_path),
            "visual_verification_timeout_seconds": server_env.get("OBSERVATION_VISUAL_VERIFICATION_TIMEOUT_SECONDS"),
            "visual_model": configured_visual_model,
            "provider_endpoint": configured_endpoint,
            "openrouter_api_key_state": "empty" if server_env.get("OPENROUTER_API_KEY", "") == "" else "set",
        },
        "env_inheritance_preflight": preflight,
        "env_intent": {
            "mode": mode,
            "openrouter_key_state_child": "empty" if server_env.get("OPENROUTER_API_KEY", "") == "" else "set",
            "budget_enabled": server_env.get("PROVIDER_CALL_BUDGET_ENABLED"),
            "budget_state_path": str(state_path),
            "budget_log_path": str(log_path),
            "visual_model_env": configured_visual_model,
            "provider_endpoint_env": configured_endpoint,
        },
    }

    if not preflight.get("passed"):
        raise ProbeAbort("child_env_inheritance_check_failed")

    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=server_env)

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            stop_result: dict[str, Any] | None = None
            process_status_after_stop: dict[str, Any] | None = None

            try:
                start_out, _ = await call_tool(
                    session,
                    "start_dashboard_and_wait_ready",
                    {
                        "timeout_seconds": 20.0,
                        "max_observation_age_seconds": 3.0,
                        "poll_interval_seconds": 0.25,
                    },
                )
                report["start_dashboard_and_wait_ready"] = start_out

                selected_message_id = None
                deadline = time.time() + 20.0
                while time.time() < deadline:
                    files = sorted(ENVELOPES.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if files and files[0].stat().st_mtime >= started_at:
                        selected_message_id = files[0].stem
                        break
                    await asyncio.sleep(0.12)

                report["selected_message_id"] = selected_message_id
                if not selected_message_id:
                    raise ProbeAbort("no_fresh_envelope_found")

                envelope_payload, envelope_info = load_envelope(selected_message_id)
                report["envelope_check"] = envelope_info
                if envelope_payload is None:
                    raise ProbeAbort("envelope_unreadable_or_missing")

                envelope_obj = envelope_payload.get("envelope")
                if not isinstance(envelope_obj, dict):
                    raise ProbeAbort("envelope_payload_missing")

                ids_equal = (
                    envelope_info.get("record_message_id") == selected_message_id
                    and envelope_info.get("envelope_message_id") == selected_message_id
                )
                report["envelope_message_id_equal"] = bool(ids_equal)
                if not ids_equal:
                    raise ProbeAbort("envelope_message_id_mismatch")

                refs_to_check: list[str] = []
                frame_ref = envelope_obj.get("frame_reference")
                if isinstance(frame_ref, str) and frame_ref.strip():
                    refs_to_check.append(frame_ref.strip())
                source_refs = envelope_obj.get("source_evidence_refs")
                if isinstance(source_refs, list):
                    refs_to_check.extend(
                        item.strip()
                        for item in source_refs
                        if isinstance(item, str) and item.strip()
                    )

                if not refs_to_check:
                    raise ProbeAbort("no_visual_references_in_envelope")

                snapshots = [reference_meta(ref) for ref in refs_to_check]
                report["reference_snapshot"] = snapshots[:8]

                now = time.time()
                valid_nonexpired = 0
                for snap in snapshots:
                    expires_at = snap.get("expires_at")
                    if (
                        bool(snap.get("valid_format"))
                        and bool(snap.get("meta_exists"))
                        and isinstance(expires_at, (int, float))
                        and float(expires_at) > now
                    ):
                        valid_nonexpired += 1
                report["valid_nonexpired_visual_reference_count"] = valid_nonexpired
                if valid_nonexpired < 1:
                    raise ProbeAbort("no_valid_nonexpired_visual_reference")

                get_before, _ = await call_tool(session, "get_tracking_visual_evidence", {"message_id": selected_message_id})
                report["get_before_verify"] = {
                    "available": get_before.get("available"),
                    "error_code": get_before.get("error_code"),
                    "gate_errors": (get_before.get("details") or {}).get("gate_errors") if isinstance(get_before, dict) else None,
                }
                if not bool(get_before.get("available")):
                    raise ProbeAbort("preverify_get_unavailable")

                get_immediate, _ = await call_tool(session, "get_tracking_visual_evidence", {"message_id": selected_message_id})
                report["get_immediate_pre_verify"] = {
                    "available": get_immediate.get("available"),
                    "error_code": get_immediate.get("error_code"),
                    "gate_errors": (get_immediate.get("details") or {}).get("gate_errors") if isinstance(get_immediate, dict) else None,
                }
                if not bool(get_immediate.get("available")):
                    raise ProbeAbort("evidence_became_unavailable_before_verify")

                stop_before_verify_result, _ = await call_tool(
                    session,
                    "stop_dashboard_and_confirm",
                    {
                        "force": False,
                        "timeout_seconds": 8.0,
                        "poll_interval_seconds": 0.1,
                    },
                )
                report["stop_before_verify"] = stop_before_verify_result

                dashboard_status_after_preverify_stop, _ = await call_tool(session, "dashboard_demo_status", {})
                report["dashboard_status_after_preverify_stop"] = dashboard_status_after_preverify_stop

                get_after_stop, _ = await call_tool(
                    session,
                    "get_tracking_visual_evidence",
                    {"message_id": selected_message_id},
                )
                report["get_after_stop_pre_verify"] = {
                    "available": get_after_stop.get("available"),
                    "error_code": get_after_stop.get("error_code"),
                    "gate_errors": (get_after_stop.get("details") or {}).get("gate_errors") if isinstance(get_after_stop, dict) else None,
                }
                if not bool(get_after_stop.get("available")):
                    raise ProbeAbort("evidence_unavailable_after_dashboard_stop")

                verify_out, _ = await call_tool(session, "verify_tracking_visual_evidence", {"message_id": selected_message_id})
                validation_payload = verify_out.get("validation") if isinstance(verify_out, dict) else None
                coordinator_proven = isinstance(validation_payload, dict) and ("valid" in validation_payload)
                report["coordinator_participation_demonstrated"] = bool(coordinator_proven)

                report["verify_result"] = {
                    "available": verify_out.get("available"),
                    "error_code": verify_out.get("error_code"),
                    "error": verify_out.get("error"),
                    "details": verify_out.get("details"),
                    "model": ((verify_out.get("llm_meta") or {}).get("model") if isinstance(verify_out, dict) else None),
                    "latency_ms": ((verify_out.get("llm_meta") or {}).get("latency_ms") if isinstance(verify_out, dict) else None),
                    "llm_status": ((verify_out.get("llm_meta") or {}).get("status") if isinstance(verify_out, dict) else None),
                    "llm_error_code": ((verify_out.get("llm_meta") or {}).get("error_code") if isinstance(verify_out, dict) else None),
                    "llm_attempts": ((verify_out.get("llm_meta") or {}).get("attempts") if isinstance(verify_out, dict) else None),
                    "llm_retry_count": ((verify_out.get("llm_meta") or {}).get("retry_count") if isinstance(verify_out, dict) else None),
                    "verification_status": ((verify_out.get("envelope") or {}).get("verification_status") if isinstance(verify_out, dict) else None),
                    "observation_id": ((verify_out.get("envelope") or {}).get("observation_id") if isinstance(verify_out, dict) else None),
                    "referenced_message_id": ((verify_out.get("envelope") or {}).get("referenced_message_id") if isinstance(verify_out, dict) else None),
                    "coordinator_validation_valid": validation_payload.get("valid") if isinstance(validation_payload, dict) else None,
                    "coordinator_validation_errors": validation_payload.get("errors") if isinstance(validation_payload, dict) else None,
                }

                if bool(verify_out.get("available")):
                    report["verification_outcome"] = "verification_succeeded"
                else:
                    report["verification_outcome"] = "verification_failed"

            except ProbeAbort as abort:
                report["error"] = str(abort)
                report["verification_outcome"] = "verification_skipped"
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}:{exc}"
                if report.get("verification_outcome") == "verification_skipped":
                    report["verification_outcome"] = "verification_failed"
            finally:
                try:
                    stop_result, _ = await call_tool(
                        session,
                        "stop_dashboard_and_confirm",
                        {
                            "force": False,
                            "timeout_seconds": 8.0,
                            "poll_interval_seconds": 0.1,
                        },
                    )
                except Exception as stop_exc:
                    stop_result = {
                        "stopped": False,
                        "stop_error": f"{type(stop_exc).__name__}:{stop_exc}",
                    }

                try:
                    process_status_after_stop, _ = await call_tool(session, "dashboard_demo_status", {})
                except Exception as status_exc:
                    process_status_after_stop = {
                        "running": None,
                        "status_error": f"{type(status_exc).__name__}:{status_exc}",
                    }

                report["stop_dashboard_and_confirm"] = stop_result
                report["dashboard_process_status_after_stop"] = process_status_after_stop
                report["dashboard_exited_confirmed"] = bool(
                    isinstance(process_status_after_stop, dict)
                    and (process_status_after_stop.get("running") is False)
                )

    report["post_shutdown_live_tracking_display_processes"] = process_check_live_tracking_display()
    report["no_live_tracking_display_processes"] = len(report["post_shutdown_live_tracking_display_processes"]) == 0

    report["budget_artifacts"] = {
        "state_exists": state_path.is_file(),
        "lock_exists": lock_path.is_file(),
        "log_exists": log_path.is_file(),
        "state_path": str(state_path),
        "lock_path": str(lock_path),
        "log_path": str(log_path),
    }

    reservation_count = None
    if state_path.is_file():
        try:
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            totals = (state_payload.get("totals") or {}) if isinstance(state_payload, dict) else {}
            per_role = (totals.get("per_role_attempted") or {}) if isinstance(totals, dict) else {}
            reservation_count = totals.get("attempted")
            report["budget_totals"] = {
                "attempted_reservations_total": reservation_count,
                "observation_visual_reserved": per_role.get("observation_visual"),
                "tracking_vlm_reserved": per_role.get("tracking_vlm"),
                "observation_text_reserved": per_role.get("observation_text"),
                "reasoning_text_reserved": per_role.get("reasoning_text"),
            }
        except Exception as exc:
            report["budget_state_parse_error"] = f"{type(exc).__name__}:{exc}"

    attempt_events: list[dict[str, Any]] = []
    blocked_events: list[dict[str, Any]] = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            compact = {
                "event_type": event.get("event_type"),
                "role": event.get("role"),
                "outcome": event.get("outcome"),
                "error_code": event.get("error_code"),
                "latency_ms": event.get("latency_ms"),
                "attempt_id": event.get("attempt_id"),
            }
            if event.get("event_type") == "provider_call_attempt":
                attempt_events.append(compact)
            elif event.get("event_type") == "provider_call_blocked":
                blocked_events.append(compact)

    report["reservation_count_total"] = reservation_count
    report["budget_reservations"] = {
        "reservation_count_total": reservation_count,
    }
    report["provider_attempt_count"] = len(attempt_events)
    report["provider_attempt_events"] = attempt_events
    report["provider_blocked_events"] = blocked_events

    verify_result = report.get("verify_result") or {}
    if isinstance(verify_result, dict):
        verify_details = verify_result.get("details") if isinstance(verify_result.get("details"), dict) else {}
        report["verification_error"] = {
            "error_code": verify_result.get("error_code"),
            "error": verify_result.get("error"),
            "upstream_error_code": verify_details.get("error_code") if isinstance(verify_details, dict) else None,
            "upstream_attempts": verify_details.get("attempts") if isinstance(verify_details, dict) else None,
            "upstream_model": verify_details.get("model") if isinstance(verify_details, dict) else None,
        }

    verification_error_code = verify_result.get("error_code") if isinstance(verify_result, dict) else None
    verification_details = verify_result.get("details") if isinstance(verify_result, dict) and isinstance(verify_result.get("details"), dict) else {}
    upstream_error_code = verification_details.get("error_code") if isinstance(verification_details, dict) else None

    failure_classification = "none"
    if report.get("verification_outcome") == "verification_failed":
        if verification_error_code == "coordinator_validation_rejected":
            failure_classification = "coordinator_validation_failure"
        elif verification_error_code == "unsupported_vision_model":
            failure_classification = "provider_compatibility_failure"
        elif verification_error_code == "provider_failure":
            if upstream_error_code == "authentication":
                failure_classification = "authentication_failure"
            elif upstream_error_code in {"invalid_response_format", "empty_response"}:
                failure_classification = "provider_compatibility_failure"
            elif upstream_error_code == "budget_exhausted":
                failure_classification = "budget_exhausted"
            else:
                failure_classification = "model_failure"
        else:
            failure_classification = "model_failure"
    report["verification_failure_classification"] = failure_classification

    verification_error_text = ""
    if isinstance(verify_result, dict) and isinstance(verify_result.get("error"), str):
        verification_error_text = verify_result["error"]
    api_key_empty_confirmed = bool((report.get("env_inheritance_preflight") or {}).get("api_key_state") == "empty")
    api_key_set_confirmed = bool((report.get("env_inheritance_preflight") or {}).get("api_key_state") == "set")
    provider_attempted = len(attempt_events) > 0
    report["external_provider_reachability"] = {
        "expected_block_reason": "empty_openrouter_api_key" if mode == "dry" else "real_mode_provider_allowed",
        "key_missing_error_observed": ("OPENROUTER_API_KEY is not set" in verification_error_text),
        "provider_attempt_count": len(attempt_events),
        "provider_attempted": provider_attempted,
        "cannot_reach_external_provider_confirmed": api_key_empty_confirmed and (not provider_attempted),
        "provider_reachability_enabled_for_real_mode": api_key_set_confirmed if mode == "real" else False,
    }

    report["inheritance_confirmation"] = {
        "budget_paths_honored": bool(report["budget_artifacts"]["state_exists"] or report["budget_artifacts"]["log_exists"]),
        "mode": mode,
        "openrouter_key_state_child": "empty" if server_env.get("OPENROUTER_API_KEY", "") == "" else "set",
        "preflight_passed": bool(preflight.get("passed")),
    }

    return report


def main() -> int:
    fallback_report_path = REPO / "captures" / "probes" / "stage2a_probe_report_error.json"
    report_path = fallback_report_path
    report: dict[str, Any] = {}
    exit_code = 0

    try:
        report = asyncio.run(run_probe())
    except ProbeAbort as exc:
        report = {
            "mode": os.getenv("MODE", ""),
            "verification_outcome": "verification_skipped",
            "error": str(exc),
        }
        exit_code = 1
    except Exception as exc:
        report = {
            "mode": os.getenv("MODE", ""),
            "verification_outcome": "verification_failed",
            "error": f"{type(exc).__name__}:{exc}",
        }
        exit_code = 1
    finally:
        report_path_raw = report.get("report_path") if isinstance(report, dict) else None
        if isinstance(report_path_raw, str) and report_path_raw.strip():
            report_path = Path(report_path_raw)
        safe_write_report(report_path, report)
        print(json.dumps({"report_path": str(report_path)}, indent=2))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
