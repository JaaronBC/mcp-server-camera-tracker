from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _safe_load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_read_float(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _safe_read_int(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            output.append(payload)
    return output


def _role_attempt_counts(state_payload: dict[str, Any] | None, budget_events: list[dict[str, Any]]) -> dict[str, int]:
    if isinstance(state_payload, dict):
        totals = state_payload.get("totals")
        if isinstance(totals, dict):
            per_role = totals.get("per_role_attempted")
            if isinstance(per_role, dict):
                return {
                    "tracking_vlm": int(per_role.get("tracking_vlm", 0) or 0),
                    "observation_text": int(per_role.get("observation_text", 0) or 0),
                    "reasoning_text": int(per_role.get("reasoning_text", 0) or 0),
                    "observation_visual": int(per_role.get("observation_visual", 0) or 0),
                }

    counts = {"tracking_vlm": 0, "observation_text": 0, "reasoning_text": 0, "observation_visual": 0}
    for event in budget_events:
        if event.get("event_type") != "provider_call_attempt":
            continue
        role = str(event.get("role", ""))
        if role in counts:
            counts[role] += 1
    return counts


def _in_window(ts: Any, start_ts: float | None, effective_end_ts: float | None) -> bool:
    if start_ts is None or effective_end_ts is None:
        return True
    if not isinstance(ts, (int, float)):
        return False
    return float(start_ts) <= float(ts) <= float(effective_end_ts)


def _partition_records_for_run_id(payload: Any, run_id: str | None) -> tuple[list[dict[str, Any]], int, int, int]:
    if not isinstance(payload, list):
        return ([], 0, 0, 0)

    accepted: list[dict[str, Any]] = []
    missing_count = 0
    mismatched_count = 0
    total = 0

    for item in payload:
        if not isinstance(item, dict):
            continue
        total += 1
        record_run_id = item.get("run_id")
        if not isinstance(record_run_id, str) or not record_run_id.strip():
            missing_count += 1
            continue
        if not run_id or record_run_id != run_id:
            mismatched_count += 1
            continue
        accepted.append(item)

    return (accepted, missing_count, mismatched_count, total)


def _partition_trigger_events_for_run_id(payload: list[dict[str, Any]], run_id: str | None) -> tuple[list[dict[str, Any]], int, int, int]:
    accepted: list[dict[str, Any]] = []
    missing_count = 0
    mismatched_count = 0
    total = 0

    for item in payload:
        if not isinstance(item, dict):
            continue
        total += 1
        record_run_id = item.get("run_id")
        if not isinstance(record_run_id, str) or not record_run_id.strip():
            missing_count += 1
            continue
        if not run_id or record_run_id != run_id:
            mismatched_count += 1
            continue
        accepted.append(item)

    return (accepted, missing_count, mismatched_count, total)


def _normalize_event_origin(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized in {"startup_activation", "production_monitor"}:
        return normalized
    return None


def _trigger_event_identifier(event: dict[str, Any]) -> str | None:
    for key in ("event_id", "task_id"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_synthetic_track(track_id: str) -> bool:
    return track_id.startswith("sim-")


def _build_correlated_chain_rows(
    queue_records: list[dict[str, Any]],
    submission_records: list[dict[str, Any]],
    trigger_events: list[dict[str, Any]],
    submission_decisions: list[dict[str, Any]],
    *,
    start_ts: float | None,
    effective_end_ts: float | None,
) -> list[dict[str, Any]]:
    queue_by_request_id: dict[str, dict[str, Any]] = {}
    for item in queue_records:
        request_id = str(item.get("request_id", "")).strip()
        if request_id:
            queue_by_request_id[request_id] = item

    submissions_by_request_id: dict[str, dict[str, Any]] = {}
    submissions_by_track: dict[str, list[dict[str, Any]]] = {}
    for item in submission_records:
        request_id = str(item.get("request_id", "")).strip()
        if request_id:
            submissions_by_request_id[request_id] = item
        track_id = str(item.get("track_id", "")).strip()
        if track_id:
            submissions_by_track.setdefault(track_id, []).append(item)

    for track_submissions in submissions_by_track.values():
        track_submissions.sort(key=lambda row: float(row.get("submitted_at")) if isinstance(row.get("submitted_at"), (int, float)) else 0.0)

    decision_by_trigger_event_id: dict[str, dict[str, Any]] = {}
    for item in submission_decisions:
        if not isinstance(item, dict):
            continue
        trigger_event_id = item.get("trigger_event_id")
        if isinstance(trigger_event_id, str) and trigger_event_id.strip():
            decision_by_trigger_event_id[trigger_event_id.strip()] = item

    run_trigger_events: list[dict[str, Any]] = []
    for event in trigger_events:
        event_type = str(event.get("event_type", "")).strip()
        if event_type not in {"reasoning_cycle", "verification_trigger", "phase16e_trigger"}:
            continue
        if not _in_window(event.get("timestamp"), start_ts, effective_end_ts):
            continue
        if event_type == "reasoning_cycle" and not bool(event.get("evidence_requested", False)):
            continue
        run_trigger_events.append(event)

    rows: list[dict[str, Any]] = []
    for event in run_trigger_events:
        track_id = str(event.get("track_id", "")).strip() or "unknown"
        trigger_timestamp = float(event.get("timestamp")) if isinstance(event.get("timestamp"), (int, float)) else None
        trigger_event_id = _trigger_event_identifier(event)
        decision = decision_by_trigger_event_id.get(trigger_event_id or "")

        event_origin = _normalize_event_origin(event.get("event_origin"))
        if event_origin is None:
            event_origin = _normalize_event_origin(decision.get("event_origin") if isinstance(decision, dict) else None)

        synthetic_track = _is_synthetic_track(track_id)
        eligible_for_genuine_chain = (
            str(event.get("event_type", "")).strip() == "reasoning_cycle"
            and bool(event.get("evidence_requested", False))
            and event_origin == "production_monitor"
            and not synthetic_track
        )

        decision_queue_request_id = None
        if isinstance(decision, dict):
            candidate_request_id = decision.get("queue_request_id")
            if isinstance(candidate_request_id, str) and candidate_request_id.strip():
                decision_queue_request_id = candidate_request_id.strip()

        matched_submission: dict[str, Any] | None = None
        fallback_submission: dict[str, Any] | None = None
        if decision_queue_request_id is not None:
            matched_submission = submissions_by_request_id.get(decision_queue_request_id)
        elif eligible_for_genuine_chain and track_id in submissions_by_track and trigger_timestamp is not None:
            for candidate in submissions_by_track[track_id]:
                submitted_at = candidate.get("submitted_at")
                if not isinstance(submitted_at, (int, float)):
                    continue
                submitted_at_float = float(submitted_at)
                if submitted_at_float < trigger_timestamp:
                    continue
                if effective_end_ts is not None and submitted_at_float > effective_end_ts:
                    continue
                if fallback_submission is None:
                    fallback_submission = candidate
                if bool(candidate.get("accepted", False)):
                    matched_submission = candidate
                    break
        if matched_submission is None:
            matched_submission = fallback_submission

        queue_item: dict[str, Any] = {}
        if matched_submission is not None:
            request_id = str(matched_submission.get("request_id", "")).strip()
            if request_id:
                queue_item = queue_by_request_id.get(request_id, {})

        queue_submission = bool(matched_submission and bool(matched_submission.get("accepted", False)))
        verification_success = str(queue_item.get("validation_outcome", "")).strip() == "accepted"
        submission_status = str(matched_submission.get("status", "")).strip() if isinstance(matched_submission, dict) else "not_attempted"
        submission_reason = (str(matched_submission.get("reason", "")).strip() or None) if isinstance(matched_submission, dict) else None
        skip_reason = (
            str(decision.get("skip_reason")).strip()
            if isinstance(decision, dict) and isinstance(decision.get("skip_reason"), str) and str(decision.get("skip_reason")).strip()
            else None
        )
        submission_attempted_from_decision = bool(decision.get("submission_attempted", False)) if isinstance(decision, dict) else False
        submission_accepted_from_decision = bool(decision.get("submission_accepted", False)) if isinstance(decision, dict) else False

        automatic_evidence_requested = (
            bool(event.get("evidence_requested", False))
            if str(event.get("event_type", "")).strip() == "reasoning_cycle"
            else bool(event.get("automatic_evidence_requested", event.get("evidence_requested", False)))
        )

        rows.append(
            {
                "trigger_event_id": trigger_event_id,
                "track_id": track_id,
                "event_origin": event_origin,
                "synthetic_track": synthetic_track,
                "trigger_timestamp": event.get("timestamp"),
                "automatic_evidence_requested": automatic_evidence_requested,
                "eligible_for_genuine_chain": eligible_for_genuine_chain,
                "submission_decision_available": isinstance(decision, dict),
                "submission_attempted": bool(matched_submission is not None or submission_attempted_from_decision),
                "submission_status": submission_status,
                "submission_reason": submission_reason or skip_reason,
                "queue_submission": queue_submission,
                "queue_request_id": matched_submission.get("request_id") if isinstance(matched_submission, dict) else queue_item.get("request_id"),
                "evidence_message_id": matched_submission.get("message_id") if isinstance(matched_submission, dict) else queue_item.get("evidence_message_id"),
                "fresh_camera_yolo_evidence": bool((matched_submission.get("message_id") if isinstance(matched_submission, dict) else queue_item.get("evidence_message_id"))),
                "verification_status": queue_item.get("status"),
                "verification_success": verification_success,
                "submission_accepted_from_decision": submission_accepted_from_decision,
                "coordinator_acceptance": bool(queue_item.get("coordinator_valid", verification_success)),
                "feedback_publication": bool(queue_item.get("feedback_published", False)),
                "tracking_feedback_consumer_acceptance": bool(queue_item.get("tracking_feedback_consumer_acceptance", False)),
                "current_not_stale": str(queue_item.get("current_applicability", "")) == "current",
            }
        )

    return rows


def _build_failure_reasons(
    *,
    run_id: str | None,
    main_exit_code: int | None,
    run_interrupted: bool,
    budget_state_exists: bool,
    budget_log_exists: bool,
    post_shutdown_processes: list[str],
    role_counts: dict[str, int],
    queue_missing_run_id_count: int,
    queue_mismatched_run_id_count: int,
    submission_missing_run_id_count: int,
    submission_mismatched_run_id_count: int,
    submission_decision_missing_run_id_count: int,
    submission_decision_mismatched_run_id_count: int,
    trigger_event_missing_run_id_count: int,
    trigger_event_mismatched_run_id_count: int,
    required_run_local_artifacts_missing: list[str],
    correlated_rows: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []

    if not run_id:
        reasons.append("run_id_missing")
    if main_exit_code is None:
        reasons.append("main_exit_code_missing")
    elif main_exit_code != 0:
        reasons.append(f"main_exit_nonzero:{main_exit_code}")
    if run_interrupted:
        reasons.append("run_interrupted")
    if not budget_state_exists:
        reasons.append("budget_state_missing")
    if not budget_log_exists:
        reasons.append("budget_log_missing")

    if queue_missing_run_id_count > 0:
        reasons.append(f"queue_records_missing_run_id:{queue_missing_run_id_count}")
    if queue_mismatched_run_id_count > 0:
        reasons.append(f"queue_records_mismatched_run_id:{queue_mismatched_run_id_count}")
    if submission_missing_run_id_count > 0:
        reasons.append(f"submission_records_missing_run_id:{submission_missing_run_id_count}")
    if submission_mismatched_run_id_count > 0:
        reasons.append(f"submission_records_mismatched_run_id:{submission_mismatched_run_id_count}")
    if submission_decision_missing_run_id_count > 0:
        reasons.append(f"submission_decisions_missing_run_id:{submission_decision_missing_run_id_count}")
    if submission_decision_mismatched_run_id_count > 0:
        reasons.append(f"submission_decisions_mismatched_run_id:{submission_decision_mismatched_run_id_count}")
    if trigger_event_missing_run_id_count > 0:
        reasons.append(f"trigger_events_missing_run_id:{trigger_event_missing_run_id_count}")
    if trigger_event_mismatched_run_id_count > 0:
        reasons.append(f"trigger_events_mismatched_run_id:{trigger_event_mismatched_run_id_count}")

    for artifact_name in required_run_local_artifacts_missing:
        reasons.append(f"run_local_artifact_missing:{artifact_name}")

    if post_shutdown_processes:
        reasons.append("post_shutdown_processes_not_empty")
    if role_counts.get("observation_visual", 0) != 1:
        reasons.append("observation_visual_attempt_count_not_equal_to_one")
    if role_counts.get("tracking_vlm", 0) != 0:
        reasons.append("tracking_vlm_attempt_count_not_zero")
    if role_counts.get("observation_text", 0) != 0:
        reasons.append("observation_text_attempt_count_not_zero")
    if role_counts.get("reasoning_text", 0) != 0:
        reasons.append("reasoning_text_attempt_count_not_zero")

    eligible_rows = [row for row in correlated_rows if bool(row.get("eligible_for_genuine_chain", False))]
    if not eligible_rows:
        reasons.append("correlated_chain_missing")

    for row in eligible_rows:
        track_id = str(row.get("track_id", "unknown"))
        if not bool(row.get("queue_submission", False)):
            reasons.append(f"queue_submission_missing:{track_id}")
        if not bool(row.get("feedback_publication", False)):
            reasons.append(f"feedback_publication_missing:{track_id}")
        if not bool(row.get("tracking_feedback_consumer_acceptance", False)):
            reasons.append(f"feedback_consumer_acceptance_missing:{track_id}")

    return reasons


def generate_report(run_dir: Path) -> dict[str, Any]:
    start_ts = _safe_read_float(run_dir / "start_ts.txt")
    end_ts = _safe_read_float(run_dir / "end_ts.txt")
    main_exit_code = _safe_read_int(run_dir / "main_exit_code.txt")
    run_interrupted = (_safe_read_int(run_dir / "run_interrupted.txt") or 0) == 1
    run_id = (run_dir / "run_id.txt").read_text(encoding="utf-8").strip() if (run_dir / "run_id.txt").is_file() else None
    if isinstance(run_id, str) and not run_id.strip():
        run_id = None

    bounded_session_seconds = float(os.getenv("BOUNDED_SESSION_SECONDS", "60") or 60)
    verification_drain_seconds = float(os.getenv("VERIFICATION_DRAIN_SECONDS", "20") or 20)
    watchdog_margin_seconds = float(os.getenv("PHASE16E_WATCHDOG_MARGIN_SECONDS", "15") or 15)
    effective_end_ts = (end_ts + verification_drain_seconds) if isinstance(end_ts, float) else None

    budget_state_path = run_dir / "provider_call_budget_state.json"
    budget_log_path = run_dir / "provider_call_budget_events.jsonl"
    post_shutdown_path = run_dir / "post_shutdown_processes.txt"
    main_log_path = run_dir / "main.log"
    trigger_events_path = run_dir / "agent_events.jsonl"

    budget_state = _safe_load_json(budget_state_path)
    budget_events = _load_jsonl(budget_log_path)
    trigger_events_payload = _load_jsonl(trigger_events_path)
    submission_decisions_payload = _load_jsonl(run_dir / "verification_submission_decisions.jsonl")
    queue_payload = _safe_load_json(run_dir / "latest_verification_queue_results.json")
    submission_payload = _safe_load_json(run_dir / "latest_verification_queue_submissions.json")

    queue_records, queue_missing_run_id_count, queue_mismatched_run_id_count, queue_total = _partition_records_for_run_id(queue_payload, run_id)
    submission_records, submission_missing_run_id_count, submission_mismatched_run_id_count, submission_total = _partition_records_for_run_id(submission_payload, run_id)
    submission_decisions, submission_decision_missing_run_id_count, submission_decision_mismatched_run_id_count, submission_decision_total = _partition_records_for_run_id(submission_decisions_payload, run_id)
    trigger_events, trigger_event_missing_run_id_count, trigger_event_mismatched_run_id_count, trigger_event_total = _partition_trigger_events_for_run_id(trigger_events_payload, run_id)

    required_artifacts_status_payload = _safe_load_json(run_dir / "run_local_artifact_status.json")
    if isinstance(required_artifacts_status_payload, dict):
        required_run_local_artifacts_expected = [str(item) for item in required_artifacts_status_payload.get("expected", []) if isinstance(item, str)]
        required_run_local_artifacts_present = [str(item) for item in required_artifacts_status_payload.get("present", []) if isinstance(item, str)]
        required_run_local_artifacts_missing = [str(item) for item in required_artifacts_status_payload.get("missing", []) if isinstance(item, str)]
    else:
        required_run_local_artifacts_expected = [
            "provider_call_budget_state.json",
            "provider_call_budget_events.jsonl",
            "latest_verification_queue_results.json",
            "latest_verification_queue_submissions.json",
            "operator_tracking_feedback.json",
            "agent_events.jsonl",
        ]
        required_run_local_artifacts_present = [name for name in required_run_local_artifacts_expected if (run_dir / name).is_file()]
        required_run_local_artifacts_missing = [name for name in required_run_local_artifacts_expected if name not in required_run_local_artifacts_present]

    post_shutdown_processes: list[str] = []
    if post_shutdown_path.is_file():
        post_shutdown_processes = [line.strip() for line in post_shutdown_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    role_counts = _role_attempt_counts(budget_state if isinstance(budget_state, dict) else None, budget_events)
    correlated_rows = _build_correlated_chain_rows(
        queue_records,
        submission_records,
        trigger_events,
        submission_decisions,
        start_ts=start_ts,
        effective_end_ts=effective_end_ts,
    )
    eligible_correlated_rows = [row for row in correlated_rows if bool(row.get("eligible_for_genuine_chain", False))]
    startup_only_observed = bool(correlated_rows) and not bool(eligible_correlated_rows)

    failure_reasons = _build_failure_reasons(
        run_id=run_id,
        main_exit_code=main_exit_code,
        run_interrupted=run_interrupted,
        budget_state_exists=budget_state_path.is_file(),
        budget_log_exists=budget_log_path.is_file(),
        post_shutdown_processes=post_shutdown_processes,
        role_counts=role_counts,
        queue_missing_run_id_count=queue_missing_run_id_count,
        queue_mismatched_run_id_count=queue_mismatched_run_id_count,
        submission_missing_run_id_count=submission_missing_run_id_count,
        submission_mismatched_run_id_count=submission_mismatched_run_id_count,
        submission_decision_missing_run_id_count=submission_decision_missing_run_id_count,
        submission_decision_mismatched_run_id_count=submission_decision_mismatched_run_id_count,
        trigger_event_missing_run_id_count=trigger_event_missing_run_id_count,
        trigger_event_mismatched_run_id_count=trigger_event_mismatched_run_id_count,
        required_run_local_artifacts_missing=required_run_local_artifacts_missing,
        correlated_rows=correlated_rows,
    )

    verdict = "PASS" if not failure_reasons else "FAIL"
    if main_exit_code is None:
        verdict = "INCONCLUSIVE"
    elif startup_only_observed:
        non_infra_reasons = {"correlated_chain_missing"}
        infra_failures = [reason for reason in failure_reasons if reason not in non_infra_reasons]
        verdict = "FAIL" if infra_failures else "INCONCLUSIVE"

    report = {
        "schema_version": "1.0",
        "generated_at": time.time(),
        "run_dir": str(run_dir),
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "effective_end_ts": effective_end_ts,
            "bounded_session_seconds": bounded_session_seconds,
            "verification_drain_seconds": verification_drain_seconds,
            "watchdog_margin_seconds": watchdog_margin_seconds,
            "session_duration_seconds": (end_ts - start_ts) if isinstance(start_ts, float) and isinstance(end_ts, float) else None,
            "session_bound_seconds": bounded_session_seconds + verification_drain_seconds + watchdog_margin_seconds,
        },
        "artifacts": {
            "run_id": run_id,
            "budget_state_path": str(budget_state_path),
            "budget_log_path": str(budget_log_path),
            "main_log_path": str(main_log_path),
            "post_shutdown_processes_path": str(post_shutdown_path),
            "verification_submission_path": str(run_dir / "latest_verification_queue_submissions.json"),
            "verification_results_path": str(run_dir / "latest_verification_queue_results.json"),
            "verification_submission_decisions_path": str(run_dir / "verification_submission_decisions.jsonl"),
            "trigger_events_path": str(trigger_events_path),
            "main_exit_code_path": str(run_dir / "main_exit_code.txt"),
            "output_path": str(run_dir / "phase16e_report.json"),
        },
        "checks": {
            "run_clean_shutdown": len(post_shutdown_processes) == 0,
            "post_shutdown_processes": post_shutdown_processes,
            "main_exit_code": main_exit_code,
            "main_exit_code_ok": main_exit_code == 0,
            "run_interrupted": run_interrupted,
            "budget": {
                "state_exists": budget_state_path.is_file(),
                "log_exists": budget_log_path.is_file(),
                "role_attempt_counts": role_counts,
                "exactly_one_observation_visual_attempt": role_counts.get("observation_visual", 0) == 1,
            },
            "run_identity": {
                "run_id_present": bool(run_id),
                "queue_total": queue_total,
                "queue_accepted_for_run_id": len(queue_records),
                "queue_missing_run_id_count": queue_missing_run_id_count,
                "queue_mismatched_run_id_count": queue_mismatched_run_id_count,
                "submission_total": submission_total,
                "submission_accepted_for_run_id": len(submission_records),
                "submission_missing_run_id_count": submission_missing_run_id_count,
                "submission_mismatched_run_id_count": submission_mismatched_run_id_count,
                "submission_decision_total": submission_decision_total,
                "submission_decision_accepted_for_run_id": len(submission_decisions),
                "submission_decision_missing_run_id_count": submission_decision_missing_run_id_count,
                "submission_decision_mismatched_run_id_count": submission_decision_mismatched_run_id_count,
                "trigger_event_total": trigger_event_total,
                "trigger_event_accepted_for_run_id": len(trigger_events),
                "trigger_event_missing_run_id_count": trigger_event_missing_run_id_count,
                "trigger_event_mismatched_run_id_count": trigger_event_mismatched_run_id_count,
            },
            "submission_decisions": {
                "path": str(run_dir / "verification_submission_decisions.jsonl"),
                "records_for_run_id": len(submission_decisions),
            },
            "required_run_local_artifacts": {
                "expected": required_run_local_artifacts_expected,
                "present": required_run_local_artifacts_present,
                "missing": required_run_local_artifacts_missing,
            },
            "trigger_eligibility": {
                "startup_only_observed": startup_only_observed,
                "correlated_row_count": len(correlated_rows),
                "eligible_row_count": len(eligible_correlated_rows),
            },
        },
        "correlated_chain_rows": correlated_rows,
        "verdict": verdict,
        "failure_reasons": sorted(set(failure_reasons)),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase16E report from run artifacts")
    parser.add_argument("--run-dir", required=True, help="Phase16E run directory path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    report = generate_report(run_dir)
    output_path = run_dir / "phase16e_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(output_path), "verdict": report.get("verdict"), "failure_reasons": report.get("failure_reasons", [])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
