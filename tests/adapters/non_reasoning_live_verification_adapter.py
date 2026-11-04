from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

import main as runtime_main
from agents.orchestration.runtime import app as runtime_app
from agents.orchestration.runtime.verification_queue import (
    VerificationQueueRequest,
    build_request_from_message_id,
)
from agents.tracking.evidence_store import TrackingEvidenceResolver


@dataclass(slots=True)
class EvidenceCandidate:
    track_id: str
    message_id: str
    stored_at: float
    frame_reference: str
    detection_reference: str


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_run_dir() -> Path:
    results_path = os.getenv("PHASE16E_VERIFICATION_RESULTS_PATH", "").strip()
    if results_path:
        return Path(results_path).expanduser().resolve().parent
    return (Path(__file__).resolve().parents[2] / "captures" / "probes" / "runs").resolve()


def _parse_positive_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _read_active_track_ids(radar_state_path: Path) -> set[str]:
    payload = _load_json(radar_state_path)
    if not isinstance(payload, dict):
        return set()
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        return set()

    active: set[str] = set()
    for item in tracks:
        if isinstance(item, dict):
            track_id = item.get("track_id")
            if isinstance(track_id, str) and track_id.strip():
                active.add(track_id.strip())
    return active


def _extract_detection_reference(source_refs: list[str]) -> str | None:
    for ref in source_refs:
        if ref.startswith("evidence://detection/"):
            return ref
    return None


def select_current_run_evidence_candidate(
    *,
    evidence_root_dir: Path,
    radar_state_path: Path,
    run_started_at_epoch: float,
    now_epoch: float,
) -> EvidenceCandidate | None:
    active_track_ids = _read_active_track_ids(radar_state_path)
    if not active_track_ids:
        return None

    envelopes_dir = evidence_root_dir / "envelopes"
    if not envelopes_dir.is_dir():
        return None

    resolver = TrackingEvidenceResolver(evidence_root_dir)
    candidates: list[EvidenceCandidate] = []
    for envelope_path in envelopes_dir.glob("*.json"):
        payload = _load_json(envelope_path)
        if not isinstance(payload, dict):
            continue

        validation = payload.get("validation")
        if not isinstance(validation, dict) or not bool(validation.get("valid", False)):
            continue

        stored_at = payload.get("stored_at")
        if not isinstance(stored_at, (int, float)):
            continue
        stored_at_value = float(stored_at)
        if stored_at_value < run_started_at_epoch:
            continue

        envelope = payload.get("envelope")
        if not isinstance(envelope, dict):
            continue

        track_id = envelope.get("track_id")
        message_id = envelope.get("message_id")
        frame_reference = envelope.get("frame_reference")
        source_refs = envelope.get("source_evidence_refs")
        detection_id = envelope.get("detection_id")

        if not isinstance(track_id, str) or not track_id.strip():
            continue
        track_id = track_id.strip()
        if track_id not in active_track_ids:
            continue

        if not isinstance(message_id, str) or not message_id.strip():
            continue
        message_id = message_id.strip()

        if not isinstance(frame_reference, str) or not frame_reference.strip():
            continue
        frame_reference = frame_reference.strip()

        if not isinstance(source_refs, list):
            continue
        normalized_refs = [item.strip() for item in source_refs if isinstance(item, str) and item.strip()]
        detection_reference = _extract_detection_reference(normalized_refs)
        if detection_reference is None:
            continue

        frame_resolution = resolver.resolve_reference(frame_reference, now_timestamp=now_epoch)
        if not bool(frame_resolution.get("valid", False)):
            continue

        detection_resolution = resolver.resolve_reference(
            detection_reference,
            expected_track_id=track_id,
            expected_detection_id=(detection_id if isinstance(detection_id, str) and detection_id.strip() else None),
            expected_frame_reference=frame_reference,
            now_timestamp=now_epoch,
        )
        if not bool(detection_resolution.get("valid", False)):
            continue

        candidates.append(
            EvidenceCandidate(
                track_id=track_id,
                message_id=message_id,
                stored_at=stored_at_value,
                frame_reference=frame_reference,
                detection_reference=detection_reference,
            )
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stored_at, reverse=True)
    return candidates[0]


class SingleVerificationRequestSource:
    def __init__(
        self,
        *,
        evidence_root_dir: Path,
        radar_state_path: Path,
        run_started_at_epoch: float,
    ) -> None:
        self._evidence_root_dir = evidence_root_dir
        self._radar_state_path = radar_state_path
        self._run_started_at_epoch = run_started_at_epoch
        self._submitted = False
        self.selected_candidate: EvidenceCandidate | None = None
        self.submitted_request: VerificationQueueRequest | None = None

    def __call__(self) -> list[VerificationQueueRequest]:
        if self._submitted:
            return []

        candidate = select_current_run_evidence_candidate(
            evidence_root_dir=self._evidence_root_dir,
            radar_state_path=self._radar_state_path,
            run_started_at_epoch=self._run_started_at_epoch,
            now_epoch=time.time(),
        )
        if candidate is None:
            return []

        request = build_request_from_message_id(
            candidate.track_id,
            candidate.message_id,
            source="external_source",
        )
        self.selected_candidate = candidate
        self.submitted_request = request
        self._submitted = True
        return [request]


def _find_submission_record(records: list[dict[str, Any]], request_id: str) -> dict[str, Any] | None:
    for item in records:
        if isinstance(item, dict) and item.get("request_id") == request_id:
            return item
    return None


def evaluate_boundaries(
    *,
    run_dir: Path,
    selected_candidate: EvidenceCandidate | None,
    submitted_request: VerificationQueueRequest | None,
) -> dict[str, Any]:
    request_id = submitted_request.request_id if submitted_request is not None else None
    message_id = selected_candidate.message_id if selected_candidate is not None else None
    track_id = selected_candidate.track_id if selected_candidate is not None else None

    submissions_payload = _load_json(run_dir / "latest_verification_queue_submissions.json")
    submissions = submissions_payload if isinstance(submissions_payload, list) else []
    submission_record = _find_submission_record(submissions, request_id) if request_id else None

    results_payload = _load_json(run_dir / "latest_verification_queue_results.json")
    results = results_payload if isinstance(results_payload, list) else []
    result_record = _find_submission_record(results, request_id) if request_id else None

    feedback_payload = _load_json(run_dir / "operator_tracking_feedback.json")
    feedback_records_payload = feedback_payload.get("records") if isinstance(feedback_payload, dict) else None
    feedback_records = feedback_records_payload if isinstance(feedback_records_payload, list) else []

    feedback_record = None
    if request_id is not None:
        expected_feedback_id = f"obsfeedback-{request_id}"
        for item in feedback_records:
            if isinstance(item, dict) and item.get("feedback_id") == expected_feedback_id:
                feedback_record = item
                break

    diagnostics_payload = _load_json(run_dir / "latest_tracking_estimate_diagnostics.json")
    estimates = []
    if isinstance(diagnostics_payload, dict):
        current_valid = diagnostics_payload.get("current_valid_estimates")
        if isinstance(current_valid, list):
            estimates = [item for item in current_valid if isinstance(item, dict)]

    visual_success = False
    visual_reason = "result_record_missing"
    verification_observation_id = None
    if isinstance(result_record, dict):
        details = result_record.get("details") if isinstance(result_record.get("details"), dict) else {}
        verification_observation_id = details.get("verification_observation_id")
        referenced_message_id = details.get("verification_referenced_message_id")
        visual_success = (
            result_record.get("status") == "completed"
            and result_record.get("validation_outcome") == "accepted"
            and bool(result_record.get("coordinator_valid", False))
            and isinstance(verification_observation_id, str)
            and bool(verification_observation_id)
            and isinstance(referenced_message_id, str)
            and referenced_message_id == message_id
        )
        visual_reason = "ok" if visual_success else "result_record_not_accepted"

    publication_success = False
    publication_reason = "feedback_record_missing"
    if isinstance(feedback_record, dict):
        publication_success = (
            feedback_record.get("verification_request_id") == request_id
            and feedback_record.get("track_id") == track_id
            and feedback_record.get("evidence_message_id") == message_id
        )
        publication_reason = "ok" if publication_success else "feedback_record_mismatch"

    consumption_success = False
    consumption_reason = "tracking_diagnostics_missing"
    if isinstance(verification_observation_id, str) and verification_observation_id:
        for estimate in estimates:
            source_observation_id = estimate.get("source_verification_observation_id")
            source_tracking_message_ids = estimate.get("source_tracking_evidence_ids")
            if source_observation_id != verification_observation_id:
                continue
            if isinstance(source_tracking_message_ids, list) and message_id in source_tracking_message_ids:
                consumption_success = True
                consumption_reason = "ok"
                break
        if not consumption_success and consumption_reason != "ok":
            consumption_reason = (
                "likely_stale_or_mismatched_latest_message"
                if publication_success
                else "verification_not_consumed"
            )

    return {
        "schema_version": "1.0",
        "generated_at": time.time(),
        "selected": {
            "track_id": track_id,
            "message_id": message_id,
            "request_id": request_id,
            "verification_observation_id": verification_observation_id,
        },
        "submission": {
            "found": submission_record is not None,
            "record": submission_record,
        },
        "visual_verification": {
            "success": visual_success,
            "reason": visual_reason,
            "record": result_record,
        },
        "feedback_publication": {
            "success": publication_success,
            "reason": publication_reason,
            "record": feedback_record,
        },
        "feedback_consumption": {
            "success": consumption_success,
            "reason": consumption_reason,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


async def run_adapter() -> int:
    run_dir = _resolve_run_dir()
    run_started_at_epoch = time.time()
    bounded_session_seconds = _parse_positive_float(os.getenv("BOUNDED_SESSION_SECONDS"), 60.0)
    verification_drain_seconds = _parse_positive_float(os.getenv("VERIFICATION_DRAIN_SECONDS"), 20.0)

    evidence_root_dir = runtime_main.PROJECT_ROOT / "captures" / "tracking_evidence"
    radar_state_path = runtime_main.PROJECT_ROOT / "captures" / "latest_radar_state.json"
    request_source = SingleVerificationRequestSource(
        evidence_root_dir=evidence_root_dir,
        radar_state_path=radar_state_path,
        run_started_at_epoch=run_started_at_epoch,
    )

    config = runtime_main.load_config(["--disable-agent-ops-heartbeat"])
    logger = runtime_app.configure_logging(config)
    services = await runtime_app.initialize_services(
        config,
        logger,
        server_path=runtime_main.SERVER_PATH,
        python_executable=sys.executable,
        stdio_client_fn=stdio_client,
        client_session_cls=ClientSession,
        startup_runner=runtime_main.initialize_runtime_services,
    )

    monitor_task: asyncio.Task[Any] | None = None
    terminal_result_at: float | None = None
    try:
        await services.start()

        monitor_task = asyncio.create_task(
            runtime_main.monitor_movement(
                services.session,
                enable_agent_ops_heartbeat=config.enable_agent_ops_heartbeat,
                agent_ops_interval_seconds=config.agent_ops_interval_seconds,
                agent_ops_max_tracks=config.agent_ops_max_tracks,
                agent_ops_summary_every_cycles=config.agent_ops_summary_every_cycles,
                agent_ops_summary_window_size=config.agent_ops_summary_window_size,
                handoff_degraded_warn_rate=config.handoff_degraded_warn_rate,
                handoff_degraded_critical_rate=config.handoff_degraded_critical_rate,
                depth_coverage_summary_enabled=config.depth_coverage_summary_enabled,
                depth_coverage_summary_every_cycles=config.depth_coverage_summary_every_cycles,
                depth_coverage_sample_limit=config.depth_coverage_sample_limit,
                verification_request_source_fn=request_source,
            )
        )

        submission_deadline = time.monotonic() + bounded_session_seconds
        while time.monotonic() < submission_deadline:
            await asyncio.sleep(0.25)
            if monitor_task.done():
                break

            if request_source.submitted_request is None:
                continue

            results_payload = _load_json(run_dir / "latest_verification_queue_results.json")
            if not isinstance(results_payload, list):
                continue
            for item in results_payload:
                if not isinstance(item, dict):
                    continue
                if item.get("request_id") != request_source.submitted_request.request_id:
                    continue
                if item.get("status") in {"completed", "failed", "rejected", "cancelled"}:
                    terminal_result_at = time.monotonic()
                    break
            if terminal_result_at is not None:
                break

        if terminal_result_at is not None:
            drain_deadline = min(
                time.monotonic() + verification_drain_seconds,
                submission_deadline + verification_drain_seconds,
            )
            while time.monotonic() < drain_deadline and monitor_task is not None and not monitor_task.done():
                await asyncio.sleep(0.25)

    finally:
        if monitor_task is not None and not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        await services.stop()

    evaluation = evaluate_boundaries(
        run_dir=run_dir,
        selected_candidate=request_source.selected_candidate,
        submitted_request=request_source.submitted_request,
    )

    _write_json(run_dir / "adapter_selected_ids.json", evaluation.get("selected", {}))
    _write_json(run_dir / "adapter_non_reasoning_verification_summary.json", evaluation)

    visual_success = bool(evaluation.get("visual_verification", {}).get("success", False))
    publication_success = bool(evaluation.get("feedback_publication", {}).get("success", False))
    if visual_success and publication_success:
        return 0
    return 2


async def _main_async() -> int:
    parser = argparse.ArgumentParser(description="Test-only non-reasoning live verification adapter")
    parser.parse_args()
    return await run_adapter()


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
