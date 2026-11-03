from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_FRESHNESS_SECONDS = 5.0


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if not path.is_file():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except Exception:
                continue
            if isinstance(item, dict):
                rows.append(item)
    except Exception:
        return rows
    return rows


def _is_non_sim_track_snapshot(snapshot: dict[str, Any], event_track_id: str) -> bool:
    track_id = snapshot.get("track_id")
    if not isinstance(track_id, str) or not track_id.strip() or track_id.startswith("sim-"):
        return False
    if track_id != event_track_id:
        return False

    position_kind = snapshot.get("position_kind")
    if position_kind == "simulated_ground_track":
        return False

    quality_flags = snapshot.get("quality_flags")
    if isinstance(quality_flags, list) and any(flag == "simulated_data" for flag in quality_flags):
        return False

    ground = snapshot.get("ground_position")
    if not isinstance(ground, dict):
        return False
    return isinstance(ground.get("x"), (int, float)) and isinstance(ground.get("z"), (int, float))


def _has_tracking_provenance(tracking_context: dict[str, Any], event_track_id: str) -> bool:
    if not bool(tracking_context.get("available")):
        return False
    if tracking_context.get("source_agent") != "visual_tracking":
        return False

    updates = tracking_context.get("selected_updates")
    if not isinstance(updates, list) or not updates:
        return False

    for update in updates:
        if not isinstance(update, dict):
            continue
        resolved_track_id = update.get("resolved_track_id")
        accepted = update.get("accepted") is True
        evidence = update.get("supporting_evidence")
        if (
            accepted
            and isinstance(resolved_track_id, str)
            and resolved_track_id == event_track_id
            and isinstance(evidence, list)
            and "yolo_candidate_track" in evidence
        ):
            return True
    return False


def _event_is_camera_derived(
    event: dict[str, Any],
    *,
    run_started_at: float,
    publish_detected_at: float,
    freshness_seconds: float,
) -> bool:
    event_track_id = event.get("track_id")
    event_timestamp = event.get("timestamp")
    if not isinstance(event_track_id, str) or not event_track_id.strip() or event_track_id.startswith("sim-"):
        return False
    if not isinstance(event_timestamp, (int, float)):
        return False
    if float(event_timestamp) <= float(publish_detected_at):
        return False
    if float(event_timestamp) < float(run_started_at):
        return False

    evidence_trace = event.get("evidence_trace")
    if not isinstance(evidence_trace, dict):
        return False
    snapshot = evidence_trace.get("latest_track_snapshot")
    coordinate_context = evidence_trace.get("coordinate_context")
    tracking_context = evidence_trace.get("tracking_agent_context")
    if not isinstance(snapshot, dict) or not isinstance(coordinate_context, dict) or not isinstance(tracking_context, dict):
        return False

    if not bool(snapshot.get("available")):
        return False
    if not _is_non_sim_track_snapshot(snapshot, event_track_id):
        return False

    if coordinate_context.get("coordinate_system") == "simulation_fallback":
        return False
    if coordinate_context.get("camera_status") != "connected":
        return False
    if coordinate_context.get("detector_status") != "running":
        return False

    last_obs_ts = coordinate_context.get("last_observation_timestamp")
    if not isinstance(last_obs_ts, (int, float)):
        return False
    if float(last_obs_ts) < float(publish_detected_at):
        return False
    if float(event_timestamp) - float(last_obs_ts) > float(freshness_seconds):
        return False

    if not _has_tracking_provenance(tracking_context, event_track_id):
        return False

    return True


def evaluate_camera_only_handoff_run(
    run_dir: Path,
    *,
    freshness_seconds: float = DEFAULT_FRESHNESS_SECONDS,
) -> dict[str, Any]:
    summary_path = run_dir / "camera_only_handoff_summary.json"
    events_path = run_dir / "agent_events.jsonl"

    summary = _load_json(summary_path)
    if summary is None:
        return {
            "available": False,
            "error": "summary_missing_or_invalid",
            "summary_path": str(summary_path),
        }

    evidence = summary.get("evidence") if isinstance(summary.get("evidence"), dict) else {}
    started_at = summary.get("started_at")
    publish_detected_at = evidence.get("selected_radar_detected_at")
    if not isinstance(started_at, (int, float)) or not isinstance(publish_detected_at, (int, float)):
        return {
            "available": False,
            "error": "missing_publish_timing",
            "summary_path": str(summary_path),
        }

    transitions = summary.get("transitions") if isinstance(summary.get("transitions"), dict) else {}
    if not bool(transitions.get("real_radar_publication_during_run", False)):
        return {
            "available": True,
            "classification": "INCONCLUSIVE",
            "reason": "no_genuine_real_radar_publication",
            "match_event": None,
            "summary_path": str(summary_path),
            "events_path": str(events_path),
        }

    events = _load_jsonl(events_path)
    post_publish_prod_events = [
        event
        for event in events
        if event.get("event_type") == "reasoning_cycle"
        and event.get("event_origin") == "production_monitor"
        and isinstance(event.get("timestamp"), (int, float))
        and float(event["timestamp"]) > float(publish_detected_at)
    ]

    for event in post_publish_prod_events:
        if _event_is_camera_derived(
            event,
            run_started_at=float(started_at),
            publish_detected_at=float(publish_detected_at),
            freshness_seconds=float(freshness_seconds),
        ):
            return {
                "available": True,
                "classification": "PASS",
                "reason": "qualifying_camera_derived_production_event_found",
                "match_event": event,
                "post_publish_production_event_count": len(post_publish_prod_events),
                "summary_path": str(summary_path),
                "events_path": str(events_path),
            }

    return {
        "available": True,
        "classification": "FAIL",
        "reason": "no_qualifying_camera_derived_production_event",
        "match_event": None,
        "post_publish_production_event_count": len(post_publish_prod_events),
        "summary_path": str(summary_path),
        "events_path": str(events_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate camera-only handoff run artifacts with provenance-based matching.")
    parser.add_argument("run_dir", type=Path, help="Path to run artifact directory")
    parser.add_argument("--freshness-seconds", type=float, default=DEFAULT_FRESHNESS_SECONDS)
    args = parser.parse_args()

    result = evaluate_camera_only_handoff_run(args.run_dir, freshness_seconds=args.freshness_seconds)
    print(json.dumps(result, indent=2))
