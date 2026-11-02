from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def main() -> int:
    now = time.time()

    budget_state_path = Path(os.environ["PROVIDER_CALL_BUDGET_STATE_PATH"])
    budget_log_path = Path(os.environ["PROVIDER_CALL_BUDGET_LOG_PATH"])

    state_payload = {
        "schema_version": "1.0",
        "updated_at": now,
        "limits": {
            "total_limit": int(os.getenv("PROVIDER_CALL_BUDGET_TOTAL_LIMIT", "1") or 1),
            "per_role_limits": {
                "tracking_vlm": int(os.getenv("PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT", "0") or 0),
                "observation_text": int(os.getenv("PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT", "0") or 0),
                "reasoning_text": int(os.getenv("PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT", "0") or 0),
                "observation_visual": int(os.getenv("PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT", "1") or 1),
            },
        },
        "totals": {
            "attempted": 1,
            "per_role_attempted": {
                "tracking_vlm": 0,
                "observation_text": 0,
                "reasoning_text": 0,
                "observation_visual": 1,
            },
        },
    }
    budget_state_path.parent.mkdir(parents=True, exist_ok=True)
    budget_state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    budget_state_path.with_suffix(".lock").touch(exist_ok=True)

    _append_jsonl(
        budget_log_path,
        {
            "event_type": "provider_call_attempt",
            "timestamp": now,
            "attempt_id": "attempt-mock-observation-visual",
            "role": "observation_visual",
            "outcome": "ok",
            "reserved_at": now,
            "latency_ms": 1.0,
            "usage": {"total_tokens": 1},
            "reason": None,
            "error_code": None,
            "pid": os.getpid(),
        },
    )

    captures = Path("captures")
    _append_jsonl(
        captures / "agent_events.jsonl",
        {
            "event_type": "phase16e_trigger",
            "timestamp": now,
            "track_id": "mock-track-1",
            "automatic_evidence_requested": True,
        },
    )

    queue_results_path = Path(os.environ["PROVIDER_CALL_BUDGET_STATE_PATH"]).parent / "latest_verification_queue_results.json"
    queue_results_path.write_text(
        json.dumps(
            [
                {
                    "track_id": "mock-track-1",
                    "request_id": "req-mock-1",
                    "message_id": "trkmsg-mock-1",
                    "verification_status": "supported",
                    "coordinator_acceptance": True,
                    "feedback_published": True,
                    "tracking_feedback_consumer_acceptance": True,
                    "current_not_stale": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    bounded = float(os.getenv("BOUNDED_SESSION_SECONDS", "60") or 60)
    drain = float(os.getenv("VERIFICATION_DRAIN_SECONDS", "20") or 20)
    sleep_seconds = min(0.5, max(0.1, (bounded + drain) / 320.0))
    time.sleep(sleep_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
