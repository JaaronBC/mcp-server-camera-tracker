from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv

from agents.contracts import BoundingBox, DetectionCandidate
from agents.tracking.agent_tracking import (
    DEFAULT_OPENAI_BASE_URL,
    LLMVisualTrackingAgent,
    TRACKING_VLM_MODEL,
    encode_frame_as_data_url,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "captures" / "diagnostics" / "tracking_agent_latency_report.json"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "captures" / "diagnostics" / "tracking_agent_latency_events.jsonl"
PRODUCTION_EVENTS_PATH = PROJECT_ROOT / "captures" / "latency_events.jsonl"

FIXED_TIMESTAMP = 1735689600.0
FIXED_EVENT_REASONS = ["diagnostic_latency_probe", "fixed_representative_input"]


def _build_fixed_inline_image_data_url() -> str:
    frame = np.full((96, 128, 3), 245, dtype=np.uint8)
    cv2.rectangle(frame, (8, 8), (52, 52), (0, 0, 255), thickness=-1)
    cv2.circle(frame, (96, 28), 20, (0, 255, 0), thickness=-1)
    triangle = np.array([[28, 88], [64, 60], [88, 88]], dtype=np.int32)
    cv2.fillConvexPoly(frame, triangle, (255, 0, 0))
    return encode_frame_as_data_url(frame)


FIXED_INLINE_IMAGE_DATA_URL = _build_fixed_inline_image_data_url()


@dataclass(frozen=True, slots=True)
class DiagnosticCase:
    name: str
    timeout_seconds: float
    include_image: bool
    request_count: int


def _build_fixed_detection_candidates() -> list[DetectionCandidate]:
    return [
        DetectionCandidate(
            detection_id="diag-det-1",
            timestamp=FIXED_TIMESTAMP,
            yolo_track_id="yolo-11",
            label="person",
            detection_confidence=0.93,
            pixel_bbox=BoundingBox(x1=230.0, y1=120.0, x2=380.0, y2=455.0),
            pixel_center_x=305.0,
            pixel_center_y=455.0,
            relative_depth=0.31,
            ground_x=1.2,
            ground_z=2.9,
            ground_coordinate_unit="relative_units",
            position_kind="relative_ground_estimate",
            motion_history=[
                {"timestamp": FIXED_TIMESTAMP - 0.2, "ground_x": 1.17, "ground_z": 2.96},
                {"timestamp": FIXED_TIMESTAMP - 0.1, "ground_x": 1.19, "ground_z": 2.93},
                {"timestamp": FIXED_TIMESTAMP, "ground_x": 1.2, "ground_z": 2.9},
            ],
        ),
        DetectionCandidate(
            detection_id="diag-det-2",
            timestamp=FIXED_TIMESTAMP,
            yolo_track_id="yolo-7",
            label="person",
            detection_confidence=0.88,
            pixel_bbox=BoundingBox(x1=500.0, y1=150.0, x2=620.0, y2=430.0),
            pixel_center_x=560.0,
            pixel_center_y=430.0,
            relative_depth=0.44,
            ground_x=2.4,
            ground_z=3.4,
            ground_coordinate_unit="relative_units",
            position_kind="relative_ground_estimate",
            motion_history=[
                {"timestamp": FIXED_TIMESTAMP - 0.2, "ground_x": 2.35, "ground_z": 3.47},
                {"timestamp": FIXED_TIMESTAMP - 0.1, "ground_x": 2.38, "ground_z": 3.43},
                {"timestamp": FIXED_TIMESTAMP, "ground_x": 2.4, "ground_z": 3.4},
            ],
        ),
    ]


def _build_cases(requests_per_case: int) -> list[DiagnosticCase]:
    return [
        DiagnosticCase(
            name="structured_only_timeout_10s",
            timeout_seconds=10.0,
            include_image=False,
            request_count=requests_per_case,
        ),
        DiagnosticCase(
            name="structured_plus_image_timeout_10s",
            timeout_seconds=10.0,
            include_image=True,
            request_count=requests_per_case,
        ),
        DiagnosticCase(
            name="structured_plus_image_timeout_3s",
            timeout_seconds=3.0,
            include_image=True,
            request_count=requests_per_case,
        ),
        DiagnosticCase(
            name="structured_plus_image_timeout_1s",
            timeout_seconds=1.0,
            include_image=True,
            request_count=requests_per_case,
        ),
    ]


def _filter_cases(cases: list[DiagnosticCase], case_name: str | None) -> list[DiagnosticCase]:
    if case_name is None:
        return cases
    selected = [item for item in cases if item.name == case_name]
    if not selected:
        raise ValueError(f"Unknown case name: {case_name}")
    return selected


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _latest_event_for_correlation(
    *,
    records: list[dict[str, Any]],
    correlation_id: str | None,
    event_type: str,
) -> dict[str, Any] | None:
    if not correlation_id:
        return None
    for record in reversed(records):
        if record.get("event_type") != event_type:
            continue
        if record.get("correlation_id") != correlation_id:
            continue
        return record
    return None


def _get_usage_from_event(event: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(event, dict):
        return None
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            normalized[key] = value
    return normalized or None


def _estimate_cost_usd(
    *,
    usage: dict[str, int] | None,
    input_cost_per_million_tokens: float | None,
    output_cost_per_million_tokens: float | None,
) -> float | None:
    if usage is None:
        return None
    if input_cost_per_million_tokens is None or output_cost_per_million_tokens is None:
        return None

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None

    estimated = (
        (prompt_tokens / 1_000_000.0) * input_cost_per_million_tokens
        + (completion_tokens / 1_000_000.0) * output_cost_per_million_tokens
    )
    return round(estimated, 8)


def _is_timeout(result_error: str | None, uncertainty_flags: list[str]) -> bool:
    return result_error == "vlm_timeout" or "vlm_timeout" in uncertainty_flags


def _is_truncation(result_error: str | None, uncertainty_flags: list[str]) -> bool:
    return result_error == "vlm_output_truncated" or "vlm_output_truncated" in uncertainty_flags


def _is_invalid_json(result_error: str | None, uncertainty_flags: list[str]) -> bool:
    return result_error == "vlm_invalid_json_response" or "vlm_invalid_json_response" in uncertainty_flags


async def _run_single_case(
    *,
    case: DiagnosticCase,
    agent: LLMVisualTrackingAgent,
    detections: list[DetectionCandidate],
    image_data_url: str | None,
    events_path: Path,
    input_cost_per_million_tokens: float | None,
    output_cost_per_million_tokens: float | None,
) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    for index in range(case.request_count):
        started = time.perf_counter()
        result = await agent.propose_associations(
            frame_reference=image_data_url,
            detections=detections,
            event_reasons=FIXED_EVENT_REASONS,
            request_timeout_seconds=case.timeout_seconds,
        )
        wall_latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        records = _load_jsonl(events_path)
        completed_event = _latest_event_for_correlation(
            records=records,
            correlation_id=result.llm_correlation_id,
            event_type="llm_request_completed",
        )

        usage = _get_usage_from_event(completed_event)
        estimated_cost = _estimate_cost_usd(
            usage=usage,
            input_cost_per_million_tokens=input_cost_per_million_tokens,
            output_cost_per_million_tokens=output_cost_per_million_tokens,
        )

        uncertainty_flags = list(result.uncertainty_flags)
        timeout_flag = _is_timeout(result.error, uncertainty_flags)
        truncation_flag = _is_truncation(result.error, uncertainty_flags)
        invalid_json_flag = _is_invalid_json(result.error, uncertainty_flags)
        schema_complete = "schema_complete_response" in uncertainty_flags
        schema_incomplete = "schema_incomplete_response" in uncertainty_flags

        requests.append(
            {
                "request_index": index + 1,
                "wall_latency_ms": wall_latency_ms,
                "agent_reported_latency_ms": None if result.latency_ms is None else round(float(result.latency_ms), 3),
                "api_success": bool(result.available),
                "timeout": timeout_flag,
                "truncation": truncation_flag,
                "invalid_json": invalid_json_flag,
                "schema_complete": schema_complete,
                "schema_incomplete": schema_incomplete,
                "proposal_count": len(result.proposals),
                "error": result.error,
                "uncertainty_flags": uncertainty_flags,
                "llm_correlation_id": result.llm_correlation_id,
                "token_usage": usage,
                "estimated_cost_usd": estimated_cost,
                "event_status": completed_event.get("status") if isinstance(completed_event, dict) else None,
                "event_error_code": completed_event.get("error_code") if isinstance(completed_event, dict) else None,
                "event_sdk_exception_class": completed_event.get("sdk_exception_class") if isinstance(completed_event, dict) else None,
                "event_http_status_code": completed_event.get("http_status_code") if isinstance(completed_event, dict) else None,
                "event_provider_error_code": completed_event.get("provider_error_code") if isinstance(completed_event, dict) else None,
            }
        )

    wall_latencies = [item["wall_latency_ms"] for item in requests if isinstance(item.get("wall_latency_ms"), (int, float))]
    summary = {
        "request_count": len(requests),
        "api_success_count": sum(1 for item in requests if item["api_success"]),
        "timeout_count": sum(1 for item in requests if item["timeout"]),
        "truncation_count": sum(1 for item in requests if item["truncation"]),
        "invalid_json_count": sum(1 for item in requests if item["invalid_json"]),
        "schema_complete_count": sum(1 for item in requests if item["schema_complete"]),
        "schema_incomplete_count": sum(1 for item in requests if item["schema_incomplete"]),
        "wall_latency_ms": {
            "min": min(wall_latencies) if wall_latencies else None,
            "max": max(wall_latencies) if wall_latencies else None,
            "mean": round(sum(wall_latencies) / len(wall_latencies), 3) if wall_latencies else None,
        },
        "estimated_cost_usd_total": round(
            sum(item["estimated_cost_usd"] for item in requests if isinstance(item.get("estimated_cost_usd"), (int, float))),
            8,
        ),
    }

    return {
        "case_name": case.name,
        "timeout_seconds": case.timeout_seconds,
        "include_image": case.include_image,
        "requests": requests,
        "summary": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated tracking-agent latency diagnostic runner.")
    parser.add_argument(
        "--requests-per-case",
        type=int,
        default=3,
        help="Number of requests per diagnostic case (must be >=1).",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Optional single case selector (e.g. structured_plus_image_timeout_10s).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for diagnostic report output JSON.",
    )
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        default=DEFAULT_EVENTS_PATH,
        help="Path for isolated diagnostic latency events JSONL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TRACKING_VLM_MODEL,
        help="Model name to pass to LLMVisualTrackingAgent.",
    )
    parser.add_argument(
        "--input-cost-per-million-tokens",
        type=float,
        default=None,
        help="Assumed USD cost per 1M prompt tokens (optional; required for cost estimate).",
    )
    parser.add_argument(
        "--output-cost-per-million-tokens",
        type=float,
        default=None,
        help="Assumed USD cost per 1M completion tokens (optional; required for cost estimate).",
    )
    parser.add_argument(
        "--max-total-requests",
        type=int,
        default=12,
        help="Hard cap on total model requests across all cases.",
    )
    return parser.parse_args()


def _build_agent_with_retries_disabled(model: str) -> tuple[LLMVisualTrackingAgent, str]:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")

    resolved_base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    client = AsyncOpenAI(
        base_url=resolved_base_url,
        api_key=api_key,
        max_retries=0,
    )
    agent = LLMVisualTrackingAgent(
        model=model,
        openai_base_url=resolved_base_url,
        client=client,
    )
    return agent, resolved_base_url


async def _run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    if args.requests_per_case < 1:
        raise ValueError("--requests-per-case must be >= 1")

    output_path: Path = args.output_json.expanduser().resolve()
    events_path: Path = args.events_jsonl.expanduser().resolve()
    if output_path == events_path:
        raise ValueError("--output-json and --events-jsonl must be different paths")
    if events_path == PRODUCTION_EVENTS_PATH.resolve():
        raise ValueError("--events-jsonl must not target production latency events path")
    if output_path == PRODUCTION_EVENTS_PATH.resolve():
        raise ValueError("--output-json must not target production latency events path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["PHASE16E_LATENCY_INSTRUMENTATION_ENABLED"] = "1"
    os.environ["PHASE16E_LATENCY_EVENTS_PATH"] = str(events_path)

    detections = _build_fixed_detection_candidates()
    cases = _filter_cases(_build_cases(args.requests_per_case), args.case_name)
    total_requests = sum(case.request_count for case in cases)
    if total_requests > args.max_total_requests:
        raise ValueError(
            f"Requested {total_requests} requests exceeds --max-total-requests={args.max_total_requests}"
        )
    image_sha = hashlib.sha256(FIXED_INLINE_IMAGE_DATA_URL.encode("utf-8")).hexdigest()

    agent, resolved_base_url = _build_agent_with_retries_disabled(args.model)

    started_at = time.time()
    case_results: list[dict[str, Any]] = []
    for case in cases:
        frame_reference = FIXED_INLINE_IMAGE_DATA_URL if case.include_image else None
        case_result = await _run_single_case(
            case=case,
            agent=agent,
            detections=detections,
            image_data_url=frame_reference,
            events_path=events_path,
            input_cost_per_million_tokens=args.input_cost_per_million_tokens,
            output_cost_per_million_tokens=args.output_cost_per_million_tokens,
        )
        case_results.append(case_result)

    ended_at = time.time()
    report = {
        "meta": {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(ended_at - started_at, 3),
            "script": "tools/tracking_agent_latency_diagnostic.py",
            "read_only": True,
            "production_settings_changed": False,
            "live_camera_used": False,
            "fixed_input_used": True,
            "total_requests_expected": total_requests,
        },
        "equivalence": {
            "entrypoint": "LLMVisualTrackingAgent.propose_associations",
            "same_model_constant_as_production": True,
            "same_prompt_construction": True,
            "same_openai_compatible_api_call": True,
            "same_timeout_wrapper_asyncio_wait_for": True,
            "same_json_parser": True,
            "resolved_model": args.model,
            "resolved_base_url": resolved_base_url,
            "automatic_retries_disabled_for_budget_control": True,
            "notes": [
                "Diagnostic bypasses live frame capture and dashboard runtime orchestration.",
                "Diagnostic does not submit or apply proposals to live tracking state.",
                "Diagnostic uses fixed representative DetectionCandidate payload and fixed inline image sample.",
            ],
        },
        "inputs": {
            "event_reasons": FIXED_EVENT_REASONS,
            "detection_count": len(detections),
            "detection_ids": [item.detection_id for item in detections],
            "image_reference": {
                "included_in_image_cases": True,
                "kind": "fixed_embedded_data_url",
                "sha256_prefix": image_sha[:16],
                "data_url_logged": False,
            },
        },
        "logging": {
            "events_path": str(events_path),
            "events_isolated_from_production_default": str(events_path) != str(
                (PROJECT_ROOT / "captures" / "latency_events.jsonl").resolve()
            ),
            "instrumentation_enabled": True,
        },
        "cost_assumptions": {
            "input_cost_per_million_tokens_usd": args.input_cost_per_million_tokens,
            "output_cost_per_million_tokens_usd": args.output_cost_per_million_tokens,
            "estimation_rule": "cost = prompt_tokens/1e6*input_rate + completion_tokens/1e6*output_rate",
            "token_usage_source": "llm_request_completed usage field when present",
            "prices_are_estimates_only": True,
        },
        "cases": case_results,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    report = asyncio.run(_run_diagnostic(args))
    print(json.dumps({
        "output_json": str(args.output_json),
        "events_jsonl": str(args.events_jsonl),
        "total_requests_expected": report["meta"]["total_requests_expected"],
    }, indent=2))


if __name__ == "__main__":
    main()
