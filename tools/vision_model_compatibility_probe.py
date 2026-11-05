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

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import llm_call
from agents.tracking.agent_tracking import DEFAULT_OPENAI_BASE_URL
from tools.tracking_agent_latency_diagnostic import FIXED_INLINE_IMAGE_DATA_URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_EVENTS_PATH = PROJECT_ROOT / "captures" / "latency_events.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "captures" / "diagnostics" / "vision_model_compatibility_report.json"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "captures" / "diagnostics" / "vision_model_compatibility_events.jsonl"

DEFAULT_MODELS = ["qwen-3.8-27b", "muse-glimmer"]
DEFAULT_PROMPT = "Describe what you see in this image in one sentence."


@dataclass(frozen=True, slots=True)
class ProbeResult:
    model: str
    request_index: int
    wall_latency_ms: float
    api_success: bool
    timeout: bool
    response_valid: bool
    finish_reason: str | None
    response_text: str | None
    response_preview: str | None
    token_usage: dict[str, int] | None
    sdk_exception_class: str | None
    http_status_code: int | None
    provider_error_code: str | None


def _sanitize_provider_error_code(raw_code: object) -> str | None:
    if not isinstance(raw_code, str):
        return None
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789._:-"
    lowered = raw_code.strip().lower()
    if not lowered:
        return None
    normalized = "".join(ch if ch in allowed else "_" for ch in lowered).strip("_")
    if not normalized:
        return None
    return normalized[:64]


def _response_preview(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 200:
        return compact
    return compact[:200] + "..."


def _looks_like_image_understanding(text: str) -> bool:
    lowered = text.lower()
    refusal_markers = [
        "i can't view",
        "i cannot view",
        "i can't see",
        "i cannot see",
        "unable to view",
        "unable to see",
        "cannot access image",
        "can't access image",
    ]
    if any(marker in lowered for marker in refusal_markers):
        return False
    return bool(text.strip())


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        return


def _build_client() -> tuple[AsyncOpenAI, str]:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    return AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0), base_url


async def _probe_model(
    *,
    client: AsyncOpenAI,
    model: str,
    timeout_seconds: float,
    prompt: str,
    image_data_url: str,
    events_path: Path,
) -> ProbeResult:
    started = time.perf_counter()
    request_index = 1

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    _append_jsonl(
        events_path,
        {
            "event_type": "vision_probe_started",
            "timestamp": time.time(),
            "model": model,
            "timeout_seconds": timeout_seconds,
            "request_index": request_index,
            "retry_limit": 0,
            "image_reference_present": True,
        },
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        _append_jsonl(
            events_path,
            {
                "event_type": "vision_probe_completed",
                "timestamp": time.time(),
                "model": model,
                "timeout_seconds": timeout_seconds,
                "request_index": request_index,
                "status": "error",
                "error_code": "timeout",
                "duration_ms": latency_ms,
            },
        )
        return ProbeResult(
            model=model,
            request_index=request_index,
            wall_latency_ms=latency_ms,
            api_success=False,
            timeout=True,
            response_valid=False,
            finish_reason=None,
            response_text=None,
            response_preview=None,
            token_usage=None,
            sdk_exception_class="TimeoutError",
            http_status_code=None,
            provider_error_code=None,
        )
    except Exception as error:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        status_code = getattr(error, "status_code", None)
        provider_error_code = _sanitize_provider_error_code(getattr(error, "code", None))
        _append_jsonl(
            events_path,
            {
                "event_type": "vision_probe_completed",
                "timestamp": time.time(),
                "model": model,
                "timeout_seconds": timeout_seconds,
                "request_index": request_index,
                "status": "error",
                "error_code": "runtime_error",
                "duration_ms": latency_ms,
                "sdk_exception_class": type(error).__name__,
                "http_status_code": status_code if isinstance(status_code, int) else None,
                "provider_error_code": provider_error_code,
            },
        )
        return ProbeResult(
            model=model,
            request_index=request_index,
            wall_latency_ms=latency_ms,
            api_success=False,
            timeout=False,
            response_valid=False,
            finish_reason=None,
            response_text=None,
            response_preview=None,
            token_usage=None,
            sdk_exception_class=type(error).__name__,
            http_status_code=status_code if isinstance(status_code, int) else None,
            provider_error_code=provider_error_code,
        )

    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    finish_reason = None
    content_text = ""
    if getattr(response, "choices", None):
        first_choice = response.choices[0]
        finish_reason = getattr(first_choice, "finish_reason", None)
        content_text = getattr(getattr(first_choice, "message", None), "content", "") or ""

    usage = llm_call.extract_usage_dict(response)
    response_valid = bool(content_text.strip()) and finish_reason != "length"

    _append_jsonl(
        events_path,
        {
            "event_type": "vision_probe_completed",
            "timestamp": time.time(),
            "model": model,
            "timeout_seconds": timeout_seconds,
            "request_index": request_index,
            "status": "ok",
            "duration_ms": latency_ms,
            "finish_reason": finish_reason,
            "usage": usage,
            "response_valid": response_valid,
            "image_understanding_signal": _looks_like_image_understanding(content_text),
        },
    )

    return ProbeResult(
        model=model,
        request_index=request_index,
        wall_latency_ms=latency_ms,
        api_success=True,
        timeout=False,
        response_valid=response_valid,
        finish_reason=finish_reason,
        response_text=content_text if content_text else None,
        response_preview=_response_preview(content_text) if content_text else None,
        token_usage=usage,
        sdk_exception_class=None,
        http_status_code=None,
        provider_error_code=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image compatibility probe across two models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to probe with one request each.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-total-requests",
        type=int,
        default=2,
        help="Strict cap on total requests.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Isolated report output path.",
    )
    parser.add_argument(
        "--events-jsonl",
        type=Path,
        default=DEFAULT_EVENTS_PATH,
        help="Isolated events JSONL path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="User text prompt paired with image input.",
    )
    return parser.parse_args()


async def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")

    models: list[str] = [str(item).strip() for item in args.models if str(item).strip()]
    if not models:
        raise ValueError("At least one model is required")

    total_requests = len(models)
    if total_requests > args.max_total_requests:
        raise ValueError(
            f"Requested {total_requests} requests exceeds --max-total-requests={args.max_total_requests}"
        )

    output_path = args.output_json.expanduser().resolve()
    events_path = args.events_jsonl.expanduser().resolve()
    if output_path == events_path:
        raise ValueError("--output-json and --events-jsonl must differ")
    if output_path == PRODUCTION_EVENTS_PATH.resolve() or events_path == PRODUCTION_EVENTS_PATH.resolve():
        raise ValueError("Output paths must not target production latency events path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("", encoding="utf-8")

    image_sha = hashlib.sha256(FIXED_INLINE_IMAGE_DATA_URL.encode("utf-8")).hexdigest()

    client, base_url = _build_client()
    started_at = time.time()
    results: list[ProbeResult] = []
    for model in models:
        item = await _probe_model(
            client=client,
            model=model,
            timeout_seconds=float(args.timeout_seconds),
            prompt=args.prompt,
            image_data_url=FIXED_INLINE_IMAGE_DATA_URL,
            events_path=events_path,
        )
        results.append(item)
    ended_at = time.time()

    report = {
        "meta": {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(ended_at - started_at, 3),
            "script": "tools/vision_model_compatibility_probe.py",
            "read_only": True,
            "production_settings_changed": False,
            "live_camera_used": False,
            "total_requests_expected": total_requests,
            "automatic_retries_disabled": True,
        },
        "request_config": {
            "base_url": base_url,
            "timeout_seconds": float(args.timeout_seconds),
            "prompt": args.prompt,
            "request_format": "openai_chat_completions_image_url",
            "image_reference": {
                "kind": "fixed_embedded_data_url",
                "sha256_prefix": image_sha[:16],
                "data_url_logged": False,
            },
        },
        "logging": {
            "events_path": str(events_path),
            "events_isolated_from_production_default": str(events_path) != str(PRODUCTION_EVENTS_PATH.resolve()),
        },
        "results": [
            {
                "model": item.model,
                "request_index": item.request_index,
                "wall_latency_ms": item.wall_latency_ms,
                "api_success": item.api_success,
                "timeout": item.timeout,
                "response_valid": item.response_valid,
                "finish_reason": item.finish_reason,
                "response_text": item.response_text,
                "response_preview": item.response_preview,
                "token_usage": item.token_usage,
                "sdk_exception_class": item.sdk_exception_class,
                "http_status_code": item.http_status_code,
                "provider_error_code": item.provider_error_code,
                "image_understanding_signal": (
                    _looks_like_image_understanding(item.response_preview or "")
                    if item.api_success and item.response_preview
                    else False
                ),
            }
            for item in results
        ],
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    report = asyncio.run(_run_probe(args))
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "events_jsonl": str(args.events_jsonl),
                "total_requests_expected": report["meta"]["total_requests_expected"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
