from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from openai import AsyncOpenAI

DEFAULT_OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
DEFAULT_REASONING_AGENT_MODEL = "gpt-oss-120b"
DEFAULT_LLM_TIMEOUT_SECONDS = 8.0
DEFAULT_LLM_RETRY_LIMIT = 1
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 0.1
DEFAULT_REASONING_SYSTEM_PROMPT = (
    "You are the Reasoning/Decision-Support Agent. Synthesize observation summaries and evidence "
    "into an explainable operator-facing assessment. Do not infer hostile intent from movement alone. "
    "Return strict JSON with keys: assessment_label, summary, rationale, recommendation, "
    "supporting_observation_ids, uncertainties, limitations, alternative_interpretations, "
    "missing_evidence, contradictory_evidence, request_more_observations."
)


def _resolve_model(env_name: str = "REASONING_AGENT_MODEL", default_model: str = DEFAULT_REASONING_AGENT_MODEL) -> str:
    value = os.getenv(env_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default_model


def _resolve_timeout_seconds() -> float:
    value = os.getenv("DASHBOARD_AGENT_LLM_TIMEOUT_SECONDS")
    if value is None:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    if timeout <= 0:
        return DEFAULT_LLM_TIMEOUT_SECONDS
    return timeout


def _resolve_retry_limit() -> int:
    value = os.getenv("DASHBOARD_AGENT_LLM_RETRY_LIMIT")
    if value is None:
        return DEFAULT_LLM_RETRY_LIMIT
    try:
        retry_limit = int(value)
    except ValueError:
        return DEFAULT_LLM_RETRY_LIMIT
    if retry_limit < 0:
        return DEFAULT_LLM_RETRY_LIMIT
    return retry_limit


def _resolve_retry_backoff_seconds() -> float:
    value = os.getenv("DASHBOARD_AGENT_LLM_RETRY_BACKOFF_SECONDS")
    if value is None:
        return DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    try:
        backoff_seconds = float(value)
    except ValueError:
        return DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    if backoff_seconds < 0:
        return DEFAULT_LLM_RETRY_BACKOFF_SECONDS
    return backoff_seconds


def _classify_llm_error(error: Exception) -> str:
    text = f"{type(error).__name__}:{error}".lower()
    if "timeout" in text:
        return "timeout"
    if "api_key" in text or "unauthorized" in text or "authentication" in text:
        return "authentication"
    if "llm_invalid_json_response" in text or "json" in text:
        return "invalid_response_format"
    if "llm_empty_response" in text or "empty_response" in text:
        return "empty_response"
    return "runtime_error"


def _create_openai_client() -> AsyncOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    if text.startswith("```"):
        blocks = text.split("```")
        text = ""
        for block in blocks:
            candidate = block.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
        if not text:
            return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def validate_reasoning_llm_payload(
    payload: dict[str, Any],
    track_id: str,
    created_at: float,
    valid_observation_ids: set[str],
) -> tuple[dict[str, Any], bool]:
    label = _coerce_str(payload.get("assessment_label"), "requires_more_observation")
    if label not in {"stable_motion", "requires_review", "requires_more_observation"}:
        label = "requires_more_observation"

    supporting_ids = [
        item for item in _coerce_str_list(payload.get("supporting_observation_ids"))
        if item in valid_observation_ids
    ]

    request_more_observations = _coerce_bool(
        payload.get("request_more_observations"),
        default=(label == "requires_more_observation"),
    )

    assessment = {
        "track_id": track_id,
        "created_at": created_at,
        "assessment_label": label,
        "summary": _coerce_str(payload.get("summary"), "Reasoning agent produced no summary."),
        "rationale": _coerce_str(payload.get("rationale"), "No rationale provided."),
        "recommendation": _coerce_str(payload.get("recommendation"), "requires_operator_review"),
        "supporting_observation_ids": supporting_ids,
        "uncertainties": _coerce_str_list(payload.get("uncertainties")),
        "limitations": _coerce_str_list(payload.get("limitations")),
        "alternative_interpretations": _coerce_str_list(payload.get("alternative_interpretations")),
        "missing_evidence": _coerce_str_list(payload.get("missing_evidence")),
        "contradictory_evidence": _coerce_str_list(payload.get("contradictory_evidence")),
        "source_agent": "reasoning",
        "score": None,
        "score_method": None,
    }

    return assessment, request_more_observations


def build_reasoning_handoff_payload(
    track_id: str,
    created_at: float,
    summary: dict[str, Any],
    history_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(summary, dict):
        raise ValueError("observation_summary must be a dict")
    if not isinstance(history_records, list):
        raise ValueError("history_records must be a list")

    summary_supporting_ids = summary.get("supporting_observation_ids")
    if not isinstance(summary_supporting_ids, list):
        raise ValueError("supporting_observation_ids must be a list")

    valid_history_ids: set[str] = set()
    normalized_history_records: list[dict[str, Any]] = []
    for record in history_records:
        if not isinstance(record, dict):
            raise ValueError("history record must be an object")
        observation_id = record.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id.strip():
            raise ValueError("history record missing observation_id")
        valid_history_ids.add(observation_id)
        normalized_history_records.append(record)

    normalized_supporting_ids = [
        item
        for item in _coerce_str_list(summary_supporting_ids)
        if item in valid_history_ids
    ]
    if not normalized_supporting_ids:
        raise ValueError("no valid supporting observation ids for reasoning handoff")

    normalized_summary = dict(summary)
    normalized_summary["supporting_observation_ids"] = normalized_supporting_ids

    handoff_payload = {
        "track_id": track_id,
        "created_at": created_at,
        "observation_summary": normalized_summary,
        "history_records": normalized_history_records,
    }
    return handoff_payload, {
        "status": "ok",
        "error": None,
        "error_code": None,
    }


async def invoke_llm_json_response(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    client = _create_openai_client()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
            ),
            timeout=timeout_seconds,
        )
        if not response.choices:
            raise RuntimeError("llm_empty_response")
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            raise RuntimeError("llm_invalid_json_response")
        return parsed
    finally:
        close_method = getattr(client, "close", None)
        if callable(close_method):
            close_result = close_method()
            if asyncio.iscoroutine(close_result):
                await close_result


def call_reasoning_agent_with_retry(
    user_payload: dict[str, Any],
    *,
    invoke_fn: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    retry_limit: int | None = None,
    retry_backoff_seconds: float | None = None,
    classify_error_fn: Callable[[Exception], str] | None = None,
    system_prompt: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    invoke = invoke_fn or invoke_llm_json_response
    selected_model = model or _resolve_model()
    selected_timeout = _resolve_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    selected_retry_limit = _resolve_retry_limit() if retry_limit is None else int(retry_limit)
    selected_backoff = _resolve_retry_backoff_seconds() if retry_backoff_seconds is None else float(retry_backoff_seconds)
    classify_error = classify_error_fn or _classify_llm_error
    selected_prompt = system_prompt or DEFAULT_REASONING_SYSTEM_PROMPT

    attempts = selected_retry_limit + 1
    last_error: str | None = None
    last_error_code: str | None = None
    started = time.perf_counter()
    for attempt_index in range(attempts):
        try:
            payload = asyncio.run(
                invoke(
                    model=selected_model,
                    system_prompt=selected_prompt,
                    user_payload=user_payload,
                    timeout_seconds=selected_timeout,
                )
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            return payload, {
                "status": "ok",
                "model": selected_model,
                "latency_ms": latency_ms,
                "error": None,
                "error_code": None,
                "attempts": attempt_index + 1,
                "retry_count": attempt_index,
            }
        except Exception as error:
            last_error_code = classify_error(error)
            last_error = f"{type(error).__name__}:{error}"
            has_more_attempts = attempt_index < attempts - 1
            if has_more_attempts and selected_backoff > 0:
                time.sleep(selected_backoff * (2**attempt_index))

    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return None, {
        "status": "error",
        "model": selected_model,
        "latency_ms": latency_ms,
        "error": last_error or "unknown_error",
        "error_code": last_error_code or "runtime_error",
        "attempts": attempts,
        "retry_count": max(0, attempts - 1),
    }
