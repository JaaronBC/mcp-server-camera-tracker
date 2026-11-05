from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from .. import latency_instrumentation
from .. import llm_call
from .. import provider_call_budget

DEFAULT_OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
DEFAULT_REASONING_AGENT_MODEL = "gpt-oss-120b"
DEFAULT_LLM_TIMEOUT_SECONDS = 8.0
DEFAULT_LLM_RETRY_LIMIT = 1
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = 0.1
DEFAULT_REASONING_AGENT_MAX_OUTPUT_TOKENS = 512
DEFAULT_REASONING_SYSTEM_PROMPT = (
    "You are the Reasoning/Decision-Support Agent. Synthesize observation summaries and evidence "
    "into an explainable operator-facing assessment. Do not infer hostile intent from movement alone. "
    "Return strict JSON with keys: assessment_label, summary, rationale, recommendation, "
    "supporting_observation_ids, uncertainties, limitations, alternative_interpretations, "
    "missing_evidence, contradictory_evidence, request_more_observations. "
    "Keep assessment and rationale concise, but preserve quality and all material alternative interpretations, "
    "limitations, missing evidence, contradictory evidence, and relevant supporting observation IDs. "
    "Return JSON only, with no Markdown or unnecessary explanation."
)


def _resolve_model(env_name: str = "REASONING_AGENT_MODEL", default_model: str = DEFAULT_REASONING_AGENT_MODEL) -> str:
    return llm_call.resolve_model(env_name, default_model)


def _resolve_timeout_seconds() -> float:
    return llm_call.resolve_timeout_seconds(DEFAULT_LLM_TIMEOUT_SECONDS)


def _resolve_retry_limit() -> int:
    return llm_call.resolve_retry_limit(DEFAULT_LLM_RETRY_LIMIT)


def _resolve_retry_backoff_seconds() -> float:
    return llm_call.resolve_retry_backoff_seconds(DEFAULT_LLM_RETRY_BACKOFF_SECONDS)


def _resolve_reasoning_max_output_tokens() -> int | None:
    return llm_call.resolve_max_output_tokens(
        "REASONING_AGENT_MAX_OUTPUT_TOKENS",
        DEFAULT_REASONING_AGENT_MAX_OUTPUT_TOKENS,
    )


def _classify_llm_error(error: Exception) -> str:
    return llm_call.classify_llm_error(error)


def _create_openai_client() -> AsyncOpenAI:
    return llm_call.create_openai_client(DEFAULT_OPENAI_BASE_URL)


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    payload = llm_call.extract_json_object(raw_text)
    if isinstance(payload, dict):
        return payload
    return None


def _coerce_str_list(value: Any) -> list[str]:
    return llm_call.coerce_str_list(value)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return llm_call.coerce_bool(value, default)


def _coerce_str(value: Any, default: str) -> str:
    return llm_call.coerce_str(value, default)


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
    invoke_started = time.perf_counter()
    attempt_context = latency_instrumentation.get_llm_attempt_context() or {}
    provider_label = "openai_compatible"
    reservation = provider_call_budget.reserve_or_raise(provider_call_budget.ROLE_REASONING_TEXT)
    started = time.perf_counter()
    client = _create_openai_client()
    try:
        max_output_tokens = _resolve_reasoning_max_output_tokens()
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }
        if max_output_tokens is not None:
            request_kwargs["max_tokens"] = max_output_tokens
        response = await asyncio.wait_for(
            client.chat.completions.create(**request_kwargs),
            timeout=timeout_seconds,
        )
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="ok",
            latency_ms=latency_ms,
            usage=llm_call.extract_usage_dict(response),
        )
        if not response.choices:
            raise RuntimeError("llm_empty_response")
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise RuntimeError("llm_output_truncated")
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            raise RuntimeError("llm_invalid_json_response")
        latency_instrumentation.emit_event(
            "llm_request_completed",
            stage="reasoning_agent_llm_request",
            duration_ms=round((time.perf_counter() - invoke_started) * 1000.0, 3),
            role="reasoning_agent",
            provider=provider_label,
            model=model,
            timeout_seconds=timeout_seconds,
            attempt_number=attempt_context.get("attempt_number"),
            attempts_total=attempt_context.get("attempts_total"),
            retry_limit=attempt_context.get("retry_limit"),
            status="ok",
            usage=llm_call.extract_usage_dict(response),
        )
        return parsed
    except Exception as error:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        provider_call_budget.finalize_provider_call_attempt(
            reservation,
            outcome="error",
            latency_ms=latency_ms,
            error_code=llm_call.classify_llm_error(error),
        )
        latency_instrumentation.emit_event(
            "llm_request_completed",
            stage="reasoning_agent_llm_request",
            duration_ms=round((time.perf_counter() - invoke_started) * 1000.0, 3),
            role="reasoning_agent",
            provider=provider_label,
            model=model,
            timeout_seconds=timeout_seconds,
            attempt_number=attempt_context.get("attempt_number"),
            attempts_total=attempt_context.get("attempts_total"),
            retry_limit=attempt_context.get("retry_limit"),
            status="error",
            error_code=llm_call.classify_llm_error(error),
            error=f"{type(error).__name__}:{error}",
        )
        raise
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
        attempt_started = time.perf_counter()
        previous_attempt_context = latency_instrumentation.set_llm_attempt_context(
            {
                "attempt_number": attempt_index + 1,
                "attempts_total": attempts,
                "retry_limit": selected_retry_limit,
            }
        )
        try:
            payload = asyncio.run(
                invoke(
                    model=selected_model,
                    system_prompt=selected_prompt,
                    user_payload=user_payload,
                    timeout_seconds=selected_timeout,
                )
            )
            latency_instrumentation.emit_event(
                "llm_attempt_completed",
                stage="reasoning_agent_llm_attempt",
                duration_ms=round((time.perf_counter() - attempt_started) * 1000.0, 3),
                role="reasoning_agent",
                provider="openai_compatible",
                model=selected_model,
                timeout_seconds=selected_timeout,
                attempt_number=attempt_index + 1,
                attempts_total=attempts,
                retry_limit=selected_retry_limit,
                status="ok",
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
            latency_instrumentation.emit_event(
                "llm_attempt_completed",
                stage="reasoning_agent_llm_attempt",
                duration_ms=round((time.perf_counter() - attempt_started) * 1000.0, 3),
                role="reasoning_agent",
                provider="openai_compatible",
                model=selected_model,
                timeout_seconds=selected_timeout,
                attempt_number=attempt_index + 1,
                attempts_total=attempts,
                retry_limit=selected_retry_limit,
                status="error",
                error_code=last_error_code,
                error=last_error,
            )
            has_more_attempts = attempt_index < attempts - 1
            if has_more_attempts and selected_backoff > 0:
                time.sleep(selected_backoff * (2**attempt_index))
        finally:
            latency_instrumentation.set_llm_attempt_context(previous_attempt_context)

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
