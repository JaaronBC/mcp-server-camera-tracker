from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


def resolve_model(env_name: str, default_model: str) -> str:
    value = os.getenv(env_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default_model


def resolve_timeout_seconds(default_value: float) -> float:
    value = os.getenv("DASHBOARD_AGENT_LLM_TIMEOUT_SECONDS")
    if value is None:
        return default_value
    try:
        timeout = float(value)
    except ValueError:
        return default_value
    if timeout <= 0:
        return default_value
    return timeout


def resolve_retry_limit(default_value: int) -> int:
    value = os.getenv("DASHBOARD_AGENT_LLM_RETRY_LIMIT")
    if value is None:
        return default_value
    try:
        retry_limit = int(value)
    except ValueError:
        return default_value
    if retry_limit < 0:
        return default_value
    return retry_limit


def resolve_retry_backoff_seconds(default_value: float) -> float:
    value = os.getenv("DASHBOARD_AGENT_LLM_RETRY_BACKOFF_SECONDS")
    if value is None:
        return default_value
    try:
        backoff_seconds = float(value)
    except ValueError:
        return default_value
    if backoff_seconds < 0:
        return default_value
    return backoff_seconds


def classify_llm_error(error: Exception) -> str:
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


def create_openai_client(default_openai_base_url: str) -> AsyncOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base_url = os.getenv("OPENAI_BASE_URL", default_openai_base_url)
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def extract_json_object(raw_text: str) -> dict | None:
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


def coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return default


def coerce_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default
