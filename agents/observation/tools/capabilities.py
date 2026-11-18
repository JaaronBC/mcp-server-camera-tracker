"""Observation agent tool capability metadata and autonomous decision contracts."""

from __future__ import annotations

from typing import Any

OBSERVATION_ACTION_ASSESS_NOW = "assess_now"
OBSERVATION_ACTION_PREDICT_BOUNDED = "predict_bounded"
OBSERVATION_ACTION_INCONCLUSIVE = "inconclusive"
OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE = "request_additional_tracking_evidence"

OBSERVATION_TOOL_GET_TRACK_EVIDENCE_HISTORY = "get_track_evidence_history"

OBSERVATION_TOOL_CAPABILITIES: dict[str, dict[str, object]] = {
    OBSERVATION_TOOL_GET_TRACK_EVIDENCE_HISTORY: {
        "request_action": OBSERVATION_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
        "read_only": True,
        "domain": "track_evidence_history",
        "description": "Read-only bounded track evidence history retrieval for observation follow-up decisions.",
        "model_selectable_arguments": {
            "max_messages": "positive integer",
            "max_history_records": "positive integer",
            "max_scan_records": "positive integer",
        },
        "trusted_metadata_arguments": {
            "track_id": "authoritative orchestration context",
            "now_timestamp": "runtime dispatch metadata",
            "investigation_id": "runtime correlation metadata",
            "correlation_id": "runtime correlation metadata",
        },
    },
}

OBSERVATION_REQUEST_ACTION_TO_TOOL_NAME: dict[str, str] = {
    str(metadata["request_action"]): tool_name
    for tool_name, metadata in OBSERVATION_TOOL_CAPABILITIES.items()
}

OBSERVATION_TOOL_NAME_TO_REQUEST_ACTION: dict[str, str] = {
    tool_name: request_action
    for request_action, tool_name in OBSERVATION_REQUEST_ACTION_TO_TOOL_NAME.items()
}

OBSERVATION_DECISION_REQUEST_ACTIONS = frozenset(OBSERVATION_REQUEST_ACTION_TO_TOOL_NAME.keys())

OBSERVATION_DECISION_ACTIONS = frozenset(
    {
        OBSERVATION_ACTION_ASSESS_NOW,
        OBSERVATION_ACTION_PREDICT_BOUNDED,
        OBSERVATION_ACTION_INCONCLUSIVE,
        *OBSERVATION_DECISION_REQUEST_ACTIONS,
    }
)

OBSERVATION_DECISION_TOOL_ALLOWLIST = frozenset(OBSERVATION_TOOL_CAPABILITIES.keys())


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized else None


def normalize_observation_decision(
    payload: Any,
    *,
    fallback_action: str = OBSERVATION_ACTION_INCONCLUSIVE,
    fallback_reason: str = "decision_unavailable",
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "action": fallback_action,
            "reason": fallback_reason,
            "uncertainty": ["decision_payload_not_object"],
            "tool_requests": [],
        }

    action = _nonempty_str(payload.get("action"))
    if action not in OBSERVATION_DECISION_ACTIONS:
        action = fallback_action

    decision: dict[str, Any] = {
        "action": action,
        "reason": _nonempty_str(payload.get("reason")) or fallback_reason,
        "uncertainty": [],
        "tool_requests": [],
    }

    uncertainty = payload.get("uncertainty")
    if isinstance(uncertainty, list):
        decision["uncertainty"] = [
            item.strip()
            for item in uncertainty
            if isinstance(item, str) and item.strip()
        ]

    tool_requests = payload.get("tool_requests")
    if not isinstance(tool_requests, list):
        return decision

    normalized_tool_requests: list[dict[str, Any]] = []
    for item in tool_requests[:1]:
        if not isinstance(item, dict):
            continue
        tool_name = _nonempty_str(item.get("tool_name"))
        if tool_name is None:
            tool_name = _nonempty_str(item.get("tool")) or _nonempty_str(item.get("name"))
        if tool_name not in OBSERVATION_DECISION_TOOL_ALLOWLIST:
            continue
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            arguments = item.get("tool_arguments")
        if not isinstance(arguments, dict):
            arguments = item.get("args")
        if not isinstance(arguments, dict):
            arguments = item.get("parameters")
        normalized_tool_requests.append(
            {
                "request_id": _nonempty_str(item.get("request_id")) or _nonempty_str(item.get("id")),
                "tool_name": tool_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
        )
    decision["tool_requests"] = normalized_tool_requests
    return decision
