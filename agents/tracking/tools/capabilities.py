"""Tracking agent tool capability metadata and decision mappings."""

from __future__ import annotations

TRACKING_ACTION_REQUEST_IDENTITY_SNAPSHOT = "request_identity_snapshot"
TRACKING_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE = "request_additional_tracking_evidence"
TRACKING_ACTION_PROPOSE_IDENTITY_ASSOCIATION = "propose_identity_association"
TRACKING_ACTION_INCONCLUSIVE = "inconclusive"

TRACKING_TOOL_GET_IDENTITY_SNAPSHOT = "get_identity_snapshot"
TRACKING_TOOL_GET_TRACK_EVIDENCE_HISTORY = "get_track_evidence_history"

TRACKING_TOOL_CAPABILITIES: dict[str, dict[str, object]] = {
    TRACKING_TOOL_GET_IDENTITY_SNAPSHOT: {
        "request_action": TRACKING_ACTION_REQUEST_IDENTITY_SNAPSHOT,
        "read_only": True,
        "domain": "identity_snapshot",
        "description": "Read-only identity snapshot/candidate lookup for tracking continuity decisions.",
    },
    TRACKING_TOOL_GET_TRACK_EVIDENCE_HISTORY: {
        "request_action": TRACKING_ACTION_REQUEST_ADDITIONAL_TRACKING_EVIDENCE,
        "read_only": True,
        "domain": "track_evidence_history",
        "description": "Read-only bounded track evidence history retrieval for tracking follow-up decisions.",
    },
}

TRACKING_REQUEST_ACTION_TO_TOOL_NAME: dict[str, str] = {
    str(metadata["request_action"]): tool_name
    for tool_name, metadata in TRACKING_TOOL_CAPABILITIES.items()
}

TRACKING_TOOL_NAME_TO_REQUEST_ACTION: dict[str, str] = {
    tool_name: request_action
    for request_action, tool_name in TRACKING_REQUEST_ACTION_TO_TOOL_NAME.items()
}

TRACKING_DECISION_REQUEST_ACTIONS = frozenset(TRACKING_REQUEST_ACTION_TO_TOOL_NAME.keys())

TRACKING_DECISION_ACTIONS = frozenset(
    {
        *TRACKING_DECISION_REQUEST_ACTIONS,
        TRACKING_ACTION_PROPOSE_IDENTITY_ASSOCIATION,
        TRACKING_ACTION_INCONCLUSIVE,
    }
)

TRACKING_DECISION_TOOL_ALLOWLIST = frozenset(TRACKING_TOOL_CAPABILITIES.keys())
