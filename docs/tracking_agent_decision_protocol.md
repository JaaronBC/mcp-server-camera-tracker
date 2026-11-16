# Tracking Agent Decision Protocol (Increment 2 Design)

## Scope

This document defines the proposed, bounded protocol for optional Tracking Agent decision payloads emitted by the visual association response. It is design-only for Increment 2; no autonomous tool execution or IPC bridge behavior is implemented here.

## Goals

- Preserve backward compatibility with existing tracking outputs (`proposals`, `uncertainty_flags`, `supporting_evidence`).
- Add an optional `decision` object with strict validation.
- Keep orchestration authoritative for execution, guardrails, and mutation pathways.

## Decision Envelope

The tracking response MAY include:

```json
{
  "proposals": [...],
  "uncertainty_flags": [...],
  "supporting_evidence": [...],
  "decision": {
    "action": "request_identity_snapshot | request_additional_tracking_evidence | propose_identity_association | inconclusive",
    "investigation_id": "string",
    "correlation_id": "string",
    "reason": "string",
    "uncertainty": ["string"],
    "evidence_references": {
      "frame_reference": "string",
      "detection_reference": "string",
      "tracking_message_id": "string"
    },
    "provenance": {
      "key": "string|number|bool|null"
    },
    "snapshot_constraints": {
      "min_source_generation": 0,
      "min_source_version": 0
    },
    "tool_requests": [
      {
        "request_id": "string",
        "tool_name": "get_identity_snapshot | get_track_evidence_history",
        "arguments": {}
      }
    ]
  }
}
```

## Validation Rules

- `decision.action` is required and must be in the allowed enum.
- `tool_name` is allowlisted to read-only tools:
  - `get_identity_snapshot`
  - `get_track_evidence_history`
- At most one `tool_requests` entry is allowed (`max=1`).
- For tool-request actions (`request_identity_snapshot`, `request_additional_tracking_evidence`):
  - exactly one tool request is required;
  - `investigation_id` is required.
- `request_id` is required for each tool request.
- Tool arguments must match tool-specific schemas.
- `inconclusive` requires explicit non-empty `reason`.
- `snapshot_constraints` values (if present) must be non-negative integers.
- Evidence/provenance references are optional but must be type-safe when present.

## Tool Argument Contracts

### `get_identity_snapshot`

- Exactly one selector required:
  - `app_identity_id` OR `detection_reference`
- Optional fields:
  - `include_history: bool`
  - `max_history_entries: int > 0`
  - `max_candidates: int > 0`
  - `min_source_generation: int >= 0`
  - `min_source_version: int >= 0`
  - `stale_after_seconds: number > 0`
  - `investigation_id: non-empty string`
  - `correlation_id: non-empty string`

### `get_track_evidence_history`

- Required:
  - `track_id: non-empty string`
- Optional fields:
  - `max_messages: int > 0`
  - `max_history_records: int > 0`
  - `max_scan_records: int > 0`
  - `now_timestamp: number`
  - `investigation_id: non-empty string`
  - `correlation_id: non-empty string`

## Error Contract (Validation Failure)

If `decision` exists but fails validation, the tracking response is treated as unavailable with:

- `error = "vlm_invalid_decision_contract"`
- `uncertainty_flags` containing `vlm_invalid_decision_contract` plus specific validation codes.

This fails closed and prevents downstream execution of malformed decision requests.

## Increment Boundaries

- Included in Increment 1 implementation:
  - Decision schema validation and deterministic unit coverage.
- Deferred to later increments:
  - IPC bridge dispatch.
  - Autonomous bounded execution loop.
  - Retry/backoff scheduling for evidence requests.
