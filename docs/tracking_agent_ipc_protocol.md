# Tracking Agent IPC Protocol (Design-Only, Pre-Bridge)

## Scope and Non-Goals

This document defines the bounded file-based IPC protocol for Phase 2D Increment 3 design.

- This is a design artifact only.
- No IPC bridge or autonomous MCP execution implementation is included in this increment.
- Registry mutation authority remains outside this protocol.

## Process Ownership

- Dashboard process (`agents/dashboard/tracking_loop.py`)
  - Produces bounded tool-request records derived from validated Tracking Agent `decision` payloads.
  - Consumes bounded tool-result records for matching investigations.
  - Triggers at most one follow-up reasoning pass per investigation.
  - Must remain non-blocking relative to YOLO/ByteTrack/radar loop cadence.

- Orchestration process (`main.py` runtime via `agents/orchestration/runtime/*`)
  - Reuses existing MCP `ClientSession`.
  - Consumes pending tool requests, validates contracts, executes allowlisted read-only tools.
  - Emits idempotent tool-result records.
  - Enforces terminal investigation checks and bounded retention.

## Transport Choice

Use immutable append-only JSONL event artifacts (not in-place JSONL edits):

- `captures/tracking_decision_tool_requests.jsonl`
- `captures/tracking_decision_tool_results.jsonl`

Optional compact state checkpoints are atomically replaced snapshots:

- `captures/latest_tracking_decision_request_index.json`
- `captures/latest_tracking_decision_result_index.json`

No design path uses in-place mutation of existing JSONL rows.

## Request Schema (JSONL event)

```json
{
  "schema_version": "1.0",
  "event_type": "decision_tool_request",
  "request_id": "req-...",
  "investigation_id": "inv-...",
  "iteration_index": 1,
  "llm_correlation_id": "tracking-llm-...",
  "decision_action": "request_identity_snapshot|request_additional_tracking_evidence",
  "tool_name": "get_identity_snapshot|get_track_evidence_history",
  "arguments": {},
  "snapshot_constraints": {
    "min_source_generation": 0,
    "min_source_version": 0
  },
  "created_at": 0.0,
  "expires_at": 0.0,
  "dedupe_key": "inv-...:req-..."
}
```

Required constraints:

- `request_id` non-empty string.
- `investigation_id` non-empty string.
- `iteration_index` integer in `[1, 2]`.
- At most one tool request per decision.
- `tool_name` allowlisted to read-only tools only.
- Tool arguments must pass the same schema contract validated in tracking agent decision parsing.

## Result Schema (JSONL event)

```json
{
  "schema_version": "1.0",
  "event_type": "decision_tool_result",
  "request_id": "req-...",
  "investigation_id": "inv-...",
  "iteration_index": 1,
  "tool_name": "get_identity_snapshot|get_track_evidence_history",
  "status": "ok|timeout|unavailable|rejected|expired|duplicate|terminal|incompatible|stale",
  "result_payload": {},
  "error_code": "optional_error_code",
  "error": "optional_sanitized_error",
  "started_at": 0.0,
  "finished_at": 0.0,
  "idempotency_key": "inv-...:req-...:tool"
}
```

Rules:

- `request_id` + `investigation_id` must match a request event.
- One terminal result per `idempotency_key`; duplicates become `status=duplicate` and are ignored by consumers.
- Result payload is sanitized and read-only.

## Correlation and Matching

- Primary correlation key: (`investigation_id`, `request_id`).
- Secondary safety key: (`investigation_id`, `iteration_index`).
- A result is consumable only when all keys match and investigation is non-terminal.
- Mismatched keys are rejected with `status=rejected` + `error_code=correlation_mismatch`.

## Tool Allowlist and Argument Validation

Allowlist only:

- `get_identity_snapshot`
- `get_track_evidence_history`

Validation requirements:

- Enforce the same per-tool argument schema used by tracking decision contract validation.
- Reject unknown tool names and malformed arguments before MCP call dispatch.
- Reject actions that attempt to imply mutation or authorization transfer.

## Iteration and Bounded Execution

Per investigation:

- Maximum one tool call per iteration.
- Maximum two reasoning iterations total (`iteration_index` 1 then optional 2).
- No third iteration; emit terminal inconclusive outcome.

Decision flow:

1. Tracking emits request-action decision with one tool request.
2. Orchestration executes allowlisted read-only tool and writes result.
3. Dashboard consumes result and may run one follow-up reasoning pass.
4. If still unresolved after iteration 2, force explicit `inconclusive`.

## Snapshot Freshness/Compatibility

For `get_identity_snapshot` results:

- If stale/incompatible with required generation/version constraints, classify as `status=stale` or `status=incompatible`.
- Such results do not authorize final identity association.
- Downstream action becomes explicit `inconclusive` unless independent strong evidence exists in the same iteration contract.

## Terminal Investigation Rejection

Treat investigations in `resolved|closed` as terminal.

- New requests for terminal investigations are rejected (`status=terminal`).
- Late results for terminal investigations are ignored and logged as `status=terminal`.
- Terminal state lookup should use bounded in-memory index with optional checkpoint recovery.

## Timeout and Unavailable Evidence Handling

If MCP call times out or returns unavailable evidence:

- Persist a result event with `status=timeout|unavailable`.
- Dashboard converts this to explicit `decision.action=inconclusive` with reason tags such as:
  - `tool_timeout`
  - `evidence_unavailable`
  - `snapshot_incompatible`

No silent fallback to actionable association.

## Atomic Writes and Crash Recovery

### Atomic write policy

- JSONL appends must use write+flush+fsync on a single line per event.
- Checkpoint index files (`latest_*_index.json`) must be written via temp file + atomic rename.

### Crash window of interest

Window: MCP call succeeds, but result event is not durably persisted due to crash.

Safe behavior:

- On restart, replay pending requests without durable matching result.
- Use `idempotency_key` to deduplicate re-executed results.
- If the tool call is retried and yields same logical result, write exactly one canonical consumable result and optionally a duplicate marker event.

This guarantees at-least-once execution with exactly-once consumption semantics by idempotency key.

## Request Deduplication and Result Idempotency

- Request dedupe key: `inv:request_id`.
- Result idempotency key: `inv:request_id:tool_name`.
- Duplicate requests do not dispatch duplicate MCP calls once an accepted dispatch/result exists.
- Duplicate results are retained as audit events but not re-consumed.

## Bounded Retention and Cleanup

Retention controls:

- Max JSONL records per artifact (e.g., 10k requests, 10k results).
- Max age window (e.g., 24h) for non-terminal investigations.
- Keep terminal investigation summaries in bounded index (e.g., 2k entries).

Cleanup strategy:

- Compaction writes a new JSONL file from retained events, then atomic rename.
- Never mutate individual rows in-place.

## Non-Blocking Dashboard Behavior

Dashboard constraints:

- No synchronous waits on orchestration responses in tracking frame loop.
- Request enqueue and result poll are O(1) bounded operations per frame/tick.
- Any heavy parsing/cleanup runs on separate cadence or background helper thread.

This preserves YOLO/ByteTrack/radar loop responsiveness.

## Deterministic Acceptance Criteria (for future implementation)

1. Request with invalid tool name is rejected before dispatch.
2. Request/result correlation mismatch is rejected.
3. Duplicate request is deduplicated deterministically.
4. Crash-replay after missing result persistence does not cause double-consumption.
5. Terminal investigations reject new requests and late results.
6. Iteration cap at 2 is enforced; unresolved path becomes explicit inconclusive.
7. Timeout/unavailable tool outcomes become explicit inconclusive reasons.
8. Dashboard loop remains non-blocking under pending request backlog.
9. Snapshot stale/incompatible results are not treated as actionable associations.
10. No registry mutation side effects occur from decision metadata or IPC records.
