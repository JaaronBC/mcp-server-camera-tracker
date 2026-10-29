"""Reasoning tool registration for observation summaries."""

from __future__ import annotations

from typing import Any

from ...registry import mcp
from . import _shared as _reasoning_shared


@mcp.tool()
def simulate_observation_summary(track_id: str) -> dict[str, Any]:
    prepared = _reasoning_shared._prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return _reasoning_shared._build_unavailable_track_response(track_id, prepared)

    radar_state = prepared["radar_state"]
    ingested_count = prepared["ingested_count"]
    now = prepared["now"]
    base_summary, history_records, supporting_ids, llm_inputs = _reasoning_shared._build_observation_llm_input(
        track_id,
        created_at=now,
        radar_state=radar_state,
    )
    observation_payload, observation_meta = _reasoning_shared._call_observation_agent_with_retry(llm_inputs[0])
    if observation_payload is None:
        summary = _reasoning_shared._fallback_observation_summary(base_summary)
    else:
        summary = _reasoning_shared._validate_observation_llm_payload(
            observation_payload,
            fallback_summary=base_summary,
            valid_observation_ids=set(supporting_ids),
        )

    _reasoning_shared._attach_llm_meta(summary, observation_payload, observation_meta)

    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "summary": summary,
        "agents": {
            "observation": observation_meta,
        },
        "history_records": [record.to_dict() for record in history_records],
    }
