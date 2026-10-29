"""Reasoning tool registration for full reasoning assessment."""

from __future__ import annotations

from typing import Any

from ...registry import mcp
from . import _shared as _reasoning_shared


@mcp.tool()
def simulate_reasoning_assessment(
    track_id: str,
    request_evidence_on_uncertainty: bool = True,
) -> dict[str, Any]:
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

    handoff_validation = _reasoning_shared._default_handoff_validation()
    try:
        reasoning_input, handoff_validation = _reasoning_shared._build_reasoning_handoff_payload(
            track_id=track_id,
            created_at=now,
            summary=summary,
            history_records=[record.to_dict() for record in history_records],
        )
        reasoning_payload, reasoning_meta = _reasoning_shared._call_reasoning_agent_with_retry(reasoning_input)
    except ValueError as error:
        reasoning_payload = None
        reasoning_meta = _reasoning_shared._skipped_reasoning_meta(error)
        handoff_validation = _reasoning_shared._handoff_validation_error(error)

    if reasoning_payload is None:
        assessment, request_more_observations = _reasoning_shared._fallback_reasoning_assessment(
            track_id,
            created_at=now,
            summary=summary,
        )
    else:
        assessment, request_more_observations = _reasoning_shared._validate_reasoning_llm_payload(
            reasoning_payload,
            track_id=track_id,
            created_at=now,
            valid_observation_ids=set(summary.get("supporting_observation_ids", [])),
        )

    _reasoning_shared._attach_llm_meta(assessment, reasoning_payload, reasoning_meta)

    evidence_request = None
    if request_evidence_on_uncertainty and request_more_observations:
        request = _reasoning_shared._COORDINATOR.request_evidence(
            track_id=track_id,
            reason="llm_requested_more_observations",
            requested_at=now,
        )
        if request is not None:
            evidence_request = request.to_dict()

    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "summary": summary,
        "assessment": assessment,
        "evidence_request": evidence_request,
        "agents": {
            "observation": observation_meta,
            "reasoning": reasoning_meta,
        },
        "handoff_validation": handoff_validation,
    }
