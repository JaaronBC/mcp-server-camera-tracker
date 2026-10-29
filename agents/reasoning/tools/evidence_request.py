"""Reasoning tool registration for evidence requests."""

from __future__ import annotations

from ...registry import mcp
from . import _shared as _reasoning_shared


@mcp.tool()
def simulate_evidence_request(track_id: str, reason: str) -> dict[str, object]:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")

    prepared = _reasoning_shared._prepare_track_analysis(track_id)
    if not bool(prepared.get("available")):
        return _reasoning_shared._build_unavailable_track_response(track_id, prepared)

    _radar_state = prepared["radar_state"]
    ingested_count = prepared["ingested_count"]
    now = prepared["now"]
    request = _reasoning_shared._COORDINATOR.request_evidence(
        track_id=track_id,
        reason=reason,
        requested_at=now,
    )

    return {
        "available": True,
        "track_id": track_id,
        "ingested_tracks": ingested_count,
        "request": None if request is None else request.to_dict(),
        "bounded_out": request is None,
    }
