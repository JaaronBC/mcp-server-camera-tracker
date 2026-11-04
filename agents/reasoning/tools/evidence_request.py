"""Reasoning tool registration for evidence requests."""

from __future__ import annotations

from ...registry import mcp
from . import _shared as _reasoning_shared


@mcp.tool()
def simulate_evidence_request(track_id: str, reason: str) -> dict[str, object]:
    return _reasoning_shared.simulate_evidence_request(track_id, reason)
