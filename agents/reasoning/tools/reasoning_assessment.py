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
    return _reasoning_shared.simulate_reasoning_assessment(
        track_id,
        request_evidence_on_uncertainty=request_evidence_on_uncertainty,
    )
