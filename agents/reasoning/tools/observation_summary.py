"""Reasoning tool registration for observation summaries."""

from __future__ import annotations

from typing import Any

from ...registry import mcp
from . import _shared as _reasoning_shared


@mcp.tool()
def simulate_observation_summary(track_id: str) -> dict[str, Any]:
    return _reasoning_shared.simulate_observation_summary(track_id)
