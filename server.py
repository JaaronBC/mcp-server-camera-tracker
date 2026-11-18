# Handles server behavior.
from agents.registry import mcp

# Import tools
from agents.observation.tools import evidence as _observation_evidence_tools
from agents.observation.tools import track_evidence_history as _observation_track_evidence_history_tools
from agents.orchestration.tools import orchestration as _orchestration_tools
from agents.reasoning import tools as _reasoning_tools
from agents.tracking.tools import visual_tracking as _tracking_visual_tracking_tools

if __name__ == "__main__":
    mcp.run()