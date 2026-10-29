# Handles server behavior.
from agents.registry import mcp

# Import tools
from agents.orchestration.tools import orchestration as _orchestration_tools
from agents.reasoning import tools as _reasoning_tools
from agents.tracking import tools as _tracking_tools

if __name__ == "__main__":
    mcp.run()