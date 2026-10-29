# Handles server behavior.
from tools.registry import mcp

# Import tools
from tools.orchestration import tools_orchestration
from tools.reasoning import tools_reasoning_assessment
from tools.tracking import tools_visual_tracking

if __name__ == "__main__":
    mcp.run()