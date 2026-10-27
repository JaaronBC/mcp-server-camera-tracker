# Handles server behavior.
from tools.registry import mcp

# Import tools
from tools import detectObjects
from tools import dashboardControl
from tools import dashboardCalibration
from tools import dashboardAgentOps
from tools import dashboardHealth
from tools import dashboardMetrics
from tools import dashboardReasoning
from tools import dashboardVisualTracking
from tools import liveTracking

if __name__ == "__main__":
    mcp.run()