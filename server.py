from tools.registry import mcp

# Import tools
from tools import captureWebcam
from tools import detectTrackObjects
from tools import analyzeMovement
from tools import liveTracking
from tools import depthMappingWebcam
    
if __name__ == "__main__":
    mcp.run()