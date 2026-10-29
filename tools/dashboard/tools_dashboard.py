"""Single dashboard MCP registration entrypoint.

Import this module from ``server.py`` to register all dashboard-related tools.
The modules imported here are the canonical dashboard package implementation.
"""

from . import agent_ops  # noqa: F401
from . import calibration  # noqa: F401
from . import control  # noqa: F401
from . import detect_objects_tool  # noqa: F401
from . import health  # noqa: F401
from . import live_tracking_tool  # noqa: F401
from . import metrics  # noqa: F401
