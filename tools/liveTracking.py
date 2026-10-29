"""Compatibility alias for ``tools.orchestration.live_tracking_tool``."""

import sys as _sys

from .orchestration import live_tracking_tool as _impl

_sys.modules[__name__] = _impl
