"""Compatibility alias for `tools.dashboard.live_tracking_tool`."""

import sys as _sys

from ..dashboard import live_tracking_tool as _impl

_sys.modules[__name__] = _impl
