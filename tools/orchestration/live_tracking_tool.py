"""Compatibility alias to `tools.dashboard.live_tracking_tool` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import live_tracking_tool as _impl

_sys.modules[__name__] = _impl
