"""Compatibility alias to `tools.dashboard.control` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import control as _impl

_sys.modules[__name__] = _impl
