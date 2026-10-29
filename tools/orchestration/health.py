"""Compatibility alias to `tools.dashboard.health` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import health as _impl

_sys.modules[__name__] = _impl
