"""Compatibility alias to `tools.dashboard.process` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import process as _impl

_sys.modules[__name__] = _impl
