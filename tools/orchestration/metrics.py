"""Compatibility alias to `tools.dashboard.metrics` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import metrics as _impl

_sys.modules[__name__] = _impl
