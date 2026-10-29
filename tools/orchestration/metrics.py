"""Compatibility alias for `tools.dashboard.metrics`."""

import sys as _sys

from ..dashboard import metrics as _impl

_sys.modules[__name__] = _impl
