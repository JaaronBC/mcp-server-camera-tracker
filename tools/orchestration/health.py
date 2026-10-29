"""Compatibility alias for `tools.dashboard.health`."""

import sys as _sys

from ..dashboard import health as _impl

_sys.modules[__name__] = _impl
