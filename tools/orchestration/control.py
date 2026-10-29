"""Compatibility alias for `tools.dashboard.control`."""

import sys as _sys

from ..dashboard import control as _impl

_sys.modules[__name__] = _impl
