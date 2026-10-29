"""Compatibility alias for `tools.dashboard.process`."""

import sys as _sys

from ..dashboard import process as _impl

_sys.modules[__name__] = _impl
