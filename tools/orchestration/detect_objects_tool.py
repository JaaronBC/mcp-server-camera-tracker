"""Compatibility alias for `tools.dashboard.detect_objects_tool`."""

import sys as _sys

from ..dashboard import detect_objects_tool as _impl

_sys.modules[__name__] = _impl
