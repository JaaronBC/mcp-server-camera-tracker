"""Compatibility alias to `tools.dashboard.detect_objects_tool` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import detect_objects_tool as _impl

_sys.modules[__name__] = _impl
