"""Compatibility alias to `tools.dashboard.calibration` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import calibration as _impl

_sys.modules[__name__] = _impl
