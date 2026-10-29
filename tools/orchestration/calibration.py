"""Compatibility alias for `tools.dashboard.calibration`."""

import sys as _sys

from ..dashboard import calibration as _impl

_sys.modules[__name__] = _impl
