"""Compatibility alias for `tools.components.liveTrackingDisplay`."""

import sys as _sys

from ..components import liveTrackingDisplay as _impl

_sys.modules[__name__] = _impl
