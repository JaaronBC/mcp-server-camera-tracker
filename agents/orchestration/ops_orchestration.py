"""Compatibility alias for `tools.orchestration.agent_orchestration`."""

import sys as _sys

from . import agent_orchestration as _impl

_sys.modules[__name__] = _impl
