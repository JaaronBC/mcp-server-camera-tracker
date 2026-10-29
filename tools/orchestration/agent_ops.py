"""Compatibility alias to `tools.dashboard.agent_ops` for incremental orchestration migration."""

import sys as _sys

from ..dashboard import agent_ops as _impl

_sys.modules[__name__] = _impl
