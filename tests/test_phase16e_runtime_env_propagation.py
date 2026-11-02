from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from pathlib import Path


def _install_mcp_stub() -> None:
    if "mcp" in sys.modules:
        return

    class _StubClientSession:
        pass

    class _StubStdioServerParameters:
        def __init__(self, *, command: str, args: list[str], env: dict[str, str], cwd: str):
            self.command = command
            self.args = args
            self.env = env
            self.cwd = cwd

    stub_module = types.ModuleType("mcp")
    stub_module.ClientSession = _StubClientSession
    stub_module.StdioServerParameters = _StubStdioServerParameters
    sys.modules["mcp"] = stub_module


class TestRuntimeEnvPropagation(unittest.TestCase):
    def test_server_params_include_budget_env_and_cwd(self) -> None:
        _install_mcp_stub()
        runtime_app = importlib.import_module("agents.orchestration.runtime.app")

        config = runtime_app.RuntimeConfig(
            show_startup_banner=False,
            show_startup_depth_snapshot=False,
            show_startup_pipeline_contract_snapshot=False,
            enable_agent_ops_heartbeat=False,
            agent_ops_interval_seconds=1.0,
            agent_ops_max_tracks=1,
            agent_ops_summary_every_cycles=1,
            agent_ops_summary_window_size=1,
            handoff_degraded_warn_rate=0.5,
            handoff_degraded_critical_rate=0.9,
            depth_coverage_summary_enabled=False,
            depth_coverage_summary_every_cycles=1,
            depth_coverage_sample_limit=1,
        )
        logger = runtime_app.configure_logging(config)
        server_path = Path(__file__).resolve()

        services = runtime_app.RuntimeServices(
            config,
            logger,
            server_path=server_path,
            python_executable=sys.executable,
            stdio_client_fn=lambda *_args, **_kwargs: None,
            client_session_cls=runtime_app.ClientSession,
            startup_runner=lambda *_args, **_kwargs: None,
        )
        params = services._server_params

        self.assertIsNotNone(getattr(params, "env", None))
        self.assertIsInstance(params.env, dict)
        self.assertIn("PATH", params.env)
        self.assertEqual(params.env.get("PATH"), os.environ.get("PATH"))
        self.assertIsNotNone(getattr(params, "cwd", None))
        self.assertEqual(params.cwd, str(server_path.parent))


if __name__ == "__main__":
    unittest.main()
