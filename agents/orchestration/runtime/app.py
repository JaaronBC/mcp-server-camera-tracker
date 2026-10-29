# Runtime application wiring helpers for initialization-first async entrypoints.
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp import ClientSession, StdioServerParameters


@dataclass(slots=True)
class RuntimeConfig:
    show_startup_banner: bool
    show_startup_depth_snapshot: bool
    show_startup_pipeline_contract_snapshot: bool
    enable_agent_ops_heartbeat: bool
    agent_ops_interval_seconds: float
    agent_ops_max_tracks: int
    agent_ops_summary_every_cycles: int
    agent_ops_summary_window_size: int
    handoff_degraded_warn_rate: float
    handoff_degraded_critical_rate: float
    depth_coverage_summary_enabled: bool
    depth_coverage_summary_every_cycles: int
    depth_coverage_sample_limit: int


def configure_logging(_config: RuntimeConfig) -> logging.Logger:
    # Configures runtime logger for initialization-first flow.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger("camera_tracker.runtime")


class RuntimeServices:
    def __init__(
        self,
        config: RuntimeConfig,
        logger: logging.Logger,
        *,
        server_path: Path,
        python_executable: str,
        stdio_client_fn: Callable[..., Any],
        client_session_cls: type[ClientSession],
        startup_runner: Callable[[ClientSession, RuntimeConfig], Awaitable[None]],
    ) -> None:
        # Performs init logic for runtime service lifecycle ownership.
        self.config = config
        self.logger = logger
        self.tracking_agent = "tracking_agent"
        self.observation_agent = "observation_agent"
        self.reasoning_agent = "reasoning_agent"
        self.observation_source: ClientSession | None = None

        self._server_params = StdioServerParameters(
            command=python_executable,
            args=[str(server_path)],
        )
        self._stdio_client_fn = stdio_client_fn
        self._client_session_cls = client_session_cls
        self._startup_runner = startup_runner

        self._stdio_context = None
        self._session_context = None
        self.session: ClientSession | None = None

    async def connect(self) -> None:
        # Connects to MCP server and initializes the session.
        self._stdio_context = self._stdio_client_fn(self._server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()

        self._session_context = self._client_session_cls(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()
        self.observation_source = self.session

    async def start(self) -> None:
        # Starts runtime services and startup checks before operator run loop.
        if self.session is None:
            raise RuntimeError("Runtime services are not connected")
        await self._startup_runner(self.session, self.config)

    async def stop(self) -> None:
        # Stops runtime services and closes MCP session resources.
        if self._session_context is not None:
            await self._session_context.__aexit__(None, None, None)
            self._session_context = None
            self.session = None
            self.observation_source = None

        if self._stdio_context is not None:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None


async def initialize_services(
    config: RuntimeConfig,
    logger: logging.Logger,
    *,
    server_path: Path,
    python_executable: str,
    stdio_client_fn: Callable[..., Any],
    client_session_cls: type[ClientSession],
    startup_runner: Callable[[ClientSession, RuntimeConfig], Awaitable[None]],
) -> RuntimeServices:
    # Initializes and returns connected runtime services.
    services = RuntimeServices(
        config,
        logger,
        server_path=server_path,
        python_executable=python_executable,
        stdio_client_fn=stdio_client_fn,
        client_session_cls=client_session_cls,
        startup_runner=startup_runner,
    )
    await services.connect()
    return services


class AgentCoordinator:
    def __init__(
        self,
        *,
        tracking_agent: Any,
        observation_agent: Any,
        reasoning_agent: Any,
        services: RuntimeServices,
        run_loop: Callable[[ClientSession, RuntimeConfig], Awaitable[None]],
    ) -> None:
        # Performs init logic for operator runtime coordinator wrapper.
        self.tracking_agent = tracking_agent
        self.observation_agent = observation_agent
        self.reasoning_agent = reasoning_agent
        self._services = services
        self._run_loop = run_loop

    async def run(self) -> None:
        # Runs the operator monitoring loop through existing runtime functions.
        if self._services.session is None:
            raise RuntimeError("Runtime services are not connected")
        await self._run_loop(self._services.session, self._services.config)

    async def stop(self) -> None:
        # Stops coordinator-owned runtime operations.
        return None


class OrchestrationAgent:
    def __init__(self, *, coordinator: AgentCoordinator, observation_source: Any) -> None:
        # Performs init logic for orchestration runtime wrapper.
        self._coordinator = coordinator
        self.observation_source = observation_source

    async def run(self) -> None:
        # Runs orchestration loop via coordinator.
        await self._coordinator.run()

    async def stop(self) -> None:
        # Stops orchestration loop via coordinator.
        await self._coordinator.stop()
