import json
from typing import Any, Callable

from .agent_tracking import (
    AsyncVisualTrackingRuntime,
    LLMVisualTrackingAgent,
    TRACKING_VLM_MODEL,
    VisualTrackingAgentResult,
    VisualTrackingAgentProtocol,
    encode_frame_as_data_url,
    load_tracking_agent_skill,
)

LIVE_TRACKING_TOOL_NAME = "live_tracking"
DETECT_OBJECTS_TOOL_NAME = "detect_objects"


def get_result_text(result: Any) -> str:
    # Gets result text values for downstream consumers.
    return "\n".join(
        item.text
        for item in (result.content or [])
        if hasattr(item, "text") and isinstance(item.text, str)
    )


def get_result_json_object(result: Any) -> dict[str, Any] | None:
    # Gets result json object values for downstream consumers.
    for item in (result.content or []):
        for attribute in ("json", "data", "value"):
            payload = getattr(item, attribute, None)
            if isinstance(payload, dict):
                return payload

        if hasattr(item, "text") and isinstance(item.text, str):
            text = item.text.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

    return None


def _emit_message(message: str, emit: Callable[[str], None] | None) -> None:
    # Emits user-facing runtime messages through provided sink.
    if emit is None:
        print(message, flush=True)
    else:
        emit(message)


async def verify_required_tools(
    session,
    *,
    live_tracking_tool_name: str = LIVE_TRACKING_TOOL_NAME,
    detect_objects_tool_name: str = DETECT_OBJECTS_TOOL_NAME,
) -> None:
    # Verifies tracking tool contract before runtime starts.
    tool_list = await session.list_tools()
    available = {tool.name for tool in tool_list.tools}
    required = {live_tracking_tool_name, detect_objects_tool_name}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            f"Required MCP tools are not registered: {missing}. "
            f"Available tools: {sorted(available)}"
        )


async def start_live_tracking(
    session,
    *,
    tool_name: str = LIVE_TRACKING_TOOL_NAME,
    result_text_fn: Callable[[Any], str] = get_result_text,
    emit: Callable[[str], None] | None = None,
) -> None:
    # Starts live tracking through MCP tool and emits startup feedback.
    result = await session.call_tool(tool_name, arguments={})
    message = result_text_fn(result)
    if getattr(result, "is_error", False):
        raise RuntimeError(f"Could not start live tracking: {message}")
    _emit_message(
        message
        or "Live tracking started. Press 'q' in the webcam window to stop.",
        emit,
    )


async def run_tracking_cycle(
    session,
    *,
    address: list[Any],
    detect_objects_tool_name: str = DETECT_OBJECTS_TOOL_NAME,
    result_text_fn: Callable[[Any], str] = get_result_text,
    emit: Callable[[str], None] | None = None,
) -> None:
    # Runs one tracking analysis cycle and emits movement output.
    try:
        result = await session.call_tool(
            detect_objects_tool_name,
            arguments={"address": address},
        )
    except Exception as error:
        _emit_message(
            "Movement analysis tool call failed "
            f"({type(error).__name__}): {error}",
            emit,
        )
        return

    frame_data = result_text_fn(result)
    if getattr(result, "is_error", False):
        _emit_message(f"Movement analysis failed: {frame_data}", emit)
    elif not frame_data:
        _emit_message("Movement analysis returned no frame data.", emit)
    else:
        _emit_message(f"\nYOLO frame:\n{frame_data}", emit)


__all__ = [
    "AsyncVisualTrackingRuntime",
    "LLMVisualTrackingAgent",
    "TRACKING_VLM_MODEL",
    "VisualTrackingAgentResult",
    "VisualTrackingAgentProtocol",
    "encode_frame_as_data_url",
    "get_result_json_object",
    "get_result_text",
    "run_tracking_cycle",
    "start_live_tracking",
    "verify_required_tools",
    "load_tracking_agent_skill",
]
