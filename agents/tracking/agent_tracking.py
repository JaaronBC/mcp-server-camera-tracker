# Visual tracking agent runtime that invokes VLM association proposals and converts them into validated tracking observations.
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ..contracts import DetectionCandidate, VisualAssociationProposal
from ..coordinator import AgentCoordinator
from .. import latency_instrumentation
from .. import llm_call
from .. import provider_call_budget

TRACKING_VLM_MODEL = "qwen-3.8-27b"
DEFAULT_OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"
DEFAULT_TRACKING_AGENT_MAX_OUTPUT_TOKENS = 384
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKING_AGENT_SKILL_PATH = PROJECT_ROOT / "TrackingAgentSkill.md"
TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK = (
    "You are a visual tracking agent for continuity-preserving identity association. "
    "Given structured detections and optional frame evidence, return conservative, evidence-grounded "
    "association proposals with calibrated uncertainty and explicit conflict flags. "
    "Never invent measurements or unsupported identities, and always return strict JSON only."
)

_PROVIDER_ERROR_CODE_ALLOWED_PATTERN = re.compile(r"[^a-z0-9._:-]+")


def _sanitize_provider_error_code(raw_code: object) -> str | None:
    if not isinstance(raw_code, str):
        return None
    normalized = raw_code.strip().lower()
    if not normalized:
        return None
    normalized = _PROVIDER_ERROR_CODE_ALLOWED_PATTERN.sub("_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        return None
    return normalized[:64]


def _extract_exception_telemetry_fields(error: Exception) -> dict[str, object]:
    fields: dict[str, object] = {
        "sdk_exception_class": type(error).__name__,
    }

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        fields["http_status_code"] = status_code

    provider_error_code = _sanitize_provider_error_code(getattr(error, "code", None))
    if provider_error_code is not None:
        fields["provider_error_code"] = provider_error_code

    return fields


def load_tracking_agent_skill() -> str:
    # Loads tracking agent skill data from storage for downstream processing.
    if TRACKING_AGENT_SKILL_PATH.is_file():
        loaded = TRACKING_AGENT_SKILL_PATH.read_text(encoding="utf-8").strip()
        if loaded:
            return loaded
    return TRACKING_AGENT_SYSTEM_PROMPT_FALLBACK


def _resolve_tracking_max_output_tokens() -> int | None:
    return llm_call.resolve_max_output_tokens(
        "TRACKING_AGENT_MAX_OUTPUT_TOKENS",
        DEFAULT_TRACKING_AGENT_MAX_OUTPUT_TOKENS,
    )


@dataclass(slots=True)
class VisualTrackingAgentResult:
    available: bool
    proposals: list[VisualAssociationProposal]
    uncertainty_flags: list[str]
    supporting_evidence: list[str]
    latency_ms: float | None
    error: str | None = None
    llm_correlation_id: str | None = None


class VisualTrackingAgentProtocol(Protocol):
    model: str

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    # Implements propose associations behavior for this module.
    ) -> VisualTrackingAgentResult: ...


class LLMVisualTrackingAgent:
    def __init__(
        self,
        model: str = TRACKING_VLM_MODEL,
        openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
        system_prompt: str | None = None,
    ) -> None:
        # Implements init behavior for this module.
        load_dotenv()
        self.model = model
        self._openai_base_url = os.getenv("OPENAI_BASE_URL", openai_base_url)
        self._api_key = api_key
        self._client = client
        self._system_prompt = (
            system_prompt.strip()
            if isinstance(system_prompt, str) and system_prompt.strip()
            else load_tracking_agent_skill()
        )

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult:
        # Implements propose associations behavior for this module.
        request_correlation_id = latency_instrumentation.new_correlation_id("tracking-llm")
        if not detections:
            return VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["no_detections"],
                supporting_evidence=[],
                latency_ms=0.0,
                llm_correlation_id=request_correlation_id,
            )

        if self._client is None:
            load_dotenv()
            api_key = self._api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                return VisualTrackingAgentResult(
                    available=False,
                    proposals=[],
                    uncertainty_flags=["vlm_api_key_missing"],
                    supporting_evidence=[],
                    latency_ms=None,
                    error="vlm_api_key_missing",
                    llm_correlation_id=request_correlation_id,
                )
            self._client = AsyncOpenAI(base_url=self._openai_base_url, api_key=api_key)

        payload = {
            "event_reasons": event_reasons or [],
            "detections": [item.to_dict() for item in detections],
            "instructions": {
                "do_not_invent_measurements": True,
                "do_not_override_reliable_tracking_without_evidence": True,
                "return_json_only": True,
            },
        }

        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Analyze visual continuity and propose track associations. "
                    "Return JSON object with keys: proposals, uncertainty_flags, supporting_evidence. "
                    "Each proposal should include detection_id, proposed_track_id, association_confidence, "
                    "occlusion_likelihood, id_switch_risk, depth_conflict_flag, position_conflict_flag, "
                    "uncertainties, supporting_evidence. Preserve one proposal per detection requiring evaluation; "
                    "do not omit detections because of output length. Keep uncertainty/supporting_evidence entries concise "
                    "short phrases while preserving material information. Return JSON only, with no Markdown or extra explanation.\n"
                    f"Structured input:\n{json.dumps(payload)}"
                ),
            }
        ]
        if frame_reference:
            user_content.append({"type": "image_url", "image_url": {"url": frame_reference}})

        try:
            reservation = provider_call_budget.reserve_or_raise(provider_call_budget.ROLE_TRACKING_VLM)
        except provider_call_budget.ProviderCallBudgetExceeded:
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_budget_exhausted"],
                supporting_evidence=[],
                latency_ms=0.0,
                error="vlm_budget_exhausted",
                llm_correlation_id=request_correlation_id,
            )
        started = time.perf_counter()
        latency_instrumentation.emit_event(
            "llm_request_started",
            stage="tracking_agent_llm_request",
            duration_ms=0.0,
            role="tracking_agent",
            provider="openai_compatible",
            model=self.model,
            timeout_seconds=request_timeout_seconds,
            retry_limit=0,
            attempt_number=1,
            attempts_total=1,
            correlation_id=request_correlation_id,
            detection_count=len(detections),
            frame_reference_present=bool(frame_reference),
        )
        try:
            max_output_tokens = _resolve_tracking_max_output_tokens()
            request_kwargs: dict[str, object] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._system_prompt,
                    },
                    {"role": "user", "content": user_content},
                ],
            }
            if max_output_tokens is not None:
                request_kwargs["max_tokens"] = max_output_tokens
            response = await asyncio.wait_for(
                self._client.chat.completions.create(**request_kwargs),
                timeout=request_timeout_seconds,
            )
            provider_call_budget.finalize_provider_call_attempt(
                reservation,
                outcome="ok",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                usage=llm_call.extract_usage_dict(response),
            )
            latency_instrumentation.emit_event(
                "llm_request_completed",
                stage="tracking_agent_llm_request",
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                role="tracking_agent",
                provider="openai_compatible",
                model=self.model,
                timeout_seconds=request_timeout_seconds,
                retry_limit=0,
                attempt_number=1,
                attempts_total=1,
                correlation_id=request_correlation_id,
                status="ok",
                usage=llm_call.extract_usage_dict(response),
            )
        except asyncio.TimeoutError:
            provider_call_budget.finalize_provider_call_attempt(
                reservation,
                outcome="error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error_code="timeout",
            )
            latency_instrumentation.emit_event(
                "llm_request_completed",
                stage="tracking_agent_llm_request",
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                role="tracking_agent",
                provider="openai_compatible",
                model=self.model,
                timeout_seconds=request_timeout_seconds,
                retry_limit=0,
                attempt_number=1,
                attempts_total=1,
                correlation_id=request_correlation_id,
                status="error",
                error_code="timeout",
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_timeout"],
                supporting_evidence=[],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="vlm_timeout",
                llm_correlation_id=request_correlation_id,
            )
        except Exception as error:
            provider_call_budget.finalize_provider_call_attempt(
                reservation,
                outcome="error",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error_code="runtime_error",
            )
            try:
                exception_fields = _extract_exception_telemetry_fields(error)
            except Exception:
                exception_fields = {}
            latency_instrumentation.emit_event(
                "llm_request_completed",
                stage="tracking_agent_llm_request",
                duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                role="tracking_agent",
                provider="openai_compatible",
                model=self.model,
                timeout_seconds=request_timeout_seconds,
                retry_limit=0,
                attempt_number=1,
                attempts_total=1,
                correlation_id=request_correlation_id,
                status="error",
                error_code="runtime_error",
                **exception_fields,
            )
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_unavailable"],
                supporting_evidence=[],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="vlm_unavailable",
                llm_correlation_id=request_correlation_id,
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        if not response.choices:
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_empty_response"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_empty_response",
                llm_correlation_id=request_correlation_id,
            )

        content = response.choices[0].message.content or ""
        if getattr(response.choices[0], "finish_reason", None) == "length":
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_output_truncated"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_output_truncated",
                llm_correlation_id=request_correlation_id,
            )
        parsed = _extract_json_object(content)
        if parsed is None:
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_invalid_json_response"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_invalid_json_response",
                llm_correlation_id=request_correlation_id,
            )

        raw_proposals = parsed.get("proposals")
        uncertainty_flags = parsed.get("uncertainty_flags")
        supporting_evidence = parsed.get("supporting_evidence")

        schema_complete = (
            isinstance(raw_proposals, list)
            and isinstance(uncertainty_flags, list)
            and isinstance(supporting_evidence, list)
        )
        if not isinstance(raw_proposals, list):
            raw_proposals = []
        if not isinstance(uncertainty_flags, list):
            uncertainty_flags = []
        if not isinstance(supporting_evidence, list):
            supporting_evidence = []
        normalized_proposals: list[VisualAssociationProposal] = []
        for item in raw_proposals:
            if isinstance(item, dict):
                normalized = _normalize_proposal(item)
                if normalized is not None:
                    normalized_proposals.append(normalized)

        normalized_uncertainty_flags = [str(value) for value in uncertainty_flags]
        if schema_complete:
            normalized_uncertainty_flags.append("schema_complete_response")
        else:
            normalized_uncertainty_flags.append("schema_incomplete_response")

        return VisualTrackingAgentResult(
            available=True,
            proposals=normalized_proposals,
            uncertainty_flags=normalized_uncertainty_flags,
            supporting_evidence=[str(value) for value in supporting_evidence],
            latency_ms=latency_ms,
            llm_correlation_id=request_correlation_id,
        )


class AsyncVisualTrackingRuntime:
    def __init__(
        self,
        coordinator: AgentCoordinator,
        agent: VisualTrackingAgentProtocol,
        min_invoke_interval_seconds: float = 0.5,
    ) -> None:
        # Implements init behavior for this module.
        self._coordinator = coordinator
        self._agent = agent
        self._min_invoke_interval_seconds = min_invoke_interval_seconds
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future: Future | None = None
        self._last_started_at = 0.0
        self._completed_observations: deque = deque()

    def _consume_completed_future_locked(self) -> None:
        pending_future = self._pending_future
        if pending_future is None or not pending_future.done():
            return

        self._pending_future = None
        try:
            observation = pending_future.result()
        except Exception:
            observation = None

        if observation is not None:
            self._completed_observations.append(observation)

    def submit_if_needed(
        self,
        now: float,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> bool:
        # Implements submit if needed behavior for this module.
        with self._lock:
            self._consume_completed_future_locked()
            if self._pending_future is not None and not self._pending_future.done():
                return False
            if (now - self._last_started_at) < self._min_invoke_interval_seconds and not event_reasons:
                return False
            self._last_started_at = now
            observation_id = latency_instrumentation.new_correlation_id("tracking-observation")
            self._pending_future = self._executor.submit(
                self._run_once,
                now,
                observation_id,
                frame_reference,
                detections,
                event_reasons or [],
                request_timeout_seconds,
            )
            return True

    def poll_latest_observation(self):
        # Implements poll latest observation behavior for this module.
        with self._lock:
            self._consume_completed_future_locked()
            if not self._completed_observations:
                return None
            return self._completed_observations.popleft()

    def is_busy(self) -> bool:
        # Implements is busy behavior for this module.
        with self._lock:
            return self._pending_future is not None and not self._pending_future.done()

    def shutdown(self) -> None:
        # Implements shutdown behavior for this module.
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_once(
        self,
        now: float,
        observation_id: str,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str],
        request_timeout_seconds: float,
    ):
        # Runs once workflow steps and returns status outcomes.
        result = asyncio.run(
            self._agent.propose_associations(
                frame_reference=frame_reference,
                detections=detections,
                event_reasons=event_reasons,
                request_timeout_seconds=request_timeout_seconds,
            )
        )

        uncertainties = list(result.uncertainty_flags)
        if not result.available and result.error is not None:
            uncertainties.append(result.error)

        return self._coordinator.build_tracking_agent_observation(
            detections=detections,
            proposals=result.proposals,
            timestamp=now,
            vlm_model=self._agent.model,
            frame_reference=frame_reference,
            proposal_latency_ms=result.latency_ms,
            observation_id=observation_id,
            llm_correlation_id=result.llm_correlation_id,
            additional_uncertainties=uncertainties,
        )


def encode_frame_as_data_url(frame) -> str:
    # Implements encode frame as data url behavior for this module.
    ok, encoded = __import__("cv2").imencode(".jpg", frame)
    if not ok:
        raise ValueError("unable_to_encode_frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _extract_json_object(raw_text: str) -> dict | None:
    # Extracts json object from structured payloads.
    text = raw_text.strip()
    if not text:
        return None

    if text.startswith("```"):
        parts = text.split("```")
        text = ""
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
        if not text:
            return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def _normalize_proposal(payload: dict) -> VisualAssociationProposal | None:
    # Normalizes proposal into contract-safe values.
    detection_id = payload.get("detection_id")
    if not isinstance(detection_id, str):
        return None

    proposed_track_id = payload.get("proposed_track_id")
    if proposed_track_id is not None and not isinstance(proposed_track_id, str):
        proposed_track_id = None

    association_confidence = payload.get("association_confidence")
    if not isinstance(association_confidence, (int, float)):
        association_confidence = None

    occlusion_likelihood = payload.get("occlusion_likelihood")
    if not isinstance(occlusion_likelihood, (int, float)):
        occlusion_likelihood = None

    id_switch_risk = payload.get("id_switch_risk")
    if not isinstance(id_switch_risk, (int, float)):
        id_switch_risk = None

    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list):
        uncertainties = []

    supporting_evidence = payload.get("supporting_evidence")
    if not isinstance(supporting_evidence, list):
        supporting_evidence = []

    return VisualAssociationProposal(
        detection_id=detection_id,
        proposed_track_id=proposed_track_id,
        association_confidence=None if association_confidence is None else float(association_confidence),
        occlusion_likelihood=None if occlusion_likelihood is None else float(occlusion_likelihood),
        id_switch_risk=None if id_switch_risk is None else float(id_switch_risk),
        depth_conflict_flag=bool(payload.get("depth_conflict_flag", False)),
        position_conflict_flag=bool(payload.get("position_conflict_flag", False)),
        uncertainties=[str(value) for value in uncertainties],
        supporting_evidence=[str(value) for value in supporting_evidence],
    )

