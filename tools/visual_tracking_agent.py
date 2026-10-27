from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .agent_contracts import DetectionCandidate, VisualAssociationProposal
from .agent_coordinator import AgentCoordinator

GEMINI_TRACKING_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_BASE_URL = "https://128.171.10.83:9443/v1"


@dataclass(slots=True)
class VisualTrackingAgentResult:
    available: bool
    proposals: list[VisualAssociationProposal]
    uncertainty_flags: list[str]
    supporting_evidence: list[str]
    latency_ms: float | None
    error: str | None = None


class VisualTrackingAgentProtocol(Protocol):
    model: str

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult: ...


class GeminiVisualTrackingAgent:
    def __init__(
        self,
        model: str = GEMINI_TRACKING_MODEL,
        openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        load_dotenv()
        self.model = model
        self._openai_base_url = os.getenv("OPENAI_BASE_URL", openai_base_url)
        self._api_key = api_key
        self._client = client

    async def propose_associations(
        self,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> VisualTrackingAgentResult:
        if not detections:
            return VisualTrackingAgentResult(
                available=True,
                proposals=[],
                uncertainty_flags=["no_detections"],
                supporting_evidence=[],
                latency_ms=0.0,
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
                    "uncertainties, supporting_evidence.\n"
                    f"Structured input:\n{json.dumps(payload)}"
                ),
            }
        ]
        if frame_reference:
            user_content.append({"type": "image_url", "image_url": {"url": frame_reference}})

        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a visual tracking agent. Focus on association continuity, occlusions, "
                                "ID-switch risk, and conflicts between depth/position evidence. "
                                "Never invent physical measurements."
                            ),
                        },
                        {"role": "user", "content": user_content},
                    ],
                ),
                timeout=request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_timeout"],
                supporting_evidence=[],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="vlm_timeout",
            )
        except Exception:
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_unavailable"],
                supporting_evidence=[],
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="vlm_unavailable",
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
            )

        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        if parsed is None:
            return VisualTrackingAgentResult(
                available=False,
                proposals=[],
                uncertainty_flags=["vlm_invalid_json_response"],
                supporting_evidence=[],
                latency_ms=latency_ms,
                error="vlm_invalid_json_response",
            )

        raw_proposals = parsed.get("proposals")
        normalized_proposals: list[VisualAssociationProposal] = []
        if isinstance(raw_proposals, list):
            for item in raw_proposals:
                if isinstance(item, dict):
                    normalized = _normalize_proposal(item)
                    if normalized is not None:
                        normalized_proposals.append(normalized)

        uncertainty_flags = parsed.get("uncertainty_flags")
        if not isinstance(uncertainty_flags, list):
            uncertainty_flags = []

        supporting_evidence = parsed.get("supporting_evidence")
        if not isinstance(supporting_evidence, list):
            supporting_evidence = []

        return VisualTrackingAgentResult(
            available=True,
            proposals=normalized_proposals,
            uncertainty_flags=[str(value) for value in uncertainty_flags],
            supporting_evidence=[str(value) for value in supporting_evidence],
            latency_ms=latency_ms,
        )


class AsyncVisualTrackingRuntime:
    def __init__(
        self,
        coordinator: AgentCoordinator,
        agent: VisualTrackingAgentProtocol,
        min_invoke_interval_seconds: float = 0.5,
    ) -> None:
        self._coordinator = coordinator
        self._agent = agent
        self._min_invoke_interval_seconds = min_invoke_interval_seconds
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future: Future | None = None
        self._last_started_at = 0.0
        self._latest_observation = None

    def submit_if_needed(
        self,
        now: float,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str] | None = None,
        request_timeout_seconds: float = 2.0,
    ) -> bool:
        with self._lock:
            if self._pending_future is not None and not self._pending_future.done():
                return False
            if (now - self._last_started_at) < self._min_invoke_interval_seconds and not event_reasons:
                return False
            self._last_started_at = now
            self._pending_future = self._executor.submit(
                self._run_once,
                now,
                frame_reference,
                detections,
                event_reasons or [],
                request_timeout_seconds,
            )
            return True

    def poll_latest_observation(self):
        with self._lock:
            if self._pending_future is not None and self._pending_future.done():
                self._latest_observation = self._pending_future.result()
                self._pending_future = None
            value = self._latest_observation
            self._latest_observation = None
            return value

    def is_busy(self) -> bool:
        with self._lock:
            return self._pending_future is not None and not self._pending_future.done()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run_once(
        self,
        now: float,
        frame_reference: str | None,
        detections: list[DetectionCandidate],
        event_reasons: list[str],
        request_timeout_seconds: float,
    ):
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
            additional_uncertainties=uncertainties,
        )


def encode_frame_as_data_url(frame) -> str:
    ok, encoded = __import__("cv2").imencode(".jpg", frame)
    if not ok:
        raise ValueError("unable_to_encode_frame")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _extract_json_object(raw_text: str) -> dict | None:
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
