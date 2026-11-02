from __future__ import annotations

import json
import os
import tempfile
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp.client import stdio as mcp_stdio

from agents.observation.agent_observation import (
    ObservationVisualVerificationError,
    ObservationVisualVerificationRequest,
    verify_tracking_evidence_for_observation,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_envelope_payload(*, message_id: str, frame_reference: str, detection_reference: str) -> dict:
    return {
        "schema_version": "1.0",
        "message_id": message_id,
        "track_id": "track-1",
        "timestamp": 1000.0,
        "frame_id": "frame-1",
        "detection_id": "det-1",
        "pixel_bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
        "pixel_center": {"x": 20.0, "y": 40.0},
        "relative_depth": 0.3,
        "ground_position": {"x": 1.0, "z": 2.0, "unit": "meters"},
        "tracking_confidence": None,
        "quality_flags": ["position_kind:calibrated_ground_plane"],
        "track_history": [
            {
                "observation_id": "obs-a",
                "track_id": "track-1",
                "timestamp": 998.0,
                "source": "tracking_frame",
                "pixel_x": 19.0,
                "pixel_y": 39.0,
                "ground_point": {"x": 0.9, "z": 1.9, "unit": "meters"},
                "confidence": 0.7,
                "quality_flags": ["position_kind:calibrated_ground_plane"],
            },
            {
                "observation_id": "obs-b",
                "track_id": "track-1",
                "timestamp": 1000.0,
                "source": "tracking_frame",
                "pixel_x": 20.0,
                "pixel_y": 40.0,
                "ground_point": {"x": 1.0, "z": 2.0, "unit": "meters"},
                "confidence": 0.8,
                "quality_flags": ["position_kind:calibrated_ground_plane"],
            },
        ],
        "tracking_claims": {
            "claimed_resolved_track_id": "track-1",
            "association_status": "accepted",
            "possible_id_switch": False,
            "visibility_status": "unknown",
            "claim_reasons": ["vlm_association_accepted"],
        },
        "frame_reference": frame_reference,
        "source_evidence_refs": [frame_reference, detection_reference],
        "uncertainty": [],
    }


def _verification_payload() -> dict:
    return {
        "verification_status": "supported",
        "visibility_status": "visible",
        "identity_consistency": "consistent",
        "possible_id_switch": False,
        "confidence": 0.82,
        "uncertainty": [],
        "limitations": [],
        "supporting_evidence_refs": ["evidence://frame/frame-abc"],
        "contradicting_evidence_refs": [],
        "recommended_follow_up": None,
        "visual_content_inspected": True,
    }


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.choices = [_FakeChoice(json.dumps(payload))]
        self.usage = None


class _FakeCompletionsSuccess:
    def __init__(self, state: dict[str, int]) -> None:
        self._state = state

    async def create(self, *args, **kwargs):
        self._state["provider_calls"] += 1
        return _FakeResponse(_verification_payload())


class _FakeCompletionsError:
    def __init__(self, state: dict[str, int]) -> None:
        self._state = state

    async def create(self, *args, **kwargs):
        self._state["provider_calls"] += 1
        raise RuntimeError("upstream down")


class _FakeCompletionsSlow:
    def __init__(self, state: dict[str, int], *, sleep_seconds: float) -> None:
        self._state = state
        self._sleep_seconds = sleep_seconds

    async def create(self, *args, **kwargs):
        self._state["provider_calls"] += 1
        await asyncio.sleep(self._sleep_seconds)
        return _FakeResponse(_verification_payload())


class _FakeChat:
    def __init__(self, completions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions) -> None:
        self.chat = _FakeChat(completions)

    def close(self):
        return None


def _read_budget_attempt_count(state_path: Path) -> int:
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    return int(((state_payload.get("totals") or {}).get("attempted") or 0))


def _read_provider_attempt_events(log_path: Path) -> list[dict]:
    if not log_path.is_file():
        return []
    events: list[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("event_type") == "provider_call_attempt":
            events.append(payload)
    return events


class TestObservationVisualBudgetIntegration(unittest.TestCase):
    _BUDGET_ENV_KEYS = (
        "PROVIDER_CALL_BUDGET_ENABLED",
        "PROVIDER_CALL_BUDGET_TOTAL_LIMIT",
        "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT",
        "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT",
        "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT",
        "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT",
        "PROVIDER_CALL_BUDGET_STATE_PATH",
        "PROVIDER_CALL_BUDGET_LOCK_PATH",
        "PROVIDER_CALL_BUDGET_LOG_PATH",
        "OPENROUTER_API_KEY",
    )

    def setUp(self) -> None:
        self._env_backup = {key: os.environ.get(key) for key in self._BUDGET_ENV_KEYS}

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _build_valid_store(self, root: Path, *, message_id: str = "msg-1") -> Path:
        evidence_root = root / "tracking_evidence"
        refs_dir = evidence_root / "refs"
        frames_dir = evidence_root / "frames"
        envelopes_dir = evidence_root / "envelopes"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_id = "frame-abc"
        frame_reference = f"evidence://frame/{frame_id}"
        frame_path = frames_dir / f"{frame_id}.jpg"
        frame_path.write_bytes(b"fake-jpeg-bytes")

        _write_json(
            refs_dir / f"{frame_id}.json",
            {
                "kind": "frame",
                "id": frame_id,
                "reference": frame_reference,
                "timestamp": 1000.0,
                "created_at": 1000.0,
                "expires_at": 2000.0,
                "status": "available",
                "frame_path": str(frame_path),
                "width": 32,
                "height": 24,
            },
        )

        det_ref_id = "detref-abc"
        detection_reference = f"evidence://detection/{det_ref_id}"
        _write_json(
            refs_dir / f"{det_ref_id}.json",
            {
                "kind": "detection",
                "id": det_ref_id,
                "reference": detection_reference,
                "timestamp": 1000.0,
                "created_at": 1000.0,
                "expires_at": 2000.0,
                "track_id": "track-1",
                "detection_id": "det-1",
                "frame_reference": frame_reference,
                "pixel_bbox": {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0},
            },
        )

        _write_json(
            envelopes_dir / f"{message_id}.json",
            {
                "message_id": message_id,
                "stored_at": 1001.0,
                "validation": {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "message_id": message_id,
                    "track_id": "track-1",
                },
                "envelope": _valid_envelope_payload(
                    message_id=message_id,
                    frame_reference=frame_reference,
                    detection_reference=detection_reference,
                ),
            },
        )

        return evidence_root

    def _set_budget_env(self, root: Path, *, total_limit: int, visual_limit: int) -> tuple[Path, Path, Path]:
        budget_dir = root / "budget"
        budget_dir.mkdir(parents=True, exist_ok=True)
        state_path = budget_dir / "provider_call_budget_state.json"
        lock_path = budget_dir / "provider_call_budget_state.lock"
        log_path = budget_dir / "provider_call_budget_events.jsonl"

        os.environ["PROVIDER_CALL_BUDGET_ENABLED"] = "1"
        os.environ["PROVIDER_CALL_BUDGET_TOTAL_LIMIT"] = str(total_limit)
        os.environ["PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT"] = "0"
        os.environ["PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT"] = "0"
        os.environ["PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT"] = "0"
        os.environ["PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT"] = str(visual_limit)
        os.environ["PROVIDER_CALL_BUDGET_STATE_PATH"] = str(state_path)
        os.environ["PROVIDER_CALL_BUDGET_LOCK_PATH"] = str(lock_path)
        os.environ["PROVIDER_CALL_BUDGET_LOG_PATH"] = str(log_path)
        os.environ["OPENROUTER_API_KEY"] = "fake-local-key"

        return state_path, lock_path, log_path

    def test_first_visual_request_counted_second_rejected_pre_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = self._build_valid_store(root, message_id="msg-budget")
            state_path, _, log_path = self._set_budget_env(root, total_limit=1, visual_limit=1)

            call_state = {"provider_calls": 0, "client_factory_calls": 0}

            def _fake_client_factory():
                call_state["client_factory_calls"] += 1
                return _FakeClient(_FakeCompletionsSuccess(call_state))

            with patch("agents.observation.agent_observation._create_openai_client", side_effect=_fake_client_factory):
                first_result = verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-budget", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

                with self.assertRaises(ObservationVisualVerificationError) as second_error:
                    verify_tracking_evidence_for_observation(
                        ObservationVisualVerificationRequest(message_id="msg-budget", now_timestamp=1001.0),
                        evidence_root_dir=evidence_root,
                        model="qwen-3.8-27b",
                        retry_limit=0,
                    )

            self.assertTrue(first_result.validation["valid"])
            self.assertEqual(call_state["provider_calls"], 1)
            self.assertEqual(call_state["client_factory_calls"], 1)
            self.assertEqual(second_error.exception.code, "provider_failure")
            self.assertEqual(second_error.exception.details.get("error_code"), "budget_exhausted")

            self.assertTrue(state_path.is_file())
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state_payload["totals"]["attempted"], 1)
            self.assertEqual(state_payload["totals"]["per_role_attempted"]["observation_visual"], 1)

            self.assertTrue(log_path.is_file())
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            attempt_events = [event for event in events if event.get("event_type") == "provider_call_attempt"]
            blocked_events = [event for event in events if event.get("event_type") == "provider_call_blocked"]
            self.assertEqual(len(attempt_events), 1)
            self.assertEqual(attempt_events[0].get("role"), "observation_visual")
            self.assertEqual(len(blocked_events), 1)
            self.assertEqual(blocked_events[0].get("role"), "observation_visual")
            self.assertEqual(_read_budget_attempt_count(state_path), len(attempt_events))

    def test_retry_path_requires_separate_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = self._build_valid_store(root, message_id="msg-retry")
            state_path, _, log_path = self._set_budget_env(root, total_limit=2, visual_limit=2)

            call_state = {"provider_calls": 0, "client_factory_calls": 0}

            def _fake_client_factory():
                call_state["client_factory_calls"] += 1
                return _FakeClient(_FakeCompletionsError(call_state))

            with patch("agents.observation.agent_observation._create_openai_client", side_effect=_fake_client_factory):
                with self.assertRaises(ObservationVisualVerificationError) as error:
                    verify_tracking_evidence_for_observation(
                        ObservationVisualVerificationRequest(message_id="msg-retry", now_timestamp=1001.0),
                        evidence_root_dir=evidence_root,
                        model="qwen-3.8-27b",
                        retry_limit=1,
                    )

            self.assertEqual(error.exception.code, "provider_failure")
            self.assertEqual(call_state["provider_calls"], 2)
            self.assertEqual(call_state["client_factory_calls"], 2)

            self.assertTrue(state_path.is_file())
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state_payload["totals"]["attempted"], 2)
            self.assertEqual(state_payload["totals"]["per_role_attempted"]["observation_visual"], 2)

            self.assertTrue(log_path.is_file())
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            attempt_events = [event for event in events if event.get("event_type") == "provider_call_attempt"]
            self.assertEqual(len(attempt_events), 2)
            self.assertEqual(_read_budget_attempt_count(state_path), len(attempt_events))

    def test_missing_credentials_finalizes_reserved_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = self._build_valid_store(root, message_id="msg-missing-key")
            state_path, _, log_path = self._set_budget_env(root, total_limit=1, visual_limit=1)
            os.environ["OPENROUTER_API_KEY"] = ""

            with self.assertRaises(ObservationVisualVerificationError) as error:
                verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-missing-key", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

            self.assertEqual(error.exception.code, "provider_failure")
            self.assertEqual(error.exception.details.get("error_code"), "authentication")
            self.assertEqual(_read_budget_attempt_count(state_path), 1)
            attempt_events = _read_provider_attempt_events(log_path)
            self.assertEqual(len(attempt_events), 1)
            self.assertEqual(attempt_events[0].get("outcome"), "error")
            self.assertEqual(attempt_events[0].get("error_code"), "authentication")

    def test_client_initialization_failure_finalizes_reserved_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = self._build_valid_store(root, message_id="msg-client-init-fail")
            state_path, _, log_path = self._set_budget_env(root, total_limit=1, visual_limit=1)

            with patch(
                "agents.observation.agent_observation._create_openai_client",
                side_effect=RuntimeError("client_bootstrap_failed"),
            ):
                with self.assertRaises(ObservationVisualVerificationError) as error:
                    verify_tracking_evidence_for_observation(
                        ObservationVisualVerificationRequest(message_id="msg-client-init-fail", now_timestamp=1001.0),
                        evidence_root_dir=evidence_root,
                        model="qwen-3.8-27b",
                        retry_limit=0,
                    )

            self.assertEqual(error.exception.code, "provider_failure")
            self.assertEqual(error.exception.details.get("error_code"), "runtime_error")
            self.assertEqual(_read_budget_attempt_count(state_path), 1)
            attempt_events = _read_provider_attempt_events(log_path)
            self.assertEqual(len(attempt_events), 1)
            self.assertEqual(attempt_events[0].get("outcome"), "error")
            self.assertEqual(attempt_events[0].get("error_code"), "runtime_error")

    def test_successful_visual_invocation_state_and_log_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = self._build_valid_store(root, message_id="msg-success")
            state_path, _, log_path = self._set_budget_env(root, total_limit=1, visual_limit=1)

            call_state = {"provider_calls": 0, "client_factory_calls": 0}

            def _fake_client_factory():
                call_state["client_factory_calls"] += 1
                return _FakeClient(_FakeCompletionsSuccess(call_state))

            with patch("agents.observation.agent_observation._create_openai_client", side_effect=_fake_client_factory):
                result = verify_tracking_evidence_for_observation(
                    ObservationVisualVerificationRequest(message_id="msg-success", now_timestamp=1001.0),
                    evidence_root_dir=evidence_root,
                    model="qwen-3.8-27b",
                    retry_limit=0,
                )

            self.assertTrue(result.validation.get("valid"))
            self.assertEqual(call_state["provider_calls"], 1)
            self.assertEqual(call_state["client_factory_calls"], 1)
            self.assertEqual(_read_budget_attempt_count(state_path), 1)
            attempt_events = _read_provider_attempt_events(log_path)
            self.assertEqual(len(attempt_events), 1)
            self.assertEqual(attempt_events[0].get("outcome"), "ok")
            self.assertEqual(attempt_events[0].get("error_code"), None)

    def test_mcp_stdio_default_env_does_not_forward_budget_variables(self) -> None:
        os.environ["PROVIDER_CALL_BUDGET_ENABLED"] = "1"
        inherited = mcp_stdio.get_default_environment()
        self.assertIn("PATH", inherited)
        self.assertNotIn("PROVIDER_CALL_BUDGET_ENABLED", inherited)

    def test_timeout_failure_includes_sanitized_provenance_and_finalized_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evidence_root = self._build_valid_store(root, message_id="msg-timeout")
            state_path, _, log_path = self._set_budget_env(root, total_limit=1, visual_limit=1)

            call_state = {"provider_calls": 0, "client_factory_calls": 0}

            def _fake_client_factory():
                call_state["client_factory_calls"] += 1
                return _FakeClient(_FakeCompletionsSlow(call_state, sleep_seconds=0.05))

            with patch("agents.observation.agent_observation._create_openai_client", side_effect=_fake_client_factory):
                with self.assertRaises(ObservationVisualVerificationError) as error:
                    verify_tracking_evidence_for_observation(
                        ObservationVisualVerificationRequest(message_id="msg-timeout", now_timestamp=1001.0),
                        evidence_root_dir=evidence_root,
                        model="qwen-3.8-27b",
                        timeout_seconds=0.01,
                        retry_limit=0,
                    )

            self.assertEqual(error.exception.code, "provider_failure")
            details = error.exception.details
            self.assertEqual(details.get("error_code"), "timeout")
            self.assertEqual(details.get("exception_class"), "TimeoutError")
            self.assertEqual(details.get("timeout_source"), "local_asyncio_deadline")
            self.assertEqual(details.get("configured_timeout_seconds"), 0.01)
            self.assertEqual(details.get("attempts"), 1)
            self.assertGreaterEqual(float(details.get("latency_ms", 0.0)), 0.0)
            self.assertGreaterEqual(float(details.get("attempt_latency_ms", 0.0)), 0.0)

            self.assertEqual(call_state["provider_calls"], 1)
            self.assertEqual(call_state["client_factory_calls"], 1)

            self.assertEqual(_read_budget_attempt_count(state_path), 1)
            attempt_events = _read_provider_attempt_events(log_path)
            self.assertEqual(len(attempt_events), 1)
            self.assertEqual(attempt_events[0].get("outcome"), "error")
            self.assertEqual(attempt_events[0].get("error_code"), "timeout")


if __name__ == "__main__":
    unittest.main()
