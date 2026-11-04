from __future__ import annotations

import multiprocessing
import os
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import patch

from agents import provider_call_budget
from agents.observation import agent_observation


def _env_for_budget(temp_dir: str, **overrides: str) -> dict[str, str]:
    base = {
        "PROVIDER_CALL_BUDGET_ENABLED": "1",
        "PROVIDER_CALL_BUDGET_TOTAL_LIMIT": "999",
        "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT": "999",
        "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT": "999",
        "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT": "999",
        "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT": "999",
        "PROVIDER_CALL_BUDGET_STATE_PATH": str(Path(temp_dir) / "budget_state.json"),
        "PROVIDER_CALL_BUDGET_LOCK_PATH": str(Path(temp_dir) / "budget_state.lock"),
        "PROVIDER_CALL_BUDGET_LOG_PATH": str(Path(temp_dir) / "budget_events.jsonl"),
    }
    base.update(overrides)
    return base


def _cross_process_worker(env: dict[str, str], attempts: int, result_queue: multiprocessing.Queue) -> None:
    os.environ.update(env)
    allowed = 0
    blocked = 0
    for _ in range(attempts):
        reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)
        if reservation.allowed:
            allowed += 1
            provider_call_budget.finalize_provider_call_attempt(
                reservation,
                outcome="ok",
                latency_ms=1.0,
                usage={"total_tokens": 5},
            )
        else:
            blocked += 1
    result_queue.put({"allowed": allowed, "blocked": blocked})


class _FakeSuccessChoice:
    def __init__(self, payload: str) -> None:
        self.message = type("Message", (), {"content": payload})


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, payload: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> None:
        self.choices = [_FakeSuccessChoice(payload)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens)


class _FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = outcomes

    async def create(self, *args, **kwargs):
        if not self._outcomes:
            raise RuntimeError("no_outcomes")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(outcomes)})

    def close(self):
        return None


class TestProviderCallBudget(unittest.TestCase):
    def test_per_role_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT="1",
                PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT="2",
            )
            with patch.dict(os.environ, env, clear=False):
                first = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)
                second = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)
                text_first = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_OBSERVATION_TEXT)
                text_second = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_OBSERVATION_TEXT)
                text_third = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_OBSERVATION_TEXT)

        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertEqual(second.reason, "role_limit_exhausted")
        self.assertTrue(text_first.allowed)
        self.assertTrue(text_second.allowed)
        self.assertFalse(text_third.allowed)

    def test_global_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="2",
                PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT="10",
                PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT="10",
            )
            with patch.dict(os.environ, env, clear=False):
                one = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)
                two = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_OBSERVATION_TEXT)
                three = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_REASONING_TEXT)

        self.assertTrue(one.allowed)
        self.assertTrue(two.allowed)
        self.assertFalse(three.allowed)
        self.assertEqual(three.reason, "global_limit_exhausted")

    def test_concurrent_attempts_respect_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="3",
                PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT="10",
            )
            with patch.dict(os.environ, env, clear=False):
                results: Queue[bool] = Queue()

                def _reserve() -> None:
                    reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)
                    results.put(reservation.allowed)

                threads = [threading.Thread(target=_reserve) for _ in range(10)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                allowed_count = 0
                blocked_count = 0
                while not results.empty():
                    if results.get_nowait():
                        allowed_count += 1
                    else:
                        blocked_count += 1

        self.assertEqual(allowed_count, 3)
        self.assertEqual(blocked_count, 7)

    def test_cross_process_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="4",
                PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT="99",
            )
            ctx = multiprocessing.get_context("spawn")
            result_queue: multiprocessing.Queue = ctx.Queue()
            process_a = ctx.Process(target=_cross_process_worker, args=(env, 4, result_queue))
            process_b = ctx.Process(target=_cross_process_worker, args=(env, 4, result_queue))
            process_a.start()
            process_b.start()
            process_a.join(timeout=10)
            process_b.join(timeout=10)

            self.assertEqual(process_a.exitcode, 0)
            self.assertEqual(process_b.exitcode, 0)

            first = result_queue.get(timeout=2)
            second = result_queue.get(timeout=2)

        self.assertEqual(first["allowed"] + second["allowed"], 4)
        self.assertEqual(first["blocked"] + second["blocked"], 4)

    def test_retry_accounting_counts_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="2",
                PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT="2",
                DASHBOARD_AGENT_LLM_RETRY_LIMIT="1",
                DASHBOARD_AGENT_LLM_RETRY_BACKOFF_SECONDS="0",
            )
            outcomes: list[object] = [
                RuntimeError("transient_failure"),
                _FakeResponse(
                    '{"summary":"ok","measured_facts":[],"interpretations":[],"hypotheses":[],"supporting_observation_ids":[],"conflicting_evidence":[],"uncertainties":[],"needs_more_observations":false}'
                ),
            ]

            with patch.dict(os.environ, env, clear=False):
                with patch(
                    "agents.observation.agent_observation._create_openai_client",
                    side_effect=lambda: _FakeClient(outcomes),
                ):
                    payload, meta = agent_observation.call_observation_agent_with_retry(
                        {"track_id": "track-1"},
                        model="gpt-oss-120b",
                        timeout_seconds=0.5,
                    )

                totals = provider_call_budget.read_budget_totals()

        self.assertIsNotNone(payload)
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["attempts"], 2)
        self.assertEqual(totals["totals"]["attempted"], 2)
        self.assertEqual(totals["totals"]["per_role_attempted"][provider_call_budget.ROLE_OBSERVATION_TEXT], 2)

    def test_budget_exhaustion_blocks_further_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="1",
                PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT="1",
            )
            with patch.dict(os.environ, env, clear=False):
                first = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_REASONING_TEXT)
                second = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_REASONING_TEXT)

        self.assertTrue(first.allowed)
        self.assertFalse(second.allowed)
        self.assertEqual(second.reason, "global_limit_exhausted")

    def test_unrestricted_normal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_ENABLED="0",
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="1",
                PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT="1",
            )
            with patch.dict(os.environ, env, clear=False):
                reservations = [
                    provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)
                    for _ in range(10)
                ]

        self.assertTrue(all(item.allowed for item in reservations))
        self.assertTrue(all(not item.enabled for item in reservations))

    def test_enabled_budget_rejects_invalid_numeric_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="invalid",
            )
            with patch.dict(os.environ, env, clear=False):
                reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)

        self.assertFalse(reservation.allowed)
        self.assertTrue(reservation.enabled)
        self.assertEqual(reservation.reason, "invalid_limits_configuration")

    def test_enabled_budget_rejects_missing_required_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(temp_dir)
            env.pop("PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT", None)
            with patch.dict(os.environ, env, clear=False):
                os.environ.pop("PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT", None)
                reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_REASONING_TEXT)

        self.assertFalse(reservation.allowed)
        self.assertEqual(reservation.reason, "invalid_limits_configuration")

    def test_corrupted_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(temp_dir)
            state_path = Path(env["PROVIDER_CALL_BUDGET_STATE_PATH"])
            state_path.write_text("{not-json", encoding="utf-8")
            with patch.dict(os.environ, env, clear=False):
                reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)

        self.assertFalse(reservation.allowed)
        self.assertEqual(reservation.reason, "state_corrupted")

    def test_inaccessible_state_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(temp_dir)
            state_path = Path(env["PROVIDER_CALL_BUDGET_STATE_PATH"])
            state_path.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, env, clear=False):
                with patch("agents.provider_call_budget.Path.read_text", side_effect=OSError("permission denied")):
                    reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)

        self.assertFalse(reservation.allowed)
        self.assertEqual(reservation.reason, "state_unreadable")

    def test_persistence_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(temp_dir)
            with patch.dict(os.environ, env, clear=False):
                with patch("agents.provider_call_budget._write_state", side_effect=PermissionError("read-only")):
                    reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)

        self.assertFalse(reservation.allowed)
        self.assertEqual(reservation.reason, "reservation_failed")

    def test_lock_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(temp_dir)
            with patch.dict(os.environ, env, clear=False):
                with patch("agents.provider_call_budget.fcntl.flock", side_effect=OSError("lock failed")):
                    reservation = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)

        self.assertFalse(reservation.allowed)
        self.assertEqual(reservation.reason, "reservation_failed")

    def test_failed_reservation_does_not_invoke_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="not-a-number",
            )
            with patch.dict(os.environ, env, clear=False):
                with patch("agents.observation.agent_observation._create_openai_client") as create_client:
                    payload, meta = agent_observation.call_observation_agent_with_retry(
                        {"track_id": "track-1"},
                        model="gpt-oss-120b",
                        timeout_seconds=0.2,
                        retry_limit=0,
                    )

        self.assertIsNone(payload)
        self.assertEqual(meta["status"], "error")
        self.assertEqual(meta["error_code"], "budget_exhausted")
        create_client.assert_not_called()

    def test_existing_budget_history_is_not_reset_on_limit_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_initial = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="10",
                PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT="10",
            )
            with patch.dict(os.environ, env_initial, clear=False):
                first = provider_call_budget.reserve_provider_call_attempt(provider_call_budget.ROLE_TRACKING_VLM)
                self.assertTrue(first.allowed)
                totals_after_first = provider_call_budget.read_budget_totals()

            env_changed = _env_for_budget(
                temp_dir,
                PROVIDER_CALL_BUDGET_TOTAL_LIMIT="20",
                PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT="20",
            )
            with patch.dict(os.environ, env_changed, clear=False):
                totals_after_change = provider_call_budget.read_budget_totals()

        self.assertEqual(totals_after_first["totals"]["attempted"], 1)
        self.assertEqual(totals_after_change["totals"]["attempted"], 1)
        self.assertEqual(
            totals_after_change["totals"]["per_role_attempted"][provider_call_budget.ROLE_TRACKING_VLM],
            1,
        )


if __name__ == "__main__":
    unittest.main()
