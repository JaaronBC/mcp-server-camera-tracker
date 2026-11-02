from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path

from agents import provider_call_budget


def _worker(env: dict[str, str], role: str, out_queue: mp.Queue) -> None:
    os.environ.update(env)
    try:
        reservation = provider_call_budget.reserve_provider_call_attempt(role)
    except Exception as exc:
        out_queue.put({"ok": False, "error": str(exc)})
        return
    out_queue.put({"ok": bool(reservation.allowed), "reason": reservation.reason})


class TestPhase16EBudgetMultiprocess(unittest.TestCase):
    def test_exactly_one_reservation_with_shared_budget_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_budget_multi_") as tmp:
            run_dir = Path(tmp)
            state_path = run_dir / "provider_call_budget_state.json"
            log_path = run_dir / "provider_call_budget_events.jsonl"
            lock_path = run_dir / "provider_call_budget_state.lock"

            env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "1",
                "PROVIDER_CALL_BUDGET_TOTAL_LIMIT": "1",
                "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT": "0",
                "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT": "0",
                "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT": "1",
                "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT": "0",
                "PROVIDER_CALL_BUDGET_STATE_PATH": str(state_path),
                "PROVIDER_CALL_BUDGET_LOG_PATH": str(log_path),
                "PROVIDER_CALL_BUDGET_LOCK_PATH": str(lock_path),
            }

            queue: mp.Queue = mp.Queue()
            processes = [
                mp.Process(target=_worker, args=(env, "observation_visual", queue))
                for _ in range(4)
            ]
            for proc in processes:
                proc.start()
            for proc in processes:
                proc.join(timeout=20)
                self.assertFalse(proc.is_alive(), "worker process did not terminate")

            results = [queue.get(timeout=5) for _ in processes]
            success_count = sum(1 for item in results if item.get("ok"))
            failure_count = sum(1 for item in results if not item.get("ok"))

            self.assertEqual(success_count, 1)
            self.assertEqual(failure_count, len(processes) - 1)

            self.assertTrue(state_path.is_file())
            self.assertTrue(log_path.is_file())

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["totals"]["attempted"], 1)
            self.assertEqual(state["totals"]["per_role_attempted"]["observation_visual"], 1)


if __name__ == "__main__":
    unittest.main()
