from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import phase16e_run

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "tools" / "phase16e_run.py"


def _wait_for_path(path: Path, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestPhase16ERunReliability(unittest.TestCase):
    def test_load_environment_defaults_uses_repo_dotenv_without_override(self) -> None:
        repo_root = Path("/tmp/fake-phase16e-repo")
        with patch.object(phase16e_run, "load_dotenv") as load_dotenv_mock:
            phase16e_run._load_environment_defaults(repo_root)
        load_dotenv_mock.assert_called_once_with(dotenv_path=repo_root / ".env", override=False)

    def _launch_runner(
        self,
        run_dir: Path,
        child_code: str,
        *,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen[str]:
        main_command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child_code)}"
        args = [
            sys.executable,
            str(RUNNER_PATH),
            "--run-dir",
            str(run_dir),
            "--main-command",
            main_command,
            "--bounded-session-seconds",
            "60",
            "--verification-drain-seconds",
            "20",
            "--watchdog-margin-seconds",
            "15",
        ]
        if extra_args:
            args.extend(extra_args)
        return subprocess.Popen(args, cwd=str(REPO_ROOT), env=env, text=True)

    def _assert_finalized_failure(self, run_dir: Path, expected_exit: int) -> None:
        self.assertTrue((run_dir / "end_ts.txt").is_file())
        self.assertTrue((run_dir / "main_exit_code.txt").is_file())
        self.assertTrue((run_dir / "post_shutdown_processes.txt").is_file())
        self.assertTrue((run_dir / "phase16e_report.json").is_file())
        self.assertTrue((run_dir / "final_summary.json").is_file())
        self.assertTrue((run_dir / "run_interrupted.txt").is_file())

        main_exit_code = int((run_dir / "main_exit_code.txt").read_text(encoding="utf-8").strip())
        self.assertEqual(main_exit_code, expected_exit)

        report = json.loads((run_dir / "phase16e_report.json").read_text(encoding="utf-8"))
        self.assertNotEqual(report.get("verdict"), "PASS")
        self.assertTrue(report.get("checks", {}).get("run_interrupted"))
        self.assertIn("run_interrupted", report.get("failure_reasons", []))

    def test_report_subprocess_completion_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_report_ok_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "start_ts.txt").write_text(f"{time.time()}\n", encoding="utf-8")
            (run_dir / "end_ts.txt").write_text(f"{time.time()}\n", encoding="utf-8")
            (run_dir / "main_exit_code.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "run_interrupted.txt").write_text("0\n", encoding="utf-8")
            (run_dir / "post_shutdown_processes.txt").write_text("", encoding="utf-8")

            rc = phase16e_run._run_report(REPO_ROOT, run_dir, Path(sys.executable))
            self.assertEqual(rc, 0)
            self.assertTrue((run_dir / "phase16e_report.json").is_file())
            self.assertTrue((run_dir / "report_exit_code.txt").is_file())

    def test_capture_runtime_artifacts_helper_removed(self) -> None:
        self.assertFalse(hasattr(phase16e_run, "_capture_runtime_artifacts"))

    def test_run_local_artifact_status_does_not_use_global_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_run_local_only_") as tmp:
            run_dir = Path(tmp) / "run"
            stale_captures = Path(tmp) / "captures"
            run_dir.mkdir(parents=True, exist_ok=True)
            stale_captures.mkdir(parents=True, exist_ok=True)

            (stale_captures / "latest_verification_queue_results.json").write_text(
                json.dumps([{"track_id": "3", "status": "queued"}]),
                encoding="utf-8",
            )
            (stale_captures / "latest_verification_queue_submissions.json").write_text(
                json.dumps([{"request_id": "req-1", "track_id": "3", "status": "queued"}]),
                encoding="utf-8",
            )

            payload = phase16e_run._record_run_local_artifact_status(run_dir)
            missing = set(payload.get("missing", []))
            self.assertIn("latest_verification_queue_results.json", missing)
            self.assertIn("latest_verification_queue_submissions.json", missing)

    def test_start_timestamp_written_after_pre_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_start_ts_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pre_cleanup_marker = {"ts": 0.0}

            class _FakeProc:
                def __init__(self) -> None:
                    self.pid = 999999
                    self.returncode = 0

                def poll(self) -> int:
                    return 0

            def _fake_cleanup(_run_dir: Path) -> list[str]:
                pre_cleanup_marker["ts"] = time.time()
                return []

            with patch.object(
                sys,
                "argv",
                [
                    str(RUNNER_PATH),
                    "--run-dir",
                    str(run_dir),
                    "--main-command",
                    "python -c \"print('ok')\"",
                ],
            ), patch.object(phase16e_run, "_cleanup_interfering_processes", side_effect=_fake_cleanup), patch.object(
                phase16e_run.subprocess,
                "Popen",
                return_value=_FakeProc(),
            ), patch.object(phase16e_run, "_run_report", return_value=0), patch.object(
                phase16e_run,
                "_collect_post_shutdown_processes",
                return_value=[],
            ), patch.object(
                phase16e_run,
                "_record_run_local_artifact_status",
                return_value={"missing": []},
            ):
                rc = phase16e_run.main()

            self.assertEqual(rc, 0)
            start_ts = float((run_dir / "start_ts.txt").read_text(encoding="utf-8").strip())
            self.assertGreaterEqual(start_ts, pre_cleanup_marker["ts"])
            self.assertTrue((run_dir / "run_id.txt").is_file())

    def test_main_subprocess_env_includes_decision_log_and_watchdog_margin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_env_decisions_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            captured_env: dict[str, str] = {}

            class _FakeProc:
                def __init__(self) -> None:
                    self.pid = 999998
                    self.returncode = 0

                def poll(self) -> int:
                    return 0

            def _fake_popen(*_args, **kwargs):
                env = kwargs.get("env")
                if isinstance(env, dict):
                    captured_env.update({str(key): str(value) for key, value in env.items()})
                return _FakeProc()

            with patch.object(
                sys,
                "argv",
                [
                    str(RUNNER_PATH),
                    "--run-dir",
                    str(run_dir),
                    "--main-command",
                    "python -c \"print('ok')\"",
                    "--bounded-session-seconds",
                    "60",
                    "--verification-drain-seconds",
                    "20",
                    "--watchdog-margin-seconds",
                    "15",
                ],
            ), patch.object(phase16e_run, "_cleanup_interfering_processes", return_value=[]), patch.object(
                phase16e_run.subprocess,
                "Popen",
                side_effect=_fake_popen,
            ), patch.object(phase16e_run, "_run_report", return_value=0), patch.object(
                phase16e_run,
                "_collect_post_shutdown_processes",
                return_value=[],
            ), patch.object(
                phase16e_run,
                "_record_run_local_artifact_status",
                return_value={"missing": []},
            ):
                rc = phase16e_run.main()

            self.assertEqual(rc, 0)
            self.assertEqual(captured_env.get("PHASE16E_WATCHDOG_MARGIN_SECONDS"), "15.0")
            self.assertEqual(
                captured_env.get("PHASE16E_VERIFICATION_SUBMISSION_DECISIONS_PATH"),
                str((run_dir / "verification_submission_decisions.jsonl").resolve()),
            )

    def test_budget_defaults_and_zero_override(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_budget_env_defaults_") as tmp:
            default_run_dir = Path(tmp) / "default"
            default_run_dir.mkdir(parents=True, exist_ok=True)
            default_env: dict[str, str] = {}

            phase16e_run._ensure_budget_env(default_env, default_run_dir)
            phase16e_run._initialize_budget_artifacts(default_env)
            phase16e_run._write_budget_configuration_diagnostics(default_run_dir)

            default_state = json.loads((default_run_dir / "provider_call_budget_state.json").read_text(encoding="utf-8"))
            self.assertEqual(default_state["limits"]["total_limit"], 1)
            self.assertEqual(default_state["limits"]["per_role_limits"]["observation_visual"], 1)
            self.assertEqual(default_state["limits"]["per_role_limits"]["tracking_vlm"], 0)
            self.assertEqual(default_state["limits"]["per_role_limits"]["observation_text"], 0)
            self.assertEqual(default_state["limits"]["per_role_limits"]["reasoning_text"], 0)

            zero_run_dir = Path(tmp) / "zero"
            zero_run_dir.mkdir(parents=True, exist_ok=True)
            zero_env = {
                "PROVIDER_CALL_BUDGET_ENABLED": "1",
                "PROVIDER_CALL_BUDGET_TOTAL_LIMIT": "0",
                "PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT": "0",
                "PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT": "0",
                "PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT": "0",
                "PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT": "0",
            }
            phase16e_run._ensure_budget_env(zero_env, zero_run_dir)
            phase16e_run._initialize_budget_artifacts(zero_env)
            phase16e_run._write_budget_configuration_diagnostics(zero_run_dir)

            zero_state = json.loads((zero_run_dir / "provider_call_budget_state.json").read_text(encoding="utf-8"))
            self.assertEqual(zero_state["limits"]["total_limit"], 0)
            self.assertEqual(zero_state["limits"]["per_role_limits"]["tracking_vlm"], 0)
            self.assertEqual(zero_state["limits"]["per_role_limits"]["observation_text"], 0)
            self.assertEqual(zero_state["limits"]["per_role_limits"]["reasoning_text"], 0)
            self.assertEqual(zero_state["limits"]["per_role_limits"]["observation_visual"], 0)

            zero_diag = json.loads((zero_run_dir / "provider_call_budget_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(zero_diag["configured_limits"]["total_limit"], 0)
            self.assertEqual(zero_diag["attempted_provider_calls"]["state_attempted_total"], 0)
            self.assertEqual(zero_diag["blocked_provider_calls"]["blocked_event_total"], 0)

    def test_report_subprocess_failure_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_report_fail_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            fake_repo = Path(tmp) / "fake_repo"
            (fake_repo / "tools").mkdir(parents=True, exist_ok=True)

            rc = phase16e_run._run_report(fake_repo, run_dir, Path(sys.executable))
            self.assertNotEqual(rc, 0)
            self.assertTrue((run_dir / "report_exit_code.txt").is_file())
            self.assertTrue((run_dir / "report_stderr.log").is_file())
            self.assertIn("phase16e_report.py", (run_dir / "report_stderr.log").read_text(encoding="utf-8", errors="ignore"))

    def test_sighup_during_active_tracking_finalizes_and_kills_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_interrupt_tracking_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pid_path = run_dir / "child_pid.txt"
            heartbeat_path = run_dir / "tracking_heartbeat.txt"

            child_code = (
                "import os, signal, sys, time, pathlib\n"
                f"pid_path=pathlib.Path({str(pid_path)!r})\n"
                f"heartbeat=pathlib.Path({str(heartbeat_path)!r})\n"
                "pid_path.write_text(str(os.getpid()), encoding='utf-8')\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
                "signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))\n"
                "i = 0\n"
                "while True:\n"
                "  heartbeat.write_text(str(i), encoding='utf-8')\n"
                "  i += 1\n"
                "  time.sleep(0.2)\n"
            )

            proc = self._launch_runner(run_dir, child_code)
            self.assertTrue(_wait_for_path(pid_path), "child pid file not created")
            proc.send_signal(signal.SIGHUP)
            return_code = proc.wait(timeout=30)
            self.assertEqual(return_code, 128 + int(signal.SIGHUP))

            self._assert_finalized_failure(run_dir, 128 + int(signal.SIGHUP))

            child_pid = int(pid_path.read_text(encoding="utf-8").strip())
            deadline = time.time() + 6.0
            while time.time() < deadline and _pid_alive(child_pid):
                time.sleep(0.1)
            self.assertFalse(_pid_alive(child_pid), "child process group was not cleaned up")

    def test_sighup_during_pending_verification_finalizes_with_failure_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_interrupt_pending_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pid_path = run_dir / "child_pid.txt"
            queue_path = run_dir / "latest_verification_queue_results.json"

            child_code = (
                "import json, os, signal, sys, time, pathlib\n"
                f"pid_path=pathlib.Path({str(pid_path)!r})\n"
                f"queue_path=pathlib.Path({str(queue_path)!r})\n"
                "pid_path.write_text(str(os.getpid()), encoding='utf-8')\n"
                "queue_path.write_text(json.dumps([{'track_id':'pending-track','verification_status':'pending'}]), encoding='utf-8')\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
                "signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))\n"
                "while True:\n"
                "  time.sleep(0.25)\n"
            )

            proc = self._launch_runner(run_dir, child_code)
            self.assertTrue(_wait_for_path(pid_path), "child pid file not created")
            self.assertTrue(_wait_for_path(queue_path), "queue artifact not created")

            proc.send_signal(signal.SIGHUP)
            return_code = proc.wait(timeout=30)
            self.assertEqual(return_code, 128 + int(signal.SIGHUP))

            self._assert_finalized_failure(run_dir, 128 + int(signal.SIGHUP))

            report = json.loads((run_dir / "phase16e_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report.get("checks", {}).get("main_exit_code"), 128 + int(signal.SIGHUP))

    def test_preserves_externally_supplied_phase16e_run_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_external_run_id_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            external_run_id = "external-run-id-123"

            child_code = "import time; time.sleep(0.2)"
            env = dict(os.environ)
            env["PHASE16E_RUN_ID"] = external_run_id

            proc = self._launch_runner(run_dir, child_code, env=env)
            return_code = proc.wait(timeout=30)
            self.assertEqual(return_code, 0)

            run_id_value = (run_dir / "run_id.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(run_id_value, external_run_id)

    def test_sigint_during_active_tracking_finalizes_and_kills_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_interrupt_sigint_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pid_path = run_dir / "child_pid.txt"

            child_code = (
                "import os, signal, sys, time, pathlib\n"
                f"pid_path=pathlib.Path({str(pid_path)!r})\n"
                "pid_path.write_text(str(os.getpid()), encoding='utf-8')\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
                "while True:\n"
                "  time.sleep(0.25)\n"
            )

            proc = self._launch_runner(run_dir, child_code)
            self.assertTrue(_wait_for_path(pid_path), "child pid file not created")
            proc.send_signal(signal.SIGINT)
            return_code = proc.wait(timeout=30)
            self.assertEqual(return_code, 128 + int(signal.SIGINT))

            self._assert_finalized_failure(run_dir, 128 + int(signal.SIGINT))

            child_pid = int(pid_path.read_text(encoding="utf-8").strip())
            deadline = time.time() + 6.0
            while time.time() < deadline and _pid_alive(child_pid):
                time.sleep(0.1)
            self.assertFalse(_pid_alive(child_pid), "child process group was not cleaned up")

    def test_sigterm_during_active_tracking_finalizes_and_kills_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_interrupt_sigterm_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pid_path = run_dir / "child_pid.txt"

            child_code = (
                "import os, signal, sys, time, pathlib\n"
                f"pid_path=pathlib.Path({str(pid_path)!r})\n"
                "pid_path.write_text(str(os.getpid()), encoding='utf-8')\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
                "while True:\n"
                "  time.sleep(0.25)\n"
            )

            proc = self._launch_runner(run_dir, child_code)
            self.assertTrue(_wait_for_path(pid_path), "child pid file not created")
            proc.send_signal(signal.SIGTERM)
            return_code = proc.wait(timeout=30)
            self.assertEqual(return_code, 128 + int(signal.SIGTERM))

            self._assert_finalized_failure(run_dir, 128 + int(signal.SIGTERM))

            child_pid = int(pid_path.read_text(encoding="utf-8").strip())
            deadline = time.time() + 6.0
            while time.time() < deadline and _pid_alive(child_pid):
                time.sleep(0.1)
            self.assertFalse(_pid_alive(child_pid), "child process group was not cleaned up")

    def test_watchdog_timeout_finalizes_and_cleans_processes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="phase16e_watchdog_timeout_") as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            pid_path = run_dir / "child_pid.txt"

            child_code = (
                "import os, signal, sys, time, pathlib\n"
                f"pid_path=pathlib.Path({str(pid_path)!r})\n"
                "pid_path.write_text(str(os.getpid()), encoding='utf-8')\n"
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
                "signal.signal(signal.SIGINT, lambda *_: sys.exit(0))\n"
                "while True:\n"
                "  time.sleep(0.25)\n"
            )

            proc = self._launch_runner(
                run_dir,
                child_code,
                extra_args=[
                    "--bounded-session-seconds",
                    "0.1",
                    "--verification-drain-seconds",
                    "0.1",
                    "--watchdog-margin-seconds",
                    "0.1",
                ],
            )
            self.assertTrue(_wait_for_path(pid_path), "child pid file not created")
            return_code = proc.wait(timeout=30)
            self.assertEqual(return_code, 124)

            self.assertEqual((run_dir / "watchdog_triggered.txt").read_text(encoding="utf-8").strip(), "1")
            self.assertEqual((run_dir / "main_exit_code.txt").read_text(encoding="utf-8").strip(), "124")

            child_pid = int(pid_path.read_text(encoding="utf-8").strip())
            deadline = time.time() + 6.0
            while time.time() < deadline and _pid_alive(child_pid):
                time.sleep(0.1)
            self.assertFalse(_pid_alive(child_pid), "child process group was not cleaned up")


if __name__ == "__main__":
    unittest.main()
