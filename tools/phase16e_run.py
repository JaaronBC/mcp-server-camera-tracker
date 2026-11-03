from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_environment_defaults(repo_root: Path) -> None:
    load_dotenv(dotenv_path=repo_root / ".env", override=False)


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def _list_processes(pattern: str) -> list[tuple[int, str]]:
    result = _run(["pgrep", "-fl", pattern])
    if result.returncode != 0:
        return []
    matches: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split(" ", 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        matches.append((pid, cmd))
    return matches


def _kill_pid(pid: int, *, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _cleanup_interfering_processes(run_dir: Path) -> list[str]:
    patterns = [
        r"agents\.dashboard\.live_tracking_display",
        r"mcp-server-camera-tracker/.+server\.py",
    ]
    before: list[str] = []
    for pattern in patterns:
        for pid, cmd in _list_processes(pattern):
            before.append(f"{pid} {cmd}")
            _kill_pid(pid, sig=signal.SIGTERM)

    time.sleep(0.4)

    still_running: list[str] = []
    for pattern in patterns:
        for pid, cmd in _list_processes(pattern):
            still_running.append(f"{pid} {cmd}")
            _kill_pid(pid, sig=signal.SIGKILL)

    (run_dir / "pre_cleanup_processes.txt").write_text("\n".join(before) + ("\n" if before else ""), encoding="utf-8")
    (run_dir / "post_cleanup_residual_processes.txt").write_text("\n".join(still_running) + ("\n" if still_running else ""), encoding="utf-8")
    return still_running


def _ensure_budget_env(env: dict[str, str], run_dir: Path) -> None:
    env.setdefault("PROVIDER_CALL_BUDGET_ENABLED", "1")
    env.setdefault("PROVIDER_CALL_BUDGET_TOTAL_LIMIT", "1")
    env.setdefault("PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT", "0")
    env.setdefault("PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT", "0")
    env.setdefault("PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT", "0")
    env.setdefault("PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT", "1")
    env.setdefault("PROVIDER_CALL_BUDGET_STATE_PATH", str((run_dir / "provider_call_budget_state.json").resolve()))
    env.setdefault("PROVIDER_CALL_BUDGET_LOCK_PATH", str((run_dir / "provider_call_budget_state.lock").resolve()))
    env.setdefault("PROVIDER_CALL_BUDGET_LOG_PATH", str((run_dir / "provider_call_budget_events.jsonl").resolve()))


def _initialize_budget_artifacts(env: dict[str, str]) -> None:
    state_path = Path(env["PROVIDER_CALL_BUDGET_STATE_PATH"]).resolve()
    lock_path = Path(env["PROVIDER_CALL_BUDGET_LOCK_PATH"]).resolve()
    log_path = Path(env["PROVIDER_CALL_BUDGET_LOG_PATH"]).resolve()

    for path in (state_path, lock_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    limits = {
        "total_limit": int(env.get("PROVIDER_CALL_BUDGET_TOTAL_LIMIT", "0") or 0),
        "per_role_limits": {
            "tracking_vlm": int(env.get("PROVIDER_CALL_BUDGET_TRACKING_VLM_LIMIT", "0") or 0),
            "observation_text": int(env.get("PROVIDER_CALL_BUDGET_OBSERVATION_TEXT_LIMIT", "0") or 0),
            "reasoning_text": int(env.get("PROVIDER_CALL_BUDGET_REASONING_TEXT_LIMIT", "0") or 0),
            "observation_visual": int(env.get("PROVIDER_CALL_BUDGET_OBSERVATION_VISUAL_LIMIT", "0") or 0),
        },
    }

    if not state_path.exists():
        state_payload = {
            "schema_version": "1.0",
            "updated_at": time.time(),
            "limits": limits,
            "totals": {
                "attempted": 0,
                "per_role_attempted": {
                    "tracking_vlm": 0,
                    "observation_text": 0,
                    "reasoning_text": 0,
                    "observation_visual": 0,
                },
            },
        }
        state_path.write_text(json.dumps(state_payload), encoding="utf-8")

    lock_path.touch(exist_ok=True)
    log_path.touch(exist_ok=True)


def _write_budget_configuration_diagnostics(run_dir: Path) -> None:
    state_path = run_dir / "provider_call_budget_state.json"
    log_path = run_dir / "provider_call_budget_events.jsonl"

    configured_limits: dict[str, Any] = {}
    attempted_total = 0
    attempted_per_role: dict[str, int] = {
        "tracking_vlm": 0,
        "observation_text": 0,
        "reasoning_text": 0,
        "observation_visual": 0,
    }
    state_read_error: str | None = None
    try:
        if state_path.is_file():
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(state_payload, dict):
                limits_payload = state_payload.get("limits")
                if isinstance(limits_payload, dict):
                    configured_limits = limits_payload
                totals_payload = state_payload.get("totals")
                if isinstance(totals_payload, dict):
                    attempted_total = int(totals_payload.get("attempted", 0) or 0)
                    per_role_payload = totals_payload.get("per_role_attempted")
                    if isinstance(per_role_payload, dict):
                        for role in attempted_per_role:
                            attempted_per_role[role] = int(per_role_payload.get(role, 0) or 0)
    except Exception as exc:
        state_read_error = f"{type(exc).__name__}:{exc}"

    blocked_total = 0
    blocked_per_role: dict[str, int] = {}
    attempt_event_total = 0
    attempt_event_per_role: dict[str, int] = {}
    log_read_error: str | None = None
    try:
        if log_path.is_file():
            for raw in log_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    continue
                event_type = event.get("event_type")
                role_value = event.get("role")
                role = role_value.strip() if isinstance(role_value, str) and role_value.strip() else None
                if event_type == "provider_call_blocked":
                    blocked_total += 1
                    if role is not None:
                        blocked_per_role[role] = blocked_per_role.get(role, 0) + 1
                elif event_type == "provider_call_attempt":
                    attempt_event_total += 1
                    if role is not None:
                        attempt_event_per_role[role] = attempt_event_per_role.get(role, 0) + 1
    except Exception as exc:
        log_read_error = f"{type(exc).__name__}:{exc}"

    payload = {
        "schema_version": "1.0",
        "generated_at": time.time(),
        "configured_limits": configured_limits,
        "attempted_provider_calls": {
            "state_attempted_total": attempted_total,
            "state_attempted_per_role": attempted_per_role,
            "attempt_event_total": attempt_event_total,
            "attempt_event_per_role": attempt_event_per_role,
        },
        "blocked_provider_calls": {
            "blocked_event_total": blocked_total,
            "blocked_event_per_role": blocked_per_role,
        },
        "errors": {
            "state_read_error": state_read_error,
            "log_read_error": log_read_error,
        },
    }
    (run_dir / "provider_call_budget_diagnostics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _wait_for_exit(proc: subprocess.Popen[Any], timeout_seconds: float) -> int | None:
    deadline = time.time() + max(0.0, timeout_seconds)
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            return code
        time.sleep(0.1)
    return proc.poll()


def _kill_group(proc: subprocess.Popen[Any], *, term_timeout: float = 2.0) -> int:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return proc.poll() or 0
    os.killpg(pgid, signal.SIGTERM)
    code = _wait_for_exit(proc, term_timeout)
    if code is not None:
        return code
    os.killpg(pgid, signal.SIGKILL)
    final_code = _wait_for_exit(proc, 2.0)
    return -signal.SIGKILL if final_code is None else final_code


def _collect_post_shutdown_processes(run_dir: Path) -> list[str]:
    rows: list[str] = []
    for pattern in [r"agents\.dashboard\.live_tracking_display", r"mcp-server-camera-tracker/.+server\.py"]:
        for pid, cmd in _list_processes(pattern):
            rows.append(f"{pid} {cmd}")
    (run_dir / "post_shutdown_processes.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return rows


def _signal_exit_code(signum: int | None) -> int:
    if signum is None:
        return 1
    return 128 + int(signum)


def _run_report(repo_root: Path, run_dir: Path, python_path: Path) -> int:
    report_cmd = [str(python_path), str(repo_root / "tools" / "phase16e_report.py"), "--run-dir", str(run_dir)]
    result = _run(report_cmd, cwd=repo_root)
    _write(run_dir / "report_stdout.json", result.stdout if result.stdout else "{}\n")
    _write(run_dir / "report_stderr.log", result.stderr or "")
    _write(run_dir / "report_exit_code.txt", f"{result.returncode}\n")
    return result.returncode


def _record_run_local_artifact_status(run_dir: Path) -> dict[str, Any]:
    expected = [
        "provider_call_budget_state.json",
        "provider_call_budget_events.jsonl",
        "provider_call_budget_diagnostics.json",
        "latest_verification_queue_results.json",
        "latest_verification_queue_submissions.json",
        "operator_tracking_feedback.json",
        "agent_events.jsonl",
    ]
    present = [name for name in expected if (run_dir / name).is_file()]
    missing = [name for name in expected if name not in present]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": time.time(),
        "expected": expected,
        "present": present,
        "missing": missing,
    }
    _write(run_dir / "run_local_artifact_status.json", json.dumps(payload, indent=2) + "\n")
    return payload


def _collect_pids_from_process_rows(rows: list[str]) -> list[int]:
    pids: list[int] = []
    for row in rows:
        parts = row.strip().split(" ", 1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            continue
    return pids


def _enforce_post_shutdown_cleanup(run_dir: Path) -> list[str]:
    initial_rows = _collect_post_shutdown_processes(run_dir)
    for pid in _collect_pids_from_process_rows(initial_rows):
        _kill_pid(pid, sig=signal.SIGTERM)

    time.sleep(0.4)

    after_term_rows = _collect_post_shutdown_processes(run_dir)
    for pid in _collect_pids_from_process_rows(after_term_rows):
        _kill_pid(pid, sig=signal.SIGKILL)

    time.sleep(0.2)
    return _collect_post_shutdown_processes(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase16E bounded execution procedure")
    parser.add_argument("--run-dir", default="", help="Run directory path (optional)")
    parser.add_argument("--main-command", default="", help="Override main command string")
    parser.add_argument("--bounded-session-seconds", type=float, default=float(os.getenv("BOUNDED_SESSION_SECONDS", "60") or 60))
    parser.add_argument("--verification-drain-seconds", type=float, default=float(os.getenv("VERIFICATION_DRAIN_SECONDS", "20") or 20))
    parser.add_argument("--watchdog-margin-seconds", type=float, default=float(os.getenv("PHASE16E_WATCHDOG_MARGIN_SECONDS", "15") or 15))
    args = parser.parse_args()

    repo_root = _repo_root()
    _load_environment_defaults(repo_root)
    python_path = repo_root / ".venv" / "bin" / "python"
    if not python_path.is_file():
        python_path = Path(sys.executable)

    run_dir = Path(args.run_dir).resolve() if args.run_dir else (repo_root / "captures" / "probes" / "runs" / f"phase16e_live_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_run_id = os.getenv("PHASE16E_RUN_ID", "").strip()
    run_id = existing_run_id or f"phase16e-{int(time.time())}"
    _write(run_dir / "run_id.txt", f"{run_id}\n")

    main_env = dict(os.environ)
    main_env["BOUNDED_SESSION_SECONDS"] = str(args.bounded_session_seconds)
    main_env["VERIFICATION_DRAIN_SECONDS"] = str(args.verification_drain_seconds)
    main_env["PHASE16E_WATCHDOG_MARGIN_SECONDS"] = str(args.watchdog_margin_seconds)
    main_env["PHASE16E_RUN_ID"] = run_id
    main_env["PHASE16E_VERIFICATION_RESULTS_PATH"] = str((run_dir / "latest_verification_queue_results.json").resolve())
    main_env["PHASE16E_VERIFICATION_SUBMISSIONS_PATH"] = str((run_dir / "latest_verification_queue_submissions.json").resolve())
    main_env["PHASE16E_VERIFICATION_SUBMISSION_DECISIONS_PATH"] = str((run_dir / "verification_submission_decisions.jsonl").resolve())
    main_env["PHASE16E_OPERATOR_FEEDBACK_PATH"] = str((run_dir / "operator_tracking_feedback.json").resolve())
    main_env["PHASE16E_AGENT_EVENTS_PATH"] = str((run_dir / "agent_events.jsonl").resolve())
    _ensure_budget_env(main_env, run_dir)
    _initialize_budget_artifacts(main_env)

    main_command = args.main_command.strip() if isinstance(args.main_command, str) else ""
    if main_command:
        cmd = ["/bin/zsh", "-lc", main_command]
    else:
        cmd = [str(python_path), str(repo_root / "main.py")]

    interruption_signal: int | None = None
    runtime_error: str | None = None
    watchdog_triggered = False
    final_code: int | None = None
    report_exit_code = 1
    proc: subprocess.Popen[Any] | None = None

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sighup = signal.getsignal(signal.SIGHUP)

    def _mark_interruption(signum: int, _frame: Any) -> None:
        nonlocal interruption_signal
        if interruption_signal is None:
            interruption_signal = int(signum)

    signal.signal(signal.SIGINT, _mark_interruption)
    signal.signal(signal.SIGTERM, _mark_interruption)
    signal.signal(signal.SIGHUP, _mark_interruption)

    try:
        residual = _cleanup_interfering_processes(run_dir)
        if residual:
            _write(run_dir / "start_ts.txt", f"{time.time()}\n")
            final_code = 1
            _write(run_dir / "main.log", "Pre-run cleanup failed: residual processes remain.\n")
        else:
            with (run_dir / "main.log").open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    env=main_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    text=True,
                )
                _write(run_dir / "start_ts.txt", f"{time.time()}\n")

                watchdog_deadline = time.time() + args.bounded_session_seconds + args.verification_drain_seconds + args.watchdog_margin_seconds
                while True:
                    code = proc.poll()
                    if code is not None:
                        final_code = code
                        break
                    if interruption_signal is not None:
                        log_file.write(f"RUNNER_INTERRUPTED_BY_SIGNAL:{interruption_signal}\n")
                        log_file.flush()
                        _kill_group(proc)
                        final_code = _signal_exit_code(interruption_signal)
                        break
                    if time.time() >= watchdog_deadline:
                        watchdog_triggered = True
                        log_file.write("WATCHDOG_TIMEOUT_TRIGGERED\n")
                        log_file.flush()
                        _kill_group(proc)
                        final_code = 124
                        break
                    try:
                        time.sleep(0.15)
                    except KeyboardInterrupt:
                        if interruption_signal is None:
                            interruption_signal = int(signal.SIGINT)
    except KeyboardInterrupt:
        if interruption_signal is None:
            interruption_signal = int(signal.SIGINT)
        final_code = _signal_exit_code(interruption_signal)
    except BaseException as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        if final_code is None:
            final_code = 1
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGHUP, original_sighup)

        if proc is not None and proc.poll() is None:
            _kill_group(proc)

        if interruption_signal is not None:
            final_code = _signal_exit_code(interruption_signal)
        elif final_code is None and proc is not None and proc.returncode is not None:
            final_code = int(proc.returncode)
        elif final_code is None:
            final_code = 1

        _write(run_dir / "end_ts.txt", f"{time.time()}\n")
        _write(run_dir / "main_exit_code.txt", f"{final_code}\n")
        _write(run_dir / "watchdog_triggered.txt", "1\n" if watchdog_triggered else "0\n")
        _write(run_dir / "run_interrupted.txt", "1\n" if interruption_signal is not None else "0\n")
        _write(run_dir / "runner_signal.txt", f"{interruption_signal}\n" if interruption_signal is not None else "\n")
        if runtime_error:
            _write(run_dir / "runner_exception.txt", runtime_error + "\n")

        _enforce_post_shutdown_cleanup(run_dir)
        _write_budget_configuration_diagnostics(run_dir)
        run_local_artifact_status = _record_run_local_artifact_status(run_dir)
        report_exit_code = _run_report(repo_root, run_dir, python_path)

        summary = {
            "run_dir": str(run_dir),
            "main_exit": final_code,
            "watchdog_triggered": watchdog_triggered,
            "interrupted": interruption_signal is not None,
            "interrupt_signal": interruption_signal,
            "report_exit": report_exit_code,
            "run_local_artifacts_missing": run_local_artifact_status.get("missing", []),
            "runtime_error": runtime_error,
        }
        _write(run_dir / "final_summary.json", json.dumps(summary, indent=2) + "\n")

    return int(final_code)


if __name__ == "__main__":
    raise SystemExit(main())
