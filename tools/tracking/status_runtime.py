from __future__ import annotations


def build_tracking_agent_status_text(status: dict[str, object], detailed: bool = True) -> str:
    model = status.get("model", "unknown")
    submissions = status.get("submissions", 0)
    completions = status.get("completions", 0)
    unavailable_count = status.get("unavailable_count", 0)
    accepted = status.get("total_updates_accepted", 0)
    rejected = status.get("total_updates_rejected", 0)
    unmatched = status.get("total_updates_unmatched", 0)
    latency_ms = status.get("last_latency_ms")
    last_error = status.get("last_error")

    latency_text = "n/a" if not isinstance(latency_ms, (int, float)) else f"{latency_ms:.1f}ms"
    error_text = "none" if not isinstance(last_error, str) or not last_error else last_error

    if not detailed:
        return (
            "tracking_agent_status: "
            f"model={model} | completions={completions} | unavailable={unavailable_count} | "
            f"accepted={accepted} | rejected={rejected} | last_error={error_text}"
        )

    return (
        "tracking_agent_status: "
        f"model={model} | submissions={submissions} | completions={completions} | "
        f"unavailable={unavailable_count} | accepted={accepted} | rejected={rejected} | "
        f"unmatched={unmatched} | last_latency={latency_text} | last_error={error_text}"
    )


def classify_tracking_agent_health(status: dict[str, object]) -> str:
    submissions = status.get("submissions", 0)
    completions = status.get("completions", 0)
    unavailable_count = status.get("unavailable_count", 0)
    accepted = status.get("total_updates_accepted", 0)
    rejected = status.get("total_updates_rejected", 0)
    last_error = status.get("last_error")

    if not isinstance(submissions, int):
        submissions = 0
    if not isinstance(completions, int):
        completions = 0
    if not isinstance(unavailable_count, int):
        unavailable_count = 0
    if not isinstance(accepted, int):
        accepted = 0
    if not isinstance(rejected, int):
        rejected = 0

    if isinstance(last_error, str) and last_error in {"vlm_timeout", "vlm_unavailable"}:
        return "error"

    if submissions == 0:
        return "warn"

    unavailable_rate = unavailable_count / submissions
    decisions = accepted + rejected
    rejection_rate = (rejected / decisions) if decisions > 0 else 0.0

    if unavailable_rate >= 0.25 or rejection_rate >= 0.5:
        return "error"
    if unavailable_rate >= 0.1 or rejection_rate >= 0.25 or completions < submissions:
        return "warn"
    return "good"


def tracking_agent_health_color(health: str) -> tuple[int, int, int]:
    if health == "good":
        return (90, 220, 120)
    if health == "error":
        return (80, 80, 245)
    return (100, 210, 240)


def build_tracking_agent_health_legend_text() -> str:
    return "tracking health legend: good | warn | error"


def build_tracking_debug_toggle_hint(show_tracking_debug_details: bool) -> str:
    mode = "detailed" if show_tracking_debug_details else "concise"
    return f"press 'h' to toggle tracking-agent diagnostics ({mode})"
