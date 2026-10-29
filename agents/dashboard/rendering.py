from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2

from agents.tracking.geometry import ground_to_radar_screen
from agents.tracking.tracking_types import RadarTrackSnapshot
from agents.tracking.runtime_support import build_relative_depth_coverage_text, compute_relative_depth_coverage

from .config import (
    CAPTURES_DIR,
    MAX_GROUND_X,
    MAX_GROUND_Z,
    RADAR_LOST_SECONDS,
    RADAR_STALE_SECONDS,
    RADAR_TRAIL_LENGTH,
    RENDER_HZ,
    WINDOW_NAME,
)
from .state import SharedDashboardState

AGENT_EVENTS_PATH = CAPTURES_DIR / "agent_events.jsonl"
AGENT_EVENT_FRESHNESS_SECONDS = 30.0


def _load_latest_agent_event(
    last_mtime: float,
    last_event_snapshot: dict | None,
    events_path: Path = AGENT_EVENTS_PATH,
) -> tuple[float, dict | None]:
    if not events_path.is_file():
        return 0.0, None

    try:
        current_mtime = events_path.stat().st_mtime
    except OSError:
        return 0.0, None

    if current_mtime == last_mtime:
        return last_mtime, last_event_snapshot

    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return last_mtime, last_event_snapshot

    latest_payload: dict | None = None
    latest_by_track: dict[str, dict] = {}

    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        if latest_payload is None:
            latest_payload = payload

        track_id = payload.get("track_id")
        if isinstance(track_id, str) and track_id and track_id not in latest_by_track:
            latest_by_track[track_id] = payload

    if latest_payload is None:
        return current_mtime, None

    return current_mtime, {
        "latest": latest_payload,
        "latest_by_track": latest_by_track,
    }


def _select_assessment_event(
    event_snapshot: dict | None,
    selected_track_id: str | None,
) -> dict | None:
    if not isinstance(event_snapshot, dict):
        return None

    latest_by_track = event_snapshot.get("latest_by_track")
    if (
        isinstance(latest_by_track, dict)
        and isinstance(selected_track_id, str)
        and selected_track_id
    ):
        selected_event = latest_by_track.get(selected_track_id)
        if isinstance(selected_event, dict):
            return selected_event

    latest_event = event_snapshot.get("latest")
    if isinstance(latest_event, dict):
        return latest_event
    return None


def _format_assessment_summary(
    now: float,
    latest_event: dict | None,
) -> tuple[str, str, tuple[int, int, int]]:
    if not isinstance(latest_event, dict):
        return (
            "assessment: unavailable",
            "limits: event_log=missing_or_empty",
            (170, 170, 170),
        )

    event_timestamp = latest_event.get("timestamp")
    age_seconds: float | None = None
    if isinstance(event_timestamp, (int, float)):
        age_seconds = max(0.0, now - float(event_timestamp))
    is_fresh = isinstance(age_seconds, float) and age_seconds <= AGENT_EVENT_FRESHNESS_SECONDS

    event_status = latest_event.get("status")
    if isinstance(event_status, str) and event_status.strip():
        status_text = event_status.strip()
    else:
        status_text = "unknown"

    assessment_label = latest_event.get("assessment_label")
    label_text = assessment_label if isinstance(assessment_label, str) and assessment_label.strip() else "n/a"

    handoff_quality = latest_event.get("handoff_quality")
    handoff_text = handoff_quality if isinstance(handoff_quality, str) and handoff_quality.strip() else "n/a"

    obs_error_code = latest_event.get("observation_agent_error_code")
    rea_error_code = latest_event.get("reasoning_agent_error_code")
    cycle_error_code = latest_event.get("cycle_error_code")

    obs_text = obs_error_code if isinstance(obs_error_code, str) and obs_error_code.strip() else "none"
    rea_text = rea_error_code if isinstance(rea_error_code, str) and rea_error_code.strip() else "none"
    cycle_text = cycle_error_code if isinstance(cycle_error_code, str) and cycle_error_code.strip() else "none"

    freshness_text = "fresh" if is_fresh else "stale"
    age_text = f"{age_seconds:.1f}s" if isinstance(age_seconds, float) else "n/a"

    summary_line = (
        f"assessment: status={status_text} | label={label_text} | handoff={handoff_text} | {freshness_text} ({age_text})"
    )
    limits_line = f"limits: obs_err={obs_text} | rea_err={rea_text} | cycle_err={cycle_text}"

    if not is_fresh:
        color = (0, 180, 255)
    elif status_text == "error" or cycle_text != "none":
        color = (0, 0, 255)
    else:
        color = (180, 220, 180)

    return summary_line, limits_line, color


def draw_placeholder_webcam_panel(width: int, height: int, message: str):
    panel = cv2.UMat(height, width, cv2.CV_8UC3).get()
    panel[:] = (22, 22, 22)
    cv2.putText(
        panel,
        "Webcam feed unavailable",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 170, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        message,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def draw_radar_panel(
    width: int,
    height: int,
    tracks: dict[str, RadarTrackSnapshot],
    trails: dict[str, deque[tuple[float, float]]],
    now: float,
):
    panel = cv2.UMat(height, width, cv2.CV_8UC3).get()
    panel[:] = (15, 18, 22)

    margin = 40
    center_x = margin + (width - 2 * margin) // 2
    camera_y = height - margin

    cv2.line(panel, (center_x, margin), (center_x, camera_y), (70, 70, 70), 1)
    cv2.line(panel, (margin, camera_y), (width - margin, camera_y), (70, 70, 70), 1)
    cv2.putText(panel, "X", (width - margin + 8, camera_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(panel, "Z", (center_x + 8, margin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(panel, "Camera", (center_x - 25, camera_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    for track_id, trail in trails.items():
        if len(trail) < 2:
            continue

        points = [
            ground_to_radar_screen(
                x,
                z,
                width,
                height,
                MAX_GROUND_X,
                MAX_GROUND_Z,
                margin,
            )
            for (x, z) in trail
        ]

        for i in range(1, len(points)):
            cv2.line(panel, points[i - 1], points[i], (80, 130, 240), 1)

        if track_id in tracks:
            snapshot = tracks[track_id]
            marker_point = ground_to_radar_screen(
                snapshot.ground_position.x,
                snapshot.ground_position.z,
                width,
                height,
                MAX_GROUND_X,
                MAX_GROUND_Z,
                margin,
            )
            age_seconds = now - snapshot.timestamp
            if age_seconds > RADAR_LOST_SECONDS:
                marker_color = (0, 0, 255)
                status_label = "lost"
            elif age_seconds > RADAR_STALE_SECONDS:
                marker_color = (0, 180, 255)
                status_label = "stale"
            else:
                marker_color = (70, 230, 70)
                status_label = "fresh"

            cv2.circle(panel, marker_point, 6, marker_color, -1)
            cv2.putText(
                panel,
                f"{track_id} {status_label}",
                (marker_point[0] + 8, marker_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

    return panel


def render_dashboard(state: SharedDashboardState) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    latest_event_mtime = 0.0
    latest_agent_event_snapshot: dict | None = None

    while True:
        with state.lock:
            if not state.running:
                break

            camera_status = state.camera_status
            detector_status = state.detector_status
            calibration_status = state.calibration_status
            mapping_mode = state.mapping_mode
            latest_frame = state.latest_annotated_frame
            tracks = dict(state.active_tracks)
            trails = {
                key: deque(value, maxlen=RADAR_TRAIL_LENGTH)
                for key, value in state.track_trails.items()
            }
            last_obs_time = state.last_observation_time
            selected_track_id = state.selected_track_id

        now = time.time()
        latest_event_mtime, latest_agent_event_snapshot = _load_latest_agent_event(
            latest_event_mtime,
            latest_agent_event_snapshot,
        )
        latest_agent_event = _select_assessment_event(
            latest_agent_event_snapshot,
            selected_track_id,
        )
        assessment_summary_line, assessment_limits_line, assessment_color = _format_assessment_summary(
            now,
            latest_agent_event,
        )

        dashboard_width = 1380
        dashboard_height = 760
        header_height = 132
        panel_height = dashboard_height - header_height
        panel_width = dashboard_width // 2

        if latest_frame is None:
            webcam_panel = draw_placeholder_webcam_panel(
                panel_width,
                panel_height,
                f"camera_status={camera_status}",
            )
        else:
            webcam_panel = cv2.resize(latest_frame, (panel_width, panel_height))
            cv2.putText(
                webcam_panel,
                "Live Webcam (YOLO + ByteTrack)",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        radar_panel = draw_radar_panel(panel_width, panel_height, tracks, trails, now)
        cv2.putText(
            radar_panel,
            "Ground-Plane Radar (relative units)",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        header = cv2.UMat(header_height, dashboard_width, cv2.CV_8UC3).get()
        header[:] = (28, 30, 38)
        cv2.putText(
            header,
            "Counter-UAS Visual Radar Dashboard Demo",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

        active_track_count = len(tracks)
        relative_depth_coverage = compute_relative_depth_coverage(list(tracks.values()))
        relative_depth_coverage_text = build_relative_depth_coverage_text(relative_depth_coverage)
        last_obs_ago = "n/a" if last_obs_time == 0 else f"{now - last_obs_time:.2f}s ago"
        status_text = (
            f"camera={camera_status} | detector={detector_status} | "
            f"tracks={active_track_count} | last_obs={last_obs_ago} | "
            f"mode={mapping_mode} | calibration={calibration_status}"
        )
        cv2.putText(
            header,
            status_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            header,
            relative_depth_coverage_text,
            (980, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (170, 220, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            header,
            assessment_summary_line,
            (20, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            assessment_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            header,
            assessment_limits_line,
            (20, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        if tracks:
            sorted_ids = sorted(tracks.keys())
            if selected_track_id not in tracks:
                selected_track_id = sorted_ids[0]
            with state.lock:
                state.selected_track_id = selected_track_id

            selected = tracks[selected_track_id]
            detail_lines = [
                f"Selected: {selected.track_id}",
                f"timestamp: {selected.timestamp:.3f}",
                (
                    "pixel: n/a"
                    if selected.pixel_position is None
                    else f"pixel: ({selected.pixel_position.x:.1f}, {selected.pixel_position.y:.1f})"
                ),
                f"ground: ({selected.ground_position.x:.2f}, {selected.ground_position.z:.2f})",
                (
                    "relative_depth: n/a"
                    if selected.relative_depth is None
                    else f"relative_depth: {selected.relative_depth:.3f}"
                ),
                f"position_kind: {selected.position_kind}",
                f"quality_flags: {', '.join(selected.quality_flags) if selected.quality_flags else 'none'}",
            ]
            y_cursor = 20
            for line in detail_lines:
                cv2.putText(
                    radar_panel,
                    line,
                    (18, panel_height - 130 + y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                y_cursor += 18

        body = cv2.hconcat([webcam_panel, radar_panel])
        dashboard = cv2.vconcat([header, body])
        cv2.imshow(WINDOW_NAME, dashboard)

        key = cv2.waitKey(int(1000 / RENDER_HZ)) & 0xFF
        if key == ord("q"):
            with state.lock:
                state.running = False
            break

        if key in {ord("n"), ord("p")} and tracks:
            sorted_ids = sorted(tracks.keys())
            current_id = state.selected_track_id if state.selected_track_id in sorted_ids else sorted_ids[0]
            current_index = sorted_ids.index(current_id)
            if key == ord("n"):
                next_id = sorted_ids[(current_index + 1) % len(sorted_ids)]
            else:
                next_id = sorted_ids[(current_index - 1) % len(sorted_ids)]
            with state.lock:
                state.selected_track_id = next_id

    cv2.destroyAllWindows()
