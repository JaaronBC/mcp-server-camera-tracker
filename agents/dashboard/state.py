from __future__ import annotations

import threading
from collections import deque

from agents.components.dashboard_types import RadarTrackSnapshot


class SharedDashboardState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = True

        self.camera_status = "starting"
        self.detector_status = "starting"
        self.calibration_status = "unknown"
        self.mapping_mode = "relative_image_plane"

        self.latest_annotated_frame = None
        self.frame_width = 1280
        self.frame_height = 720

        self.last_live_frame_time = 0.0
        self.last_observation_time = 0.0

        self.active_tracks: dict[str, RadarTrackSnapshot] = {}
        self.track_trails: dict[str, deque[tuple[float, float]]] = {}
        self.latest_tracking_agent_observation: dict | None = None
        self.tracking_agent_model = "qwen-3.8-27b"
        self.tracking_agent_submissions = 0
        self.tracking_agent_completions = 0
        self.tracking_agent_unavailable_count = 0
        self.tracking_agent_total_updates_accepted = 0
        self.tracking_agent_total_updates_rejected = 0
        self.tracking_agent_total_updates_unmatched = 0
        self.tracking_agent_last_latency_ms: float | None = None
        self.tracking_agent_last_error: str | None = None

        self.selected_track_id: str | None = None
        self.show_tracking_debug_details = False
