from __future__ import annotations

import threading
from collections import deque
from typing import Any

from agents.tracking.tracking_types import RadarTrackSnapshot
from agents.tracking.evidence_packaging import TrackingEvidencePackager
from agents.tracking.evidence_store import TrackingEvidenceStore
from agents.tracking.estimate_proposals import TrackingEstimateProposalEngine
from agents.tracking.observation_feedback import TrackingObservationFeedbackConsumer
from agents.tracking.identity_registry import PersistentIdentityRegistry


class SharedDashboardState:
    def __init__(self, video_path: str | None = None) -> None:
        self.lock = threading.Lock()
        self.running = True

        self.camera_status = "starting"
        self.detector_status = "starting"
        self.video_path = video_path
        self.input_error: str | None = None
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
        self.tracking_agent_llm_valid_response_count = 0
        self.tracking_agent_schema_complete_count = 0
        self.tracking_agent_schema_incomplete_count = 0
        self.tracking_agent_llm_timeout_count = 0
        self.tracking_agent_llm_truncation_count = 0
        self.tracking_agent_llm_invalid_json_count = 0
        self.tracking_agent_llm_other_failure_count = 0
        self.tracking_agent_fallback_usage_count = 0
        self.tracking_agent_llm_applied_update_count = 0
        self.tracking_agent_last_latency_ms: float | None = None
        self.tracking_agent_last_error: str | None = None
        self.tracking_evidence_store: TrackingEvidenceStore | None = None
        self.tracking_evidence_packager: TrackingEvidencePackager | None = None
        self.tracking_feedback_consumer: TrackingObservationFeedbackConsumer | None = None
        self.tracking_estimate_engine: TrackingEstimateProposalEngine | None = None
        self.latest_tracking_evidence_message_by_track: dict[str, str] = {}
        self.latest_tracking_lost_evidence_message_by_track: dict[str, str] = {}
        self.latest_tracking_lost_track_expires_at_by_track: dict[str, float] = {}
        self.tracking_missing_track_grace_seconds: float = 1.5
        self.latest_tracking_observation_feedback: dict | None = None
        self.latest_tracking_estimate_proposals: dict | None = None
        self.tracking_identity_registry_enabled: bool = False
        self.tracking_identity_registry: PersistentIdentityRegistry | None = None
        self.tracking_identity_registry_generation: int = 0
        self.tracking_identity_registry_version: int = 0
        self.latest_tracking_identity_registry_summary: dict | None = None
        self.tracking_identity_snapshot_bridge_enabled: bool = False
        self.latest_tracking_identity_snapshot_metadata: dict | None = None
        self.latest_tracking_identity_snapshot_source_version: int | None = None
        self.latest_tracking_identity_snapshot_published_at: float = 0.0
        self.tracking_identity_reassociation_enabled: bool = False
        self.tracking_identity_reassociation_consumer: Any | None = None
        self.latest_tracking_identity_reassociation_summary: dict | None = None
        self.tracking_stable_identity_visualization_enabled: bool = False
        self.latest_tracking_display_identity_by_track: dict[str, dict[str, Any]] = {}
        self.latest_tracking_display_identity_summary: dict[str, Any] | None = None

        self.selected_track_id: str | None = None
        self.show_tracking_debug_details = False
