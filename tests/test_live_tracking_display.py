# Handles test live tracking display behavior.
import unittest
from collections import deque

from agents.contracts import TrackingAgentObservation, VisualAssociationUpdate
from agents.components.dashboard_types import GroundPosition, RadarTrackSnapshot
from agents.dashboard.live_tracking_display import (
    apply_tracking_agent_updates_to_tracks,
    build_relative_depth_coverage_text,
    build_tracking_debug_toggle_hint,
    build_tracking_agent_health_legend_text,
    build_tracking_agent_status_text,
    build_tracking_detection_id,
    build_tracking_message,
    classify_tracking_agent_health,
    compute_relative_depth_coverage,
    estimate_relative_depth_from_image_geometry,
    normalize_position_metadata,
    tracking_agent_health_color,
)


class TestLiveTrackingDisplayMessage(unittest.TestCase):
    def test_build_tracking_message_schema(self) -> None:
        # Performs test build tracking message schema logic.
        detections = [
            {
                "id": 7,
                "label": "person",
                "confidence": 0.91,
                "x": 320.5,
                "y": 410.2,
            }
        ]

        message = build_tracking_message(1234.5, detections)

        self.assertIn("time", message)
        self.assertIn("detections", message)
        self.assertEqual(message["time"], 1234.5)
        self.assertIsInstance(message["detections"], list)
        self.assertEqual(message["detections"][0]["id"], 7)

    def test_build_tracking_detection_id_uses_track_key(self) -> None:
        # Performs test build tracking detection id uses track key logic.
        self.assertEqual(build_tracking_detection_id("12"), "12")

    def test_apply_tracking_agent_updates_summary(self) -> None:
        # Performs test apply tracking agent updates summary logic.
        tracks = {
            "7": RadarTrackSnapshot(
                track_id="7",
                timestamp=1.0,
                pixel_position=None,
                ground_position=GroundPosition(x=0.0, z=1.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=[],
                confidence=0.8,
            )
        }
        observation = TrackingAgentObservation(
            timestamp=2.0,
            source_agent="visual_tracking",
            vlm_model="qwen-3.8-27b",
            proposal_latency_ms=12.0,
            frame_reference=None,
            updates=[
                VisualAssociationUpdate(
                    detection_id="7",
                    resolved_track_id="vlm-7",
                    accepted=True,
                    rejection_reasons=[],
                    uncertainties=[],
                    supporting_evidence=["appearance_match"],
                ),
                VisualAssociationUpdate(
                    detection_id="missing",
                    resolved_track_id="missing",
                    accepted=False,
                    rejection_reasons=["association_confidence_too_low"],
                    uncertainties=["low_association_confidence"],
                    supporting_evidence=["weak_match"],
                ),
            ],
        )

        summary = apply_tracking_agent_updates_to_tracks(tracks, observation)

        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected"], 0)
        self.assertEqual(summary["unmatched"], 1)
        self.assertIn("vlm_association_candidate:vlm-7", tracks["7"].quality_flags)

    def test_apply_tracking_agent_updates_blends_ground_estimate_with_resolved_trail(self) -> None:
        # Performs test apply tracking agent updates blends ground estimate with resolved trail logic.
        tracks = {
            "7": RadarTrackSnapshot(
                track_id="7",
                timestamp=1.0,
                pixel_position=None,
                ground_position=GroundPosition(x=0.0, z=0.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=[],
                confidence=0.8,
            )
        }
        observation = TrackingAgentObservation(
            timestamp=2.0,
            source_agent="visual_tracking",
            vlm_model="qwen-3.8-27b",
            proposal_latency_ms=12.0,
            frame_reference=None,
            updates=[
                VisualAssociationUpdate(
                    detection_id="7",
                    resolved_track_id="vlm-7",
                    accepted=True,
                    rejection_reasons=[],
                    uncertainties=[],
                    supporting_evidence=["appearance_match"],
                ),
            ],
        )

        track_trails = {
            "vlm-7": deque([(2.0, 2.0)], maxlen=20),
        }

        apply_tracking_agent_updates_to_tracks(
            tracks,
            observation,
            track_trails=track_trails,
        )

        self.assertGreater(tracks["7"].ground_position.x, 0.0)
        self.assertGreater(tracks["7"].ground_position.z, 0.0)
        self.assertIn("tracking_agent_ground_anchor_blend", tracks["7"].quality_flags)

    def test_apply_tracking_agent_updates_rejected_requires_review_and_downgrades_confidence(self) -> None:
        # Performs test apply tracking agent updates rejected requires review and downgrades confidence logic.
        tracks = {
            "7": RadarTrackSnapshot(
                track_id="7",
                timestamp=1.0,
                pixel_position=None,
                ground_position=GroundPosition(x=0.0, z=1.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=[],
                confidence=1.0,
            )
        }
        observation = TrackingAgentObservation(
            timestamp=2.0,
            source_agent="visual_tracking",
            vlm_model="qwen-3.8-27b",
            proposal_latency_ms=12.0,
            frame_reference=None,
            updates=[
                VisualAssociationUpdate(
                    detection_id="7",
                    resolved_track_id="7",
                    accepted=False,
                    rejection_reasons=["association_confidence_too_low"],
                    uncertainties=["low_association_confidence"],
                    supporting_evidence=["weak_match"],
                ),
            ],
        )

        apply_tracking_agent_updates_to_tracks(tracks, observation)

        self.assertIn("vlm_association_rejected", tracks["7"].quality_flags)
        self.assertIn("cross_agent_review_required", tracks["7"].quality_flags)
        self.assertLess(tracks["7"].confidence or 1.0, 1.0)

    def test_build_tracking_agent_status_text(self) -> None:
        # Performs test build tracking agent status text logic.
        status_text = build_tracking_agent_status_text(
            {
                "model": "qwen-3.8-27b",
                "submissions": 12,
                "completions": 10,
                "unavailable_count": 2,
                "total_updates_accepted": 20,
                "total_updates_rejected": 4,
                "total_updates_unmatched": 1,
                "last_latency_ms": 44.3,
                "last_error": "vlm_timeout",
            }
        )

        self.assertIn("model=qwen-3.8-27b", status_text)
        self.assertIn("submissions=12", status_text)
        self.assertIn("completions=10", status_text)
        self.assertIn("last_latency=44.3ms", status_text)
        self.assertIn("last_error=vlm_timeout", status_text)

    def test_build_tracking_agent_status_text_concise(self) -> None:
        # Performs test build tracking agent status text concise logic.
        status_text = build_tracking_agent_status_text(
            {
                "model": "qwen-3.8-27b",
                "submissions": 12,
                "completions": 10,
                "unavailable_count": 2,
                "total_updates_accepted": 20,
                "total_updates_rejected": 4,
                "total_updates_unmatched": 1,
                "last_latency_ms": 44.3,
                "last_error": "vlm_timeout",
            },
            detailed=False,
        )

        self.assertIn("model=qwen-3.8-27b", status_text)
        self.assertIn("completions=10", status_text)
        self.assertNotIn("submissions=12", status_text)
        self.assertNotIn("last_latency", status_text)

    def test_build_tracking_agent_status_text_with_defaults(self) -> None:
        # Performs test build tracking agent status text with defaults logic.
        status_text = build_tracking_agent_status_text({})
        self.assertIn("last_latency=n/a", status_text)
        self.assertIn("last_error=none", status_text)

    def test_classify_tracking_agent_health_good(self) -> None:
        # Performs test classify tracking agent health good logic.
        health = classify_tracking_agent_health(
            {
                "submissions": 10,
                "completions": 10,
                "unavailable_count": 0,
                "total_updates_accepted": 30,
                "total_updates_rejected": 3,
                "last_error": None,
            }
        )
        self.assertEqual(health, "good")

    def test_classify_tracking_agent_health_warn_when_in_flight(self) -> None:
        # Performs test classify tracking agent health warn when in flight logic.
        health = classify_tracking_agent_health(
            {
                "submissions": 10,
                "completions": 9,
                "unavailable_count": 0,
                "total_updates_accepted": 20,
                "total_updates_rejected": 2,
            }
        )
        self.assertEqual(health, "warn")

    def test_classify_tracking_agent_health_error_on_timeout(self) -> None:
        # Performs test classify tracking agent health error on timeout logic.
        health = classify_tracking_agent_health(
            {
                "submissions": 10,
                "completions": 10,
                "unavailable_count": 1,
                "total_updates_accepted": 20,
                "total_updates_rejected": 2,
                "last_error": "vlm_timeout",
            }
        )
        self.assertEqual(health, "error")

    def test_tracking_agent_health_color_mapping(self) -> None:
        # Performs test tracking agent health color mapping logic.
        self.assertEqual(tracking_agent_health_color("good"), (90, 220, 120))
        self.assertEqual(tracking_agent_health_color("warn"), (100, 210, 240))
        self.assertEqual(tracking_agent_health_color("error"), (80, 80, 245))

    def test_build_tracking_agent_health_legend_text(self) -> None:
        # Performs test build tracking agent health legend text logic.
        legend = build_tracking_agent_health_legend_text()
        self.assertIn("good", legend)
        self.assertIn("warn", legend)
        self.assertIn("error", legend)

    def test_build_tracking_debug_toggle_hint(self) -> None:
        # Performs test build tracking debug toggle hint logic.
        concise_hint = build_tracking_debug_toggle_hint(False)
        detailed_hint = build_tracking_debug_toggle_hint(True)

        self.assertIn("press 'h'", concise_hint)
        self.assertIn("concise", concise_hint)
        self.assertIn("detailed", detailed_hint)

    def test_radar_track_snapshot_to_dict_includes_relative_depth(self) -> None:
        # Performs test radar track snapshot to dict includes relative depth logic.
        snapshot = RadarTrackSnapshot(
            track_id="depth-1",
            timestamp=10.0,
            pixel_position=None,
            ground_position=GroundPosition(x=1.2, z=3.4),
            ground_coordinate_unit="relative_units",
            position_kind="relative_ground_estimate",
            quality_flags=["relative_depth_estimate:image_geometry"],
            confidence=0.9,
            relative_depth=0.42,
        )

        payload = snapshot.to_dict()

        self.assertIn("relative_depth", payload)
        self.assertEqual(payload["relative_depth"], 0.42)
        self.assertEqual(payload["ground_coordinate_unit"], "relative_units")

    def test_estimate_relative_depth_from_image_geometry_returns_depth(self) -> None:
        # Performs test estimate relative depth from image geometry returns depth logic.
        depth = estimate_relative_depth_from_image_geometry(
            x1=100.0,
            y1=80.0,
            x2=220.0,
            y2=260.0,
            frame_width=1280,
            frame_height=720,
        )

        self.assertIsNotNone(depth)
        assert depth is not None
        self.assertGreaterEqual(depth, 0.0)
        self.assertLessEqual(depth, 1.0)

    def test_estimate_relative_depth_from_image_geometry_near_vs_far(self) -> None:
        # Performs test estimate relative depth from image geometry near vs far logic.
        far_depth = estimate_relative_depth_from_image_geometry(
            x1=300.0,
            y1=60.0,
            x2=380.0,
            y2=180.0,
            frame_width=1280,
            frame_height=720,
        )
        near_depth = estimate_relative_depth_from_image_geometry(
            x1=300.0,
            y1=360.0,
            x2=440.0,
            y2=700.0,
            frame_width=1280,
            frame_height=720,
        )

        assert far_depth is not None
        assert near_depth is not None
        self.assertGreater(far_depth, near_depth)

    def test_estimate_relative_depth_from_image_geometry_invalid_bbox(self) -> None:
        # Performs test estimate relative depth from image geometry invalid bbox logic.
        depth = estimate_relative_depth_from_image_geometry(
            x1=220.0,
            y1=200.0,
            x2=180.0,
            y2=180.0,
            frame_width=1280,
            frame_height=720,
        )

        self.assertIsNone(depth)

    def test_compute_relative_depth_coverage(self) -> None:
        # Performs test compute relative depth coverage logic.
        tracks = [
            RadarTrackSnapshot(
                track_id="1",
                timestamp=1.0,
                pixel_position=None,
                ground_position=GroundPosition(x=0.0, z=1.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=[],
                confidence=0.8,
                relative_depth=0.2,
            ),
            RadarTrackSnapshot(
                track_id="2",
                timestamp=1.0,
                pixel_position=None,
                ground_position=GroundPosition(x=1.0, z=2.0),
                ground_coordinate_unit="relative_units",
                position_kind="relative_ground_estimate",
                quality_flags=[],
                confidence=0.7,
                relative_depth=None,
            ),
        ]

        metrics = compute_relative_depth_coverage(tracks)

        self.assertEqual(metrics["tracks_with_depth"], 1)
        self.assertEqual(metrics["total_tracks"], 2)
        self.assertEqual(metrics["coverage_ratio"], 0.5)
        self.assertEqual(metrics["coverage_percent"], 50.0)

    def test_build_relative_depth_coverage_text(self) -> None:
        # Performs test build relative depth coverage text logic.
        text = build_relative_depth_coverage_text(
            {
                "tracks_with_depth": 3,
                "total_tracks": 4,
                "coverage_percent": 75.0,
            }
        )

        self.assertIn("depth coverage", text)
        self.assertIn("3/4", text)
        self.assertIn("75.0%", text)

    def test_normalize_position_metadata_relative_requires_relative_units(self) -> None:
        # Performs test normalize position metadata relative requires relative units logic.
        kind, unit, flags = normalize_position_metadata(
            position_kind="relative_ground_estimate",
            ground_coordinate_unit="meters",
            quality_flags=["estimated_ground_from_image"],
        )

        self.assertEqual(kind, "relative_ground_estimate")
        self.assertEqual(unit, "relative_units")
        self.assertIn("ground_unit_normalized:relative_units", flags)

    def test_normalize_position_metadata_calibrated_requires_non_relative_unit(self) -> None:
        # Performs test normalize position metadata calibrated requires non relative unit logic.
        kind, unit, flags = normalize_position_metadata(
            position_kind="calibrated_ground_plane",
            ground_coordinate_unit="relative_units",
            quality_flags=["ground_plane_homography"],
        )

        self.assertEqual(kind, "relative_ground_estimate")
        self.assertEqual(unit, "relative_units")
        self.assertIn("position_kind_normalized:invalid_calibrated_unit", flags)

    def test_normalize_position_metadata_unknown_kind_downgrades(self) -> None:
        # Performs test normalize position metadata unknown kind downgrades logic.
        kind, unit, flags = normalize_position_metadata(
            position_kind="simulated_ground_track",
            ground_coordinate_unit="meters",
            quality_flags=[],
        )

        self.assertEqual(kind, "relative_ground_estimate")
        self.assertEqual(unit, "relative_units")
        self.assertIn("position_kind_normalized:unknown_kind", flags)


if __name__ == "__main__":
    unittest.main()
