import unittest

from tools.agent_contracts import TrackingAgentObservation, VisualAssociationUpdate
from tools.components.dashboard_types import GroundPosition, RadarTrackSnapshot
from tools.components.liveTrackingDisplay import (
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
    tracking_agent_health_color,
)


class TestLiveTrackingDisplayMessage(unittest.TestCase):
    def test_build_tracking_message_schema(self) -> None:
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
        self.assertEqual(build_tracking_detection_id("12"), "12")

    def test_apply_tracking_agent_updates_summary(self) -> None:
        tracks = {
            "7": RadarTrackSnapshot(
                track_id="7",
                timestamp=1.0,
                pixel_position=None,
                ground_position=GroundPosition(x=0.0, z=1.0),
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

    def test_build_tracking_agent_status_text(self) -> None:
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
        status_text = build_tracking_agent_status_text({})
        self.assertIn("last_latency=n/a", status_text)
        self.assertIn("last_error=none", status_text)

    def test_classify_tracking_agent_health_good(self) -> None:
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
        self.assertEqual(tracking_agent_health_color("good"), (90, 220, 120))
        self.assertEqual(tracking_agent_health_color("warn"), (100, 210, 240))
        self.assertEqual(tracking_agent_health_color("error"), (80, 80, 245))

    def test_build_tracking_agent_health_legend_text(self) -> None:
        legend = build_tracking_agent_health_legend_text()
        self.assertIn("good", legend)
        self.assertIn("warn", legend)
        self.assertIn("error", legend)

    def test_build_tracking_debug_toggle_hint(self) -> None:
        concise_hint = build_tracking_debug_toggle_hint(False)
        detailed_hint = build_tracking_debug_toggle_hint(True)

        self.assertIn("press 'h'", concise_hint)
        self.assertIn("concise", concise_hint)
        self.assertIn("detailed", detailed_hint)

    def test_radar_track_snapshot_to_dict_includes_relative_depth(self) -> None:
        snapshot = RadarTrackSnapshot(
            track_id="depth-1",
            timestamp=10.0,
            pixel_position=None,
            ground_position=GroundPosition(x=1.2, z=3.4),
            position_kind="relative_ground_estimate",
            quality_flags=["relative_depth_estimate:image_geometry"],
            confidence=0.9,
            relative_depth=0.42,
        )

        payload = snapshot.to_dict()

        self.assertIn("relative_depth", payload)
        self.assertEqual(payload["relative_depth"], 0.42)

    def test_estimate_relative_depth_from_image_geometry_returns_depth(self) -> None:
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
        tracks = [
            RadarTrackSnapshot(
                track_id="1",
                timestamp=1.0,
                pixel_position=None,
                ground_position=GroundPosition(x=0.0, z=1.0),
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


if __name__ == "__main__":
    unittest.main()
