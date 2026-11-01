from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agents.dashboard.state import SharedDashboardState
from agents.dashboard.tracking_loop import _publish_tracking_estimate_diagnostics
from agents.tracking.tracking_types import GroundPosition, PixelPosition, RadarTrackSnapshot


def _accepted_entry(
    *,
    track_id: str,
    estimate_id: str,
    now: float,
    expires_at: float,
    validation_valid: bool = True,
    coordinate_system: str = "calibrated_ground_plane",
) -> dict:
    return {
        "proposal": {
            "track_id": track_id,
            "estimate_id": estimate_id,
            "estimated_pixel_center": {"x": 140.0, "y": 150.0},
            "estimated_ground_position": {"x": 1.2, "z": 4.8, "unit": "meters"},
            "coordinate_system": coordinate_system,
            "estimated_at": now - 0.1,
            "predicted_for_timestamp": now + 0.4,
            "expires_at": expires_at,
            "source_tracking_message_ids": ["msg-1"],
            "source_verification_observation_id": "verify-msg-1",
            "source_verification_message_id": "msg-1",
        },
        "validation": {
            "valid": validation_valid,
            "freshness_status": "fresh",
            "plausibility_status": "plausible",
            "errors": [],
            "warnings": [],
        },
    }


class TestPhase16ATrackingEstimateDiagnostics(unittest.TestCase):
    def _read_payload(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_accepted_current_estimate_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latest_tracking_estimate_diagnostics.json"
            now = 2000.0

            state = SharedDashboardState()
            state.latest_tracking_estimate_proposals = {
                "accepted": [_accepted_entry(track_id="track-1", estimate_id="est-1", now=now, expires_at=now + 1.0)],
                "accepted_count": 1,
                "rejected": [],
                "rejected_count": 0,
                "generated_at": now,
            }

            _publish_tracking_estimate_diagnostics(state=state, now=now, diagnostics_path=output_path)

            payload = self._read_payload(output_path)
            self.assertEqual(payload["classification"], "predicted_estimate_diagnostics")
            self.assertEqual(payload["current_valid_count"], 1)

            item = payload["current_valid_estimates"][0]
            self.assertEqual(item["track_id"], "track-1")
            self.assertEqual(item["estimate_id"], "est-1")
            self.assertEqual(item["classification"], "predicted_estimate")
            self.assertIsInstance(item["estimated_pixel_position"], dict)
            self.assertIsInstance(item["estimated_ground_position"], dict)
            self.assertTrue(item["coordinator_validation"]["valid"])

    def test_rejected_estimate_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latest_tracking_estimate_diagnostics.json"
            now = 2000.0

            state = SharedDashboardState()
            state.latest_tracking_estimate_proposals = {
                "accepted": [_accepted_entry(track_id="track-1", estimate_id="est-1", now=now, expires_at=now + 1.0, validation_valid=False)],
                "accepted_count": 1,
                "rejected": [],
                "rejected_count": 0,
                "generated_at": now,
            }

            _publish_tracking_estimate_diagnostics(state=state, now=now, diagnostics_path=output_path)

            payload = self._read_payload(output_path)
            self.assertEqual(payload["current_valid_count"], 0)
            self.assertEqual(len(payload["current_valid_estimates"]), 0)
            self.assertGreaterEqual(payload["rejected_or_invalid_count"], 1)

    def test_expired_estimate_removed_from_current_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latest_tracking_estimate_diagnostics.json"
            now = 2000.0

            state = SharedDashboardState()
            state.latest_tracking_estimate_proposals = {
                "accepted": [_accepted_entry(track_id="track-1", estimate_id="est-1", now=now, expires_at=now - 0.01)],
                "accepted_count": 1,
                "rejected": [],
                "rejected_count": 0,
                "generated_at": now,
            }

            _publish_tracking_estimate_diagnostics(state=state, now=now, diagnostics_path=output_path)

            payload = self._read_payload(output_path)
            self.assertEqual(payload["current_valid_count"], 0)
            self.assertEqual(payload["expired_count"], 1)

    def test_empty_estimate_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latest_tracking_estimate_diagnostics.json"
            now = 2000.0

            state = SharedDashboardState()
            state.latest_tracking_estimate_proposals = None

            _publish_tracking_estimate_diagnostics(state=state, now=now, diagnostics_path=output_path)

            payload = self._read_payload(output_path)
            self.assertEqual(payload["current_valid_count"], 0)
            self.assertEqual(payload["expired_count"], 0)
            self.assertEqual(payload["rejected_or_invalid_count"], 0)

    def test_non_calibrated_estimate_omits_ground_position(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latest_tracking_estimate_diagnostics.json"
            now = 2000.0

            state = SharedDashboardState()
            state.latest_tracking_estimate_proposals = {
                "accepted": [
                    _accepted_entry(
                        track_id="track-1",
                        estimate_id="est-1",
                        now=now,
                        expires_at=now + 1.0,
                        coordinate_system="relative_image_plane",
                    )
                ],
                "accepted_count": 1,
                "rejected": [],
                "rejected_count": 0,
                "generated_at": now,
            }

            _publish_tracking_estimate_diagnostics(state=state, now=now, diagnostics_path=output_path)

            payload = self._read_payload(output_path)
            item = payload["current_valid_estimates"][0]
            self.assertIsNone(item["estimated_ground_position"])

    def test_diagnostics_publish_does_not_mutate_measured_tracking_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "latest_tracking_estimate_diagnostics.json"
            now = 2000.0

            state = SharedDashboardState()
            measured_track = RadarTrackSnapshot(
                track_id="track-measured",
                timestamp=now,
                pixel_position=PixelPosition(x=90.0, y=110.0),
                ground_position=GroundPosition(x=1.0, z=2.0),
                ground_coordinate_unit="meters",
                position_kind="calibrated_ground_plane",
                quality_flags=["position_kind:calibrated_ground_plane"],
                confidence=0.95,
                relative_depth=0.4,
            )
            state.active_tracks = {"track-measured": measured_track}
            state.last_observation_time = now - 0.25
            state.latest_tracking_estimate_proposals = {
                "accepted": [_accepted_entry(track_id="track-1", estimate_id="est-1", now=now, expires_at=now + 1.0)],
                "accepted_count": 1,
                "rejected": [],
                "rejected_count": 0,
                "generated_at": now,
            }

            active_tracks_before = {
                track_id: snapshot.to_dict()
                for track_id, snapshot in state.active_tracks.items()
            }
            last_observation_before = state.last_observation_time

            _publish_tracking_estimate_diagnostics(state=state, now=now, diagnostics_path=output_path)

            active_tracks_after = {
                track_id: snapshot.to_dict()
                for track_id, snapshot in state.active_tracks.items()
            }
            self.assertEqual(active_tracks_after, active_tracks_before)
            self.assertEqual(state.last_observation_time, last_observation_before)


if __name__ == "__main__":
    unittest.main()
