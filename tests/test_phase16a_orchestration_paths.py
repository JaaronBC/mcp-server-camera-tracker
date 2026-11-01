from __future__ import annotations

import unittest
from pathlib import Path

from agents.orchestration.tools import health as health_tools
from agents.orchestration.tools import metrics as metrics_tools
from agents.orchestration.tools import ops as ops_tools


class TestPhase16AOrchestrationToolPaths(unittest.TestCase):
    def test_paths_resolve_from_repository_root(self) -> None:
        expected_root = Path(__file__).resolve().parents[1]
        expected_captures = expected_root / "captures"

        self.assertEqual(health_tools.PROJECT_ROOT, expected_root)
        self.assertEqual(metrics_tools.PROJECT_ROOT, expected_root)
        self.assertEqual(ops_tools.PROJECT_ROOT, expected_root)

        self.assertEqual(health_tools.RADAR_STATE_PATH, expected_captures / "latest_radar_state.json")
        self.assertEqual(metrics_tools.CAPTURES_DIR, expected_captures)
        self.assertEqual(ops_tools.AGENT_EVENTS_PATH, expected_captures / "agent_events.jsonl")

        self.assertNotIn("/agents/captures", str(health_tools.RADAR_STATE_PATH))
        self.assertNotIn("/agents/captures", str(metrics_tools.CAPTURES_DIR))
        self.assertNotIn("/agents/captures", str(ops_tools.AGENT_EVENTS_PATH))


if __name__ == "__main__":
    unittest.main()
