# Handles test orchestration reasoning live llm behavior.
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agents.reasoning.tools import _shared as orchestrationReasoning


class TestOrchestrationReasoningLiveLLM(unittest.TestCase):
    """Optional live-endpoint tests for Observation/Reasoning LLM calls.

    These tests are skipped unless explicitly enabled:
    - RUN_LIVE_LLM_TESTS=1
    - OPENROUTER_API_KEY set
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Performs setUpClass logic.
        super().setUpClass()
        if os.getenv("RUN_LIVE_LLM_TESTS") != "1":
            raise unittest.SkipTest("Set RUN_LIVE_LLM_TESTS=1 to run live LLM tests")
        if not os.getenv("OPENROUTER_API_KEY"):
            raise unittest.SkipTest("OPENROUTER_API_KEY is required for live LLM tests")

    def tearDown(self) -> None:
        # Performs tearDown logic.
        orchestrationReasoning._reset_reasoning_state()

    def _write_radar_payload(self, radar_path: Path) -> None:
        # Performs write radar payload logic.
        payload = {
            "timestamp": 7000.0,
            "tracks": [
                {
                    "track_id": "track-live-1",
                    "timestamp": 7000.0,
                    "ground_position": {"x": 0.0, "z": 0.0},
                    "confidence": 0.9,
                },
                {
                    "track_id": "track-live-1",
                    "timestamp": 7001.0,
                    "ground_position": {"x": 1.5, "z": 0.8},
                    "confidence": 0.92,
                },
            ],
        }
        radar_path.write_text(json.dumps(payload), encoding="utf-8")

    def test_live_observation_agent_call(self) -> None:
        # Performs test live observation agent call logic.
        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            self._write_radar_payload(radar_path)

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                result = orchestrationReasoning.simulate_observation_summary("track-live-1")

        self.assertTrue(result["available"])
        self.assertIn("summary", result)
        self.assertIn("agents", result)
        self.assertIn("observation", result["agents"])

        observation_meta = result["agents"]["observation"]
        self.assertEqual(observation_meta.get("status"), "ok")
        self.assertIsInstance(observation_meta.get("latency_ms"), (int, float))
        self.assertTrue(result["summary"].get("llm_available"))

    def test_live_reasoning_agent_call(self) -> None:
        # Performs test live reasoning agent call logic.
        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            self._write_radar_payload(radar_path)

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                result = orchestrationReasoning.simulate_reasoning_assessment("track-live-1")

        self.assertTrue(result["available"])
        self.assertIn("summary", result)
        self.assertIn("assessment", result)
        self.assertIn("agents", result)

        observation_meta = result["agents"].get("observation", {})
        reasoning_meta = result["agents"].get("reasoning", {})

        self.assertEqual(observation_meta.get("status"), "ok")
        self.assertEqual(reasoning_meta.get("status"), "ok")
        self.assertTrue(result["summary"].get("llm_available"))
        self.assertTrue(result["assessment"].get("llm_available"))

        evidence_request = result.get("evidence_request")
        if evidence_request is not None:
            self.assertIsInstance(evidence_request, dict)
            self.assertEqual(evidence_request.get("track_id"), "track-live-1")
            self.assertIn("reason", evidence_request)
            self.assertIn("request_index", evidence_request)

    def test_live_evidence_request_budget_is_bounded(self) -> None:
        # Performs test live evidence request budget is bounded logic.
        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "latest_radar_state.json"
            self._write_radar_payload(radar_path)

            with patch("agents.reasoning.tools._shared.RADAR_STATE_PATH", radar_path):
                responses = []
                for index in range(4):
                    response = orchestrationReasoning.simulate_evidence_request(
                        "track-live-1",
                        f"live-budget-check-{index}",
                    )
                    responses.append(response)

        self.assertTrue(all(item.get("available") for item in responses))
        self.assertIsNotNone(responses[0].get("request"))
        self.assertIsNotNone(responses[1].get("request"))
        self.assertIsNotNone(responses[2].get("request"))
        self.assertIsNone(responses[3].get("request"))
        self.assertTrue(responses[3].get("bounded_out"))


if __name__ == "__main__":
    unittest.main()
