from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.reasoning.tools import evidence_request, observation_summary, reasoning_assessment


class TestReasoningToolWrappers(unittest.TestCase):
    def test_reasoning_assessment_delegates_to_shared(self) -> None:
        expected = {
            "available": True,
            "track_id": "track-1",
            "summary": {},
            "assessment": {},
            "evidence_request": None,
            "agents": {},
            "handoff_validation": {"status": "ok", "error": None, "error_code": None},
        }

        with patch(
            "agents.reasoning.tools._shared.simulate_reasoning_assessment",
            return_value=expected,
        ) as shared_mock:
            result = reasoning_assessment.simulate_reasoning_assessment(
                "track-1",
                request_evidence_on_uncertainty=False,
            )

        shared_mock.assert_called_once_with(
            "track-1",
            request_evidence_on_uncertainty=False,
        )
        self.assertIs(result, expected)

    def test_observation_summary_delegates_to_shared(self) -> None:
        expected = {
            "available": True,
            "track_id": "track-1",
            "summary": {},
            "agents": {},
            "history_records": [],
        }

        with patch(
            "agents.reasoning.tools._shared.simulate_observation_summary",
            return_value=expected,
        ) as shared_mock:
            result = observation_summary.simulate_observation_summary("track-1")

        shared_mock.assert_called_once_with("track-1")
        self.assertIs(result, expected)

    def test_evidence_request_delegates_to_shared(self) -> None:
        expected = {
            "available": True,
            "track_id": "track-1",
            "ingested_tracks": 1,
            "request": None,
            "bounded_out": True,
        }

        with patch(
            "agents.reasoning.tools._shared.simulate_evidence_request",
            return_value=expected,
        ) as shared_mock:
            result = evidence_request.simulate_evidence_request("track-1", "need_more")

        shared_mock.assert_called_once_with("track-1", "need_more")
        self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
