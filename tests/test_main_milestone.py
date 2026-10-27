from __future__ import annotations

import unittest

import main


class FakeItem:
    def __init__(self, text=None, json_payload=None):
        self.text = text
        self.json = json_payload


class FakeResult:
    def __init__(self, content):
        self.content = content


class TestMainMilestoneHelpers(unittest.TestCase):
    def test_get_result_json_object_from_text_json(self) -> None:
        result = FakeResult([FakeItem(text='{"phase": "milestone_1", "completion_percent": 85}')])

        payload = main.get_result_json_object(result)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["phase"], "milestone_1")
        self.assertEqual(payload["completion_percent"], 85)

    def test_get_result_json_object_from_json_attribute(self) -> None:
        result = FakeResult([FakeItem(json_payload={"status": "on_track"})])

        payload = main.get_result_json_object(result)

        self.assertEqual(payload, {"status": "on_track"})

    def test_format_milestone_snapshot(self) -> None:
        snapshot = main.format_milestone_snapshot(
            {
                "phase": "milestone_1_dashboard_flow",
                "status": "on_track",
                "milestone_finished": False,
                "completion_percent": 90.0,
                "completed_checks": 9,
                "total_checks": 10,
            }
        )

        self.assertIn("milestone_1_dashboard_flow", snapshot)
        self.assertIn("on_track", snapshot)
        self.assertIn("90.0%", snapshot)
        self.assertIn("9/10", snapshot)
        self.assertIn("finished=False", snapshot)

    def test_format_milestone_completion_notice(self) -> None:
        notice = main.format_milestone_completion_notice(
            {
                "phase": "milestone_1_dashboard_flow",
                "milestone_finished": True,
            }
        )
        self.assertEqual(notice, "✅ Milestone finished: milestone_1_dashboard_flow")

        no_notice = main.format_milestone_completion_notice(
            {
                "phase": "milestone_1_dashboard_flow",
                "milestone_finished": False,
            }
        )
        self.assertIsNone(no_notice)


if __name__ == "__main__":
    unittest.main()
