"""F009 private tracking and screen completion semantics; no audio or GPU."""
from __future__ import annotations

from copy import deepcopy
import unittest

from speaker_id.training.runner import _terminal_status, _tracking_config


class F009RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolved = {
            "experiment": {"retention": {"mlflow_upload_private_reports": False}},
            "model": {"architecture": "campp"},
            "labels": ["unknown", "private-speaker"],
            "manifest": [{"audio_file": "private.wav", "speaker_id": "private-speaker"}],
            "roles": [{"audio_file": "private.wav", "speaker_id": "private-speaker"}],
            "folds": [{"audio_file": "private.wav", "speaker_id": "private-speaker"}],
            "input_hashes": {"manifest": "a" * 64},
        }

    def test_private_safe_tracking_config_removes_file_level_payloads(self) -> None:
        original = deepcopy(self.resolved)
        safe = _tracking_config(self.resolved, {"retention": {"mlflow_upload_private_reports": False}})
        self.assertEqual(self.resolved, original)
        for field in ("labels", "manifest", "roles", "folds"):
            self.assertNotIn(field, safe)
        self.assertFalse(safe["private_labels_uploaded"])
        self.assertFalse(safe["private_role_rows_uploaded"])
        self.assertEqual(safe["model"], self.resolved["model"])
        self.assertEqual(safe["input_hashes"], self.resolved["input_hashes"])

    def test_legacy_tracking_config_keeps_compatibility(self) -> None:
        self.assertEqual(_tracking_config(self.resolved, {"retention": {}}), self.resolved)

    def test_screen_has_distinct_terminal_state(self) -> None:
        self.assertEqual(_terminal_status(False), "screen_complete")
        self.assertEqual(_terminal_status(True), "complete")


if __name__ == "__main__":
    unittest.main()
