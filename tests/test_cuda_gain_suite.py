"""C002 orchestration guardrails without a server or model."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training import cuda_gain_suite as cuda


class Tracker:
    def __init__(self):
        self.artifacts = []
        self.events = []

    def add_artifact(self, path, name=None):
        self.artifacts.append((Path(path), name))

    def log_metrics(self, *args, **kwargs):
        self.events.append("metrics")

    def flush(self, *, strict):
        self.assertTrue(strict)

    def assertTrue(self, value):
        if not value:
            raise AssertionError


class CUDASuiteTests(unittest.TestCase):
    def test_selector_excludes_historical_control_and_requires_both_folds(self):
        selected = {"frontend": "identity", "advanced_weight": 0.0, "calibration": {}}
        frozen = {
            "selected": {0: selected, 1: selected},
            "inner_fits": {0: {"identity": {"selected": selected}}, 1: {"identity": {"selected": selected}}},
            "no_fresh_outer_evaluation_performed_yet": True,
            "historical_control_excluded_from_selection": True,
        }
        self.assertEqual(cuda.selected_cuda_policy("C002d", 0, frozen)["frontend"], "identity")
        frozen["selected"][0] = {**selected, "frontend": "historical_control"}
        with self.assertRaises(ValueError):
            cuda.selected_cuda_policy("C002d", 0, frozen)

    def test_callback_tracks_json_only_and_rejects_embeddings(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = Tracker()
            callback = cuda.extraction_callback(Path(temporary), tracker)
            callback("progress", {"completed_pairs": 1, "total": 2, "elapsed_seconds": 0.5})
            self.assertEqual(tracker.artifacts[0][0].suffix, ".json")
            self.assertTrue(tracker.artifacts[0][1].startswith("cuda_extraction/"))
            with self.assertRaises(ValueError):
                callback("complete", {"embedding": np.zeros(512, np.float32)})


if __name__ == "__main__":
    unittest.main()
