"""S014 suite contract checks; no MLflow run or project data mutation."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speaker_id.postprocessing import transductive_scoring as scoring
from speaker_id.postprocessing import transductive_suite as suite


class TransductiveSuiteTests(unittest.TestCase):
    def test_committed_config_matches_fixed_contract(self):
        config = json.loads(
            (ROOT / "configs/postprocessing/campp_s014.json").read_text(
                encoding="utf-8"))
        suite.validate_config(config)
        changed = json.loads(json.dumps(config))
        changed["alignment"]["known_strength"] = 0.51
        with self.assertRaisesRegex(ValueError, "formula"):
            suite.validate_config(changed)

    def test_cuda_arithmetic_preserves_decisions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        rng = np.random.default_rng(14014)
        probabilities = rng.dirichlet(
            np.ones(scoring.TOTAL_CLASSES), size=32)
        valid = np.ones(32, dtype=bool)
        valid[[3, 17]] = False
        aligned = scoring.align_probabilities(probabilities, valid)
        parity = suite._cuda_policy_parity(aligned)
        self.assertTrue(parity["exact_predictions"])
        self.assertLess(parity["maximum_probability_absolute_difference"],
                        1e-12)

    def test_outer_speaker_ids_are_removed_before_reconstruction(self):
        manifest = [
            {"audio_file": "a.wav", "speaker_id": "known_a"},
            {"audio_file": "b.wav", "speaker_id": "known_b"},
            {"audio_file": "c.wav", "speaker_id": "unknown"},
        ]
        folds = [
            {"audio_file": "a.wav", "fold": "0"},
            {"audio_file": "b.wav", "fold": "1"},
            {"audio_file": "c.wav", "fold": "0"},
        ]
        masked, receipt = suite._without_outer_speaker_ids(
            manifest, folds, 0)
        self.assertNotIn("speaker_id", masked[0])
        self.assertEqual(masked[1]["speaker_id"], "known_b")
        self.assertNotIn("speaker_id", masked[2])
        self.assertEqual(receipt["own_fold_speaker_id_fields_removed"], 2)
        self.assertFalse(
            receipt["own_fold_speaker_ids_supplied_to_reconstruction"])
        self.assertTrue(all("speaker_id" in row for row in manifest))

    def test_pinned_real_cache_preserves_fixed_policy_decision_hashes(self):
        config = json.loads(
            (ROOT / "configs/postprocessing/campp_s014.json").read_text(
                encoding="utf-8"))
        for outer in (0, 1):
            entry = config["reconstruction_control"]["folds"][str(outer)]
            path = ROOT / entry["path"]
            self.assertEqual(suite.file_sha256(path), entry["sha256"])
            with np.load(path, allow_pickle=False) as saved:
                probabilities = saved["probabilities"]
                valid = ~(
                    (probabilities[:, 0] == 1)
                    & np.all(probabilities[:, 1:] == 0, axis=1))
                prediction = scoring.align_probabilities(
                    probabilities, valid)["adjusted_probabilities"].argmax(1)
            self.assertEqual(
                suite._array_sha(prediction),
                entry["powered_prediction_array_sha256"])

if __name__ == "__main__":
    unittest.main()
