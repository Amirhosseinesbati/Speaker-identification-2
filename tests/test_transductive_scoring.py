"""Synthetic S014 checks; no project labels, caches or MLflow are read."""
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speaker_id.postprocessing import transductive_scoring as scoring


class TransductiveScoringTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(140)
        self.probabilities = rng.dirichlet(
            np.ones(scoring.TOTAL_CLASSES), size=12)
        self.valid = np.asarray(
            [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
        self.labels = ["unknown"] + [
            f"speaker_{index}" for index in range(scoring.KNOWN_CLASSES)]

    def test_api_is_label_free_and_uses_absolute_design_prior(self):
        self.assertEqual(
            set(inspect.signature(scoring.align_probabilities).parameters),
            {"probabilities", "valid"})
        result = scoring.align_probabilities(
            self.probabilities, self.valid)
        normalized = self.probabilities / self.probabilities.sum(
            axis=1, keepdims=True)
        observed = normalized[self.valid].mean(axis=0)
        factors = np.ones(scoring.TOTAL_CLASSES)
        factors[1:] = np.power((0.5 / 446) / observed[1:], 0.5)
        expected = normalized[self.valid] * factors
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            result["adjusted_probabilities"][self.valid], expected,
            rtol=0, atol=2e-15)
        np.testing.assert_allclose(
            result["factors"], factors, rtol=0, atol=5e-16)
        self.assertEqual(result["factors"][0], 1.0)
        self.assertFalse(result["policy"]["organizer_guarantees_this_prior"])
        conditional = observed[1:] / observed[1:].sum()
        conditional_factor = np.sqrt((1 / 446) / conditional)
        self.assertFalse(np.allclose(
            result["factors"][1:], conditional_factor, rtol=1e-12))

    def test_order_equivariance_and_invalid_exact_unknown(self):
        original = scoring.align_probabilities(
            self.probabilities, self.valid)
        permutation = np.asarray(
            [7, 0, 11, 3, 6, 2, 9, 1, 4, 10, 8, 5])
        permuted = scoring.align_probabilities(
            self.probabilities[permutation], self.valid[permutation])
        np.testing.assert_allclose(
            permuted["adjusted_probabilities"],
            original["adjusted_probabilities"][permutation],
            rtol=0, atol=2e-15)
        invalid = original["adjusted_probabilities"][~self.valid]
        np.testing.assert_array_equal(
            invalid, np.eye(1, scoring.TOTAL_CLASSES, 0).repeat(
                len(invalid), axis=0))
        for value in original.values():
            if isinstance(value, np.ndarray):
                self.assertFalse(value.flags.writeable)

    def test_both_folds_and_selection_are_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scoring.seal_fold(
                root / "fold_0", 0, self.probabilities, self.valid,
                [f"fold0_{i}.wav" for i in range(12)], self.labels)
            with self.assertRaises(ValueError):
                scoring.seal_selection(root)
            scoring.seal_fold(
                root / "fold_1", 1, self.probabilities, self.valid,
                [f"fold1_{i}.wav" for i in range(12)], self.labels)
            receipt = scoring.seal_selection(root)
            loaded_receipt, folds = scoring.load_selection(root)
            self.assertEqual(
                loaded_receipt["selection_sha256"],
                receipt["selection_sha256"])
            self.assertEqual(set(folds), {0, 1})
            with self.assertRaises(ValueError):
                folds[0]["adjusted_probabilities"][0, 0] = 0
            path = root / "fold_0" / "seal.json"
            path.write_text(
                path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection receipt"):
                scoring.load_selection(root)

    def test_bootstrap_is_deterministic_and_promotion_requires_parity(self):
        truth = np.repeat(
            np.arange(scoring.TOTAL_CLASSES, dtype=np.int64), 2)
        prediction = truth.copy()
        groups = np.asarray([f"group_{i}" for i in range(len(truth))])
        first = scoring.speaker_group_bootstrap(
            truth, prediction, prediction, groups, samples=100)
        second = scoring.speaker_group_bootstrap(
            truth, prediction, prediction, groups, samples=100)
        np.testing.assert_array_equal(first["deltas"], second["deltas"])
        self.assertTrue(np.all(first["deltas"] == 0))
        base = {"macro_f1": .95, "accuracy": .95,
                "errors": {"unknown_to_known": 10,
                           "known_to_other_known": 4}}
        adjusted = {"macro_f1": .952, "accuracy": .951,
                    "errors": {"unknown_to_known": 12,
                               "known_to_other_known": 5}}
        base_folds = [{"macro_f1": .94}, {"macro_f1": .96}]
        new_folds = [{"macro_f1": .941}, {"macro_f1": .961}]
        bootstrap = {"lower": 0.0}
        accepted = scoring.promotion_decision(
            base, adjusted, base_folds, new_folds, bootstrap, True)
        rejected = scoring.promotion_decision(
            base, adjusted, base_folds, new_folds, bootstrap, False)
        self.assertTrue(accepted["promoted"])
        self.assertFalse(rejected["promoted"])
        self.assertFalse(rejected["gates"]["cpu_cuda_prediction_parity"])

    def test_bootstrap_handles_cross_label_content_conflicts_within_strata(self):
        truth = np.repeat(
            np.arange(scoring.TOTAL_CLASSES, dtype=np.int64), 2)
        prediction = truth.copy()
        groups = np.asarray(["shared_zero_group"] * len(truth))
        result = scoring.speaker_group_bootstrap(
            truth, prediction, prediction, groups, samples=100)
        self.assertTrue(np.all(result["deltas"] == 0))
        self.assertEqual(result["speaker_content_group_units"],
                         scoring.TOTAL_CLASSES)
        self.assertTrue(result[
            "cross_label_content_groups_are_resampled_within_each_true_speaker"])
        self.assertEqual(result["cross_label_content_groups"], 1)
        self.assertEqual(result["rows_in_cross_label_content_groups"], len(truth))


if __name__ == "__main__":
    unittest.main()
