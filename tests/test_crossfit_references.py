"""Behavioral leakage tests for frozen-embedding crossfit reference scoring."""

import unittest

import numpy as np

from speaker_id.training.crossfit_references import crossfit_scores


class OuterLabelForbidden(dict):
    def __getitem__(self, key):
        if key == "speaker_id":
            raise AssertionError("Outer label was accessed")
        return super().__getitem__(key)


class CrossfitReferenceTests(unittest.TestCase):
    def fixture(self):
        # A has two groups, with two copies in its first group. B is a singleton.
        # Unknown also has a two-copy group and a second independent group.
        rows = [
            ("a1", "A", "ga1", 1, [1, 0, 0], True),
            ("a1-copy", "A", "ga1", 1, [1, 0, 0], True),
            ("a2", "A", "ga2", 1, [0, 1, 0], True),
            ("b1", "B", "gb1", 1, [0, 0, 1], True),
            ("u1", "unknown", "gu1", 1, [1, 0, 0], True),
            ("u1-copy", "unknown", "gu1", 1, [1, 0, 0], True),
            ("u2", "unknown", "gu2", 1, [0, 1, 0], True),
            ("outer-a", "PRIVATE", "goa", 0, [1, 0, 0], True),
            ("outer-zero", "PRIVATE", "goz", 0, [0, 0, 0], False),
        ]
        manifest = [(OuterLabelForbidden if fold == 0 else dict)(audio_file=name, speaker_id=label)
                    for name, label, group, fold, vector, valid in rows]
        folds = [{"audio_file": name, "group_id": group, "fold": fold,
                  "train_eligible": str(valid), "evaluation_included": "True"}
                 for name, label, group, fold, vector, valid in rows]
        embeddings = np.asarray([row[4] for row in rows], dtype=np.float32)
        valid = np.asarray([row[5] for row in rows], dtype=bool)
        return embeddings, valid, manifest, folds

    def run_method(self, method):
        return crossfit_scores(*self.fixture(), outer=0, method=method, classes=2)

    def test_known_query_excludes_entire_duplicate_group_for_both_methods(self):
        for method in ("prototype", "max_reference"):
            with self.subTest(method=method):
                result = self.run_method(method)
                for query in (0, 1):
                    position = result["calibration_indices"].tolist().index(query)
                    # Its duplicate must not leave a cosine-1 reference behind.
                    self.assertAlmostEqual(float(result["inner_known_scores"][position, 0]), 0.0)
                    self.assertEqual(result["reference_counts"]["inner_known_files_per_class"][position, 0], 1)

    def test_unknown_query_excludes_entire_duplicate_group(self):
        result = self.run_method("prototype")
        for query in (4, 5):
            position = result["calibration_indices"].tolist().index(query)
            self.assertAlmostEqual(float(result["inner_unknown_similarity"][position]), 0.0)
            self.assertEqual(result["reference_counts"]["inner_unknown_files"][position], 1)

    def test_singleton_stays_in_gallery_but_never_becomes_calibration_query(self):
        result = self.run_method("prototype")
        self.assertNotIn(3, result["calibration_indices"])
        self.assertIn(3, result["provenance"]["reference_indices"])
        self.assertEqual(result["known_labels"], ["A", "B"])
        self.assertEqual(result["reference_counts"]["singleton_known_labels"], ["B"])
        np.testing.assert_array_equal(result["reference_counts"]["known_files_per_class"], [3, 1])

    def test_outer_rows_are_never_references_and_outer_labels_are_not_accessed(self):
        for method in ("prototype", "max_reference"):
            result = self.run_method(method)
            np.testing.assert_array_equal(result["outer_indices"], [7, 8])
            self.assertFalse(set(result["outer_indices"]) & set(result["provenance"]["reference_indices"]))
            self.assertFalse(result["provenance"]["outer_labels_accessed"])

    def test_outer_uses_full_references_and_preserves_invalid_rows(self):
        prototype = self.run_method("prototype")
        maximum = self.run_method("max_reference")
        # Full A prototype = normalize([2, 1, 0]); no calibration holdout persists.
        self.assertAlmostEqual(float(prototype["outer_known_scores"][0, 0]), 2 / np.sqrt(5), places=6)
        self.assertAlmostEqual(float(maximum["outer_known_scores"][0, 0]), 1.0)
        np.testing.assert_array_equal(prototype["outer_valid"], [True, False])
        np.testing.assert_array_equal(prototype["outer_known_scores"][1], [0, 0])
        self.assertEqual(float(prototype["outer_unknown_similarity"][1]), 0.0)

    def test_unknown_queries_use_all_known_references(self):
        result = self.run_method("prototype")
        query = result["calibration_indices"].tolist().index(4)
        self.assertAlmostEqual(float(result["inner_known_scores"][query, 0]), 2 / np.sqrt(5), places=6)
        np.testing.assert_array_equal(result["reference_counts"]["inner_known_files_per_class"][query], [3, 1])

    def test_group_crossing_outer_folds_is_rejected(self):
        inputs = list(self.fixture())
        inputs[3][7]["group_id"] = "ga1"
        with self.assertRaisesRegex(ValueError, "crosses outer"):
            crossfit_scores(*inputs, outer=0, classes=2)

    def test_validity_and_training_eligibility_both_gate_references(self):
        inputs = list(self.fixture())
        inputs[1][1] = False
        inputs[3][5]["train_eligible"] = "False"
        result = crossfit_scores(*inputs, outer=0, classes=2)
        self.assertNotIn(1, result["provenance"]["reference_indices"])
        self.assertNotIn(5, result["provenance"]["reference_indices"])
        np.testing.assert_array_equal(result["reference_counts"]["known_files_per_class"], [2, 1])

    def test_differently_ordered_fold_table_keeps_manifest_indices(self):
        inputs = list(self.fixture())
        inputs[3] = list(reversed(inputs[3]))
        actual = crossfit_scores(*inputs, outer=0, classes=2)
        expected = self.run_method("prototype")
        np.testing.assert_array_equal(actual["calibration_indices"], expected["calibration_indices"])
        np.testing.assert_allclose(actual["inner_known_scores"], expected["inner_known_scores"])

    def test_empty_unknown_leave_group_out_pool_fails_explicitly(self):
        inputs = list(self.fixture())
        inputs[3][6]["train_eligible"] = "False"
        with self.assertRaisesRegex(ValueError, "at least two eligible unknown groups"):
            crossfit_scores(*inputs, outer=0, classes=2)


if __name__ == "__main__":
    unittest.main()
