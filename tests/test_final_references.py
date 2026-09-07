"""Final enrollment must remove whole query groups without reserving an outer fold."""

import copy
import unittest

import numpy as np

from speaker_id.training.crossfit_references import all_training_crossfit_scores
from speaker_id.training.final_references import prepare_final_references


class FinalReferenceTests(unittest.TestCase):
    def fixture(self):
        rows = [("a0", "A", "a", 0, [1, 0, 0]),
                ("a0copy", "A", "a", 0, [1, 0, 0]),
                ("a1", "A", "a1", 1, [0, 1, 0]),
                ("b0", "B", "b0", 0, [0, 0, 1]),
                ("b1", "B", "b1", 1, [0, .1, 1]),
                ("u0", "unknown", "u", 0, [1, 0, 0]),
                ("u0copy", "unknown", "u", 0, [1, 0, 0]),
                ("u1", "unknown", "u1", 1, [0, 1, 0]),
                ("zero", "A", "z", 0, [0, 0, 0])]
        manifest = [{"audio_file": name, "speaker_id": label} for name, label, *_ in rows]
        folds = [{"audio_file": name, "speaker_id": label, "group_id": group,
                  "fold": fold, "train_eligible": name != "zero", "evaluation_included": True}
                 for name, label, group, fold, vector in rows]
        return np.asarray([r[4] for r in rows], dtype=np.float32), np.asarray([r[0] != "zero" for r in rows]), manifest, folds

    def test_full_data_uses_both_folds_and_excludes_whole_query_group(self):
        values = self.fixture(); original = copy.deepcopy(values[3])
        scores = all_training_crossfit_scores(*values, classes=2)
        np.testing.assert_array_equal(scores["calibration_indices"], np.arange(8))
        self.assertEqual(scores["outer_indices"].size, 0)
        self.assertEqual(scores["outer_known_scores"].shape, (0, 2))
        self.assertEqual(scores["inner_known_scores"][0, 0], 0)
        self.assertEqual(scores["inner_unknown_similarity"][5], 0)
        self.assertEqual(values[3], original)
        self.assertEqual(scores["provenance"]["scope"], "all_training_final_calibration")

    def test_final_gallery_restores_all_references_and_marks_metric_training_only(self):
        result = prepare_final_references(*self.fixture(), ["unknown", "A", "B"],
                                          unknown_weights=[0, .5], margin_weights=[0, .5], candidates=5)
        gallery = result["gallery"]
        self.assertEqual(gallery["known_embeddings"].shape, (5, 3))
        self.assertEqual(gallery["unknown_embeddings"].shape, (3, 3))
        np.testing.assert_array_equal(gallery["known_targets"], [1, 1, 1, 2, 2])
        np.testing.assert_allclose(np.linalg.norm(gallery["known_embeddings"], axis=1), 1)
        self.assertEqual(result["report"]["calibration_query_files"], 8)
        self.assertEqual(result["report"]["encoder_updates"], 0)
        self.assertIn("not OOF", result["calibration"]["score_scope"])
        self.assertNotIn("inner_macro_f1_447", result["calibration"])

    def test_invalid_label_order_fails(self):
        with self.assertRaisesRegex(ValueError, "labels"):
            prepare_final_references(*self.fixture(), ["unknown", "B", "A"],
                                     unknown_weights=[0], margin_weights=[0], candidates=3)

    def test_singleton_remains_gallery_but_is_not_a_calibration_query(self):
        values = list(self.fixture()); values[3][4]["train_eligible"] = False
        result = prepare_final_references(*values, ["unknown", "A", "B"],
                                          unknown_weights=[0], margin_weights=[0], candidates=3)
        self.assertIn(3, result["reference_indices"])
        self.assertNotIn(3, result["calibration_indices"])
        self.assertEqual(result["report"]["singleton_known_labels_skipped_as_queries"], ["B"])


if __name__ == "__main__":
    unittest.main()
