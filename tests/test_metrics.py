import math
import unittest

from speaker_id.evaluation.metrics import predictions_from_probabilities, score_predictions


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.labels = ["unknown"] + [f"speaker-{i}" for i in range(446)]
        self.reference = [{"audio_file": f"file-{i}", "speaker_id": label}
                          for i, label in enumerate(self.labels)]

    def test_perfect_predictions_align_by_filename_and_include_all_labels(self):
        result = score_predictions(self.reference, list(reversed(self.reference)), self.labels)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(len(result["per_class"]), 447)

    def test_all_unknown_prediction_does_not_omit_known_classes(self):
        predictions = [{**row, "speaker_id": "unknown"} for row in self.reference]
        result = score_predictions(self.reference, predictions, self.labels)
        self.assertAlmostEqual(result["macro_f1"], (2 / 448) / 447)
        self.assertEqual(result["errors"]["known_to_unknown"], 446)
        self.assertEqual(result["per_class"][1]["f1"], 0.0)

    def test_missing_extra_duplicate_and_invalid_label_predictions_rejected(self):
        bad_submissions = [self.reference[:-1], self.reference + [self.reference[0]],
                           [{**r, "audio_file": "extra"} if i == 0 else r for i, r in enumerate(self.reference)],
                           [{**r, "speaker_id": "invalid"} if i == 0 else r for i, r in enumerate(self.reference)]]
        for predictions in bad_submissions:
            with self.subTest(predictions=predictions[:1]):
                with self.assertRaises(ValueError):
                    score_predictions(self.reference, predictions, self.labels)
        with self.assertRaises(ValueError):
            score_predictions(self.reference + [self.reference[0]], self.reference, self.labels)
        with self.assertRaises(ValueError):
            score_predictions([], [], self.labels)

    def test_pooled_f1_differs_from_mean_fold_f1(self):
        actual = [{"audio_file": f"x-{i}", "speaker_id": self.labels[1]} for i in range(10)]
        predicted = [{**r, "speaker_id": self.labels[1] if i == 0 else "unknown"} for i, r in enumerate(actual)]
        pooled = score_predictions(actual, predicted, self.labels)["macro_f1"]
        fold_mean = sum(score_predictions(a, p, self.labels)["macro_f1"] for a, p in
                        ((actual[:1], predicted[:1]), (actual[1:], predicted[1:]))) / 2
        self.assertAlmostEqual(pooled, (2 / 11) / 447)
        self.assertAlmostEqual(fold_mean, 0.5 / 447)
        self.assertNotAlmostEqual(pooled, fold_mean)

    def test_probability_contract_and_first_index_tie(self):
        values = [0.0] * 447
        values[0] = values[2] = 0.5
        self.assertEqual(predictions_from_probabilities(["x"], [values], self.labels)[0]["speaker_id"], "unknown")
        malformed = [values[:-1], [math.nan] + values[1:], [math.inf] + values[1:],
                     [-0.1] + values[1:], [0.0] * 447, [1.1] + [0.0] * 446]
        for row in malformed:
            with self.assertRaises(ValueError):
                predictions_from_probabilities(["x"], [row], self.labels)
        with self.assertRaises(ValueError):
            predictions_from_probabilities(["x", "x"], [values, values], self.labels)


if __name__ == "__main__":
    unittest.main()
