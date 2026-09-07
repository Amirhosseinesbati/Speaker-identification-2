import unittest
import numpy as np
from speaker_id.training.reference_scoring import known_scores, calibrate_gate, gate_scores, reference_probabilities
from speaker_id.training.scoring import fit_threshold, score_probabilities


class ReferenceScoringTests(unittest.TestCase):
    def test_control_reproduces_original_decisions_and_threshold(self):
        rng = np.random.default_rng(831)
        scores = rng.uniform(-1, 1, (60, 446)).astype(np.float32)
        truth = np.r_[np.zeros(30, dtype=int), scores[30:].argmax(axis=1) + 1]
        selected, _ = calibrate_gate(scores, truth, np.zeros(60), [0], [0], 21)
        threshold, _ = fit_threshold(scores, truth, 21)
        self.assertEqual(selected['threshold'], threshold)
        valid = np.ones(60, bool)
        np.testing.assert_array_equal(reference_probabilities(scores, np.zeros(60), selected, valid).argmax(1),
                                      score_probabilities(scores, threshold, valid=valid).argmax(1))

    def test_unknown_reference_separates_equal_known_maxima(self):
        scores = np.asarray([[.9, .1], [.1, .9], [.9, .1], [.1, .9]])
        truth = np.asarray([1, 2, 0, 0])
        unknown = np.asarray([.1, .1, .99, .99])
        selected, _ = calibrate_gate(scores, truth, unknown, [0, 1], [0], 11, classes=3)
        self.assertEqual(selected['unknown_weight'], 1)
        probabilities = reference_probabilities(scores, unknown, selected, np.ones(4, bool))
        np.testing.assert_array_equal(probabilities.argmax(1), truth)
        np.testing.assert_allclose(probabilities.sum(1), 1)

    def test_boundary_ties_and_zero_signal_choose_unknown(self):
        scores = np.asarray([[.9, .1], [.9, .1], [.9, .1]])
        calibration = {'threshold': .8, 'unknown_weight': 1., 'margin_weight': 0.}
        result = reference_probabilities(scores, np.asarray([.1, .09, .09]), calibration,
                                         np.asarray([True, True, False]))
        np.testing.assert_array_equal(result.argmax(1), [0, 1, 0])

    def test_max_reference_retains_separate_recordings_and_requires_coverage(self):
        references = np.asarray([[1., 0.], [-1., 0.], [0., 1.]])
        labels = np.asarray([1, 1, 2])
        scores = known_scores(np.asarray([[1., 0.]]), references, labels, 'max_reference', classes=2)
        np.testing.assert_array_equal(scores, [[1., 0.]])
        with self.assertRaises(ValueError):
            known_scores(references, references[:2], labels[:2], 'max_reference', classes=2)

    def test_margin_increases_only_with_known_separation(self):
        scores = np.asarray([[.8, .79], [.8, .1]])
        gate = gate_scores(scores, np.zeros(2), 0, .5)
        self.assertGreater(gate[1], gate[0])


if __name__ == '__main__':
    unittest.main()
