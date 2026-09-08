"""Boundary, feature-exclusion and nested-selection checks using synthetic data."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from speaker_id.postprocessing import decision_scoring as ds
from speaker_id.postprocessing.scoring import prepare_fold, select_baseline
from speaker_id.postprocessing.nested_cases import make_nested_cases
from speaker_id.postprocessing.tree_models import MODEL_SPECS
from speaker_id.training.reference_scoring import gate_scores
from speaker_id.training.scoring import macro_f1_indices
from test_nested_decision_cases import fixture


class DecisionScoringTests(unittest.TestCase):
    def test_cascades_preserve_outside_band_and_direction_including_zero(self):
        margin = np.array([-.1, -.025, 0., .025, .1])
        confidence = np.array([1., 1., .5, 0., 0.])
        np.testing.assert_array_equal(ds.apply_cascade(margin, confidence, 'veto', .025, .5),
                                      [-.1, -.025, 0., -.5, .1])
        np.testing.assert_array_equal(ds.apply_cascade(margin, confidence, 'rescue', .025, .5),
                                      [-.1, .5, 0., .025, .1])
        np.testing.assert_array_equal(ds.apply_cascade(margin, confidence, 'two_way', .025, .5),
                                      [-.1, .5, 0., -.5, .1])
        np.testing.assert_array_equal(ds.apply_cascade(margin, confidence, 'full', None, .5),
                                      [.5, .5, 0., -.5, -.5])
        with self.assertRaises(ValueError):
            ds.apply_cascade(margin, confidence, 'full', .025, .5)
        with self.assertRaises(ValueError):
            ds.apply_cascade(margin, confidence * 2, 'veto', .025, .5)

    def test_probability_argmax_matches_margin_at_zero_and_preserves_known_order(self):
        known = np.array([[.8, .4, .8], [.1, .9, .3], [.9, .1, .2], [.5, .6, .2]])
        margin = np.array([0., 1e-10, -1e-10, 1.])
        valid = np.array([True, True, True, False])
        p = ds.decision_probabilities(known, margin, valid)
        np.testing.assert_array_equal(p.argmax(axis=1), [0, 2, 0, 0])
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-15)
        np.testing.assert_array_equal(p[-1], [1., 0., 0., 0.])
        self.assertAlmostEqual(float(p[0, 1] / p[0, 2]), float(np.exp(8)), places=9)
        tiny = .001 - np.nextafter(.001, -np.inf)
        boundary = ds.decision_probabilities(np.array([[.4, .1]] * 3),
                                             np.array([tiny, -tiny, 0.]), np.ones(3, dtype=bool))
        np.testing.assert_array_equal(boundary.argmax(axis=1), [1, 0, 0])

    def test_features_never_receive_truth_and_background_removes_entire_group(self):
        public, advanced, valid, manifest, folds, labels, count = fixture()
        prepared = prepare_fold(public, advanced, valid, manifest, folds, labels, 0)
        baseline = select_baseline(prepared)
        del prepared['inner_truth']
        for scope in ('inner', 'outer'):
            values = ds.decision_features(prepared, baseline, scope, 'cpu')
            self.assertEqual(values['feature_names'], ds.FEATURE_NAMES)
            self.assertEqual(values['features'].shape[1], 28)
            self.assertNotIn('truth', values)
            cal = baseline['calibration']
            expected_margin = gate_scores(values['known_scores'], values['unknown_similarity'],
                                          cal['unknown_weight'], cal['margin_weight']) - cal['threshold']
            np.testing.assert_array_equal(values['features'][:, 0], expected_margin)
            np.testing.assert_array_equal(values['margin'], expected_margin)
            source = prepared['values_by_alpha'][baseline['policy']['advanced_weight']].astype(np.float64)
            source /= np.linalg.norm(source, axis=1, keepdims=True)
            for row, query in enumerate(values['indices']):
                refs = [i for i, target in zip(prepared['references'], prepared['reference_targets'])
                        if target == 0 and prepared['groups'][i] != prepared['groups'][query]]
                background = sorted((float(source[query] @ source[i]) for i in refs), reverse=True)
                # Scalar independent oracle also catches duplicate query-group inclusion.
                np.testing.assert_allclose(values['features'][row, 8:12],
                    [background[0], np.mean(background[:3]), np.mean(background[:10]), np.mean(background[:50])],
                    atol=3e-7, rtol=0)
        self.assertEqual(len(ds.feature_columns('scores_only')), 26)
        self.assertEqual(len(ds.feature_columns('scores_quality')), 28)

    def test_threshold_curve_is_exact_and_ties_favor_fewer_changes_then_higher_threshold(self):
        truth = np.array([0, 1, 2, 0, 1, 2])
        guess = np.array([1, 1, 2, 1, 1, 2])
        margin = np.array([-.02, .02, .02, .02, -.02, .02])
        confidence = np.array([.2, .8, .9, .2, .8, .9])
        folds = np.array([0, 1, 2, 0, 1, 2])
        best, curve = ds.select_curve(truth, guess, margin, confidence, folds, 'full', None, 3)
        base = np.where(margin > 0, guess, 0)
        for row in curve:
            pred = np.where(confidence > row['threshold'], guess, 0)
            self.assertEqual(row['meta_macro_f1_447'], macro_f1_indices(truth, pred, 3))
            self.assertEqual(row['changed_from_meta_baseline'], int((pred != base).sum()))
        self.assertEqual(best['meta_macro_f1_447'], 1.)
        expected = sorted(curve, key=lambda r: (-r['meta_macro_f1_447'], r['changed_from_meta_baseline'], -r['threshold']))[0]
        self.assertEqual(best, expected)
        # An empty uncertainty band cannot change predictions, regardless of the threshold.
        fixed, rows = ds.select_curve(truth, guess, margin * 10, confidence, folds, 'veto', .025, 3)
        self.assertTrue(all(r['changed_from_meta_baseline'] == 0 for r in rows))
        self.assertEqual(fixed['threshold'], max(r['threshold'] for r in rows))

    def test_primary_falls_back_on_small_gain_or_large_single_fold_loss(self):
        rows = [dict(id='weak', family='decision_tree', meta_macro_f1_447=.9005,
                     meta_fold_macro_f1_447=[.901]*3, changed_from_meta_baseline=1),
                dict(id='unstable', family='random_forest', meta_macro_f1_447=.91,
                     meta_fold_macro_f1_447=[.895, .92, .92], changed_from_meta_baseline=20)]
        families, selected = ds.select_policy_summary(deepcopy(rows), .9, [.9]*3)
        self.assertIsNone(selected)
        self.assertEqual(set(families), {'decision_tree', 'random_forest'})
        rows += [dict(id='stable', family='decision_tree', meta_macro_f1_447=.903,
                      meta_fold_macro_f1_447=[.903]*3, changed_from_meta_baseline=3)]
        _, selected = ds.select_policy_summary(rows, .9, [.9]*3)
        self.assertEqual(selected['id'], 'stable')

    def test_real_nested_tree_path_is_outer_label_blind_and_binds_model_payload(self):
        public, advanced, valid, manifest, folds, labels, count = fixture()
        prepared = prepare_fold(public, advanced, valid, manifest, folds, labels, 0)
        baseline = select_baseline(prepared)
        nested = make_nested_cases(public, advanced, valid, manifest, folds, labels, 0,
            lambda p, b, s: ds.decision_features(p, b, s, 'cpu'), prepared=prepared)
        # Real fitting/export on synthetic data, one small preregistered learner.
        with patch('speaker_id.postprocessing.tree_models.MODEL_SPECS', MODEL_SPECS[:1]):
            result = ds.evaluate_decisions(prepared, baseline, nested, device='cpu')
        self.assertEqual(len(result['candidate_summary']), 14)
        self.assertTrue(result['selection']['meta_gallery_and_baseline_refitted_per_split'])
        self.assertFalse(result['selection']['outer_labels_used'])
        np.testing.assert_array_equal(result['meta_predictions']['query_global_indices'], np.arange(count))
        family = result['results']['decision_tree']
        self.assertEqual(family['policy']['model_sha256'], ds.payload_sha256(family['model']))
        expected = np.where(family['scores']['outer_decision_margin'] > 0,
                            family['scores']['outer_known_scores'].argmax(axis=1) + 1, 0)
        np.testing.assert_array_equal(family['probabilities'].argmax(axis=1), expected)


if __name__ == '__main__':
    unittest.main()
