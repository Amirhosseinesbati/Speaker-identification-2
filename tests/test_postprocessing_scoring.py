"""Synthetic leakage and arithmetic checks for the frozen S011 scorers."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.postprocessing import scoring


class OuterLabelForbidden(dict):
    def __getitem__(self, key):
        if key == 'speaker_id':
            raise AssertionError('An outer label was accessed')
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == 'speaker_id':
            raise AssertionError('An outer label was accessed')
        return super().get(key, default)


def prepared_fixture(forbid_outer_labels=False):
    rows = [('A', 'a0', 1), ('A', 'a0', 1), ('A', 'a1', 1), ('A', 'a2', 1),
            ('B', 'b0', 1), ('unknown', 'u0', 1), ('unknown', 'u0', 1),
            ('unknown', 'u1', 1), ('unknown', 'u2', 1), ('unknown', 'u3', 1),
            ('A', 'outer0', 0), ('B', 'outer1', 0), ('unknown', 'outerzero', 0)]
    rng = np.random.default_rng(1013)
    views = []
    for dimension in (512, 192):
        values = rng.standard_normal((len(rows), dimension)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        values[1] = values[0]
        values[6] = values[5]
        values[-1] = 0
        views.append(values)
    valid = np.ones(len(rows), dtype=bool)
    valid[-1] = False
    manifest = []
    folds = []
    for i, (label, group, fold) in enumerate(rows):
        kind = OuterLabelForbidden if fold == 0 and forbid_outer_labels else dict
        manifest.append(kind(audio_file=f'file-{i}', speaker_id=label,
                             duration_seconds=3. + i, mono_rms_dbfs=-55. + i))
        folds.append({'audio_file': f'file-{i}', 'group_id': group, 'fold': fold,
                      'train_eligible': bool(valid[i]), 'evaluation_included': True})
    return scoring.prepare_fold(*views, valid, manifest, folds, ['unknown', 'A', 'B'], 0)


def finite_top_stats(values, k):
    """Independent scalar oracle: remove forbidden members before sorting."""
    allowed = [float(v) for v in values if np.isfinite(v)]
    if not allowed:
        raise ValueError('No allowed reference')
    selected = np.asarray(sorted(allowed, reverse=True)[:k], dtype=np.float64)
    return selected.mean(), selected.std()


class PostprocessingScoringTests(unittest.TestCase):
    def test_prepare_fold_never_reads_outer_labels_and_keeps_all_outer_rows(self):
        expected = prepared_fixture()
        actual = prepared_fixture(forbid_outer_labels=True)
        np.testing.assert_array_equal(actual['inner_truth'], expected['inner_truth'])
        np.testing.assert_array_equal(actual['reference_targets'], expected['reference_targets'])
        np.testing.assert_array_equal(actual['quality'], expected['quality'])
        for alpha in scoring.ALPHAS:
            for name, values in expected['scores_by_alpha'][alpha].items():
                if isinstance(values, np.ndarray):
                    np.testing.assert_array_equal(actual['scores_by_alpha'][alpha][name], values)
        scores = actual['scores_by_alpha'][.5]
        np.testing.assert_array_equal(scores['outer_indices'], [10, 11, 12])
        np.testing.assert_array_equal(scores['outer_valid'], [True, True, False])
        self.assertFalse(set(actual['references']) & set(scores['outer_indices']))
        self.assertNotIn(4, scores['calibration_indices'])  # Singleton B stays enrolled.
        self.assertIn(4, actual['references'])

    def test_pair_context_removes_every_copy_of_query_and_reference_groups(self):
        prepared = prepared_fixture()
        context = scoring.pair_context(prepared, .5)
        scores = prepared['scores_by_alpha'][.5]
        query_indices = np.r_[scores['calibration_indices'], scores['outer_indices']]
        reference_groups = prepared['groups'][prepared['references']]
        for row, query in enumerate(query_indices):
            expected = reference_groups == prepared['groups'][query]
            np.testing.assert_array_equal(np.isneginf(context['similarities'][row]), expected)
        for row, group in enumerate(reference_groups):
            expected = context['cohort_groups'] == group
            np.testing.assert_array_equal(np.isneginf(context['ref_cohort'][row]), expected)
        duplicate_unknown_row = list(query_indices).index(5)
        self.assertEqual(np.isneginf(context['similarities'][duplicate_unknown_row]).sum(), 2)

    def test_reference_side_cohort_stats_exclude_whole_unknown_query_group(self):
        context = scoring.pair_context(prepared_fixture(), .5)
        # All k regimes exercise duplicate-group padding and finite-only fallback.
        for k in (1, 2, 50):
            qm, qs, rm, rs = scoring.cohort_statistics(context, k)
            for q, query_group in enumerate(context['query_groups']):
                expected_q = finite_top_stats(context['similarities'][q, context['unknown_columns']], k)
                np.testing.assert_allclose([qm[q], qs[q]], expected_q, atol=2e-8, rtol=2e-6)
                for ref in range(context['ref_cohort'].shape[0]):
                    allowed = context['ref_cohort'][ref].copy()
                    allowed[context['cohort_groups'] == query_group] = -np.inf
                    expected_ref = finite_top_stats(allowed, k)
                    np.testing.assert_allclose([rm[q, ref], rs[q, ref]], expected_ref, atol=2e-8, rtol=2e-6)
            self.assertIs(scoring.cohort_statistics(context, k), context['normalization_cache'][k])
        # Prove this fixture would detect the omitted reference-side exclusion.
        naive = np.asarray([finite_top_stats(row, 2)[0] for row in context['ref_cohort']])
        unknown_query = list(context['query_groups']).index('u0')
        corrected = scoring.cohort_statistics(context, 2)[2][unknown_query]
        self.assertGreater(float(np.max(np.abs(naive - corrected))), 1e-3)

    def test_pooling_matches_hand_values_with_singletons_and_excluded_copies(self):
        similarities = np.asarray([[.9, .3, -np.inf, .4, .2, -.1],
                                   [-np.inf, .3, -np.inf, -.6, .5, .1]], dtype=np.float32)
        targets = np.asarray([1, 1, 1, 2, 0, 0])
        for kind in ('top2', 'logmeanexp'):
            for blend in (.25, .5):
                known, unknown = scoring.pool_scores(similarities, targets, 2, kind=kind,
                                                     blend=blend, temperature=.1)
                alternative = .6 if kind == 'top2' else .9 + .1 * np.log((1 + np.exp(-6)) / 2)
                expected = [[(1-blend)*.9 + blend*alternative, .4], [.3, -.6]]
                np.testing.assert_allclose(known, expected, atol=1e-7)
                np.testing.assert_allclose(unknown, [.2, .5], atol=1e-7)
                self.assertTrue(np.isfinite(known).all())
        with self.assertRaises(ValueError):
            scoring.pool_scores(similarities[:, targets != 2], targets[targets != 2], 2)
        with self.assertRaises(ValueError):
            scoring.pool_scores(np.full_like(similarities, -np.inf), targets, 2, kind='top2', blend=.5)

    def test_top_statistics_respects_k_and_rejects_empty_permitted_population(self):
        values = np.asarray([[.1, .9, -.3, -np.inf], [.4, -np.inf, -np.inf, -np.inf]], dtype=np.float32)
        mean, std = scoring._top_statistics(values, 2)
        np.testing.assert_allclose(mean, [.5, .4], atol=1e-7)
        np.testing.assert_allclose(std, [.4, 0.], atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'no permitted reference'):
            scoring._top_statistics(np.asarray([[-np.inf, -np.inf]]), 2)

    def test_asnorm_retains_scores_outside_cosine_range(self):
        prepared = {'classes': 2, 'scores_by_alpha': {.5: {}}}
        similarities = np.asarray([[.9, -.8, .1, .11, .12], [.8, -.9, .12, .13, .14]], dtype=np.float32)
        context = {'similarities': similarities, 'unknown_columns': np.asarray([2, 3, 4]),
                   'ref_cohort': np.tile(np.asarray([.1, .11, .12], dtype=np.float32), (5, 1)),
                   'query_groups': np.asarray(['q0', 'q1']), 'cohort_groups': np.asarray(['u0', 'u1', 'u2']),
                   'reference_targets': np.asarray([1, 2, 0, 0, 0]), 'inner_length': 1,
                   'normalization_cache': {}}
        candidate = {'kind': 'asnorm', 'cohort_topk': 3, 'std_floor': .02}
        result = scoring.policy_scores(prepared, .5, candidate, context)
        qm = similarities[:, 2:].mean(axis=1)
        expected = .5 * ((similarities[:, :2] - qm[:, None])/.02 + (similarities[:, :2]-.11)/.02)
        np.testing.assert_allclose(np.r_[result['inner_known_scores'], result['outer_known_scores']], expected, atol=1e-5)
        self.assertGreater(result['inner_known_scores'][0, 0], 1.)
        self.assertLess(result['inner_known_scores'][0, 1], -1.)

    def test_cpu_cuda_new_scores_agree_on_tiny_synthetic_embeddings(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CUDA PyTorch is unavailable')
        if not torch.cuda.is_available():
            self.skipTest('CUDA device is unavailable')
        prepared = prepared_fixture()
        cpu = scoring.pair_context(prepared, .5, 'cpu')
        cuda = scoring.pair_context(prepared, .5, 'cuda')
        for key in ('similarities', 'ref_cohort'):
            np.testing.assert_array_equal(np.isneginf(cpu[key]), np.isneginf(cuda[key]))
            finite = np.isfinite(cpu[key])
            np.testing.assert_allclose(cpu[key][finite], cuda[key][finite], atol=1e-6, rtol=1e-5)
        policies = [dict(kind='top2', blend=.5), dict(kind='logmeanexp', blend=.5, temperature=.05),
                    dict(kind='unknown_top3', blend=.5), dict(kind='density', cohort_topk=2, beta=1.),
                    dict(kind='asnorm', cohort_topk=2, std_floor=.02), dict(kind='late_fusion')]
        for policy in policies:
            with self.subTest(kind=policy['kind']):
                expected = scoring.policy_scores(prepared, .5, policy, cpu)
                actual = scoring.policy_scores(prepared, .5, policy, cuda)
                for key in ('inner_known_scores', 'outer_known_scores', 'inner_unknown_similarity', 'outer_unknown_similarity'):
                    np.testing.assert_allclose(actual[key], expected[key], atol=2e-5, rtol=2e-5)

    def test_quality_gate_meta_validation_groups_are_never_used_to_fit_its_model(self):
        n = 36
        groups = np.asarray([f'quality-group-{i//2}' for i in range(n)] + ['outer-1', 'outer-2'])
        known = np.column_stack((.6 + np.arange(n)*.001, np.full(n, .2)))
        scores = {'inner_known_scores': known, 'inner_unknown_similarity': np.tile([.1, .7], n//2),
                  'calibration_indices': np.arange(n), 'outer_indices': np.asarray([n, n+1]),
                  'outer_known_scores': np.asarray([[.75, .1], [.1, .75]]),
                  'outer_unknown_similarity': np.asarray([.2, .3]), 'outer_valid': np.asarray([True, False])}
        prepared = {'classes': 2, 'groups': groups, 'inner_truth': np.tile([1, 0], n//2),
                    'quality': np.column_stack((np.linspace(1., 3., n+2), np.linspace(-60., -20., n+2))),
                    'scores_by_alpha': {alpha: deepcopy(scores) for alpha in (0., .5, 1.)}}
        candidate = {'id': 'quality_ridge_0.1', 'family': 'quality_gate', 'kind': 'quality_gate',
                     'penalty': .1, 'group_folds': 3}
        original_fit, original_apply = scoring.fit_logistic, scoring.apply_logistic
        calls = []
        def observed_fit(features, target, truth, penalty):
            model = original_fit(features, target, truth, penalty)
            calls.append({'fit_feature_ids': features[:, 0].copy(), 'model': model})
            return model
        def observed_apply(features, model):
            self.assertIs(calls[-1]['model'], model)
            calls[-1]['apply_feature_ids'] = features[:, 0].copy()
            return original_apply(features, model)
        with patch.object(scoring, 'fit_logistic', side_effect=observed_fit), patch.object(scoring, 'apply_logistic', side_effect=observed_apply):
            result = scoring.quality_gate(prepared, .5, candidate)
        self.assertEqual(len(calls), 4)
        identifiers = {value: i for i, value in enumerate(known[:, 0])}
        seen = []
        for fold, call in enumerate(calls[:3]):
            fit_rows = [identifiers[value] for value in call['fit_feature_ids']]
            validation_rows = [identifiers[value] for value in call['apply_feature_ids']]
            self.assertFalse(set(groups[fit_rows]) & set(groups[validation_rows]))
            expected = [i for i in range(n) if int(hashlib.sha256(('S011-quality-v1/' + groups[i]).encode()).hexdigest()[:8], 16) % 3 == fold]
            self.assertEqual(validation_rows, expected)
            self.assertEqual(result['policy']['meta_cv'][fold]['validation_rows'], len(expected))
            self.assertTrue(result['policy']['meta_cv'][fold]['group_disjoint'])
            seen.extend(validation_rows)
        self.assertEqual(sorted(seen), list(range(n)))
        self.assertEqual(len(calls[-1]['fit_feature_ids']), n)
        np.testing.assert_allclose(result['probabilities'].sum(axis=1), 1.)
        np.testing.assert_array_equal(result['probabilities'][1], [1., 0., 0.])
        self.assertTrue(all(row['model']['converged'] for row in result['policy']['meta_cv']))


if __name__ == '__main__':
    unittest.main()
