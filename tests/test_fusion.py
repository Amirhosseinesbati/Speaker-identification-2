from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training.fusion import (CANDIDATES, UNKNOWN_WEIGHTS, MARGIN_WEIGHTS, dual_view_scores,
    independent_score_pair, select_inner_policy, weighted_embedding_pair)
from speaker_id.training.fusion_suite import validate_fusion_config, verify_control_run, verify_predictions
from speaker_id.training.reference_scoring import known_scores, reference_probabilities

ROOT = Path(__file__).resolve().parents[1]


class OuterLabelForbidden(dict):
    def __getitem__(self, key):
        if key == 'speaker_id':
            raise AssertionError('Outer label accessed before reporting')
        return super().__getitem__(key)


class FusionTests(unittest.TestCase):
    def test_paired_reference_cosine_differs_from_independent_maxima(self):
        short = np.asarray([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]], dtype=np.float32)
        full = np.asarray([[1., 0.], [0., 1.], [1., 0.], [-1., 0.]], dtype=np.float32)
        pairs = weighted_embedding_pair(short, full, np.ones(4, bool), .5)
        np.testing.assert_allclose(pairs @ pairs.T, .5 * (short @ short.T) + .5 * (full @ full.T), atol=1e-7)
        targets = np.asarray([1, 1, 2])
        paired = known_scores(pairs[:1], pairs[1:], targets, 'max_reference', classes=2)
        independent = .5 * known_scores(short[:1], short[1:], targets, 'max_reference', classes=2) + \
                      .5 * known_scores(full[:1], full[1:], targets, 'max_reference', classes=2)
        self.assertAlmostEqual(paired[0, 0], .5, places=6)
        self.assertAlmostEqual(independent[0, 0], 1., places=6)

    def fixture(self):
        labels = ['A', 'A', 'B', 'B', 'unknown', 'unknown', 'A', 'unknown']
        manifest = [{'audio_file': f'{i}.wav', 'speaker_id': label} for i, label in enumerate(labels)]
        manifest[6] = OuterLabelForbidden(manifest[6])
        manifest[7] = OuterLabelForbidden(manifest[7])
        folds = [{'audio_file': f'{i}.wav', 'group_id': f'g{i}', 'fold': 1 if i < 6 else 0,
                  'train_eligible': i != 7} for i in range(8)]
        full = np.asarray([[1,0,0], [.9,.1,0], [0,1,0], [.1,.9,0], [0,0,1], [0,.1,.9], [1,.1,0], [0,0,0]], dtype=np.float32)
        short = full.copy()
        short[1] = [.8, .2, 0]
        short[3] = [.2, .8, 0]
        valid = np.asarray([True] * 7 + [False])
        return full, short, valid, manifest, folds

    def test_outer_labels_and_outer_features_cannot_change_inner_selection(self):
        full, short, valid, manifest, folds = self.fixture()
        before = dual_view_scores(full, short, valid, manifest, folds, 0, classes=2)
        full[6] = [0, 0, 1]
        short[6] = [0, 1, 0]
        after = dual_view_scores(full, short, valid, manifest, folds, 0, classes=2)
        self.assertEqual(list(before), [row['id'] for row in CANDIDATES])
        for name in before:
            for key in ('inner_known_scores', 'inner_unknown_similarity', 'calibration_indices'):
                np.testing.assert_array_equal(before[name][key], after[name][key])
        make_inner = lambda scores: {name: {'known': item['inner_known_scores'], 'unknown': item['inner_unknown_similarity']}
                                     for name, item in scores.items()}
        selected, _ = select_inner_policy(make_inner(before), np.asarray([1,1,2,2,0,0]), classes=3)
        selected_after, _ = select_inner_policy(make_inner(after), np.asarray([1,1,2,2,0,0]), classes=3)
        self.assertEqual(selected, selected_after)
        chosen = before[selected['id']]
        probabilities = reference_probabilities(chosen['outer_known_scores'], chosen['outer_unknown_similarity'],
                                                 selected['calibration'], chosen['outer_valid'])
        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_array_equal(probabilities[1], [1, 0, 0])

    def test_exact_inner_ties_choose_full_and_outer_arrays_are_rejected(self):
        scores = np.asarray([[.9,.1], [.1,.9], [.5,.1], [.1,.5]])
        candidates = {row['id']: {'known': scores, 'unknown': np.asarray([.1,.1,.9,.9])} for row in CANDIDATES}
        chosen, _ = select_inner_policy(candidates, np.asarray([1,2,0,0]), classes=3)
        self.assertEqual(chosen['id'], 'full')
        candidates['full']['outer_truth'] = np.asarray([1])
        with self.assertRaisesRegex(ValueError, 'INNER scores only'):
            select_inner_policy(candidates, np.asarray([1,2,0,0]), classes=3)

    def test_independent_scores_reject_misaligned_query_groups_or_label_columns(self):
        full, short, valid, manifest, folds = self.fixture()
        scores = dual_view_scores(full, short, valid, manifest, folds, 0, classes=2)
        for key, value in [('known_labels', ['B','A']), ('calibration_indices', np.asarray([3,2,1]))]:
            changed = dict(scores['short'])
            changed[key] = value
            with self.assertRaises(ValueError):
                independent_score_pair(changed, scores['full'], .5)

    def test_candidate_grid_cannot_expand_or_silently_drop_source_controls(self):
        suite = json.loads((ROOT / 'configs/train/campp_dualview_scoring.json').read_text())
        validate_fusion_config(suite)
        for key, value in [('candidates', suite['candidates'][1:]), ('unknown_weights', [0.,2.]),
                           ('threshold_candidates', 501), ('readiness_config', 'configs/train/campp_finetune_fp32.json')]:
            changed = deepcopy(suite)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_fusion_config(changed)

    def test_exact_prediction_control_rejects_duplicates_missing_rows_and_changed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.csv'
            path.write_text('audio_file,speaker_id\na.wav,A\nb.wav,unknown\n')
            rows = [{'audio_file':'a.wav','speaker_id':'A'}, {'audio_file':'b.wav','speaker_id':'unknown'}]
            self.assertTrue(verify_predictions(path, list(reversed(rows)))['exact_prediction_reproduction'])
            for changed in (rows[:1], rows + rows[:1], [rows[0], {'audio_file':'b.wav','speaker_id':'B'}]):
                with self.assertRaises(ValueError):
                    verify_predictions(path, changed)

    def test_completed_control_identity_grid_and_cache_bytes_are_required(self):
        suite = json.loads((ROOT / 'configs/train/campp_dualview_scoring.json').read_text())
        source = suite['sources']['full']
        recipe = {'id':'S002f','method':'max_reference','calibration_protocol':'leave_content_group_out',
                  'unknown_weights': UNKNOWN_WEIGHTS, 'margin_weights': MARGIN_WEIGHTS}
        provenance = {'source_parent_run_id':source['source_parent_run_id'], 'source_signature':'signature',
                      'files':[{'audio_file':'a.wav','cache_sha256':'original'}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / source['control_run']
            (control / 'tracking/artifacts').mkdir(parents=True)
            (control / 'S002f').mkdir()
            def write(path, value):
                path.write_text(json.dumps(value))
            state = {'status':'complete','parent_run_id':source['control_parent_run_id']}
            write(control/'experiment_state.json',state)
            write(control/'experiment_report.json',state)
            original_suite = {key:source[key] for key in ('source_parent_run_id','source_run','baseline_config')}
            original_suite.update(threshold_candidates=201,probability_temperature=.05,recipes=[recipe])
            write(control/'tracking/artifacts/resolved_config.json',{'suite':original_suite})
            write(control/'S002f/experiment_report.json',{'recipe':recipe})
            write(control/'cache_provenance.json',provenance)
            self.assertEqual(verify_control_run(root,source,provenance)['status'],'complete')
            write(control/'experiment_state.json',{**state,'status':'failed'})
            with self.assertRaises(ValueError):
                verify_control_run(root,source,provenance)
            write(control/'experiment_state.json',state)
            changed = deepcopy(provenance)
            changed['files'][0]['cache_sha256'] = 'different'
            with self.assertRaises(ValueError):
                verify_control_run(root,source,changed)
            original_suite['recipes'][0]['margin_weights'] = [0.]
            write(control/'tracking/artifacts/resolved_config.json',{'suite':original_suite})
            with self.assertRaises(ValueError):
                verify_control_run(root,source,provenance)


if __name__ == '__main__':
    unittest.main()
