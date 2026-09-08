"""Synthetic protocol/identity tests for the ignored S009 draft."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

STAGED = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('draft_s009', STAGED / 'src/speaker_id/training/adapted_candidate_fusion.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def export(prefix, parent):
    return {'run': f'artifacts/training/{prefix}_20990101T000000Z_01234567', 'parent_run_id': parent * 32,
        'git_commit': 'a' * 40, 'export_manifest_paths': [f'artifacts/exports/{prefix}_synthetic.json'], 'export_manifest_sha256': 'b' * 64}


def suite_fixture():
    verification = {'path': 'artifacts/infrastructure/synthetic_verified.json', 'sha256': 'c' * 64}
    adapted = {**export('F004', '1'), 'config': 'configs/train/campp_finetune_head600.json', 'signature': 'd' * 64,
        'completed_steps': 1100, 'folds': {'0': {'child_run_id': '2' * 32, 'checkpoint_sha256': 'e' * 64},
                                         '1': {'child_run_id': '3' * 32, 'checkpoint_sha256': 'f' * 64}}}
    advanced = {**export('S007', '4'), 'source_signature': '1' * 64, 'children': {'S007a': '5' * 32, 'S007b': '6' * 32}}
    return {'schema_version': 1, 'experiment_code': 'S009', 'run_name': 'S009-f004-advanced192-heldout-paired-reference',
        'readiness_config': 'configs/train/campp_coverage.json', 'output_root': 'artifacts/training',
        'source_adapted': adapted, 'source_advanced': advanced, 'candidate_verification': verification,
        'control_adapted': {**export('S006', '7'), 'recipe_id': 'S006f', 'child_run_id': '8' * 32, 'verification': verification},
        'prerequisite_fusion': {**export('S008', '9'), 'children': {'S008a': 'a' * 32, 'S008b': 'b' * 32, 'S008c': 'c' * 32}, 'verification': verification},
        'recipes': deepcopy(module.RECIPES), 'alphas': list(module.ALPHAS), 'alpha_tie_order': list(module.TIE_ORDER),
        'unknown_weights': list(module.UNKNOWN_WEIGHTS), 'margin_weights': list(module.MARGIN_WEIGHTS),
        'threshold_candidates': 201, 'probability_temperature': .05,
        'selection_policy': module.SELECTION_POLICY, 'execution_policy': module.EXECUTION_POLICY}


class OuterLabelForbidden(dict):
    def __getitem__(self, key):
        if key == 'speaker_id':
            raise AssertionError('Outer labels may not enter score construction or selection')
        return super().__getitem__(key)


def score_fixture():
    targets = ['A', 'B', 'unknown', 'A', 'A', 'B', 'unknown', 'unknown', 'A', 'unknown']
    groups = ['fit_a', 'fit_b', 'fit_u', 'query_a', 'query_a', 'query_b', 'query_u', 'query_u', 'outer_a', 'outer_zero']
    short = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.8, .6, 0], [.8, .6, 0], [.6, .8, 0],
                        [1, 0, 0], [1, 0, 0], [.9, .1, 0], [0, 0, 0]], dtype=np.float32)
    short[:-1] /= np.linalg.norm(short[:-1], axis=1, keepdims=True)
    values, advanced = np.zeros((10, 512), dtype=np.float32), np.zeros((10, 192), dtype=np.float32)
    values[:, :3] = short
    advanced[:, :3] = short[:, [1, 2, 0]]
    valid = np.asarray([True] * 9 + [False])
    manifest = [{'audio_file': f'{i}.wav', 'speaker_id': label} for i, label in enumerate(targets)]
    manifest[8], manifest[9] = OuterLabelForbidden(manifest[8]), OuterLabelForbidden(manifest[9])
    folds = [{'audio_file': f'{i}.wav', 'fold': int(i < 8), 'group_id': group, 'train_eligible': i < 9} for i, group in enumerate(groups)]
    roles = [{'audio_file': f'{i}.wav', 'outer_fold': 0, 'group_id': group, 'encoder_fit_allowed': i < 3,
        'enrollment_allowed': i < 2, 'calibration_query': 3 <= i < 8, 'outer_evaluation_included': i >= 8} for i, group in enumerate(groups)]
    contract = {'config': {'fold_ids': [0, 1]}, 'manifest': manifest, 'folds': folds, 'roles': roles, 'labels': ['unknown', 'A', 'B']}
    endpoints = {'adapted': module.heldout_reference_scores(contract, values, valid, 0),
                 'advanced_matched': module.heldout_reference_scores(contract, advanced, valid, 0)}
    return contract, values, advanced, valid, endpoints


def completed_control():
    return {'exact_prediction_reproduction': True, 'exact_pooled_metrics': True,
        'folds': {str(i): {key: True for key in ('exact_prediction_reproduction', 'exact_metrics_reproduction',
            'exact_calibration_reproduction', 'exact_probabilities', 'exact_full_calibration_curve')} for i in (0, 1)}}


class AdaptedCandidateFusionTests(unittest.TestCase):
    def test_exact_recipe_order_role_protocol_and_no_placeholder_sources(self):
        suite = suite_fixture()
        module.validate_suite(suite)
        mutations = [('recipes', list(reversed(suite['recipes']))), ('alphas', [0., .5, 1.]),
            ('alpha_tie_order', list(reversed(suite['alpha_tie_order']))), ('execution_policy', 'run now')]
        for key, value in mutations:
            changed = deepcopy(suite)
            changed[key] = value
            with self.assertRaises(ValueError):
                module.validate_suite(changed)
        changed = deepcopy(suite)
        changed['source_adapted']['folds']['1'] = deepcopy(changed['source_adapted']['folds']['0'])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            module.validate_suite(changed)
        changed = deepcopy(suite)
        changed['source_advanced']['source_signature'] = None
        with self.assertRaises(ValueError):
            module.validate_suite(changed)

    def test_interior_uses_only_own_fold_and_never_frozen_crossfit_or_outer_labels(self):
        contract, values, advanced, valid, endpoints = score_fixture()
        class OppositeFoldForbidden:
            def __iter__(self):
                raise AssertionError('Opposite-fold source accessed')
        sources = {0: (values, valid), 1: OppositeFoldForbidden()}
        with patch.object(module, 'crossfit_scores', side_effect=AssertionError('Adapted crossfit forbidden')):
            scores = module.matched_scores(contract, sources, advanced, valid, 0, .5, endpoints)
        np.testing.assert_array_equal(scores['calibration_indices'], [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(scores['inner_unknown_similarity'][-2:], [0., 0.])
        self.assertEqual(scores['reference_counts']['removed_query_group_files'].tolist(), [2, 2, 1, 2, 2])
        self.assertEqual(scores['provenance']['adapted_outer_fold'], 0)
        self.assertFalse(scores['provenance']['outer_labels_accessed'])
        self.assertNotIn(8, scores['provenance']['reference_indices'])

    def test_endpoints_preserve_exact_objects_and_do_not_claim_full_advanced_control(self):
        endpoints = {'adapted': object(), 'advanced_matched': object()}
        with patch.object(module, 'weighted_encoder_pair', side_effect=AssertionError('Endpoint changed')):
            for alpha, key in ((0., 'adapted'), (1., 'advanced_matched')):
                self.assertIs(module.matched_scores(None, {0: None, 1: None}, None, None, 0, alpha, endpoints), endpoints[key])
        self.assertEqual(module.RECIPES[1]['control'], 'S007b')
        self.assertEqual(module.RECIPES[1]['protocol'], 'leave_content_group_out')
        self.assertIsNone(module.RECIPES[2]['control'])
        self.assertEqual(module.RECIPES[2]['protocol'], module.PROTOCOL)

    def test_query_fit_and_duplicate_group_leakage_fail_closed(self):
        contract, values, advanced, valid, endpoints = score_fixture()
        contract['roles'][3]['encoder_fit_allowed'] = True
        with self.assertRaises(ValueError):
            module.matched_scores(contract, {0: (values, valid), 1: (values, valid)}, advanced, valid, 0, .25, endpoints)

    def test_full_controls_both_folds_and_pooled_gate_all_new_results(self):
        checks = {'S006f': completed_control(), 'S007b': completed_control()}
        module.require_controls(checks)
        for code in checks:
            for key in ('exact_calibration_reproduction', 'exact_full_calibration_curve', 'exact_probabilities'):
                changed = deepcopy(checks)
                changed[code]['folds']['1'][key] = False
                with self.assertRaises(ValueError):
                    module.require_controls(changed)
        with self.assertRaises(ValueError):
            module.require_controls({'S006f': completed_control()})

    def test_inner_selection_rejects_outer_data_and_has_exact_tie_order(self):
        values = np.asarray([[.9, .1], [.1, .9], [.1, .1]], dtype=np.float32)
        unknown = np.asarray([.1, .1, .9], dtype=np.float32)
        candidates = {alpha: {'known': values, 'unknown': unknown} for alpha in module.ALPHAS}
        selected, _ = module.select_inner_alpha(candidates, np.asarray([1, 2, 0]), classes=3)
        self.assertEqual(selected['advanced_weight'], 0.)
        candidates[.25]['outer'] = values
        with self.assertRaises(ValueError):
            module.select_inner_alpha(candidates, np.asarray([1, 2, 0]), classes=3)

    def test_conditional_dispatch_needs_final_owned_receipt_and_strict_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = suite_fixture()['prerequisite_fusion']
            metric = {'row_count': 4529, 'class_count': 447, 'macro_f1': .94}
            report = {'results': [{'recipe': 'S008c', 'oof': metric}]}
            runs = [{'run_id': source['parent_run_id'], 'status': 'FINISHED', 'git_commit': source['git_commit'], 'name': 'S008-parent'}]
            runs += [{'run_id': rid, 'status': 'FINISHED', 'git_commit': source['git_commit'], 'name': code + '-synthetic'} for code, rid in source['children'].items()]
            audit = {'status': 'verified', 'parent_run_id': source['parent_run_id'], 'run_name': Path(source['run']).name,
                'git_commit': source['git_commit'], 'export_manifest_sha256': source['export_manifest_sha256'],
                'runs': runs, 'metrics': {'S008c': metric}}
            path = root / source['verification']['path']
            path.parent.mkdir(parents=True)
            def check(value):
                path.write_text(json.dumps(value))
                pin = {'path': source['verification']['path'], 'sha256': module.file_sha256(path)}
                return module.verify_completed_audit(root, pin, source, report, 'S008c', 4)
            self.assertEqual(check(audit)['pooled_macro_f1'], .94)
            for mutation in ('pending', 'target', 'wrong_export', 'running_child'):
                altered = deepcopy(audit)
                if mutation == 'pending':
                    altered['status'] = 'local_verified_remote_pending'
                elif mutation == 'target':
                    # Both saved report and independent receipt honestly reach the target.
                    report['results'][0]['oof']['macro_f1'] = .965
                    altered['metrics']['S008c']['macro_f1'] = .965
                elif mutation == 'wrong_export':
                    altered['export_manifest_sha256'] = 'f' * 64
                else:
                    altered['runs'][1]['status'] = 'RUNNING'
                with self.assertRaises(ValueError):
                    check(altered)
                report['results'][0]['oof']['macro_f1'] = .94

    def test_complete_curve_and_probability_bytes_are_control_requirements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current, previous = root / 'current', root / 'old/fold_0'
            current.mkdir()
            previous.mkdir(parents=True)
            calibration = {'selected': {'threshold': .4}, 'curve': [{'threshold': .3}, {'threshold': .4}]}
            for folder in (current, previous):
                (folder / 'calibration.json').write_text(json.dumps(calibration))
                np.savez(folder / 'outer_probabilities.npz', probabilities=np.asarray([[.1, .9]]), audio_files=['a'], labels=['unknown', 'A'])
            module.verify_probability_control(current, previous.parent, 0)
            changed = deepcopy(calibration)
            changed['curve'][0]['threshold'] = .2
            (current / 'calibration.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'complete calibration curve'):
                module.verify_probability_control(current, previous.parent, 0)


if __name__ == '__main__':
    unittest.main()
