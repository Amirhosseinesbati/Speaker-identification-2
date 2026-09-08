"""Synthetic selected-release tests; no real caches, checkpoints, or source runs."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'pyproject.toml').is_file())
sys.path.insert(0, str(ROOT / 'src'))
from speaker_id.training import heldout_final_references as helper
from speaker_id.packaging import selected, selected_sources
from speaker_id.inference.selected_policy import ADAPTED_PROTOCOL, FROZEN_PROTOCOL, F004_SOURCE


def config_fixture():
    source = {'run': 'artifacts/training/S009_fixture', 'parent_run_id': '0' * 32, 'git_commit': 'a' * 40,
        'children': {code: str(i) * 32 for i, code in enumerate(('S009a', 'S009b', 'S009c', 'S009d'), 1)},
        'export_manifest_paths': ['artifacts/exports/fixture.json'], 'export_manifest_sha256': 'b' * 64}
    return {**deepcopy(selected_sources.FIXED), 'selection': {'recipe_id': 'S009d', 'family': 'adapted_advanced',
        'source': source, 'report_sha256': 'c' * 64,
        'verification': {'path': 'artifacts/audit.json', 'sha256': 'd' * 64}}}


def role_fixture():
    rows = [('aq', 'a', 'aq', 'query'), ('aq_copy', 'a', 'aq', 'query'), ('af', 'a', 'af', 'fit'),
        ('ao', 'a', 'ao', 'outer'), ('bq', 'b', 'bq', 'query'), ('bf', 'b', 'bf', 'fit'),
        ('bo', 'b', 'bo', 'outer'), ('uq', 'unknown', 'uq', 'query'), ('uf', 'unknown', 'uf', 'fit'),
        ('uo', 'unknown', 'uo', 'outer'), ('zero', 'unknown', 'z', 'outer')]
    manifest, folds, roles = [], [], []
    for name, label, group, role in rows:
        manifest.append({'audio_file': name, 'speaker_id': label})
        folds.append({'audio_file': name, 'group_id': group, 'fold': 0 if role == 'outer' else 1, 'train_eligible': name != 'zero'})
        roles.append({'audio_file': name, 'outer_fold': 0, 'group_id': group, 'encoder_fit_allowed': role == 'fit',
            'enrollment_allowed': role == 'fit', 'calibration_query': role == 'query', 'outer_evaluation_included': role == 'outer'})
    random = np.random.default_rng(123)
    vectors = {}
    for key, dimension in (('adapted', 512), ('public', 512), ('advanced', 192)):
        value = random.normal(size=(len(rows), dimension)).astype(np.float32)
        value /= np.linalg.norm(value, axis=1, keepdims=True)
        value[1] = value[0]
        value[-1] = 0
        vectors[key] = value
    return {'manifest': manifest, 'folds': folds, 'roles': roles, 'labels': ['unknown', 'a', 'b']}, vectors, np.asarray([True] * 10 + [False])


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self.value


class SelectedReleaseTests(unittest.TestCase):
    def test_schema_rejects_missing_identity_and_policy_drift(self):
        config = config_fixture()
        selected_sources.validate_config(config)
        for mutate in (lambda c: c.update(encoder_fold=1), lambda c: c.update(alphas=[0., 1.]),
                       lambda c: c['selection'].update(family='public_advanced'),
                       lambda c: c['selection']['source'].update(parent_run_id=None),
                       lambda c: c['selection']['verification'].update(sha256=None)):
            invalid = deepcopy(config)
            mutate(invalid)
            with self.assertRaises((ValueError, TypeError)):
                selected_sources.validate_config(invalid)

    def test_pruning_preserves_procedure_protocol(self):
        q0 = selected.release_policy('adapted_advanced', 1.0)
        frozen = selected.release_policy('public_advanced', 1.0)
        self.assertEqual(q0['kind'], frozen['kind'])
        self.assertEqual(q0['model_configs'], frozen['model_configs'])
        self.assertEqual(q0['calibration_protocol'], ADAPTED_PROTOCOL)
        self.assertEqual(frozen['calibration_protocol'], FROZEN_PROTOCOL)
        with self.assertRaises(ValueError): selected.release_policy('adapted_only', 1.0)

    def test_q0_alpha1_is_not_refit_on_all_rows(self):
        contract, vectors, valid = role_fixture()
        captured = []
        def select(candidates, truth, *, classes):
            captured.append((candidates, truth.copy()))
            calibration, curve = selected.calibrate_gate(candidates[1.0]['known'], truth, candidates[1.0]['unknown'],
                [0.0], [0.0], 5, classes)
            return {'advanced_weight': 1.0, 'calibration': calibration}, {'1.0': {'selected': calibration, 'curve': curve}}
        with patch.object(selected, 'select_inner_alpha', side_effect=select), patch.object(selected, 'frozen_final_scores', side_effect=AssertionError('Adapted family entered all-row fitter')):
            final = selected.fit_final_scorer(contract, vectors, valid, 'adapted_advanced')
        self.assertEqual(final['scores']['calibration_indices'].tolist(), [0, 1, 4, 7])
        self.assertEqual(final['report']['calibration_query_files'], 4)
        self.assertEqual(final['policy']['kind'], 'advanced_public_192')
        self.assertEqual(final['policy']['calibration_protocol'], ADAPTED_PROTOCOL)
        self.assertEqual(set(captured[0][0]), {0., .25, .5, .75, 1.})
        self.assertTrue(all(set(row) == {'known', 'unknown'} and len(row['known']) == 4 for row in captured[0][0].values()))
        self.assertEqual(final['report']['metric_scope'], 'fitted_calibration_not_oof')
        self.assertIs(final['vectors'], vectors['advanced'])

    def test_endpoints_bypass_concat_and_mixed_dtype_is_fp32(self):
        contract, vectors, valid = role_fixture()
        self.assertIs(selected.family_vectors(vectors, valid, 'adapted_advanced', 0.), vectors['adapted'])
        self.assertIs(selected.family_vectors(vectors, valid, 'adapted_advanced', 1.), vectors['advanced'])
        pair = selected.family_vectors(vectors, valid, 'adapted_advanced', .25)
        self.assertEqual(pair.shape, (11, 704))
        self.assertEqual(pair.dtype, np.float32)
        self.assertFalse(np.any(pair[-1]))

    def test_frozen_alpha0_retains_all_training_queries(self):
        contract, vectors, valid = role_fixture()
        def choose(candidates, truth, *, classes):
            self.assertEqual(len(truth), 10)
            calibration, curve = selected.calibrate_gate(candidates[0.]['known'], truth, candidates[0.]['unknown'], [0.], [0.], 5, classes)
            return {'advanced_weight': 0., 'calibration': calibration}, {'0.0': {'selected': calibration, 'curve': curve}}
        with patch.object(selected, 'select_inner_alpha', side_effect=choose), patch.object(selected, 'heldout_final_scores', side_effect=AssertionError('Frozen family entered Q0 fitter')):
            final = selected.fit_final_scorer(contract, vectors, valid, 'public_advanced')
        self.assertEqual(final['report']['calibration_query_files'], 10)
        self.assertEqual(final['policy']['kind'], 'public_voxceleb_512')
        self.assertEqual(final['policy']['calibration_protocol'], FROZEN_PROTOCOL)

    def test_cached_portable_parity_for_mixed_final_gallery(self):
        contract, vectors, valid = role_fixture()
        values = selected.family_vectors(vectors, valid, 'adapted_advanced', .5)
        scores = helper.heldout_final_scores(contract, values, valid)
        proof = selected.verify_cached_parity({'vectors': values, 'gallery': scores['gallery'],
            'calibration': {'unknown_weight': .25, 'margin_weight': .5, 'threshold': .2, 'temperature': .05}}, valid)
        self.assertTrue(proof['argmax_exact'])
        self.assertLessEqual(proof['maximum_absolute_probability_error'], 1e-6)

    def test_independent_audit_requires_every_finished_run_but_no_target_ceiling(self):
        selection = config_fixture()['selection']
        source = selection['source']
        runs = [{'run_id': rid, 'status': 'FINISHED', 'git_commit': source['git_commit']}
                for rid in [source['parent_run_id'], *source['children'].values()]]
        audit = {'status': 'verified', 'parent_run_id': source['parent_run_id'], 'run_name': Path(source['run']).name,
            'git_commit': source['git_commit'], 'export_manifest_sha256': source['export_manifest_sha256'], 'runs': runs,
            'metrics': {'S009d': {'macro_f1': .97}}}
        report = {'oof': {'macro_f1': .97}}
        self.assertEqual(selected_sources.verify_selected_audit(audit, selection, report)['precursor_oof_macro_f1_447'], .97)
        for change in (lambda a: a['runs'][0].update(status='RUNNING'), lambda a: a.update(runs=a['runs'][:-1]),
                       lambda a: a.update(export_manifest_sha256='e' * 64), lambda a: a['metrics']['S009d'].update(macro_f1=.96)):
            invalid = deepcopy(audit)
            change(invalid)
            with self.assertRaises(ValueError): selected_sources.verify_selected_audit(invalid, selection, report)

    def test_plain_encoder_export_preserves_backbone_head_and_excludes_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, output = Path(folder) / 'last.pt', Path(folder) / 'encoder.pt'
            checkpoint.write_bytes(b'synthetic checkpoint')
            state = {'head.conv.weight': FakeTensor([1., 2.]), 'block3.weight': FakeTensor([3.])}
            saved = {'format_version': 2, 'outer_fold': 0, 'completed_steps': 1100, 'signature': 's',
                     'encoder': state, 'head': {'classifier': 'discard'}, 'optimizer': {'state': 'discard'}}
            captured = {}
            def load(path): return saved if path == checkpoint else captured['state']
            def write(value, handle): captured['state'] = dict(value); handle.write(b'plain tensor fixture')
            actual_sha = selected.file_sha256
            def sha(path): return F004_SOURCE['checkpoint_sha256'] if path == checkpoint else actual_sha(path)
            with patch.object(selected, 'file_sha256', side_effect=sha):
                result = selected.export_encoder_only(checkpoint, output, expected_sha=F004_SOURCE['checkpoint_sha256'],
                    expected_signature='s', loader=load, writer=write, tensor_predicate=lambda item: isinstance(item, FakeTensor))
            self.assertEqual(set(captured['state']), set(state))
            self.assertIn('head.conv.weight', captured['state'])
            self.assertNotIn('optimizer', captured['state'])
            self.assertEqual(result['encoder_state_sha256'], selected.state_dict_sha256(state))

    def test_encoder_export_rejects_changed_tensor_readback(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint, output = Path(folder) / 'last.pt', Path(folder) / 'encoder.pt'
            checkpoint.write_bytes(b'synthetic checkpoint')
            state = {'head.conv.weight': FakeTensor([1.])}
            saved = {'format_version': 2, 'outer_fold': 0, 'completed_steps': 1100, 'signature': 's', 'encoder': state}
            def load(path): return saved if path == checkpoint else {'head.conv.weight': FakeTensor([2.])}
            with patch.object(selected, 'file_sha256', return_value=F004_SOURCE['checkpoint_sha256']):
                with self.assertRaisesRegex(ValueError, 'changed tensors'):
                    selected.export_encoder_only(checkpoint, output, expected_sha=F004_SOURCE['checkpoint_sha256'], expected_signature='s',
                        loader=load, writer=lambda value, handle: handle.write(b'changed'), tensor_predicate=lambda item: isinstance(item, FakeTensor))


if __name__ == '__main__':
    unittest.main()
