"""S010 contract, inner selection and immutable cache tests using synthetic arrays."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.audio.gain import GAIN_POLICY
from speaker_id.training import gain_suite as gain


def fixture_suite():
    return {'schema_version': 1, 'experiment_code': 'S010',
        'run_name': 'S010-campp-fixed-rms-boost-controlled',
        'readiness_config': 'configs/train/campp_coverage.json', 'output_root': 'artifacts/training',
        'source_release_config': gain.SOURCE_CONFIG, 'source_release_config_sha256': gain.SOURCE_CONFIG_SHA,
        'gain_policy': deepcopy(GAIN_POLICY), 'alphas': list(gain.ALPHAS), 'alpha_tie_order': list(gain.TIE_ORDER),
        'unknown_weights': [0.0, .25, .5, .75, 1.0], 'margin_weights': [0.0, .5],
        'threshold_candidates': 201, 'probability_temperature': .05, 'selection_policy': gain.SELECTION,
        'decision_rule': deepcopy(gain.DECISION),
        'recipes': ['S010a_identity_control', 'S010b_gain_only', 'S010c_inner_frontend_choice']}


def score_fixture(known=None):
    if known is None:
        known = np.asarray([[.9, .1], [.1, .9], [.2, .2]], dtype=np.float32)
    unknown = np.asarray([.1, .1, .9], dtype=np.float32)
    return {alpha: {'known': known.copy(), 'unknown': unknown.copy()} for alpha in gain.ALPHAS}


def cache_fixture(root):
    root.mkdir(exist_ok=True)
    identity = {'embedding_dims': {'public': 512, 'advanced': 192},
        'model_sources': {name: {'weights_sha256': digest * 64} for name, digest in (('public', 'b'), ('advanced', 'c'))}}
    identity['signature'] = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    manifest, files = [], []
    for name, valid, digest in [('valid.wav', True, 'd'), ('zero.mp3', False, 'e')]:
        source = {'audio_file': name, 'input_sha256': digest * 64, 'has_nonzero_signal': valid}
        manifest.append(source)
        vectors = {key: np.zeros(dim, dtype=np.float32) for key, dim in identity['embedding_dims'].items()}
        if valid:
            vectors['public'][2] = vectors['advanced'][4] = 1
        path = root / (Path(name).stem + '.npz')
        np.savez_compressed(path, **vectors, signature=identity['signature'], audio_file=name,
            audio_sha256=source['input_sha256'], valid=valid)
        files.append({'audio_file': name, 'audio_sha256': source['input_sha256'], 'cache_file': path.name,
            'bytes': path.stat().st_size, 'cache_sha256': gain.file_sha256(path), 'valid': valid})
    weights = {key: value['weights_sha256'] for key, value in identity['model_sources'].items()}
    states = {'public': 'f' * 64, 'advanced': '0' * 64}
    receipt = {'identity': identity, 'file_count': 2, 'files': files,
        'encoder_state_sha256_before': states.copy(), 'encoder_state_sha256_after': states.copy(),
        'weight_file_sha256_before': weights.copy(), 'weight_file_sha256_after': weights.copy()}
    return identity, manifest, receipt


def rewrite_cache(root, receipt, row_index, **changes):
    row = receipt['files'][row_index]
    path = root / row['cache_file']
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values.update(changes)
    np.savez_compressed(path, **values)
    row.update(bytes=path.stat().st_size, cache_sha256=gain.file_sha256(path))


class GainSuiteTests(unittest.TestCase):
    def test_fixed_contract_rejects_policy_source_grid_or_outer_selection_changes(self):
        original = fixture_suite(); gain.validate_gain_suite(original)
        for key, value in [('outer_tuning', True), ('selection_policy', 'choose outer maximum'),
                ('source_release_config_sha256', '0' * 64), ('threshold_candidates', 101),
                ('recipes', original['recipes'][1:]), ('alpha_tie_order', list(reversed(gain.TIE_ORDER))),
                ('decision_rule', {'minimum_pooled_improvement': 0, 'maximum_fold_decline': 1})]:
            with self.subTest(key=key):
                changed = deepcopy(original); changed[key] = value
                with self.assertRaises(ValueError): gain.validate_gain_suite(changed)

    def test_bool_cannot_impersonate_numeric_configuration(self):
        for field, index, value in [('schema_version', None, True), ('threshold_candidates', None, True),
                ('alphas', 0, False), ('alphas', -1, True), ('alpha_tie_order', 0, False),
                ('unknown_weights', 0, False), ('margin_weights', 0, False)]:
            with self.subTest(field=field, index=index):
                config = fixture_suite()
                if index is None: config[field] = value
                else: config[field][index] = value
                with self.assertRaises(ValueError): gain.validate_gain_suite(config)

    def test_frontend_and_alpha_exact_ties_prefer_identity_public_endpoint(self):
        candidates = {'identity': score_fixture(), 'gain': score_fixture()}
        before = {name: {alpha: {key: a.tobytes() for key, a in values.items()}
            for alpha, values in scores.items()} for name, scores in candidates.items()}
        selected, fits = gain.select_inner_frontend(candidates, np.asarray([1, 2, 0]), classes=3)
        self.assertEqual(selected['frontend'], 'identity')
        self.assertEqual(selected['advanced_weight'], 0.0)
        self.assertEqual(selected['calibration']['inner_macro_f1_447'], 1.0)
        self.assertEqual(fits['identity'], fits['gain'])
        after = {name: {alpha: {key: a.tobytes() for key, a in values.items()}
            for alpha, values in scores.items()} for name, scores in candidates.items()}
        self.assertEqual(before, after)

    def test_inner_quality_can_select_gain_without_any_outer_arrays(self):
        poor = np.asarray([[.1, .9], [.9, .1], [.2, .2]], dtype=np.float32)
        selected, fits = gain.select_inner_frontend({'identity': score_fixture(poor), 'gain': score_fixture()},
            np.asarray([1, 2, 0]), classes=3)
        self.assertEqual(selected['frontend'], 'gain')
        self.assertGreater(fits['gain']['selected']['calibration']['inner_macro_f1_447'],
                           fits['identity']['selected']['calibration']['inner_macro_f1_447'])

    def test_selector_rejects_extra_frontend_or_outer_fields(self):
        for case in ('frontend', 'known', 'labels'):
            candidates = {'identity': score_fixture(), 'gain': score_fixture()}
            if case == 'frontend': candidates['outer_winner'] = score_fixture()
            else: candidates['gain'][.25]['outer_' + case] = np.zeros((3, 2))
            with self.subTest(case=case), self.assertRaises(ValueError):
                gain.select_inner_frontend(candidates, np.asarray([1, 2, 0]), classes=3)

    def test_inner_array_adapter_never_reads_outer_fields(self):
        class Guarded(dict):
            def __getitem__(self, key):
                if key.startswith('outer'): raise AssertionError('Outer arrays accessed')
                return super().__getitem__(key)
        known, unknown = np.ones((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
        scored = {alpha: Guarded(inner_known_scores=known, inner_unknown_similarity=unknown,
                                 outer_known_scores='forbidden', outer_unknown_similarity='forbidden') for alpha in gain.ALPHAS}
        inner = gain.inner_arrays(scored)
        self.assertTrue(all(set(value) == {'known', 'unknown'} for value in inner.values()))
        self.assertIs(inner[.5]['known'], known)

    def test_candidate_signature_binds_model_data_frontend_and_actual_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'scripts').mkdir(); launcher = root / 'scripts/score_gain.py'; launcher.write_text('version1')
            suite = fixture_suite()
            contract = {'config': {'inference': {'seconds': 180., 'maximum_windows': 1}},
                'input_hashes': {'manifest': 'a' * 64}, 'labels': ['unknown', 'A'], 'code_hashes': {'src/fake.py': 'b' * 64}}
            sources = {'assets': {name: {'source_record': {'weights_sha256': digest * 64}}
                for name, digest in [('public', 'c'), ('advanced', 'd')]}}
            baseline = gain.gain_identity(root, suite, contract, sources)
            expected = hashlib.sha256(json.dumps({k: v for k, v in baseline.items() if k != 'signature'},
                sort_keys=True, allow_nan=False).encode()).hexdigest()
            self.assertEqual(baseline['signature'], expected)
            variants = []
            changed = deepcopy(sources); changed['assets']['advanced']['source_record']['weights_sha256'] = 'e' * 64
            variants.append(gain.gain_identity(root, suite, contract, changed))
            changed = deepcopy(contract); changed['input_hashes']['manifest'] = 'f' * 64
            variants.append(gain.gain_identity(root, suite, changed, sources))
            changed = deepcopy(contract); changed['code_hashes']['src/fake.py'] = '0' * 64
            variants.append(gain.gain_identity(root, suite, changed, sources))
            launcher.write_text('version2'); variants.append(gain.gain_identity(root, suite, contract, sources))
            self.assertTrue(all(v['signature'] != baseline['signature'] for v in variants))

    def test_control_failure_blocks_identity_creation_and_any_extraction(self):
        flags = ('exact_prediction_reproduction', 'exact_pooled_metrics', 'exact_inner_alpha_curves',
                 'exact_probability_and_support_arrays')
        for key in flags:
            control = {name: True for name in flags}
            control[key] = False
            with self.subTest(key=key), patch.object(gain, 'gain_identity', side_effect=AssertionError('Extraction started')):
                with self.assertRaises(ValueError):
                    gain.extract_gain_cache(None, None, None, None, None, None, control)

    @staticmethod
    def control_arrays_fixture(root):
        actual, historical = root / 'actual', root / 'historical'
        actual.mkdir(); historical.mkdir()
        probability = {'probabilities': np.asarray([[.1, .8, .1], [1., 0., 0.]], dtype=np.float64),
                       'audio_files': np.asarray(['sample.wav', 'zero.wav']), 'labels': np.asarray(['unknown', 'A', 'B'])}
        support = {'calibration_indices': np.asarray([2, 4], dtype=np.int64),
                   'known_labels': np.asarray(['A', 'B']), 'known_files_per_class': np.asarray([2, 3], dtype=np.int64)}
        for directory in (actual, historical):
            np.savez_compressed(directory / 'outer_probabilities.npz', **probability)
            np.savez_compressed(directory / 'reference_support.npz', **support)
        return actual, historical, probability, support

    def test_exact_probability_and_support_arrays_control_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual, historical, probability, support = self.control_arrays_fixture(Path(tmp))
            # Container compression is irrelevant; complete arrays and dtypes are the control.
            np.savez(actual / 'outer_probabilities.npz', **probability)
            self.assertEqual(gain.verify_saved_control_arrays(actual, historical),
                {'exact_probability_arrays': True, 'exact_reference_support_arrays': True})

    def test_probability_control_rejects_tamper_even_when_argmax_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual, historical, probability, support = self.control_arrays_fixture(Path(tmp))
            old_predictions = probability['probabilities'].argmax(1)
            probability['probabilities'][0, 0] += 1e-8
            probability['probabilities'][0, 2] -= 1e-8
            np.testing.assert_array_equal(probability['probabilities'].argmax(1), old_predictions)
            np.savez_compressed(actual / 'outer_probabilities.npz', **probability)
            with self.assertRaisesRegex(ValueError, 'probability/support'):
                gain.verify_saved_control_arrays(actual, historical)

    def test_support_control_rejects_changed_query_order_or_gallery_counts(self):
        for key in ('calibration_indices', 'known_files_per_class'):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                actual, historical, probability, support = self.control_arrays_fixture(Path(tmp))
                support[key] = support[key][::-1].copy()
                np.savez_compressed(actual / 'reference_support.npz', **support)
                with self.assertRaisesRegex(ValueError, 'probability/support'):
                    gain.verify_saved_control_arrays(actual, historical)

    def test_array_control_rejects_extra_fields_or_changed_dtype(self):
        for change in ('extra', 'dtype'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                actual, historical, probability, support = self.control_arrays_fixture(Path(tmp))
                if change == 'extra': probability['unrecorded'] = np.asarray([1])
                else: probability['probabilities'] = probability['probabilities'].astype(np.float32)
                np.savez_compressed(actual / 'outer_probabilities.npz', **probability)
                with self.assertRaises(ValueError): gain.verify_saved_control_arrays(actual, historical)

    def test_verified_cache_preserves_dimension_order_and_exact_zero_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp); identity, manifest, receipt = cache_fixture(cache)
            arrays, valid = gain.verify_gain_cache(cache, identity, manifest, receipt)
            self.assertEqual(arrays['public'].shape, (2, 512)); self.assertEqual(arrays['advanced'].shape, (2, 192))
            self.assertEqual(valid.tolist(), [True, False]); self.assertEqual(valid.dtype, np.bool_)
            self.assertEqual(int(np.argmax(arrays['public'][0])), 2)
            self.assertTrue(all(a.dtype == np.float32 and not a[1].any() for a in arrays.values()))

    def test_cache_byte_tamper_is_rejected_without_updating_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp); identity, manifest, receipt = cache_fixture(cache)
            path = cache / receipt['files'][0]['cache_file']; path.write_bytes(path.read_bytes() + b'tamper')
            with self.assertRaises(ValueError): gain.verify_gain_cache(cache, identity, manifest, receipt)

    def test_cache_source_signature_and_raw_hash_must_match_even_after_rehash(self):
        for name, value in [('signature', 'f' * 64), ('audio_file', 'other.wav'), ('audio_sha256', 'f' * 64)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                cache = Path(tmp); identity, manifest, receipt = cache_fixture(cache)
                rewrite_cache(cache, receipt, 0, **{name: value})
                with self.assertRaises(ValueError): gain.verify_gain_cache(cache, identity, manifest, receipt)

    def test_cache_requires_exact_npz_schema_and_scalar_boolean_valid(self):
        for changes in ({'unrecorded_extra': np.asarray([1])}, {'valid': np.asarray('yes')},
                        {'valid': np.asarray([True])}, {'valid': np.asarray(1)}):
            with self.subTest(changes=list(changes)), tempfile.TemporaryDirectory() as tmp:
                cache = Path(tmp); identity, manifest, receipt = cache_fixture(cache)
                rewrite_cache(cache, receipt, 0, **changes)
                with self.assertRaises(ValueError): gain.verify_gain_cache(cache, identity, manifest, receipt)

    def test_zero_or_nonfinite_or_unnormalized_embeddings_are_not_silently_fixed(self):
        cases = [(0, np.zeros(512, dtype=np.float32)), (0, np.ones(512, dtype=np.float32)),
                 (0, np.full(512, np.nan, dtype=np.float32)), (0, np.eye(1, 512, dtype=np.float64)[0]),
                 (1, np.eye(1, 512, dtype=np.float32)[0])]
        for index, vector in cases:
            with self.subTest(index=index, dtype=str(vector.dtype)), tempfile.TemporaryDirectory() as tmp:
                cache = Path(tmp); identity, manifest, receipt = cache_fixture(cache)
                rewrite_cache(cache, receipt, index, public=vector)
                with self.assertRaises(ValueError): gain.verify_gain_cache(cache, identity, manifest, receipt)

    def test_cache_rejects_duplicate_missing_or_outside_rows_and_changed_encoder(self):
        for change in ('duplicate', 'missing', 'traversal', 'encoder', 'empty_weights'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                cache = Path(tmp); identity, manifest, receipt = cache_fixture(cache)
                if change == 'duplicate': receipt['files'][1] = receipt['files'][0].copy()
                elif change == 'missing': receipt['files'].pop()
                elif change == 'traversal': receipt['files'][0]['cache_file'] = '../valid.npz'
                elif change == 'encoder': receipt['encoder_state_sha256_after']['advanced'] = '9' * 64
                else:
                    receipt['weight_file_sha256_before'] = {}; receipt['weight_file_sha256_after'] = {}
                with self.assertRaises(ValueError): gain.verify_gain_cache(cache, identity, manifest, receipt)

    def test_cli_default_validates_without_execution(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('_test_score_gain_cli', root / 'scripts/score_gain.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'suite.json'; config.write_text(json.dumps(fixture_suite()))
            stream = io.StringIO()
            with patch.object(sys, 'argv', ['score_gain.py', '--config', 'synthetic']), \
                 patch('speaker_id.training.fusion_suite.project_path', return_value=config), \
                 patch.object(gain, 'load_gain_inputs', return_value=({}, {}, {})), \
                 patch.object(gain, 'execute_gain_suite', side_effect=AssertionError('Unexpected GPU work')), \
                 patch('sys.stdout', stream):
                module.main()
            result = json.loads(stream.getvalue())
            self.assertEqual(result['status'], 'validated_no_experiment_started')
            self.assertFalse(result['encoder_training'])


if __name__ == '__main__':
    unittest.main()
