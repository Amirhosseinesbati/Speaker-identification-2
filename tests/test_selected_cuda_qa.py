"""Synthetic CUDA-QA/report fixtures; no model, CUDA, source-cache or remote execution."""
from copy import deepcopy
from contextlib import redirect_stdout
import ast
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

STAGED = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, STAGED / ('scripts/checks/' + name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cuda = load('check_selected_release_cuda')
verify = load('verify_selected_qa_reports')


def fixture():
    names = [str(i) + '.wav' for i in range(7)]
    valid = np.asarray([False] + [True] * 6)
    vectors = np.zeros((7, 192), np.float32)
    for i in range(1, 7):
        vectors[i, i] = 1
    probabilities = np.zeros((7, 447), np.float64)
    probabilities[np.arange(7), np.arange(7)] = 1
    report = {'release_id': 'fixture', 'build_parent_run_id': 'a' * 32, 'selection': {'recipe_id': 'S007b'},
        'policy': {'kind': 'advanced_public_192', 'embedding_dim': 192}, 'archive': {'archive_sha256': 'b' * 64},
        'source_model_bindings': {'advanced': 'fixture'}, 'build_report_sha256': 'c' * 64,
        'resolved_build_config_sha256': 'd' * 64, 'source_provenance_sha256': 'e' * 64,
        'cases': [{'audio_file': name, 'input_sha256': str(i) * 64} for i, name in enumerate(names)],
        'cross_device_tolerance': dict(cuda.TOLERANCE)}
    source = {'audio_file': np.asarray(names), 'embedding': vectors, 'valid': valid}
    actual = {**deepcopy(source), 'probabilities': probabilities}
    predictions = [{'audio_file': name, 'speaker_id': 'unknown' if i == 0 else 'speaker' + str(i)} for i, name in enumerate(names)]
    predictions.append({'audio_file': '__qa_corrupt__.wav', 'speaker_id': 'unknown'})
    left = {'report': report, 'source_cache_vectors': source, 'vectors': actual, 'predictions': predictions, 'report_sha256': 'f' * 64}
    return left, deepcopy(left)


class CudaQATests(unittest.TestCase):
    def test_default_cannot_execute_cuda_or_read_sources(self):
        with patch.object(cuda, 'execute', side_effect=AssertionError('no actual execution')), redirect_stdout(io.StringIO()):
            result = cuda.main(['--build-dir', 'missing', '--report', 'missing.json'])
        self.assertEqual(result['forward_calls'], 0)
        self.assertFalse(result['source_caches_read'])

    def test_authorized_instance_guard_precedes_torch_or_source_load(self):
        with patch.dict(cuda.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'authorized instance'):
                cuda.execute(Path('missing'), Path('missing.json'))
        ast.parse(cuda.WORKER)
        self.assertIn("assert os.environ.get('VAST_INSTANCE_ID') == '50079023'", cuda.WORKER)
        self.assertIn("'3090' in torch.cuda.get_device_name(0)", cuda.WORKER)
        self.assertIn("runtime._load_models(model_config, root, 'cuda')", cuda.WORKER)
        self.assertNotIn("'device': 'cpu'", cuda.WORKER)
        self.assertEqual(cuda.TOLERANCE['maximum_absolute_embedding_difference'], .002)
        self.assertEqual(cuda.TOLERANCE['minimum_embedding_cosine'], .9999)
        env = cuda.isolated_environment({'DAGSHUB_USER_TOKEN': 'secret'}, Path('/fresh'))
        self.assertNotIn('DAGSHUB_USER_TOKEN', env)
        self.assertEqual(env['VAST_INSTANCE_ID'], '50079023')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')

    def test_exact_cpu_cuda_cache_evidence_passes(self):
        left, right = fixture()
        result = verify.compare(left, right)
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['fresh_cpu_cuda_and_source_cache_decisions_exact'])
        self.assertEqual(len(result['comparisons']), 3)

    def test_different_selection_source_audio_or_archive_fails(self):
        for field in ('selection', 'archive', 'source_model_bindings', 'cases'):
            left, right = fixture(); right['report'][field] = None
            with self.assertRaises(ValueError): verify.compare(left, right)

    def test_source_cache_or_zero_mutation_fails(self):
        left, right = fixture(); right['source_cache_vectors']['embedding'][1, 2] = .01
        with self.assertRaises(ValueError): verify.compare(left, right)
        left, right = fixture(); right['vectors']['embedding'][0, 0] = 1e-9
        with self.assertRaises(ValueError): verify.compare(left, right)

    def test_changed_decision_and_relaxed_numeric_parity_fail(self):
        left, right = fixture()
        right['vectors']['probabilities'][1] = 0; right['vectors']['probabilities'][1, 2] = 1
        with self.assertRaises(ValueError): verify.compare(left, right)
        left, right = fixture()
        right['vectors']['embedding'][1, 1] = np.float32(np.sqrt(1 - .003 ** 2))
        right['vectors']['embedding'][1, 2] = .003
        with self.assertRaises(ValueError): verify.compare(left, right)

    def test_invalid_probability_contract_or_missing_corrupt_row_fails(self):
        left, right = fixture(); right['vectors']['probabilities'] = np.ones((7, 446))
        with self.assertRaises(ValueError): verify.compare(left, right)
        left, right = fixture(); right['predictions'].pop()
        with self.assertRaises(ValueError): verify.compare(left, right)


if __name__ == '__main__':
    unittest.main(verbosity=2)
