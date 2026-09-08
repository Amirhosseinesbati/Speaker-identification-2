"""Synthetic offline QA fixtures; no model assets, audio forward or remote calls."""
import ast
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selected_qa_draft', ROOT / 'scripts/checks/check_selected_release_offline.py')
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def archive(path, files, *, extra=(), symlink=None):
    manifest = {'schema_version': 1, 'release_id': 'synthetic', 'files': {
        name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()} for name, value in files.items()}}
    with zipfile.ZipFile(path, 'w') as saved:
        saved.writestr('manifest.json', json.dumps(manifest))
        for name, value in files.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = ((stat.S_IFLNK if name == symlink else stat.S_IFREG) | 0o644) << 16
            saved.writestr(info, value)
        for name, value in extra:
            saved.writestr(name, value)
    return manifest


class OfflineQATests(unittest.TestCase):
    def test_default_validation_cannot_read_sources_or_run_forward(self):
        with patch.object(qa, 'execute', side_effect=AssertionError('execution is forbidden')), redirect_stdout(io.StringIO()):
            result = qa.main(['--build-dir', 'nonexistent/draft', '--report', 'nonexistent/report.json'])
        self.assertEqual(result['status'], 'arguments_validated_only')
        self.assertEqual(result['forward_calls'], 0)
        self.assertFalse(result['source_caches_read'])

    def test_execute_is_an_explicit_dispatch(self):
        with patch.object(qa, 'execute', return_value={'status': 'synthetic'}) as call, redirect_stdout(io.StringIO()):
            qa.main(['--build-dir', 'draft', '--report', 'report.json', '--execute'])
        call.assert_called_once_with(Path('draft'), Path('report.json'), project_root=qa.ROOT)

    def test_environment_removes_credentials_and_python_path(self):
        env = qa.isolated_environment({'PATH': 'system', 'MLFLOW_TRACKING_TOKEN': 'secret', 'VAST_API_KEY': 'secret',
            'DAGSHUB_USER_TOKEN': 'secret', 'GITHUB_TOKEN': 'secret', 'PYTHONPATH': 'checkout', 'HOME': 'private'}, Path('/fresh'))
        self.assertFalse(any('TOKEN' in name or 'KEY' in name or name == 'PYTHONPATH' for name in env))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['TORCH_HOME'], str(Path('/fresh/t')))

    def test_exact_archive_extract_and_existing_destination_rejected(self):
        with tempfile.TemporaryDirectory(prefix='p2t_') as tmp:
            root = Path(tmp); manifest = archive(root / 's.zip', {'submission.py': b'pass\n', 'assets/a.json': b'{}'})
            qa.extract_verified_archive(root / 's.zip', root / 'p', manifest)
            self.assertEqual((root / 'p/submission.py').read_bytes(), b'pass\n')
            with self.assertRaises(ValueError): qa.extract_verified_archive(root / 's.zip', root / 'p', manifest)

    def test_archive_rejects_traversal_links_extra_members_and_wrong_hash(self):
        with tempfile.TemporaryDirectory(prefix='p2t_') as tmp:
            root = Path(tmp)
            for number, files, kwargs in [
                (0, {'../escape.py': b'x'}, {}),
                (1, {'submission.py': b'x'}, {'symlink': 'submission.py'}),
                (2, {'submission.py': b'x'}, {'extra': [('secret.txt', b'x')]}),
                (3, {'Submission.py': b'x', 'submission.py': b'x'}, {}),
            ]:
                path = root / f'{number}.zip'; manifest = archive(path, files, **kwargs)
                with self.assertRaises(ValueError): qa.extract_verified_archive(path, root / str(number), manifest)
                self.assertFalse((root / str(number)).exists())
            manifest = archive(root / '4.zip', {'submission.py': b'x'})
            manifest['files']['submission.py']['sha256'] = '0' * 64
            with self.assertRaises(ValueError): qa.extract_verified_archive(root / '4.zip', root / '4', manifest)

    def test_original_p001_tolerance_preserved_for_all_dimensions(self):
        self.assertEqual(qa.TOLERANCE, {'maximum_absolute_embedding_difference': .002, 'minimum_embedding_cosine': .9999,
                                      'speaker_decisions_must_match': True, 'bit_identical_embeddings_required': False})
        for dimension in (192, 512, 704):
            x = np.zeros((2, dimension), np.float32); x[0, 0] = 1
            valid = np.asarray([True, False])
            result = qa.check_parity(x, x.copy(), valid)
            self.assertEqual(result[0]['cosine'], 1)
            self.assertIsNone(result[1]['cosine'])
            changed = x.copy(); changed[0, 1] = .0021
            with self.assertRaises(ValueError): qa.check_parity(changed, x, valid)
            changed = x.copy(); changed[1, 0] = 1e-9
            with self.assertRaises(ValueError): qa.check_parity(changed, x, valid)

    def test_parity_refuses_nonfinite_shapes_and_non_f32(self):
        x = np.zeros((1, 192), np.float32); valid = np.asarray([False])
        for actual in (np.full_like(x, np.nan), x.astype(np.float64), x[:, :10]):
            with self.assertRaises(ValueError): qa.check_parity(actual, x, valid)

    def test_all_five_kinds_use_exact_captured_endpoints_or_f32_pair(self):
        # Import published pure helpers; synthetic vectors never invoke fit/build entrypoints.
        sys.path.insert(0, str(ROOT / 'src'))
        from speaker_id.packaging.selected import release_policy
        from speaker_id.training.candidate_fusion import weighted_encoder_pair
        values = {}
        rng = np.random.default_rng(77)
        valid = np.asarray([True, False])
        for key, dim in [('adapted', 512), ('public', 512), ('advanced', 192)]:
            array = rng.normal(size=(2, dim)).astype(np.float32)
            array /= np.linalg.norm(array, axis=1, keepdims=True)
            array[1] = 0
            values[key] = array
        seen = set()
        for family, alpha, left in [('adapted_only', 0.0, 'adapted'), ('public_advanced', 0.0, 'public'),
            ('advanced_only', 1.0, 'advanced'), ('adapted_advanced', .25, 'adapted'), ('public_advanced', .75, 'public')]:
            policy = release_policy(family, alpha)
            seen.add(policy['kind'])
            actual = qa.selected_vectors(policy, {'family': family, 'vectors': values, 'valid': valid})
            if alpha in (0, 1):
                self.assertIs(actual, values[left])
            else:
                expected = weighted_encoder_pair(values[left], values['advanced'], valid, alpha)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual.shape, (2, 704))
            self.assertFalse(actual[1].any())
        self.assertEqual(len(seen), 5)

    def test_csv_requires_exact_coverage_labels_and_columns(self):
        with tempfile.TemporaryDirectory(prefix='p2t_') as tmp:
            path = Path(tmp) / 'p.csv'
            path.write_text('audio_file,speaker_id\na.wav,unknown\nb.wav,id\n', encoding='utf-8')
            self.assertEqual(qa.read_predictions(path, ['unknown', 'id'], ['a.wav', 'b.wav']), {'a.wav': 'unknown', 'b.wav': 'id'})
            for text in ('audio_file,speaker_id\na.wav,unknown\na.wav,id\n',
                         'audio_file,speaker_id\na.wav,invalid\nb.wav,id\n', 'speaker_id,audio_file\nunknown,a.wav\n'):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError): qa.read_predictions(path, ['unknown', 'id'], ['a.wav', 'b.wav'])

    def test_worker_blocks_network_and_backward_before_runtime_and_uses_selected_only(self):
        ast.parse(qa.WORKER)
        self.assertLess(qa.WORKER.index('socket.socket.connect = blocked'), qa.WORKER.index('import torch'))
        self.assertLess(qa.WORKER.index('torch.Tensor.backward = no_backward'), qa.WORKER.index('from speaker_id.inference'))
        self.assertIn('from speaker_id.inference import selected_runtime as runtime', qa.WORKER)
        self.assertNotIn('from speaker_id.training', qa.WORKER)
        self.assertNotIn('from speaker_id.tracking', qa.WORKER)
        self.assertIn("assert out.read_bytes() == previous_output", qa.WORKER)
        self.assertIn("'--data-dir'", qa.WORKER)
        self.assertIn("'--predictions-file-path'", qa.WORKER)

    def test_metadata_selector_matches_the_existing_seven_unique_examples(self):
        # Metadata only: no audio is read or extracted by this fixture.
        with (qa.ROOT / 'data/processed/eda_v1/audio_manifest.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        examples = qa.select_examples(rows)
        self.assertEqual(len(examples), 7)
        self.assertEqual(len({row['row']['audio_file'] for row in examples}), 7)
        self.assertEqual(sum(item['row']['has_nonzero_signal'] == 'False' or item['row']['has_nonzero_signal'] == 'false' for item in examples), 1)
        cases = {case for item in examples for case in item['cases']}
        self.assertTrue({'zero', 'shortest_nonzero', 'longest', 'lowest_rms_nonzero', 'known', 'unknown',
                         'real_mp3', 'container_extension_mismatch'} <= cases)


if __name__ == '__main__':
    unittest.main(verbosity=2)
