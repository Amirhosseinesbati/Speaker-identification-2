"""Synthetic portable-runtime checks; no real model weights or forward calls."""
import ast
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
from speaker_id.inference import selected_policy as policy, selected_runtime as runtime


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf-8')


def selection_policy(kind, alpha=None, protocol=None):
    dimension, components, default = policy.POLICIES[kind]
    return {'schema_version': 1, 'kind': kind, 'embedding_dim': dimension,
        'advanced_weight': alpha if alpha is not None else 1.0 if kind == 'advanced_public_192' else .5 if dimension == 704 else 0.0,
        'inference': dict(policy.INFERENCE), 'model_configs': {key: policy.MODEL_PATHS[key] for key in components},
        'calibration_protocol': protocol or default}


def adapted_config(payload):
    from speaker_id.models.campp import EXPECTED_FRONTEND
    return {'schema_version': 1, 'encoder_kind': 'adapted_f004_fold0', 'architecture': 'CAMPPlus',
        'embedding_dim': 512, 'sample_rate': 16000, 'fbank_bins': 80, 'frontend': deepcopy(EXPECTED_FRONTEND),
        'architecture_kwargs': dict(policy.ARCHITECTURE_512), 'inference': dict(policy.INFERENCE),
        'source': dict(policy.F004_SOURCE), 'weights_path': 'assets/f004_fold0_encoder.pt',
        'weights_bytes': len(payload), 'weights_sha256': hashlib.sha256(payload).hexdigest(), 'encoder_state_sha256': 'a' * 64}


def provenance(root, chosen, models, family):
    mapping = {'public_only': ('S002', 'S002f'), 'adapted_only': ('S006', 'S006f'), 'advanced_only': ('S007', 'S007b'),
               'public_advanced': ('S008', 'S008c'), 'adapted_advanced': ('S009', 'S009d')}
    experiment, recipe = mapping[family]
    sources = {}
    for key, config in models.items():
        if key == 'adapted':
            sources[key] = {**policy.F004_SOURCE, 'weights_sha256': config['weights_sha256'], 'encoder_state_sha256': config['encoder_state_sha256']}
        else:
            sources[key] = {'encoder_kind': 'public_voxceleb_512' if key == 'public' else 'advanced_public_192',
                'weights_sha256': config['weights_sha256'], 'encoder_updates': 0, 'source_parent_run_id': '1' * 32,
                'source_signature': '2' * 64, 'source_git_commit': '3' * 40}
    q0 = chosen['calibration_protocol'] == policy.ADAPTED_PROTOCOL
    value = {'schema_version': 1, 'release_id': 'synthetic', 'policy': chosen,
        'model_config_sha256': {key: runtime._sha(root / chosen['model_configs'][key]) for key in models}, 'model_sources': sources,
        'selection': {'family': family, 'experiment_code': experiment, 'recipe_id': recipe, 'parent_run_id': '4' * 32, 'report_sha256': '5' * 64},
        'calibration': {'protocol': chosen['calibration_protocol'], 'source_files': 4529, 'known_reference_files': 2217,
            'unknown_reference_files': 2223, 'zero_signal_files': 89, 'calibration_query_files': 999 if q0 else 4440,
            'roles_sha256': policy.ROLES_SHA256, 'encoder_fit_queries_used': 0, 'whole_query_group_excluded': True,
            'metric_scope': 'fitted_calibration_not_oof'}}
    if q0:
        value['procedure_history'] = {'adaptation_source': policy.F004_SOURCE, 'roles_sha256': policy.ROLES_SHA256,
                                      'final_encoder_excludes_adapted': 'adapted' not in models}
    return value


def manifest(root, updates=500):
    value = {'schema_version': 1, 'release_id': 'synthetic', 'encoder_updates': updates,
        'files': {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': runtime._sha(path)}
                  for path in root.rglob('*') if path.is_file() and path.name != 'manifest.json'},
        'provenance': {'policy_sha256': runtime._sha(root / 'assets/policy.json'), 'provenance_sha256': runtime._sha(root / 'assets/provenance.json')}}
    write(root / 'manifest.json', value)
    return value


def package(root):
    chosen = selection_policy('adapted_f004_fold0_512')
    for name in {'submission.py'} | runtime._source_files('speaker_id', {'adapted'}):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('# Synthetic payload; never executed.\n')
    payload = b'synthetic encoder fixture bytes, never torch-loaded in inference tests'
    config = adapted_config(payload)
    weights = root / config['weights_path']
    weights.parent.mkdir(exist_ok=True)
    weights.write_bytes(payload)
    write(root / 'assets/adapted_model_config.json', config)
    write(root / 'assets/policy.json', chosen)
    write(root / 'assets/provenance.json', provenance(root, chosen, {'adapted': config}, 'adapted_only'))
    labels = ['unknown'] + [str(uuid.UUID(int=index)) for index in range(1, 447)]
    write(root / 'assets/labels.json', {'labels': labels, 'unknown_index': 0})
    write(root / 'assets/calibration.json', {'unknown_weight': 0.0, 'margin_weight': 0.0, 'threshold': .5, 'temperature': .05, 'inference': dict(policy.INFERENCE)})
    known, unknown = np.zeros((2217, 512), np.float32), np.zeros((2223, 512), np.float32)
    known[:, 0], unknown[:, -1] = 1, 1
    np.savez_compressed(root / 'assets/gallery.npz', known_embeddings=known, unknown_embeddings=unknown,
                        known_targets=np.arange(2217, dtype=np.int64) % 446 + 1)
    manifest(root)
    return labels


class PolicyTests(unittest.TestCase):
    def test_five_explicit_kinds_reject_silent_model_or_dimension_substitution(self):
        for kind in policy.POLICIES:
            chosen = selection_policy(kind)
            self.assertEqual(policy.validate_policy(chosen), chosen)
            changed = deepcopy(chosen)
            changed['embedding_dim'] += 1
            with self.assertRaises(ValueError):
                policy.validate_policy(changed)
        chosen = selection_policy('paired_f004_advanced_704')
        chosen['model_configs'] = {'public': policy.MODEL_PATHS['public'], 'advanced': policy.MODEL_PATHS['advanced']}
        with self.assertRaises(ValueError):
            policy.validate_policy(chosen)

    def test_fp32_pair_exactly_matches_historical_s008_and_fixed_rounding(self):
        from speaker_id.training.candidate_fusion import weighted_encoder_pair
        rng = np.random.default_rng(77)
        left, right = rng.normal(size=(5, 512)).astype(np.float32), rng.normal(size=(5, 192)).astype(np.float32)
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        left[-1], right[-1] = 0, 0
        valid = np.asarray([True] * 4 + [False])
        for alpha in (.25, .5, .75):
            expected = weighted_encoder_pair(left, right, valid, alpha)
            observed = np.asarray([policy.paired_embedding(a, b, mask, alpha) for a, b, mask in zip(left, right, valid)])
            np.testing.assert_array_equal(observed, expected)
        left, right = np.zeros(512, np.float32), np.zeros(192, np.float32)
        left[:2], right[:2] = [.6, .8], [.8, .6]
        self.assertEqual(policy.paired_embedding(left, right, True, .25)[[0, 1, 512, 513]].view(np.uint32).tolist(),
                         [1057293697, 1060199596, 1053609165, 1050253722])

    def test_pair_refuses_invalid_nonzero_or_half_vectors(self):
        for left, right, valid in ((np.zeros(512, np.float16), np.zeros(192, np.float32), False),
                                   (np.ones(512, np.float32), np.zeros(192, np.float32), False),
                                   (np.zeros(512, np.float32), np.zeros(192, np.float32), True)):
            with self.assertRaises(ValueError):
                policy.paired_embedding(left, right, valid, .5)

    def test_adapted_architecture_original_checkpoint_and_fold_are_fixed(self):
        config = adapted_config(b'fixture')
        policy.validate_adapted_config(config)
        for changed in ({**config, 'embedding_dim': 192}, {**config, 'source': {**config['source'], 'outer_fold': 1}},
                        {**config, 'source': {**config['source'], 'checkpoint_sha256': '0' * 64}}):
            with self.assertRaises(ValueError):
                policy.validate_adapted_config(changed)

    def test_advanced_endpoint_preserves_q0_family_after_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = json.loads((ROOT / 'configs/model/campp_advanced.json').read_text())
            write(root / policy.MODEL_PATHS['advanced'], config)
            chosen = selection_policy('advanced_public_192', protocol=policy.ADAPTED_PROTOCOL)
            proof = provenance(root, chosen, {'advanced': config}, 'adapted_advanced')
            runtime._provenance(proof, {'release_id': 'synthetic'}, chosen, {'advanced': config}, root)
            for bad in ('queries', 'family', 'roles', 'history'):
                changed = deepcopy(proof)
                if bad == 'queries': changed['calibration']['calibration_query_files'] = 4440
                if bad == 'family': changed['selection']['family'] = 'advanced_only'
                if bad == 'roles': changed['calibration']['roles_sha256'] = '0' * 64
                if bad == 'history': changed.pop('procedure_history')
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    runtime._provenance(changed, {'release_id': 'synthetic'}, chosen, {'advanced': config}, root)

    def test_endpoint_extraction_returns_original_bytes_and_wrong_dimension_fails(self):
        vector = np.zeros(192, np.float32)
        vector[1] = 1
        chosen = selection_policy('advanced_public_192')
        details = {'nonzero_signal': True, 'seconds': 3.0}
        with patch('speaker_id.candidates.campp_advanced.extract_advanced_embedding', return_value=(vector, details)):
            result, _ = runtime._extract({'advanced': object()}, chosen, Path('synthetic.wav'), 'cpu')
            self.assertIs(result, vector)
        with patch('speaker_id.candidates.campp_advanced.extract_advanced_embedding', return_value=(np.zeros(512, np.float32), details)):
            with self.assertRaises(ValueError):
                runtime._extract({'advanced': object()}, chosen, Path('synthetic.wav'), 'cpu')

    def test_portable_modules_have_no_training_tracking_or_network_imports(self):
        for path in (ROOT / 'src/speaker_id/inference/selected_policy.py', ROOT / 'src/speaker_id/inference/selected_runtime.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
                self.assertFalse(any(any(part in name.split('.') for part in ('training', 'tracking', 'mlflow', 'requests', 'modelscope')) for name in names))


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.package, self.audio = self.base / 'package', self.base / 'audio'
        self.package.mkdir(); self.audio.mkdir()
        self.labels = package(self.package)

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_synthetic_payload_passes_then_changed_weight_fails(self):
        self.assertEqual(runtime.verify_payload(self.package)['policy']['kind'], 'adapted_f004_fold0_512')
        (self.package / 'assets/f004_fold0_encoder.pt').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            runtime.verify_payload(self.package)

    def test_unlisted_and_manifest_listed_training_files_both_rejected(self):
        forbidden = self.package / 'speaker_id/training/unwanted.py'
        forbidden.parent.mkdir()
        forbidden.write_text('# No training in portable package')
        with self.assertRaises(ValueError):
            runtime.verify_payload(self.package)
        manifest(self.package)
        with self.assertRaises(ValueError):
            runtime.verify_payload(self.package)

    def test_manifest_false_zero_adaptation_claim_and_path_traversal_rejected(self):
        manifest(self.package, updates=0)
        with self.assertRaises(ValueError):
            runtime.verify_payload(self.package)
        value = manifest(self.package)
        value['files']['../outside'] = {'bytes': 0, 'sha256': '0' * 64}
        write(self.package / 'manifest.json', value)
        with self.assertRaises(ValueError):
            runtime.verify_payload(self.package)

    def test_csv_zero_decode_fallback_and_repeated_output(self):
        for name in ('zero.wav', 'known.mp3', 'bad.wav'):
            (self.audio / name).write_bytes(b'fixture')
        def extract(encoders, chosen, path, device):
            if path.name == 'bad.wav': raise OSError('synthetic decode error')
            vector = np.zeros(512, np.float32)
            if path.name == 'known.mp3': vector[0] = 1
            return vector, {'nonzero_signal': bool(vector.any())}
        output = self.package / 'predictions.csv'
        with patch.object(runtime, '_load_models', return_value=({'adapted': object()}, 'cpu')), patch.object(runtime, '_extract', side_effect=extract):
            for _ in range(2):
                result = runtime.run_submission(self.package, self.audio, output)
                self.assertEqual(result['files'], 3)
                with output.open(newline='') as stream:
                    guessed = list(csv.DictReader(stream))
                self.assertEqual(guessed, [{'audio_file': 'bad.wav', 'speaker_id': 'unknown'},
                    {'audio_file': 'known.mp3', 'speaker_id': self.labels[1]}, {'audio_file': 'zero.wav', 'speaker_id': 'unknown'}])

    def test_fatal_model_error_preserves_prior_output_and_inputs_assets_are_protected(self):
        audio = self.audio / 'known.mp3'; audio.write_bytes(b'original audio')
        output = self.base / 'predictions.csv'; output.write_bytes(b'prior result')
        with patch.object(runtime, '_load_models', return_value=({'adapted': object()}, 'cpu')), patch.object(runtime, '_extract', side_effect=RuntimeError('synthetic model failure')):
            with self.assertRaises(RuntimeError): runtime.run_submission(self.package, self.audio, output)
        self.assertEqual(output.read_bytes(), b'prior result')
        for unsafe in (audio, self.package / 'assets/gallery.npz', self.package / 'manifest.json'):
            with self.assertRaises(ValueError): runtime.run_submission(self.package, self.audio, unsafe)
        self.assertEqual(audio.read_bytes(), b'original audio')

    def test_symlink_output_cannot_target_an_unrelated_file(self):
        target = self.base / 'unrelated.txt'; target.write_text('preserve')
        linked = self.base / 'output.csv'
        try:
            linked.symlink_to(target)
        except OSError:
            self.skipTest('Creating symlinks requires unavailable Windows privilege')
        with self.assertRaises(ValueError): runtime.run_submission(self.package, self.audio, linked)
        self.assertEqual(target.read_text(), 'preserve')


class AdaptedLoaderTests(unittest.TestCase):
    def test_plain_state_weights_only_strict_architecture_and_invalid_checkpoint_rejection(self):
        class Tensor:
            def __init__(self, array): self.array = np.asarray(array); self.dtype = self.array.dtype; self.ndim = self.array.ndim
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self.array
        class Encoder:
            kwargs = None
            strict = None
            def __init__(self, **kwargs): Encoder.kwargs = kwargs; self.training = True
            def load_state_dict(self, state, strict): Encoder.strict = strict; self.state = state
            def state_dict(self): return self.state
            def to(self, **kwargs): return self
            def requires_grad_(self, enabled): self.trainable = enabled; return self
            def eval(self): self.training = False; return self
        state = {'head.conv1.weight': Tensor(np.ones((2, 2), np.float32)), 'head.bn1.num_batches_tracked': Tensor(np.asarray(0, dtype=np.int64))}
        fake = types.SimpleNamespace(Tensor=Tensor, float32=np.dtype('float32'), int64=np.dtype('int64'), isfinite=lambda t: np.isfinite(t.array))
        calls = []
        def torch_load(path, **kwargs): calls.append(kwargs); return state
        fake.load = torch_load
        vendor = types.SimpleNamespace(CAMPPlus=Encoder)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); payload = b'synthetic tensor container'
            config = adapted_config(payload); config['encoder_state_sha256'] = policy.state_dict_sha256(state)
            target = root / config['weights_path']; target.parent.mkdir(); target.write_bytes(payload)
            with patch.dict(sys.modules, {'torch': fake, 'speaker_id.models.vendor.campplus.DTDNN': vendor}):
                encoder = runtime.load_adapted_encoder(config, root)
                self.assertFalse(encoder.training); self.assertFalse(encoder.trainable)
                self.assertEqual(Encoder.kwargs, policy.ARCHITECTURE_512); self.assertIs(Encoder.strict, True)
                self.assertEqual(calls, [{'map_location': 'cpu', 'weights_only': True}])
                fake.load = lambda *args, **kwargs: {'encoder': state, 'head': {}, 'optimizer': {}}
                with self.assertRaises(ValueError): runtime.load_adapted_encoder(config, root)
                fake.load = lambda *args, **kwargs: {'head.conv1.weight': Tensor(np.ones((2, 2), np.float16))}
                with self.assertRaises(ValueError): runtime.load_adapted_encoder(config, root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
