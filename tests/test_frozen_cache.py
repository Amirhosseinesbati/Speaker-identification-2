import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training.frozen_suite import validated_cache, validate_suite


class FrozenCacheTests(unittest.TestCase):
    def test_suite_rejects_typo_protocol_and_unsafe_or_duplicate_ids(self):
        import copy
        root = Path(__file__).resolve().parents[1]
        suite = json.loads((root / 'configs/train/campp_scoring_suite.json').read_text())
        validate_suite(suite)
        for key, value in [('calibration_protocol', 'typo'), ('id', '../../outside'), ('unknown_weights', [float('nan')])]:
            changed = copy.deepcopy(suite)
            changed['recipes'][0][key] = value
            with self.assertRaises(ValueError):
                validate_suite(changed)
        suite['recipes'][1]['id'] = suite['recipes'][0]['id']
        with self.assertRaises(ValueError):
            validate_suite(suite)

    def fixture(self, path):
        source = path / 'source'
        (source / 'frozen_embedding_cache').mkdir(parents=True)
        snapshot = source / 'tracking/attempt/artifacts'
        snapshot.mkdir(parents=True)
        (snapshot / 'source_manifest.json').write_text(json.dumps({'git_commit': 'original'}))
        config = {'mode': 'frozen_baseline', 'inference': {'seconds': 6, 'maximum_windows': 3}}
        code = {'src/speaker_id/models/campp.py': 'model-code', 'src/speaker_id/training/runner.py': 'loop-code'}
        payload = {'config': config, 'model': {'weights_sha256': 'public'}, 'input_hashes': {'manifest': 'raw'}, 'code_hashes': code}
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        original = {**payload, 'experiment': payload['config'], 'signature': signature}
        del original['config']
        (source / 'resolved_config.json').write_text(json.dumps(original))
        (source / 'experiment_state.json').write_text(json.dumps({'status': 'complete', 'parent_run_id': 'parent', 'signature': signature}))
        vector = np.zeros(512, np.float32); vector[0] = 1
        np.savez(source / 'frozen_embedding_cache/audio.npz', embedding=vector, valid=True, signature=signature, audio_sha256='audio-sha')
        contract = {**payload, 'manifest': [{'audio_file': 'audio.mp3', 'input_sha256': 'audio-sha', 'has_nonzero_signal': True}]}
        return source, contract

    def test_cache_can_survive_unrelated_analysis_code_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source, contract = self.fixture(Path(directory))
            contract['code_hashes'] = {**contract['code_hashes'], 'src/speaker_id/evaluation/new.py': 'new'}
            vectors, valid, provenance = validated_cache(Path(directory), source, contract, 'parent')
            self.assertEqual(vectors.shape, (1, 512))
            self.assertTrue(valid[0])
            self.assertTrue(provenance['feature_implementation_unchanged'])

    def test_feature_or_extraction_loop_change_invalidates_cache(self):
        for name in ['src/speaker_id/models/campp.py', 'src/speaker_id/training/runner.py']:
            with tempfile.TemporaryDirectory() as directory:
                source, contract = self.fixture(Path(directory))
                contract['code_hashes'][name] = 'modified'
                with self.assertRaisesRegex(ValueError, 'implementation changed'):
                    validated_cache(Path(directory), source, contract, 'parent')

    def test_tampered_source_configuration_fails_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            source, contract = self.fixture(Path(directory))
            original = json.loads((source / 'resolved_config.json').read_text())
            original['experiment']['inference']['seconds'] = 30
            (source / 'resolved_config.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'cache signature'):
                validated_cache(Path(directory), source, contract, 'parent')
