import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.training.frozen_suite import validated_cache, validate_suite, verify_source_predictions


class FrozenCacheTests(unittest.TestCase):
    def test_s002_keeps_exactly_control_and_unchanged_s001f_grid(self):
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root / 'configs/train/campp_scoring_suite.json').read_text())
        coverage = json.loads((root / 'configs/train/campp_coverage_scoring.json').read_text())
        validate_suite(original)
        validate_suite(coverage)
        self.assertEqual([r['id'] for r in coverage['recipes']], ['S002a', 'S002f'])
        self.assertEqual(coverage['baseline_config'], 'configs/train/campp_coverage.json')
        self.assertEqual(coverage['source_run'], 'artifacts/training/B002_20260907T150724Z_9b11fe4b')
        self.assertEqual(coverage['source_parent_run_id'], '9b5ef17e61d24b9ba438128bf7f24a7a')
        old_f = next(r for r in original['recipes'] if r['id'] == 'S001f')
        for key in ('method', 'calibration_protocol', 'unknown_weights', 'margin_weights'):
            self.assertEqual(coverage['recipes'][1][key], old_f[key])
        self.assertEqual(coverage['threshold_candidates'], original['threshold_candidates'])
        self.assertEqual(coverage['probability_temperature'], original['probability_temperature'])

    def test_reproduction_control_cannot_be_missing_moved_or_redefined(self):
        import copy
        root = Path(__file__).resolve().parents[1]
        suite = json.loads((root / 'configs/train/campp_coverage_scoring.json').read_text())
        invalid = []
        changed = copy.deepcopy(suite)
        changed['recipes'] = changed['recipes'][1:]
        invalid.append(changed)
        changed = copy.deepcopy(suite)
        changed['recipes'].reverse()
        invalid.append(changed)
        for key, value in [('id', 'S001a'), ('method', 'max_reference'),
                           ('calibration_protocol', 'leave_content_group_out'),
                           ('unknown_weights', [0., .5]), ('margin_weights', [.5])]:
            changed = copy.deepcopy(suite)
            changed['recipes'][0][key] = value
            invalid.append(changed)
        changed = copy.deepcopy(suite)
        changed['experiment_code'] = '../S002'
        invalid.append(changed)
        for index, changed in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_suite(changed)

    def test_source_reproduction_checks_all_rows_labels_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            fold = source / 'fold_0'
            fold.mkdir()
            path = fold / 'predictions.csv'
            path.write_text('audio_file,speaker_id\na.mp3,A\nb.mp3,unknown\n')
            expected = [{'audio_file': 'a.mp3', 'speaker_id': 'A'},
                        {'audio_file': 'b.mp3', 'speaker_id': 'unknown'}]
            proof = verify_source_predictions(source, 0, list(reversed(expected)))
            self.assertTrue(proof['exact_prediction_reproduction'])
            self.assertEqual(proof['files'], 2)
            self.assertEqual(proof['source_prediction_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            for changed in (expected[:1], expected + [expected[0]],
                            [expected[0], {'audio_file': 'b.mp3', 'speaker_id': 'B'}]):
                with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                    verify_source_predictions(source, 0, changed)
            path.write_text('audio_file,speaker_id\na.mp3,A\nb.mp3,unknown\na.mp3,A\n')
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                verify_source_predictions(source, 0, expected)

    def test_s002_supervisor_is_explicitly_inactive(self):
        import configparser
        root = Path(__file__).resolve().parents[1]
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(root / 'configs/infra/supervisor_campp_coverage_scoring.conf')
        program = parser['program:speaker_id_campp_s002']
        self.assertFalse(program.getboolean('autostart'))
        self.assertFalse(program.getboolean('autorestart'))
        self.assertEqual(program['startretries'], '0')
        self.assertTrue(program['command'].endswith('configs/train/campp_coverage_scoring.json'))
        installer = (root / 'scripts/infra/install_experiment_supervisor.sh').read_text()
        launcher = (root / 'scripts/infra/run_campp_experiment.sh').read_text()
        self.assertIn('PROGRAM=speaker_id_campp_s002', installer)
        self.assertNotIn('supervisorctl start', installer)
        self.assertIn('configs/train/campp_coverage_scoring.json)', launcher)

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

    def test_fine_tuned_contract_cannot_consume_frozen_scoring_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            source, contract = self.fixture(Path(directory))
            contract['config'] = {**contract['config'], 'mode': 'fine_tune'}
            with self.assertRaisesRegex(ValueError, 'frozen baseline contract'):
                validated_cache(Path(directory), source, contract, 'parent')

    def test_tampered_source_configuration_fails_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            source, contract = self.fixture(Path(directory))
            original = json.loads((source / 'resolved_config.json').read_text())
            original['experiment']['inference']['seconds'] = 30
            (source / 'resolved_config.json').write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, 'cache signature'):
                validated_cache(Path(directory), source, contract, 'parent')
