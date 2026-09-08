"""C001 orchestration safety using synthetic arrays and a local fake tracker."""
from copy import deepcopy
from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.training import cpu_gain_suite as cpu
from speaker_id.training import cpu_pair_contract as protocol
from test_gain_suite import cache_fixture, rewrite_cache


class Tracker:
    def __init__(self, failure=False):
        self.artifacts, self.events = [], []
        self.failure = failure

    def add_artifact(self, path, name=None):
        self.artifacts.append((Path(path), name))
        self.events.append('artifact')

    def log_metrics(self, metrics, **kwargs):
        self.events.append(('metrics', metrics, kwargs))

    def flush(self, *, strict):
        assert strict
        self.events.append('strict_flush')
        if self.failure:
            raise ConnectionError('Synthetic tracking failure')

    def verify_artifacts(self):
        self.events.append('artifact_roundtrip')

    def verify_remote_metadata(self):
        self.events.append('metadata_readback')


def scored_fixture():
    class InnerOnly(dict):
        def __getitem__(self, key):
            if key.startswith('outer'):
                raise AssertionError('Fresh outer scores read before policies sealed')
            return super().__getitem__(key)
    return {alpha: InnerOnly(calibration_indices=np.asarray([0, 1, 2]),
        inner_known_scores=np.asarray([[.9, .1], [.1, .9], [.2, .2]], dtype=np.float32),
        inner_unknown_similarity=np.asarray([.1, .1, .9], dtype=np.float32)) for alpha in cpu.gain.ALPHAS}


def extracted_fixture(root):
    result = {'vectors': {}, 'identities': {}, 'receipts': {}, 'execution_report': {}}
    for frontend in cpu.FRONTENDS:
        directory = root / (frontend + '_embedding_cache')
        identity, manifest, receipt = cache_fixture(directory)
        if frontend == 'gain':
            import hashlib
            identity['frontend'] = 'gain'
            identity['signature'] = hashlib.sha256(json.dumps({key: value for key, value in identity.items()
                if key != 'signature'}, sort_keys=True, allow_nan=False).encode()).hexdigest()
            for index in range(len(manifest)):
                rewrite_cache(directory, receipt, index, signature=identity['signature'])
        values, valid = cpu.gain.verify_gain_cache(directory, identity, manifest, receipt)
        result['vectors'][frontend] = values
        result['identities'][frontend] = identity
        result['receipts'][frontend] = receipt
        result['valid'] = valid
    return result, {'manifest': manifest}


class CPUOrchestrationTests(unittest.TestCase):
    def test_metadata_rejects_arrays_bytes_numpy_scalars_and_nonfinite(self):
        for value in (np.zeros(512, np.float32), b'weights', np.float32(.5), float('nan'), float('inf')):
            with self.subTest(type=type(value)), self.assertRaises(ValueError):
                cpu._metadata({'nested': [value]})

    def test_callback_tracks_json_inventories_not_feature_npzs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tracker = Path(tmp), Tracker()
            callback = cpu.extraction_callback(root, tracker)
            callback('identities', {'identity': {'signature': 'a' * 64}})
            callback('progress', {'completed_pairs': 50, 'total': 4529, 'elapsed_seconds': 12.5})
            callback('complete', {'files': [{'cache_file': 'example.npz', 'sha256': 'b' * 64}],
                                  'embedding_artifacts_uploaded': False})
            self.assertEqual(len(tracker.artifacts), 3)
            self.assertEqual(tracker.events.count('strict_flush'), 3)
            for path, alias in tracker.artifacts:
                self.assertEqual(path.suffix, '.json')
                self.assertTrue(alias.startswith('cpu_extraction/'))
                json.loads(path.read_text())
            before = len(tracker.artifacts)
            with self.assertRaises(ValueError):
                callback('progress', {'public': np.zeros(512, np.float32)})
            self.assertEqual(before, len(tracker.artifacts))

    def test_callback_strict_failure_preserves_evidence_and_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = Tracker(failure=True)
            callback = cpu.extraction_callback(Path(tmp), tracker)
            with self.assertRaises(ConnectionError):
                callback('progress', {'completed_pairs': 50, 'total': 4529, 'elapsed_seconds': 5.})
            self.assertTrue(tracker.artifacts[0][0].is_file())

    def test_existing_evidence_directory_is_not_resumed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpu.extraction_callback(root, Tracker())
            with self.assertRaises(FileExistsError):
                cpu.extraction_callback(root, Tracker())

    def test_both_inner_choices_seal_before_return_without_outer_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tracker = Path(tmp), Tracker()
            extracted = {'vectors': {name: {'public': None, 'advanced': None} for name in cpu.FRONTENDS}, 'valid': None}
            query = {fold: np.asarray([0, 1, 2]) for fold in (0, 1)}
            truth = {fold: np.asarray([1, 2, 0]) for fold in (0, 1)}
            events = []
            def scores(*args):
                events.append(('scores', args[-1]))
                return scored_fixture()
            selector = cpu.gain.select_inner_frontend
            with patch.object(cpu.gain, 'family_scores', side_effect=scores), \
                    patch.object(cpu.gain, 'select_inner_frontend', side_effect=lambda a, b: selector(a, b, classes=3)):
                _, frozen = cpu.freeze_cpu_choices(root, tracker, {}, extracted, query, truth)
            self.assertEqual(events, [('scores', 0), ('scores', 0), ('scores', 1), ('scores', 1)])
            self.assertEqual(set(frozen['selected']), {0, 1})
            self.assertTrue(all(choice['frontend'] == 'identity' and choice['advanced_weight'] == 0.
                                for choice in frozen['selected'].values()))
            self.assertEqual(tracker.events[-3:], ['strict_flush', 'artifact_roundtrip', 'metadata_readback'])
            self.assertTrue((root / 'frozen_inner_choices.json').is_file())
            for fold in (0, 1):
                self.assertEqual(cpu.selected_cpu_policy('C001d', fold, frozen)['calibration'],
                                 cpu.selected_cpu_policy('C001b', fold, frozen)['calibration'])

    def test_query_tamper_blocks_choice_sealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tracker = Path(tmp), Tracker()
            vectors = {'vectors': {name: {'public': None, 'advanced': None} for name in cpu.FRONTENDS}, 'valid': None}
            with patch.object(cpu.gain, 'family_scores', return_value=scored_fixture()):
                with self.assertRaises(ValueError):
                    cpu.freeze_cpu_choices(root, tracker, {}, vectors,
                        {0: np.asarray([2, 1, 0]), 1: np.asarray([0, 1, 2])}, {0: None, 1: None})
            self.assertFalse(tracker.artifacts)

    def test_remote_choice_readback_failure_prevents_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, tracker = Path(tmp), Tracker()
            vectors = {'vectors': {name: {'public': None, 'advanced': None} for name in cpu.FRONTENDS}, 'valid': None}
            query = {fold: np.asarray([0, 1, 2]) for fold in (0, 1)}
            choice = {'frontend': 'identity', 'advanced_weight': 0., 'calibration': {}}
            with patch.object(cpu.gain, 'family_scores', return_value=scored_fixture()), \
                    patch.object(cpu.gain, 'select_inner_frontend', return_value=(choice, {})), \
                    patch.object(tracker, 'verify_artifacts', side_effect=ValueError('Synthetic SHA mismatch')):
                with self.assertRaises(ValueError):
                    cpu.freeze_cpu_choices(root, tracker, {}, vectors, query, {0: None, 1: None})
            self.assertTrue((root / 'frozen_inner_choices.json').is_file())
            self.assertNotIn('metadata_readback', tracker.events)

    def test_incomplete_or_gpu_policy_cannot_be_evaluated(self):
        selected = {'frontend': 'identity', 'advanced_weight': 0., 'calibration': {}}
        frozen = {'selected': {0: selected, 1: selected}, 'inner_fits': {0: {}, 1: {}},
                  'no_fresh_outer_evaluation_performed_yet': True, 'historical_GPU_excluded_from_selection': True}
        for change in ('fold', 'GPU', 'recipe'):
            candidate = deepcopy(frozen)
            if change == 'fold': candidate['selected'].pop(1)
            if change == 'GPU': candidate['selected'][0]['frontend'] = 'historical_gpu'
            with self.subTest(change=change), self.assertRaises(ValueError):
                cpu.selected_cpu_policy('C001a' if change == 'recipe' else 'C001d', 0, candidate)

    def test_cache_disk_values_zero_mask_and_returned_arrays_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extracted, contract = extracted_fixture(root)
            cpu.verify_cpu_extraction(root, extracted, contract, extracted['valid'])
            self.assertEqual(extracted['valid'].tolist(), [True, False])
            bad = deepcopy(extracted)
            bad['vectors']['identity']['public'][0, 2] = .5
            with self.assertRaises(ValueError): cpu.verify_cpu_extraction(root, bad, contract, extracted['valid'])
            bad = deepcopy(extracted)
            bad['valid'][1] = True
            with self.assertRaises(ValueError): cpu.verify_cpu_extraction(root, bad, contract, extracted['valid'])
            bad = deepcopy(extracted)
            bad['identities']['gain']['signature'] = 'f' * 64
            with self.assertRaises(ValueError): cpu.verify_cpu_extraction(root, bad, contract, extracted['valid'])

    def test_historical_curve_mismatch_stops_before_outer_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / 'S008c/fold_0'
            history.mkdir(parents=True)
            (history / 'inner_alpha_calibration.json').write_text('{}')
            contract = {'labels': ['unknown', 'A', 'B'], 'manifest': [{'speaker_id': x} for x in ('A', 'B', 'unknown')]}
            sources = {'selection': {'directory': root}, 'vectors': {'public': None, 'advanced': None}, 'valid': None}
            with patch.object(cpu.gain, 'family_scores', return_value=scored_fixture()), \
                    patch.object(cpu.gain, 'select_inner_alpha', return_value=({'advanced_weight': 0.}, {})), \
                    patch.object(cpu.gain, '_record_fold', side_effect=AssertionError('Outer record unexpectedly started')):
                with self.assertRaises(ValueError): cpu.historical_control(root, Tracker(), contract, sources)

    def test_cli_default_does_not_prepare_execution_or_load_models(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/score_gain_cpu.py'
        spec = importlib.util.spec_from_file_location('cpu_gain_cli_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'configs/train/cpu.json'
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps(protocol.FIXED))
            with patch.object(module, 'ROOT', root), patch('sys.argv', ['score_gain_cpu.py', '--config', 'configs/train/cpu.json']), \
                    patch.object(protocol, 'load_cpu_gain_inputs', return_value=({}, {}, {})), \
                    patch.object(cpu, 'execute_cpu_gain_suite', side_effect=AssertionError('Executed in validate mode')), \
                    patch('sys.stdout', new_callable=io.StringIO) as out:
                module.main()
            self.assertEqual(json.loads(out.getvalue())['status'], 'validated_no_experiment_started')

    def test_fake_orchestration_orders_control_and_remote_choices_before_fresh_outer(self):
        self._exercise_fake_run(False)

    def test_fake_orchestration_control_failure_blocks_worker_and_redacts_local_error(self):
        self._exercise_fake_run(True)

    def _exercise_fake_run(self, fail_control):
        from speaker_id import tracking
        from speaker_id.training import cpu_pair_worker
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = root / 'binding.json'
            binding.write_text('{"binding": {}}')
            suite = deepcopy(protocol.FIXED)
            contract = {'config': {name: 'unused' for name in ('manifest', 'folds', 'roles', 'label_map', 'model_config')},
                'model': {}, 'input_hashes': {}, 'code_hashes': {}, 'signature': 'synthetic',
                'manifest': [{'audio_file': name, 'speaker_id': label} for name, label in [('a.wav', 'A'), ('b.wav', 'B'), ('z.wav', 'unknown')]],
                'labels': ['unknown', 'A', 'B']}
            actions, runs = [], []
            remote_choices = [False]

            class Run(Tracker):
                @classmethod
                def prepare(cls, **kwargs):
                    obj = cls()
                    obj.run_id, obj.client = 'run' + str(len(runs)), object()
                    obj.redactor = SimpleNamespace(text=lambda value: value.replace('private-token', '[redacted]'))
                    obj.spool = kwargs['spool_dir']
                    runs.append(obj)
                    return obj

                def verify_remote_metadata(self):
                    super().verify_remote_metadata()
                    if any(path.name == 'frozen_inner_choices.json' for path, _ in self.artifacts):
                        remote_choices[0] = True

                def write_report(self, *args, **kwargs):
                    pass

                def finish(self, status, *, strict):
                    self.status = status

            predictions = [{'audio_file': row['audio_file'], 'speaker_id': row['speaker_id']} for row in contract['manifest']]
            fold_report = {'outer': {'macro_f1': 1.}}
            def result(recipe):
                return {'recipe': recipe, 'oof': {'macro_f1': 1.}, 'folds': [deepcopy(fold_report), deepcopy(fold_report)]}
            query = {fold: np.asarray([0, 1, 2]) for fold in (0, 1)}
            truths = {fold: np.asarray([1, 2, 0]) for fold in (0, 1)}
            def historical(*args):
                actions.append('historical')
                if fail_control:
                    raise ValueError('Control failed private-token')
                return result('C001a'), predictions, {'passed': True}, query, truths
            extracted = {'vectors': {name: {'public': None, 'advanced': None} for name in cpu.FRONTENDS},
                'valid': np.asarray([True, True, False]), 'identities': {}, 'execution_report': {}}
            sources = {'contract': contract, 'remote_requests': [], 'proof': {}, 'valid': extracted['valid']}
            def worker(*args, **kwargs):
                actions.append('worker')
                self.assertEqual(actions, ['historical', 'worker'])
                return extracted
            selected = {'advanced_weight': 0., 'calibration': {}}
            fitted = {name: {'selected': deepcopy(selected), 'candidates': {'0.0': {'curve': []}}} for name in cpu.FRONTENDS}
            def record(directory, *args):
                self.assertTrue(remote_choices[0])
                sealed = json.loads((directory.parents[1] / 'frozen_inner_choices.json').read_text())
                self.assertEqual(set(sealed['selected']), {'0', '1'})
                actions.append('fresh_outer')
                directory.mkdir()
                cpu.gain.write_csv(directory / 'predictions.csv', predictions)
                np.savez(directory / 'outer_probabilities.npz', probabilities=np.eye(3), labels=np.asarray(contract['labels']))
                np.savez(directory / 'reference_support.npz', support=np.arange(3))
                return predictions, deepcopy(fold_report), None
            with ExitStack() as stack:
                def mocked(owner, name, **kwargs):
                    return stack.enter_context(patch.object(owner, name, **kwargs))
                mocked(protocol, 'prepare_cpu_execution', return_value={'backend': {}, 'capacity': {}, 'cp001_evidence': {}})
                mocked(tracking, 'ExperimentBinding', side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
                mocked(tracking, 'DurableMLflowRun', new=Run)
                mocked(cpu.gain, 'load_sources', return_value=sources)
                mocked(cpu.gain, '_verify_remote_evidence', return_value={})
                mocked(cpu, 'historical_control', side_effect=historical)
                worker_mock = mocked(cpu_pair_worker, 'extract_cpu_pair_caches', side_effect=worker)
                mocked(cpu, 'verify_cpu_extraction', return_value=None)
                mocked(cpu.gain, 'family_scores', return_value=scored_fixture())
                mocked(cpu.gain, 'select_inner_frontend', return_value=({'frontend': 'identity', **selected}, fitted))
                mocked(cpu.gain, '_record_fold', side_effect=record)
                mocked(cpu.gain, 'finish_recipe', side_effect=lambda p, t, c, rows, folds, recipe, control: result(recipe))
                mocked(cpu.gain, 'paired_diagnostics', return_value={})
                if fail_control:
                    with self.assertRaises(ValueError):
                        cpu.execute_cpu_gain_suite(root, root / 'config.json', suite, contract, {}, binding)
                    worker_mock.assert_not_called()
                    failure = json.loads(next((root / suite['output_root']).glob('*/failure.json')).read_text())
                    self.assertEqual(failure['error'], 'Control failed [redacted]')
                    self.assertEqual(runs[0].status, 'FAILED')
                else:
                    report = cpu.execute_cpu_gain_suite(root, root / 'config.json', suite, contract, {}, binding)
                    self.assertEqual(set(report['results']), set(cpu.RECIPES))
                    self.assertEqual(actions, ['historical', 'worker'] + ['fresh_outer'] * 6)
                    self.assertEqual(runs[0].status, 'FINISHED')
                    for run in runs:
                        for path, _ in run.artifacts:
                            self.assertNotIn('embedding_cache', path.parts)
                            self.assertNotEqual(path.suffix, '.zip')


if __name__ == '__main__':
    unittest.main()
