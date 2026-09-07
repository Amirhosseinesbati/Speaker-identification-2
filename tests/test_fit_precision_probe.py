"""Validate the bounded diagnostic without Torch, CUDA, backward or optimization."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/checks/probe_fit_precision.py'
spec = importlib.util.spec_from_file_location('precision_probe', SCRIPT)
PROBE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROBE)


class PrecisionProbeTests(unittest.TestCase):
    def fixture(self):
        config = json.loads((ROOT / 'configs/train/campp_finetune_warmup.json').read_text())
        labels = ['unknown'] + [f'L{i:03}' for i in range(446)]
        roles = [{'outer_fold': 0, 'speaker_id': label, 'audio_file': label + '.wav',
                  'encoder_fit_allowed': True} for label in labels[1:]]
        return config, labels, roles

    def test_batch_is_deterministic_balanced_and_excludes_forbidden_rows(self):
        config, labels, roles = self.fixture()
        expected = PROBE.batch_plan(config, roles, labels)
        self.assertEqual(len(expected), 32)
        self.assertEqual(len({row['speaker_id'] for row in expected}), 32)
        forbidden = [{'outer_fold': 0, 'speaker_id': labels[1], 'audio_file': 'query.wav', 'encoder_fit_allowed': False},
                     {'outer_fold': 1, 'speaker_id': labels[1], 'audio_file': 'outer.wav', 'encoder_fit_allowed': True},
                     {'outer_fold': 0, 'speaker_id': 'unknown', 'audio_file': 'unknown.wav', 'encoder_fit_allowed': True}]
        self.assertEqual(PROBE.batch_plan(config, roles + forbidden, labels), expected)
        for row in expected:
            self.assertEqual(labels[row['target'] + 1], row['speaker_id'])
            self.assertEqual(row['audio_file'], row['speaker_id'] + '.wav')
            self.assertGreaterEqual(row['crop_position'], 0)
            self.assertLess(row['crop_position'], 1)

    def test_wrong_phase_or_missing_identity_cannot_be_probed(self):
        config, labels, roles = self.fixture()
        with self.assertRaises(ValueError):
            PROBE.batch_plan(config, roles[1:], labels)
        changed = copy.deepcopy(config)
        changed['fit']['adaptation_schedule']['head_only_steps'] = 200
        with self.assertRaises(ValueError):
            PROBE.batch_plan(changed, roles, labels)

    def test_import_has_no_torch_dependency_and_no_optimizer_operation_exists(self):
        result = subprocess.run([sys.executable, '-c',
            'import runpy,sys; runpy.run_path(sys.argv[1]); assert "torch" not in sys.modules', str(SCRIPT)],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        tree = ast.parse(SCRIPT.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {'optim', 'step', 'step_', 'save'})

    def test_checkpoint_and_report_paths_cannot_escape_project_scopes(self):
        with self.assertRaises(ValueError):
            PROBE.confined(ROOT / '.env', 'artifacts/training')
        with self.assertRaises(ValueError):
            PROBE.confined(Path('../outside.json'), 'artifacts/infrastructure')


if __name__ == '__main__':
    unittest.main()
