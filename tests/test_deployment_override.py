"""Deployment input validation without contacting Vast or an SSH endpoint."""
from __future__ import annotations

import configparser
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deploy_workspace_for_tests", ROOT / "scripts/infra/deploy_workspace.py")
DEPLOY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY)


class DeploymentOverrideTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.directory = self.root / "configs/train"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "example.json"
        self.path.write_text(json.dumps({"schema_version": 1, "mode": "frozen_baseline"}))
        self.default = "configs/train/campp_baseline.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_verify_override_normalizes_relative_and_absolute_paths_without_mutation(self):
        original = self.path.read_bytes()
        for path in (Path("configs/train/example.json"), self.path):
            with self.subTest(path=path):
                actual = DEPLOY.resolve_training_config(self.root, "verify", path, self.default)
                self.assertEqual(actual, "configs/train/example.json")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(DEPLOY.resolve_training_config(self.root, "deploy", None, self.default), self.default)

    def test_override_is_rejected_for_every_nonverify_action(self):
        for action in ("inspect", "deploy", "bootstrap", "upload-assets", "resume-raw", "download-evidence"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, "only by the verify"):
                    DEPLOY.resolve_training_config(self.root, action, self.path, self.default)

    def test_override_cannot_escape_config_directory(self):
        outside = self.root / "outside.json"
        outside.write_text("{}")
        for path in (outside, Path("outside.json"), Path("configs/train/../outside.json"), Path("../outside.json")):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    DEPLOY.resolve_training_config(self.root, "verify", path, self.default)

    def test_override_must_be_an_existing_json_object_file(self):
        cases = [("array.json", "[]"), ("broken.json", "not JSON"), ("text.txt", "{}")]
        for name, text in cases:
            path = self.directory / name
            path.write_text(text)
            with self.subTest(path=name):
                with self.assertRaises(ValueError):
                    DEPLOY.resolve_training_config(self.root, "verify", path, self.default)
        with self.assertRaises(OSError):
            DEPLOY.resolve_training_config(self.root, "verify", self.directory / "missing.json", self.default)
        with self.assertRaises(ValueError):
            DEPLOY.resolve_training_config(self.root, "verify", self.directory, self.default)

    def test_registered_experiments_are_manual_and_bound_to_exact_configs(self):
        programs = {
            "scoring_suite": ("speaker_id_campp_s001", "campp_scoring_suite.json"),
            "coverage": ("speaker_id_campp_b002", "campp_coverage.json"),
            "finetune": ("speaker_id_campp_f001", "campp_finetune.json"),
        }
        for suffix, (program, config) in programs.items():
            with self.subTest(program=program):
                parsed = configparser.ConfigParser()
                parsed.read(ROOT / f"configs/infra/supervisor_campp_{suffix}.conf")
                section = parsed[f"program:{program}"]
                self.assertFalse(section.getboolean("autostart"))
                self.assertFalse(section.getboolean("autorestart"))
                self.assertEqual(section["command"], "/bin/bash /workspace/Speaker-identification-2/scripts/infra/run_campp_experiment.sh configs/train/" + config)
                self.assertEqual(section.getint("numprocs"), 1)


if __name__ == "__main__":
    unittest.main()
