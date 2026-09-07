"""Readiness must reject stale evidence and can never upgrade a local dry run."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speaker_id.infrastructure.data import sha256_file, write_json_atomic
from speaker_id.infrastructure.readiness import (
    DEFAULT_EVIDENCE, ReadinessError, check_readiness, validate_readiness_for_execution,
)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.config_path = Path("configs/train/control.json")
        config = {"device": "cuda", "expected_source_files": 2, "evaluation_classes": 447,
                  "expected_vast_instance_id": 50079023}
        self.write(self.config_path, config)
        self.write("src/sample.py", b"value = 1\n")
        self.write("data/raw/one.mp3", b"first")
        self.write("data/raw/two.mp3", b"second")
        self.write("data/raw/labels.csv", b"speaker_id,audio_file\na,one.mp3\nunknown,two.mp3\n")
        self.write("artifacts/models/campp/model.bin", b"weights")
        self.write("Competition-Guide/leaderbordpakage.txt", b"synthetic package contract")
        self.weight_hash = sha256_file(self.root / "artifacts/models/campp/model.bin")
        self.contract = {
            "signature": "test-contract", "config": config,
            "code_hashes": {"src/sample.py": sha256_file(self.root / "src/sample.py")},
            "input_hashes": {name: "hash-" + name for name in ("manifest", "folds", "roles", "label_map", "model_config")},
            "manifest": [{"audio_file": "one.mp3", "file_bytes": 5}, {"audio_file": "two.mp3", "file_bytes": 6}],
            "model": {"weights_path": "artifacts/models/campp/model.bin", "weights_sha256": self.weight_hash},
        }
        self.binding = {"experiment_id": "23", "experiment_name": "new-campp", "scope_id": "scope", "project": "project",
                        "tracking_endpoint": "https://tracking.example.test"}
        self.write("artifacts/infrastructure/mlflow_state.json", {"binding": self.binding})
        self.write(DEFAULT_EVIDENCE["instance"], {"status": "verified", "instance_id": 50079023,
                   "verified_via": "vast_api_and_ssh", "hostname": "server-test", "remote_workspace": str(self.root)})
        self.write(DEFAULT_EVIDENCE["data"], {
            "status": "passed", "training_started": False, "manifest_sha256": "hash-manifest",
            "audio_files_verified": 2, "class_count": 447, "archive_crc_verified_members": 3,
            "output_files_verified": 3, "labels_verified": True, "archive_deleted": True,
            "output": str(self.root / "data/raw"), "labels_sha256": sha256_file(self.root / "data/raw/labels.csv"),
        })
        self.write(DEFAULT_EVIDENCE["runtime"], {
            "status": "passed", "training_started": False, "git_commit": "commit-test", "git_worktree_clean": True,
            "workspace": str(self.root), "packages": {"demo": "1"}, "disk": {"minimum_free_gb": 0},
            "cuda": {"available": True, "arithmetic_check_passed": True,
                     "devices": [{"name": "RTX 3090", "total_memory_bytes": 24 * 1024**3}]},
            "checks": {"pip_check": {"returncode": 0},
                       "audio_decode": {"status": "passed", "manifest_sha256": "hash-manifest"},
                       "leaderboard_core_versions": {
                           "source_sha256": sha256_file(self.root / "Competition-Guide/leaderbordpakage.txt"),
                           "comparisons": [{"passed": True, "guide_distribution": name}
                                           for name in ("numpy", "scipy", "soundfile", "torch", "torchaudio", "mlflow")]}},
        })
        self.write(DEFAULT_EVIDENCE["campp"], {
            "status": "passed_forward_only", "training_started": False, "device": "cuda", "gpu": "RTX 3090",
            "optimizer_steps": 0, "backward_calls": 0, "model_weight_sha256": self.weight_hash,
            "model_config_sha256": "hash-model_config", "input_hashes": self.contract["input_hashes"],
            "code_hashes": self.contract["code_hashes"], "contract_signature": "test-contract",
            "git_commit": "commit-test", "gradient_graph_constructed": True,
            "embedding_shape": [512], "fit_forward_logits_shape": [2, 446],
        })
        spool = self.root / "artifacts/infrastructure/mlflow_probe"
        artifacts = spool / "artifacts"
        self.write(artifacts / "source_snapshot.tar.gz", b"synthetic source archive")
        self.write(artifacts / "source_manifest.json", {
            "git_commit": "commit-test", "src_dirty": False,
            "archive_sha256": sha256_file(artifacts / "source_snapshot.tar.gz"),
            "files": [{"path": "src/sample.py", "sha256": self.contract["code_hashes"]["src/sample.py"]}],
        })
        fingerprints = {**self.contract["input_hashes"], "weights": self.weight_hash,
                        "resolved_config_input": sha256_file(self.root / self.config_path)}
        self.write(artifacts / "inputs_manifest.json", {key: {"sha256": value} for key, value in fingerprints.items()})
        for filename in ("report.json", "report.md", "resolved_config.json", "environment_versions.json"):
            self.write(artifacts / filename, b"{}")
        uploaded = {path.name: sha256_file(path) for path in artifacts.iterdir()}
        self.write(spool / "run_state.json", {"binding": self.binding, "run_id": "run-test", "remote_status": "FINISHED",
                                              "last_sync_error": None, "uploaded_artifacts": uploaded})
        self.write(DEFAULT_EVIDENCE["mlflow"], {
            "status": "passed", "training_started": False, "base_model": "CAM++", "experiment_id": "23",
            "experiment_name": "new-campp", "run_id": "run-test", "local_run_directory": str(spool),
            "final_artifact_roundtrip": {"status": "passed", "files_verified": 7, "artifact_paths": sorted(uploaded)},
            "final_metadata_readback": {"status": "passed", "remote_run_status": "FINISHED"},
        })
        self.patches = [
            patch.dict("os.environ", {"VAST_INSTANCE_ID": "50079023"}),
            patch("speaker_id.infrastructure.readiness._git", side_effect=lambda root, *args: "commit-test" if args[0] == "rev-parse" else ""),
            patch("speaker_id.infrastructure.readiness._package_versions", return_value={"demo": "1"}),
            patch("speaker_id.infrastructure.readiness.platform.system", return_value="Linux"),
            patch("speaker_id.infrastructure.readiness.socket.gethostname", return_value="server-test"),
            patch("speaker_id.training.contracts.load_contract", return_value=self.contract),
        ]
        for mocking in self.patches:
            mocking.start()

    def tearDown(self):
        for mocking in reversed(self.patches):
            mocking.stop()
        self.temp.cleanup()

    def write(self, path, value):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            destination.write_bytes(value)
        else:
            write_json_atomic(destination, value)

    def test_all_measured_evidence_ready_still_requires_user_start(self):
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(report["status"], "ready", report["checks"])
        self.assertIs(report["training_started"], False)
        self.assertIs(report["user_start_instruction_required"], True)
        self.assertEqual(validate_readiness_for_execution(self.root, self.contract)["status"], "ready")

    def test_local_cpu_probe_cannot_claim_server_ready(self):
        with patch("speaker_id.infrastructure.readiness.platform.system", return_value="Windows"):
            report = check_readiness(self.root, self.config_path)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any(check["check"] == "remote_instance" and check["status"] == "blocked" for check in report["checks"]))

    def test_changed_code_invalidates_uploaded_source_snapshot(self):
        self.write("src/sample.py", b"value = 2\n")
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("snapshot is stale" in check.get("reason", "") for check in report["checks"]))

    def test_changed_weights_blocks_readiness(self):
        self.write("artifacts/models/campp/model.bin", b"new weights")
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("checkpoint changed" in check.get("reason", "") for check in report["checks"]))

    def test_changed_evidence_blocks_previously_ready_execution(self):
        check_readiness(self.root, self.config_path)
        self.write(DEFAULT_EVIDENCE["runtime"], {"status": "failed"})
        with self.assertRaisesRegex(ReadinessError, "evidence changed"):
            validate_readiness_for_execution(self.root, self.contract)

    def test_missing_evidence_returns_explicit_blocked_report(self):
        (self.root / DEFAULT_EVIDENCE["mlflow"]).unlink()
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(report["status"], "blocked")
        self.assertGreater(report["blocked_checks"], 0)

    def test_runtime_package_changes_block_readiness(self):
        with patch("speaker_id.infrastructure.readiness._package_versions", return_value={"demo": "2"}):
            report = check_readiness(self.root, self.config_path)
        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("Installed packages changed" in check.get("reason", "") for check in report["checks"]))


if __name__ == "__main__":
    unittest.main()
