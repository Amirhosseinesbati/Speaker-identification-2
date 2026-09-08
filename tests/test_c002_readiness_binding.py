"""C002 readiness must bind evidence to its new GPU server before CAM++ work."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speaker_id.infrastructure.data import sha256_file, write_json_atomic
from speaker_id.infrastructure.readiness import (
    DEFAULT_EVIDENCE, ReadinessError, check_readiness, validate_readiness_for_execution,
)


class C002ReadinessBindingTests(unittest.TestCase):
    """Use a complete synthetic evidence set so target failures stay fail-closed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.config_path = Path("configs/train/campp_coverage_c002.json")
        self.config = {
            "device": "cuda", "expected_source_files": 2, "evaluation_classes": 447,
            "expected_vast_instance_id": 50288952,
            "expected_workspace": str(self.root),
            "expected_gpu": "RTX 3090",
            "minimum_gpu_memory_gib": 20,
            "data_verification": {"mode": "installed_manifest_sha256"},
        }
        self.write(self.config_path, self.config)
        self.write("src/sample.py", b"value = 1\n")
        self.write("data/raw/one.mp3", b"first")
        self.write("data/raw/two.mp3", b"second")
        self.write("data/raw/labels.csv", b"speaker_id,audio_file\na,one.mp3\nunknown,two.mp3\n")
        self.write("artifacts/models/campp/model.bin", b"weights")
        self.write("Competition-Guide/leaderbordpakage.txt", b"synthetic package contract")
        self.weight_hash = sha256_file(self.root / "artifacts/models/campp/model.bin")
        self.contract = {
            "signature": "c002-test-contract", "config": self.config,
            "code_hashes": {"src/sample.py": sha256_file(self.root / "src/sample.py")},
            "input_hashes": {name: "hash-" + name
                             for name in ("manifest", "folds", "roles", "label_map", "model_config")},
            "manifest": [{"audio_file": "one.mp3", "file_bytes": 5},
                         {"audio_file": "two.mp3", "file_bytes": 6}],
            "model": {"weights_path": "artifacts/models/campp/model.bin", "weights_sha256": self.weight_hash},
        }
        self.binding = {"experiment_id": "23", "experiment_name": "new-campp", "scope_id": "scope",
                        "project": "project", "tracking_endpoint": "https://tracking.example.test"}
        self.write("artifacts/infrastructure/mlflow_state.json", {"binding": self.binding})
        self.write(DEFAULT_EVIDENCE["instance"], {
            "status": "verified", "instance_id": 50288952, "verified_via": "vast_api_and_ssh",
            "hostname": "c002-test-host", "remote_workspace": str(self.root),
        })
        self.write(DEFAULT_EVIDENCE["data"], {
            "status": "passed", "training_started": False,
            "verification_scope": "all_installed_raw_files_manifest_sha256",
            "installed_file_sha256_verified": True, "output_bytes_verified": True,
            "manifest_sha256": "hash-manifest", "audio_files_verified": 2,
            "output_files_verified": 3, "class_count": 447, "labels_verified": True,
            "output": str(self.root / "data/raw"),
            "labels_sha256": sha256_file(self.root / "data/raw/labels.csv"),
        })
        self.write(DEFAULT_EVIDENCE["runtime"], {
            "status": "passed", "training_started": False, "git_commit": "commit-test",
            "git_worktree_clean": True, "workspace": str(self.root), "packages": {"demo": "1"},
            "disk": {"minimum_free_gb": 0},
            "cuda": {"available": True, "arithmetic_check_passed": True,
                     "devices": [{"name": "NVIDIA GeForce RTX 3090", "total_memory_bytes": 24 * 1024**3}]},
            "checks": {
                "pip_check": {"returncode": 0},
                "audio_decode": {"status": "passed", "manifest_sha256": "hash-manifest"},
                "leaderboard_core_versions": {
                    "source_sha256": sha256_file(self.root / "Competition-Guide/leaderbordpakage.txt"),
                    "comparisons": [{"passed": True, "guide_distribution": name}
                                    for name in ("numpy", "scipy", "soundfile", "torch", "torchaudio", "mlflow")]},
            },
        })
        self.write(DEFAULT_EVIDENCE["campp"], {
            "status": "passed_forward_only", "training_started": False, "device": "cuda",
            "gpu": "NVIDIA GeForce RTX 3090", "optimizer_steps": 0, "backward_calls": 0,
            "model_weight_sha256": self.weight_hash, "model_config_sha256": "hash-model_config",
            "input_hashes": self.contract["input_hashes"], "code_hashes": self.contract["code_hashes"],
            "contract_signature": "c002-test-contract", "git_commit": "commit-test",
            "gradient_graph_constructed": True, "embedding_shape": [512],
            "fit_forward_logits_shape": [2, 446],
        })
        spool = self.root / "artifacts/infrastructure/mlflow_probe"
        artifacts = spool / "artifacts"
        self.write(artifacts / "source_snapshot.zip", b"synthetic source archive")
        self.write(artifacts / "source_manifest.json", {
            "git_commit": "commit-test", "src_dirty": False,
            "archive_sha256": sha256_file(artifacts / "source_snapshot.zip"),
            "archive_format": "zip", "archive_name": "source_snapshot.zip",
            "files": [{"path": "src/sample.py", "sha256": self.contract["code_hashes"]["src/sample.py"]}],
        })
        fingerprints = {**self.contract["input_hashes"], "weights": self.weight_hash,
                        "resolved_config_input": sha256_file(self.root / self.config_path)}
        self.write(artifacts / "inputs_manifest.json", {key: {"sha256": value} for key, value in fingerprints.items()})
        for filename in ("report.json", "report.md", "resolved_config.json", "environment_versions.json"):
            self.write(artifacts / filename, b"{}")
        uploaded = {path.name: sha256_file(path) for path in artifacts.iterdir()}
        self.write(spool / "run_state.json", {
            "binding": self.binding, "run_id": "run-test", "remote_status": "FINISHED",
            "last_sync_error": None, "uploaded_artifacts": uploaded,
        })
        self.write(DEFAULT_EVIDENCE["mlflow"], {
            "status": "passed", "training_started": False, "base_model": "CAM++",
            "experiment_id": "23", "experiment_name": "new-campp", "run_id": "run-test",
            "local_run_directory": str(spool),
            "final_artifact_roundtrip": {"status": "passed", "files_verified": 7,
                                          "artifact_paths": sorted(uploaded)},
            "final_metadata_readback": {"status": "passed", "remote_run_status": "FINISHED"},
        })
        self.patches = [
            patch.dict("os.environ", {"VAST_INSTANCE_ID": "50288952"}),
            patch("speaker_id.infrastructure.readiness._git",
                  side_effect=lambda root, *args: "commit-test" if args[0] == "rev-parse" else ""),
            patch("speaker_id.infrastructure.readiness._package_versions", return_value={"demo": "1"}),
            patch("speaker_id.infrastructure.readiness.platform.system", return_value="Linux"),
            patch("speaker_id.infrastructure.readiness.socket.gethostname", return_value="c002-test-host"),
            patch("speaker_id.training.contracts.load_contract", return_value=self.contract),
        ]
        for mocked in self.patches:
            mocked.start()

    def tearDown(self):
        for mocked in reversed(self.patches):
            mocked.stop()
        self.temp.cleanup()

    def write(self, path, value):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            destination.write_bytes(value)
        else:
            write_json_atomic(destination, value)

    @staticmethod
    def finding(report, name):
        return next(check for check in report["checks"] if check["check"] == name)

    def test_c002_installed_data_evidence_and_target_are_ready(self):
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(report["status"], "ready", report["checks"])
        self.assertEqual(report["instance_id"], 50288952)
        self.assertEqual(report["configured_target"], {
            "instance_id": 50288952, "workspace": str(self.root), "expected_gpu": "RTX 3090",
            "minimum_gpu_memory_gib": 20,
        })
        self.assertEqual(self.finding(report, "full_data_verification")["status"], "passed")
        self.assertIn("VAST_INSTANCE_ID=50288952", report["next_command_after_user_start"])

    def test_marker_identity_mismatches_block_before_campp_evidence(self):
        cases = {
            "instance": {"instance_id": 50079023},
            "hostname": {"hostname": "another-host"},
            "workspace": {"remote_workspace": str(self.root / "another-workspace")},
        }
        marker_path = self.root / DEFAULT_EVIDENCE["instance"]
        for name, update in cases.items():
            with self.subTest(name=name):
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                marker.update(update)
                write_json_atomic(marker_path, marker)
                report = check_readiness(self.root, self.config_path)
                self.assertEqual(report["status"], "blocked")
                self.assertEqual(self.finding(report, "remote_instance")["status"], "blocked")
                names = [item["check"] for item in report["checks"]]
                self.assertLess(names.index("remote_instance"), names.index("campp_forward"))
                marker.update({"instance_id": 50288952, "hostname": "c002-test-host",
                               "remote_workspace": str(self.root)})
                write_json_atomic(marker_path, marker)

    def test_environment_and_configured_workspace_mismatches_fail_closed(self):
        with patch.dict("os.environ", {"VAST_INSTANCE_ID": "50079023"}):
            report = check_readiness(self.root, self.config_path)
        self.assertEqual(self.finding(report, "remote_instance")["status"], "blocked")
        self.assertIn("VAST_INSTANCE_ID=50288952", self.finding(report, "remote_instance")["reason"])

        self.config["expected_workspace"] = str(self.root / "wrong-checkout")
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(self.finding(report, "configured_target")["status"], "blocked")
        self.assertIn("expected_workspace", self.finding(report, "configured_target")["reason"])

    def test_expected_gpu_and_minimum_memory_are_config_bound(self):
        self.config["minimum_gpu_memory_gib"] = 25
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(self.finding(report, "runtime")["status"], "blocked")
        self.assertIn("25", self.finding(report, "runtime")["reason"])

        self.config["minimum_gpu_memory_gib"] = 20
        self.config["expected_gpu"] = "RTX 4090"
        report = check_readiness(self.root, self.config_path)
        self.assertEqual(self.finding(report, "runtime")["status"], "blocked")
        self.assertIn("RTX 4090", self.finding(report, "runtime")["reason"])

    def test_execution_recheck_rejects_changed_instance_environment(self):
        check_readiness(self.root, self.config_path)
        with patch.dict("os.environ", {"VAST_INSTANCE_ID": "50079023"}):
            with self.assertRaisesRegex(ReadinessError, "VAST_INSTANCE_ID=50288952"):
                validate_readiness_for_execution(self.root, self.contract)


if __name__ == "__main__":
    unittest.main()
