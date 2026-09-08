"""Fast C002 CUDA contract tests; no CUDA device, model or audio is required."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from speaker_id.training import cuda_pair_contract as cuda


def backend():
    return {
        "schema_version": 1,
        "device": "cuda",
        "device_index": 0,
        "device_name": "NVIDIA GeForce RTX 3090",
        "cuda_runtime": "12.8",
        "torch_version": "2.10.0+cu128",
        "python_version": "3.12.3",
        "tensor_dtype": "float32",
        "total_memory_bytes": 23 * 1024 ** 3,
        "visible_total_memory_bytes": 23 * 1024 ** 3,
        "free_memory_bytes": 12 * 1024 ** 3,
        "cudnn_enabled": True,
        "no_cpu_fallback": True,
        "encoder_updates": 0,
    }


class CUDAPairContractTests(unittest.TestCase):
    @staticmethod
    def compatible_contracts():
        semantic = {
            "mode": "frozen_baseline",
            "model_config": "configs/model/campp.json",
            "manifest": "data/manifest.csv",
            "folds": "data/folds.csv",
            "roles": "data/roles.csv",
            "label_map": "data/labels.json",
            "inference": {"seconds": 180.0, "maximum_windows": 1},
            "scoring": {"method": "normalized_mean_prototype"},
            "fit": {"crop_seconds": 3.0},
            "fold_ids": [0, 1],
        }
        historical = {
            "signature": "a" * 64,
            "config": {**semantic, "experiment_code": "B002", "run_name": "B002-control",
                       "hypothesis": "historical", "output_root": "artifacts/training",
                       "seed": 20260907, "expected_vast_instance_id": 50079023},
            "model": {"weights_sha256": "1" * 64},
            "input_hashes": {"manifest": "2" * 64, "model_config": "3" * 64},
            "code_hashes": {"src/model.py": "4" * 64},
            "labels": ["unknown", "speaker"],
            "manifest": [{"audio_file": "a.mp3"}],
            "folds": [{"audio_file": "a.mp3", "fold": "0"}],
            "roles": [{"audio_file": "a.mp3", "role": "outer"}],
        }
        readiness = deepcopy(historical)
        readiness["signature"] = "b" * 64
        readiness["config"].update({
            "experiment_code": "B002-C002",
            "run_name": "B002-C002-control",
            "hypothesis": "fresh infrastructure",
            "output_root": "artifacts/training/C002",
            "seed": 20260908,
            "expected_vast_instance_id": 50288952,
            "expected_workspace": "/workspace/Speaker-identification-2-c002",
            "expected_gpu": "RTX 3090",
            "minimum_gpu_memory_gib": 20,
            "mlflow_state_path": "artifacts/infrastructure/C002_preparation/mlflow_state.json",
            "data_verification": {"mode": "installed_manifest_sha256"},
        })
        return readiness, historical

    def test_historical_source_allows_only_explicit_c002_infrastructure_differences(self):
        readiness, historical = self.compatible_contracts()
        proof = cuda.verify_c002_source_compatibility(readiness, historical)
        self.assertEqual(proof["status"], "passed")
        self.assertEqual(set(proof["allowed_config_differences"]), {
            "experiment_code", "run_name", "hypothesis", "output_root", "seed",
            "expected_vast_instance_id", "expected_workspace", "expected_gpu", "minimum_gpu_memory_gib",
            "mlflow_state_path", "data_verification",
        })
        self.assertEqual(len(proof["material_identity_sha256"]), 64)
        self.assertEqual(len(proof["proof_sha256"]), 64)

    def test_historical_source_rejects_semantic_and_material_changes(self):
        cases = (
            ("semantic", lambda value: value["config"]["inference"].update(seconds=6.0)),
            ("unknown_config", lambda value: value["config"].update(unregistered_field=True)),
            ("input_hash", lambda value: value["input_hashes"].update(manifest="f" * 64)),
            ("model", lambda value: value["model"].update(weights_sha256="f" * 64)),
            ("code", lambda value: value["code_hashes"].update({"src/model.py": "f" * 64})),
            ("missing_material", lambda value: value.pop("folds")),
            ("labels", lambda value: value["labels"].reverse()),
            ("row_order", lambda value: value["manifest"].append({"audio_file": "b.mp3"})),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                readiness, historical = self.compatible_contracts()
                mutate(historical)
                with self.assertRaises(ValueError):
                    cuda.verify_c002_source_compatibility(readiness, historical)

    def test_exact_config_and_c001_output_are_rejected(self):
        root = Path(__file__).resolve().parents[1]
        suite = deepcopy(cuda.FIXED)
        cuda.validate_cuda_gain_config(suite)
        suite["target_identity"]["vast_instance_id"] = 50079023
        with self.assertRaises(ValueError):
            cuda.validate_cuda_gain_config(suite)
        with self.assertRaises(ValueError):
            cuda.require_c002_path(root, root / "artifacts/training/cpu_gain/C001_partial")

    def test_backend_rejects_cpu_fallback_and_wrong_gpu_identity(self):
        cuda.validate_cuda_backend(backend(), cuda.FIXED)
        for key, value in (("device", "cpu"), ("no_cpu_fallback", False), ("device_name", "A100"),
                           ("free_memory_bytes", 1), ("encoder_updates", 1)):
            candidate = backend()
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                cuda.validate_cuda_backend(candidate, cuda.FIXED)

    def test_backend_capacity_floor_matches_reported_rtx_3090_memory(self):
        candidate = backend()
        cuda.validate_cuda_backend(candidate, cuda.FIXED)
        candidate["visible_total_memory_bytes"] = 23 * 1024 ** 3 - 1
        with self.assertRaisesRegex(ValueError, "declared GPU identity"):
            cuda.validate_cuda_backend(candidate, cuda.FIXED)

    def test_identity_binds_new_launcher_and_declares_no_historical_parity_gate(self):
        contract = {
            "config": {"inference": {"seconds": 180.0, "maximum_windows": 1}},
            "input_hashes": {"manifest": "x"},
            "labels": ["unknown", "speaker"],
            "code_hashes": {"src/x.py": "a" * 64},
        }
        sources = {"assets": {
            "public": {"source_record": {"weights_sha256": "b" * 64}, "config": {"weights_path": "p"}},
            "advanced": {"source_record": {"weights_sha256": "c" * 64}, "config": {"weights_path": "a"}},
        }}
        with patch.object(cuda, "file_sha256", return_value="d" * 64):
            identity = cuda.build_cuda_identity(Path.cwd(), cuda.FIXED, contract, sources, "identity", backend())
            gain = cuda.build_cuda_identity(Path.cwd(), cuda.FIXED, contract, sources, "gain", backend())
        self.assertNotEqual(identity["signature"], gain["signature"])
        self.assertFalse(identity["historical_embedding_parity_required"])
        self.assertIn("scripts/score_gain_cuda.py", identity["code_hashes"])
        body = {key: value for key, value in identity.items() if key != "signature"}
        self.assertEqual(identity["signature"], hashlib.sha256(cuda.canonical(body)).hexdigest())


if __name__ == "__main__":
    unittest.main()
