from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from speaker_id.models.campp import file_sha256
from speaker_id.training import candidate_comparison as candidate
from speaker_id.training.runner import write_json

ROOT = Path(__file__).resolve().parents[1]


def complete_control():
    return {"exact_prediction_reproduction": True, "exact_pooled_metrics": True,
        "folds": {str(outer): {key: True for key in ("exact_prediction_reproduction", "exact_metrics_reproduction", "exact_calibration_reproduction")}
                  for outer in (0, 1)}}


def cache_fixture(root):
    cache = root / "candidate_embedding_cache"
    cache.mkdir()
    identity = {"signature": "a" * 64, "weights_sha256": "b" * 64, "embedding_dim": 192}
    manifest, records = [], []
    for i in range(2):
        row = {"audio_file": f"{i}.mp3", "input_sha256": str(i) * 64, "has_nonzero_signal": i == 0}
        manifest.append(row)
        vector = np.zeros(192, dtype=np.float32)
        if i == 0:
            vector[0] = 1
        path = cache / f"{i}.npz"
        np.savez_compressed(path, embedding=vector, valid=i == 0, audio_file=row["audio_file"], audio_sha256=row["input_sha256"],
                            signature=identity["signature"], model_sha256=identity["weights_sha256"], embedding_dim=np.int64(192))
        records.append({"audio_file": row["audio_file"], "audio_sha256": row["input_sha256"], "cache_file": path.name,
                        "cache_sha256": file_sha256(path), "bytes": path.stat().st_size})
    receipt = {"schema_version": 1, "identity": identity, "embedding_dim": 192, "encoder_updates": 0, "file_count": 2, "files": records}
    return cache, identity, manifest, receipt


class CandidateComparisonTests(unittest.TestCase):
    def test_exact_config_freezes_candidate_control_grid_and_manual_condition(self):
        suite = json.loads((ROOT / "configs/train/campp_advanced_scoring.json").read_text())
        candidate.validate_candidate_suite(suite)
        for key, value in (("recipes", suite["recipes"][1:]), ("unknown_weights", [0.0, 1.0]),
                           ("candidate_config", "configs/model/campp.json"), ("execution_policy", "start now")):
            changed = deepcopy(suite)
            changed[key] = value
            with self.assertRaises(ValueError):
                candidate.validate_candidate_suite(changed)
        changed = deepcopy(suite)
        changed["source"]["source_run"] = "artifacts/training/F003_fixture"
        with self.assertRaises(ValueError):
            candidate.validate_candidate_suite(changed)

    def test_candidate_signature_binds_own_model_actual_source_and_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src/speaker_id/candidates").mkdir(parents=True)
            (root / "scripts").mkdir()
            source = root / "src/speaker_id/candidates/campp_advanced.py"
            source.write_text("official loader")
            (root / "scripts/score_candidate.py").write_text("explicit CLI")
            model_path = root / "candidate.json"
            model_path.write_bytes((ROOT / "configs/model/campp_advanced.json").read_bytes())
            contract = {"input_hashes": {"manifest": "data-v1", "folds": "folds-v1", "model_config": "legacy-512"}, "labels": ["unknown", "a"]}
            original = candidate.candidate_identity(root, model_path, contract)
            self.assertEqual(original["embedding_dim"], 192)
            self.assertNotIn("model_config", original["data_input_hashes"])
            self.assertEqual(original["weights_sha256"], original["model"]["weights_sha256"])
            source.write_text("changed loader")
            self.assertNotEqual(candidate.candidate_identity(root, model_path, contract)["signature"], original["signature"])
            source.write_text("official loader")
            changed = deepcopy(contract)
            changed["input_hashes"]["folds"] = "folds-v2"
            self.assertNotEqual(candidate.candidate_identity(root, model_path, changed)["signature"], original["signature"])
            altered_model = json.loads(model_path.read_text())
            altered_model["embedding_dim"] = 512
            write_json(model_path, altered_model)
            with self.assertRaises(ValueError):
                candidate.candidate_identity(root, model_path, contract)

    def test_no_candidate_loading_or_file_access_before_all_exact_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            loader, extractor, verifier = Mock(), Mock(), Mock()
            with patch.object(candidate, "verify_candidate_weights", verifier):
                for missing in ("exact_pooled_metrics", "exact_prediction_reproduction"):
                    control = complete_control()
                    control[missing] = False
                    with self.assertRaisesRegex(ValueError, "must precede"):
                        candidate.extract_candidate_cache(Path(temporary), {}, {}, Path(temporary), Mock(), control,
                                                          loader=loader, extractor=extractor)
                control = complete_control()
                control["folds"]["1"]["exact_calibration_reproduction"] = False
                with self.assertRaises(ValueError):
                    candidate.extract_candidate_cache(Path(temporary), {}, {}, Path(temporary), Mock(), control,
                                                      loader=loader, extractor=extractor)
            loader.assert_not_called()
            extractor.assert_not_called()
            verifier.assert_not_called()

    def test_192_cache_roundtrip_preserves_zero_and_refuses_foreign_signature(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, identity, manifest, receipt = cache_fixture(Path(temporary))
            vectors, valid = candidate.verify_candidate_cache(cache, identity, manifest, receipt)
            self.assertEqual(vectors.shape, (2, 192))
            self.assertEqual(valid.tolist(), [True, False])
            self.assertFalse(vectors[1].any())
            path = cache / "0.npz"
            with np.load(path, allow_pickle=False) as saved:
                contents = {key: saved[key].copy() for key in saved.files}
            contents["signature"] = "legacy-B002-signature"
            np.savez_compressed(path, **contents)
            receipt["files"][0].update(cache_sha256=file_sha256(path), bytes=path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "own audio/model/signature"):
                candidate.verify_candidate_cache(cache, identity, manifest, receipt)

    def test_cache_refuses_changed_bytes_missing_rows_old_dimensions_or_unknown_zero_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache, identity, manifest, receipt = cache_fixture(Path(temporary))
            changed = deepcopy(receipt)
            changed["files"][1] = deepcopy(changed["files"][0])
            with self.assertRaisesRegex(ValueError, "duplicate or missing"):
                candidate.verify_candidate_cache(cache, identity, manifest, changed)
            (cache / "extra.npz").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "complete 192d"):
                candidate.verify_candidate_cache(cache, identity, manifest, receipt)
        for vector, valid, row in ((np.ones(512, dtype=np.float32), True, {"has_nonzero_signal": True}),
                                   (np.ones(192, dtype=np.float32), False, {"has_nonzero_signal": False}),
                                   (np.zeros(192, dtype=np.float32), False, {"has_nonzero_signal": True})):
            with self.assertRaises(ValueError):
                candidate.validate_candidate_vector(vector, valid, row)

    def test_one_extraction_creates_self_contained_cache_and_detects_parameter_mutation(self):
        class Tensor:
            def __init__(self):
                self.values = np.asarray([1.0], dtype=np.float32)
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self.values
        class Encoder:
            training = False
            def __init__(self): self.tensor = Tensor()
            def parameters(self): return []
            def state_dict(self): return {"frozen": self.tensor}
        for mutate in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                weights = root / "weights.bin"
                weights.write_bytes(b"test-only-model-bytes")
                identity = {"signature": "c" * 64, "weights_sha256": file_sha256(weights), "model": {},
                            "inference": {"seconds": 180.0, "maximum_windows": 1}, "embedding_dim": 192}
                contract = {"config": {"data_dir": "data/raw"}, "manifest": [
                    {"audio_file": "one.mp3", "input_sha256": "1" * 64, "has_nonzero_signal": True},
                    {"audio_file": "zero.mp3", "input_sha256": "0" * 64, "has_nonzero_signal": False}]}
                encoder, calls = Encoder(), []
                def extract(model, path, **kwargs):
                    calls.append(path.name)
                    vector = np.zeros(192, dtype=np.float32)
                    valid = path.name != "zero.mp3"
                    if valid: vector[0] = 1
                    if mutate: model.tensor.values[0] += 1
                    return vector, {"nonzero_signal": valid}
                loader = Mock(return_value=encoder)
                output = root / "output"
                with patch.object(candidate, "verify_candidate_weights", return_value=weights):
                    if mutate:
                        with self.assertRaisesRegex(ValueError, "in-memory parameters changed"):
                            candidate.extract_candidate_cache(root, contract, identity, output, Mock(), complete_control(), loader=loader, extractor=extract)
                    else:
                        vectors, valid, receipt = candidate.extract_candidate_cache(root, contract, identity, output, Mock(), complete_control(), loader=loader, extractor=extract)
                        self.assertEqual(vectors.shape, (2, 192))
                        self.assertEqual(receipt["encoder_state_sha256_before"], receipt["encoder_state_sha256_after"])
                        with zipfile.ZipFile(output / "candidate_embedding_cache.zip") as archive:
                            self.assertEqual(set(archive.namelist()), {"candidate_cache_manifest.json", "candidate_embedding_cache/one.npz", "candidate_embedding_cache/zero.npz"})
                        with self.assertRaises(FileExistsError):
                            candidate.extract_candidate_cache(root, contract, identity, output, Mock(), complete_control(), loader=loader, extractor=extract)
                self.assertEqual(calls, ["one.mp3", "zero.mp3"])
                loader.assert_called_once()

    def test_training_or_unfrozen_encoder_is_rejected(self):
        encoder = Mock(training=True)
        with self.assertRaisesRegex(ValueError, "fully frozen"):
            candidate.encoder_state_sha256(encoder)
        encoder.training = False
        encoder.parameters.return_value = [Mock(requires_grad=True)]
        with self.assertRaises(ValueError):
            candidate.encoder_state_sha256(encoder)


if __name__ == "__main__":
    unittest.main()
