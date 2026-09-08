"""Small C002 worker safety tests with synthetic vectors only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from speaker_id.audio.gain import GAIN_POLICY, IDENTITY_POLICY
from speaker_id.training import cuda_pair_worker as worker


def vectors(valid):
    value = {name: np.zeros(dimension, dtype=np.float32) for name, dimension in worker.DIMS.items()}
    if valid:
        for item in value.values():
            item[0] = 1
    return value


class CUDAWorkerTests(unittest.TestCase):
    def test_nonidentical_historical_embeddings_are_diagnostic_not_a_gate(self):
        fresh, historical = vectors(True), vectors(True)
        historical["public"][:] = 0
        historical["public"][1] = 1
        result = worker.historical_diagnostic(fresh, historical, True)
        self.assertGreater(result["public"]["l2"], 1)
        summary = worker.summarize_historical_diagnostics([result])
        self.assertTrue(summary["public"]["diagnostic_only_no_acceptance_threshold"])
        self.assertFalse(summary["public"]["historical_embedding_parity_required"])

    def test_c001_layout_and_existing_cache_are_never_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            c001 = root / "artifacts/training/cpu_gain/C001_old"
            c001.mkdir(parents=True)
            with self.assertRaises(ValueError):
                worker._require_fresh_c002_layout(root, c001)
            c002 = root / "artifacts/training/cuda_gain_c002/C002_new"
            c002.mkdir(parents=True)
            (c002 / "identity_embedding_cache").mkdir()
            with self.assertRaises(ValueError):
                worker._require_fresh_c002_layout(root, c002)

    def test_noop_gain_requires_exact_fresh_pair_and_publish_never_overwrites(self):
        identity, gain = vectors(True), vectors(True)
        info = {"gain": {"applied": False, "gain": 1.0}}
        worker.require_noop_equal(identity, gain, info)
        gain["public"][1] = 0.5
        with self.assertRaises(ValueError):
            worker.require_noop_equal(identity, gain, info)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "x.npz"
            path.write_bytes(b"original")
            with self.assertRaises(ValueError):
                worker.publish_npz(path, vectors(True), True, {"audio_file": "x.wav", "input_sha256": "a" * 64}, "b" * 64)
            self.assertEqual(path.read_bytes(), b"original")

    def test_cuda_vector_contract_rejects_bad_policy_or_zero_vector(self):
        valid = vectors(True)
        info = {"nonzero_signal": True, "gain": {"policy": IDENTITY_POLICY["name"], "applied": False, "gain": 1.0}}
        worker.validate_pair(valid, info, True, IDENTITY_POLICY)
        info["gain"]["policy"] = GAIN_POLICY["name"]
        with self.assertRaises(ValueError):
            worker.validate_pair(valid, info, True, IDENTITY_POLICY)


if __name__ == "__main__":
    unittest.main()
