"""Synthetic candidate fixtures; no Torch installation, real model or audio needed."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from speaker_id.candidates import campp_advanced as candidate
from speaker_id.models import campp as historical


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.values, axis))

    def to(self, **kwargs):
        return self

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def float(self):
        return Tensor(self.values.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeTorch(ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.Tensor = Tensor
        self.float32 = np.float32
        self.load = Mock(return_value={"fixture.weight": Tensor([1.0])})
        self.inference_active = False

    def isfinite(self, tensor):
        return np.isfinite(tensor.values)

    @contextmanager
    def inference_mode(self):
        self.inference_active = True
        try:
            yield
        finally:
            self.inference_active = False


class Encoder:
    def __init__(self, torch, output=None):
        self.torch = torch
        self.output = np.arange(1, 193, dtype=np.float32)[None, :] if output is None else output
        self.training = False
        self.load_state_dict = Mock()
        self.requires_grad_ = Mock()
        self.to = Mock(return_value=self)
        self.calls = 0

    def eval(self):
        self.training = False
        return self

    def __call__(self, features):
        if not self.torch.inference_active:
            raise AssertionError("Candidate invoked a model outside inference_mode")
        self.calls += 1
        return Tensor(self.output)


class AdvancedCAMPPTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "configs/model/campp_advanced.json").read_text(encoding="utf-8"))

    def test_candidate_pins_identity_and_cannot_replace_historical_512_contract(self):
        config = self.config()
        candidate.validate_advanced_config(config)
        with self.assertRaisesRegex(ValueError, "architecture/frontend"):
            historical.validate_model_config(config)
        self.assertIs(candidate.read_mono, historical.read_mono)
        self.assertIs(candidate.make_fbank, historical.make_fbank)
        self.assertIs(candidate.crop_waveform, historical.crop_waveform)
        for key, value in (("embedding_dim", 512), ("weights_sha256", "0" * 64),
                           ("weights_bytes", 1), ("public_revision", "master"),
                           ("checkpoint_revision", "0" * 40), ("license", "unknown")):
            changed = deepcopy(config)
            changed[key] = value
            with self.assertRaises(ValueError):
                candidate.validate_advanced_config(changed)

    def test_rejects_changed_architecture_frontend_or_silent_inference_policy(self):
        for section, key, value in (("architecture_kwargs", "growth_rate", 64),
                                    ("architecture_kwargs", "embedding_size", 512),
                                    ("frontend", "vad", True), ("frontend", "gain_normalization", True),
                                    ("inference", "seconds", 6.0), ("inference", "maximum_windows", True)):
            config = self.config()
            config[section][key] = value
            with self.assertRaises(ValueError):
                candidate.validate_advanced_config(config)
        config = self.config()
        config["download_if_missing"] = True
        with self.assertRaises(ValueError):
            candidate.validate_advanced_config(config)

    def test_requires_existing_local_confined_weights(self):
        for path in ("https://host/campplus_cn_en_common.pt", "../campplus_cn_en_common.pt",
                     "artifacts/models/../campplus_cn_en_common.pt", "artifacts/models/other.pt"):
            config = self.config()
            config["weights_path"] = path
            with self.assertRaises(ValueError):
                candidate.validate_advanced_config(config)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "must already exist"):
                candidate.load_advanced(self.config(), Path(temporary))

    def weights_fixture(self, root, *, full_size):
        config = self.config()
        path = root / config["weights_path"]
        path.parent.mkdir(parents=True)
        with path.open("wb") as handle:
            handle.truncate(config["weights_bytes"] if full_size else 16)
        return config, path

    def test_wrong_size_and_same_size_wrong_sha_rejected_before_torch_load(self):
        for full_size in (False, True):
            with self.subTest(full_size=full_size), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = self.weights_fixture(root, full_size=full_size)
                fake = FakeTorch()
                with patch.dict(sys.modules, {"torch": fake}):
                    with self.assertRaisesRegex(ValueError, "size or SHA256"):
                        candidate.load_advanced(config, root)
                fake.load.assert_not_called()

    def test_plain_strict_weights_only_loading_freezes_eval_fp32(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, path = self.weights_fixture(root, full_size=True)
            fake = FakeTorch()
            encoder = Encoder(fake)
            encoder.training = True
            vendor = ModuleType("speaker_id.models.vendor.campplus.DTDNN")
            vendor.CAMPPlus = Mock(return_value=encoder)
            with patch.dict(sys.modules, {"torch": fake, vendor.__name__: vendor}), \
                    patch.object(candidate, "file_sha256", return_value=config["weights_sha256"]):
                actual = candidate.load_advanced(config, root, "cuda:0")
            self.assertIs(actual, encoder)
            fake.load.assert_called_once_with(path.resolve(), map_location="cpu", weights_only=True)
            vendor.CAMPPlus.assert_called_once_with(**candidate.ARCHITECTURE_KWARGS)
            encoder.load_state_dict.assert_called_once_with(fake.load.return_value, strict=True)
            encoder.to.assert_called_once_with(device="cuda:0", dtype=np.float32)
            encoder.requires_grad_.assert_called_once_with(False)
            self.assertFalse(encoder.training)
            self.assertEqual(encoder.calls, 0)

    def test_wrapped_or_nonfinite_state_is_not_silently_unwrapped(self):
        for state in ({"state_dict": {}}, {"bad": Tensor([np.nan])}, {}):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = self.weights_fixture(root, full_size=True)
                fake = FakeTorch()
                fake.load.return_value = state
                vendor = ModuleType("speaker_id.models.vendor.campplus.DTDNN")
                vendor.CAMPPlus = Mock()
                with patch.dict(sys.modules, {"torch": fake, vendor.__name__: vendor}), \
                        patch.object(candidate, "file_sha256", return_value=config["weights_sha256"]):
                    with self.assertRaisesRegex(ValueError, "finite plain tensor"):
                        candidate.load_advanced(config, root)
                vendor.CAMPPlus.assert_not_called()

    def test_zero_audio_is_exact_192_fallback_without_features_or_model(self):
        encoder = Mock(side_effect=AssertionError("zero audio reached model"))
        with patch.object(candidate, "read_mono", return_value=np.zeros(17, dtype=np.float32)), \
                patch.object(candidate, "make_fbank") as fbank, patch.dict(sys.modules, {"torch": None}):
            vector, info = candidate.extract_advanced_embedding(encoder, Path("zero.wav"), device="cpu")
        self.assertEqual(vector.shape, (192,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertFalse(vector.any())
        self.assertEqual(info["window_count"], 0)
        self.assertFalse(info["nonzero_signal"])
        fbank.assert_not_called()
        encoder.assert_not_called()

    def test_single_view_centers_at_180s_and_returns_finite_fp32_unit_vector(self):
        fake = FakeTorch()
        encoder = Encoder(fake)
        signal = np.arange(181 * 16000, dtype=np.float32)
        seen = []
        def fbank(crop):
            seen.append(crop.copy())
            return Tensor(np.ones((10, 80), dtype=np.float32))
        with patch.dict(sys.modules, {"torch": fake}), patch.object(candidate, "read_mono", return_value=signal), \
                patch.object(candidate, "make_fbank", side_effect=fbank):
            vector, info = candidate.extract_advanced_embedding(encoder, Path("synthetic.wav"), device="cpu")
        self.assertEqual(encoder.calls, 1)
        self.assertEqual(seen[0].shape, (180 * 16000,))
        self.assertEqual(seen[0][0], 8000)
        self.assertEqual(info, {"nonzero_signal": True, "seconds": 181.0, "window_count": 1})
        self.assertEqual(vector.shape, (192,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_rejects_wrong_dimension_nonfinite_zero_output_and_training_encoder(self):
        fake = FakeTorch()
        for output in (np.ones((1, 512)), np.ones((192,)), np.full((1, 192), np.nan), np.zeros((1, 192))):
            with self.subTest(shape=output.shape), patch.dict(sys.modules, {"torch": fake}), \
                    patch.object(candidate, "read_mono", return_value=np.ones(16000, dtype=np.float32)), \
                    patch.object(candidate, "make_fbank", return_value=Tensor(np.ones((10, 80)))):
                with self.assertRaises(ValueError):
                    candidate.extract_advanced_embedding(Encoder(fake, output), Path("fixture.wav"), device="cpu")
        encoder = Encoder(fake)
        encoder.training = True
        with patch.dict(sys.modules, {"torch": fake}), patch.object(candidate, "read_mono", return_value=np.ones(16000)):
            with self.assertRaisesRegex(ValueError, "frozen eval"):
                candidate.extract_advanced_embedding(encoder, Path("fixture.wav"), device="cpu")
        self.assertEqual(encoder.calls, 0)

    def test_normalization_matches_historical_single_view_fp32_arithmetic(self):
        fake = FakeTorch()
        raw = np.zeros(192, dtype=np.float32)
        raw[:3] = 1.0
        first_unit = raw / np.linalg.norm(raw)
        expected = (first_unit / np.linalg.norm(first_unit)).astype(np.float32)
        previous_float64_once = (raw / float(np.linalg.norm(raw.astype(np.float64)))).astype(np.float32)
        self.assertFalse(np.array_equal(expected, previous_float64_once), "Fixture must distinguish the removed precision policy")
        with patch.dict(sys.modules, {"torch": fake}), \
                patch.object(candidate, "read_mono", return_value=np.ones(16000, dtype=np.float32)), \
                patch.object(candidate, "make_fbank", return_value=Tensor(np.ones((10, 80)))):
            actual, _ = candidate.extract_advanced_embedding(Encoder(fake, raw[None, :]), Path("fixture.wav"), device="cpu")
        np.testing.assert_array_equal(actual, expected)

    def test_inference_policy_rejected_before_audio_read(self):
        for kwargs in ({"seconds": 6.0}, {"seconds": float("nan")}, {"maximum_windows": 3}, {"maximum_windows": True}):
            with patch.object(candidate, "read_mono") as decode:
                with self.assertRaises(ValueError):
                    candidate.extract_advanced_embedding(None, Path("fixture.wav"), device="cpu", **kwargs)
                decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
