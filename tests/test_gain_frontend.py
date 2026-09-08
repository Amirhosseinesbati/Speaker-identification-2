"""Synthetic waveforms/fake encoders only; no real audio, weights or model fit."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from speaker_id.audio.gain import GAIN_POLICY, IDENTITY_POLICY, apply_gain, validate_gain_policy
from speaker_id.candidates import gain_frontend as candidate
from speaker_id.candidates import campp_advanced as historical_advanced
from speaker_id.models import campp as historical_public


class Tensor:
    def __init__(self, values): self.values = np.asarray(values)
    @property
    def shape(self): return self.values.shape
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.values, axis))
    def to(self, *args, **kwargs): return self
    def __getitem__(self, index): return Tensor(self.values[index])
    def float(self): return Tensor(self.values.astype(np.float32))
    def cpu(self): return self
    def numpy(self): return self.values


class FakeTorch(ModuleType):
    def __init__(self):
        super().__init__('torch'); self.float32 = np.float32; self.active = False
    def isfinite(self, value): return np.isfinite(value.values)
    @contextmanager
    def inference_mode(self):
        before = self.active; self.active = True
        try: yield
        finally: self.active = before


class Encoder:
    def __init__(self, torch, dimension):
        self.torch = torch; self.training = False
        self.parameter = SimpleNamespace(requires_grad=False, dtype=np.float32)
        self.child = SimpleNamespace(training=False)
        self.output = np.zeros((1, dimension), dtype=np.float32)
        self.output[0, :3] = 1.0
        self.seen = []
    def modules(self): return iter([self, self.child])
    def parameters(self): return iter([self.parameter])
    def buffers(self): return iter([])
    def __call__(self, features):
        if not self.torch.active: raise AssertionError('Model invoked outside inference_mode')
        self.seen.append(features)
        return Tensor(self.output)


class GainTests(unittest.TestCase):
    def test_exact_policy_no_unrecorded_constants_or_types(self):
        validate_gain_policy(IDENTITY_POLICY); validate_gain_policy(GAIN_POLICY)
        for key, value in [('target_rms_dbfs', -10.0), ('maximum_gain_db', 40.0),
                           ('peak_ceiling', 1.0), ('clip', True), ('schema_version', True)]:
            changed = deepcopy(GAIN_POLICY); changed[key] = value
            with self.assertRaises(ValueError): validate_gain_policy(changed)
        changed = deepcopy(GAIN_POLICY); changed['dataset_rms'] = .1
        with self.assertRaises(ValueError): validate_gain_policy(changed)

    def test_identity_and_zero_preserve_exact_bytes_including_signed_zero(self):
        for signal, policy in [(np.asarray([-.2, .01, 1.2], dtype=np.float32), IDENTITY_POLICY),
                               (np.asarray([0., -0., 0.], dtype=np.float32), GAIN_POLICY)]:
            before = signal.tobytes(); result, info = apply_gain(signal, policy)
            self.assertIs(result, signal); self.assertEqual(result.tobytes(), before)
            self.assertEqual(info['gain'], 1.0); self.assertFalse(info['applied'])
        self.assertIsNone(info['before']['rms_dbfs'])

    def test_target_gain_is_one_fp32_multiply_without_input_mutation(self):
        signal = np.tile(np.asarray([-.001, .001], dtype=np.float32), 400)
        before = signal.copy(); result, info = apply_gain(signal, GAIN_POLICY)
        np.testing.assert_array_equal(result, np.multiply(before, np.float32(info['gain']), dtype=np.float32))
        np.testing.assert_array_equal(signal, before)
        self.assertEqual(result.dtype, np.float32); self.assertFalse(np.shares_memory(result, signal))
        self.assertAlmostEqual(info['after']['rms'], .1, places=7)
        self.assertEqual(info['limiting_reason'], 'target_rms')

    def test_maximum_gain_and_peak_bound_are_both_enforced(self):
        quiet = np.full(100, np.float32(1e-30), dtype=np.float32)
        result, info = apply_gain(quiet, GAIN_POLICY)
        self.assertEqual(info['gain'], 1000.0); self.assertEqual(info['limiting_reason'], 'maximum_gain')
        self.assertTrue(np.all(result > quiet))
        transient = np.zeros(10000, dtype=np.float32); transient[0] = np.float32(.3)
        result, info = apply_gain(transient, GAIN_POLICY)
        self.assertEqual(info['limiting_reason'], 'peak_ceiling')
        self.assertLessEqual(float(np.max(np.abs(result))), .95)
        self.assertGreater(float(result[0]), .94)
        self.assertEqual(np.count_nonzero(result), 1)

    def test_loud_or_full_peak_waveform_never_attenuates_or_clips(self):
        for signal in [np.full(64, .2, dtype=np.float32),
                       np.asarray([1.1] + [0.] * 5000, dtype=np.float32)]:
            result, info = apply_gain(signal, GAIN_POLICY)
            self.assertIs(result, signal); self.assertFalse(info['applied'])
            self.assertEqual(info['before'], info['after'])

    def test_float32_rounding_never_exceeds_fixed_bound(self):
        for peak in np.linspace(.001, .949, 101, dtype=np.float32):
            signal = np.zeros(10000, dtype=np.float32); signal[0] = peak
            result, info = apply_gain(signal, GAIN_POLICY)
            self.assertLessEqual(float(np.max(np.abs(result))), .95)
            self.assertLessEqual(info['gain'], info['gain_limit_float64'])
            self.assertGreaterEqual(info['gain'], 1.0)

    def test_malformed_input_rejected_without_silent_conversion(self):
        for signal in [np.ones(4, dtype=np.float64), np.ones((4, 2), dtype=np.float32),
                       np.empty(0, dtype=np.float32), np.asarray([np.nan], dtype=np.float32),
                       np.asarray([np.inf], dtype=np.float32), [1., 2.]]:
            with self.assertRaises(ValueError): apply_gain(signal, GAIN_POLICY)


class GainFrontendTests(unittest.TestCase):
    def test_zero_short_circuits_features_torch_and_both_encoders(self):
        with patch.object(candidate, 'read_mono', return_value=np.zeros(5, dtype=np.float32)), \
             patch.object(candidate, 'make_fbank') as features, patch.dict(sys.modules, {'torch': None}):
            vectors, info = candidate.extract_gain_pair(None, None, Path('zero.wav'), device='cpu', policy=GAIN_POLICY)
        self.assertFalse(info['nonzero_signal']); self.assertEqual(info['window_count'], 0)
        self.assertEqual({k: v.shape for k, v in vectors.items()}, {'public': (512,), 'advanced': (192,)})
        self.assertTrue(all(v.dtype == np.float32 and not v.any() for v in vectors.values()))
        features.assert_not_called()

    def test_control_matches_both_original_extractors_bit_for_bit(self):
        torch = FakeTorch(); public, advanced = Encoder(torch, 512), Encoder(torch, 192)
        signal = np.linspace(-.04, .05, 16001, dtype=np.float32)
        features = Tensor(np.ones((99, 80), dtype=np.float32))
        with patch.dict(sys.modules, {'torch': torch}), \
             patch.object(candidate, 'read_mono', return_value=signal) as read, \
             patch.object(candidate, 'make_fbank', return_value=features) as make:
            vectors, info = candidate.extract_gain_pair(public, advanced, Path('fake.wav'), device='cpu', policy=IDENTITY_POLICY)
        self.assertEqual(read.call_count, 1); self.assertEqual(make.call_count, 1)
        self.assertIs(public.seen[0], advanced.seen[0])
        np.testing.assert_array_equal(make.call_args.args[0], signal)
        for name, module, encoder, method in [('public', historical_public, public, historical_public.extract_embedding),
                ('advanced', historical_advanced, advanced, historical_advanced.extract_advanced_embedding)]:
            with patch.dict(sys.modules, {'torch': torch}), patch.object(module, 'read_mono', return_value=signal), \
                 patch.object(module, 'make_fbank', return_value=features):
                original, _ = method(encoder, Path('fake.wav'), device='cpu', seconds=180., maximum_windows=1)
            np.testing.assert_array_equal(vectors[name], original)
        self.assertEqual(info['window_count'], 1)

    def test_gain_measures_original_after_decode_before_minimum_padding(self):
        torch = FakeTorch(); encoders = Encoder(torch, 512), Encoder(torch, 192)
        signal = np.full(80, .001, dtype=np.float32)
        with patch.dict(sys.modules, {'torch': torch}), patch.object(candidate, 'read_mono', return_value=signal), \
             patch.object(candidate, 'make_fbank', return_value=Tensor(np.ones((99, 80), dtype=np.float32))) as features:
            _, info = candidate.extract_gain_pair(*encoders, Path('fake.wav'), device='cpu', policy=GAIN_POLICY)
        view = features.call_args.args[0]
        self.assertEqual(len(view), 16000); self.assertEqual(info['gain']['sample_count'], 80)
        self.assertAlmostEqual(float(view[0]), .1, places=7); self.assertFalse(np.any(view[80:]))

    def test_policy_validation_precedes_audio_and_model_checks_precede_features(self):
        with patch.object(candidate, 'read_mono') as read:
            for kwargs in [{'seconds': 6.}, {'maximum_windows': True}, {'maximum_windows': 2}]:
                with self.assertRaises(ValueError):
                    candidate.extract_gain_pair(None, None, Path('fake.wav'), device='cpu', policy=GAIN_POLICY, **kwargs)
            read.assert_not_called()
        torch = FakeTorch()
        for change in [lambda e: setattr(e, 'training', True), lambda e: setattr(e.child, 'training', True),
                       lambda e: setattr(e.parameter, 'requires_grad', True), lambda e: setattr(e.parameter, 'dtype', np.float64)]:
            public, advanced = Encoder(torch, 512), Encoder(torch, 192); change(advanced)
            with patch.dict(sys.modules, {'torch': torch}), patch.object(candidate, 'read_mono', return_value=np.ones(16000, dtype=np.float32)), \
                 patch.object(candidate, 'make_fbank') as features:
                with self.assertRaises(ValueError):
                    candidate.extract_gain_pair(public, advanced, Path('fake.wav'), device='cpu', policy=GAIN_POLICY)
                features.assert_not_called(); self.assertFalse(public.seen)

    def test_invalid_encoder_embedding_is_fatal(self):
        torch = FakeTorch()
        for invalid in [np.ones((1, 191), dtype=np.float32), np.zeros((1, 192), dtype=np.float32),
                        np.full((1, 192), np.nan, dtype=np.float32)]:
            public, advanced = Encoder(torch, 512), Encoder(torch, 192); advanced.output = invalid
            with patch.dict(sys.modules, {'torch': torch}), patch.object(candidate, 'read_mono', return_value=np.ones(16000, dtype=np.float32)), \
                 patch.object(candidate, 'make_fbank', return_value=Tensor(np.ones((99, 80), dtype=np.float32))):
                with self.assertRaises(ValueError):
                    candidate.extract_gain_pair(public, advanced, Path('fake.wav'), device='cpu', policy=GAIN_POLICY)


if __name__ == '__main__': unittest.main()
