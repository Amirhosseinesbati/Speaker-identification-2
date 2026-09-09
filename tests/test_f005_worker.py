"""CPU tensor checks for the exact F005 objective and bounded waveform cache."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.training import f005_worker as worker


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is required by the training environment")
class F005WorkerTests(unittest.TestCase):
    def test_microbatch_objective_equals_full_batch_and_long_aam_has_gradient(self):
        import torch
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.linear = torch.nn.Linear(192, 446, bias=False)
            def forward(self, values, targets): return self.linear(values)
        torch.manual_seed(4)
        short = torch.randn(5, 192, dtype=torch.float32, requires_grad=True)
        long = torch.randn(5, 192, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        eligible = torch.tensor([True, False, True, True, False])
        short_len = torch.full((5,), 48000, dtype=torch.int64)
        long_len = torch.tensor([128000, 48000, 128000, 128000, 48000], dtype=torch.int64)
        arm = {"id": "treatment_mse05", "kind": "dual_aam_consistency",
               "cosine_gamma": .5, "raw_h_mse_lambda": .5}
        head_full = Head(); initial = {k: v.detach().clone() for k, v in head_full.state_dict().items()}
        full, _ = worker.f005_objective(short, long, targets, eligible, short_len, long_len,
                                        head_full, arm, consistency_active=True,
                                        consistency_scale=.37)
        full.backward()
        full_grads = (short.grad.clone(), long.grad.clone(), head_full.linear.weight.grad.clone())

        short2 = short.detach().clone().requires_grad_(); long2 = long.detach().clone().requires_grad_()
        head_micro = Head(); head_micro.load_state_dict(initial)
        pieces = []
        for selected in (slice(0, 2), slice(2, 5)):
            value, _ = worker.f005_objective(
                short2[selected], long2[selected], targets[selected], eligible[selected],
                short_len[selected], long_len[selected], head_micro, arm, consistency_active=True,
                consistency_scale=.37,
                normalization_pairs=5, normalization_eligible_pairs=3)
            pieces.append(value)
        combined = sum(pieces); combined.backward()
        self.assertTrue(torch.allclose(full, combined, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(full_grads[0], short2.grad, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(full_grads[1], long2.grad, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(full_grads[2], head_micro.linear.weight.grad, atol=2e-6, rtol=2e-6))
        self.assertGreater(float(long2.grad.abs().sum()), 0.0, "long AAM must update the long branch")

    def test_consistency_ramp_boundaries_and_effective_coefficients(self):
        import json
        import torch
        from pathlib import Path

        fit = json.loads((Path(__file__).resolve().parents[1] /
                          "configs/train/campp_f005_consistency.json").read_text())["fit"]
        self.assertEqual(worker.consistency_ramp_scale(fit, 599), 0.0)
        self.assertEqual(worker.consistency_ramp_scale(fit, 600), 0.0)
        self.assertAlmostEqual(worker.consistency_ramp_scale(fit, 698), 98 / 99)
        self.assertEqual(worker.consistency_ramp_scale(fit, 699), 1.0)
        self.assertEqual(worker.consistency_ramp_scale(fit, 700), 1.0)
        self.assertEqual(worker.consistency_ramp_scale(fit, 1099), 1.0)

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.linear = torch.nn.Linear(192, 446, bias=False)
            def forward(self, values, targets): return self.linear(values)
        short_h, long_h = torch.randn(2, 192), torch.randn(2, 192)
        targets = torch.tensor([0, 1]); eligible = torch.tensor([True, True])
        short_len, long_len = torch.tensor([3, 3]), torch.tensor([8, 8])
        treatment = {"id": "treatment_mse05", "kind": "dual_aam_consistency",
                     "cosine_gamma": .5, "raw_h_mse_lambda": .5}
        _, diagnostics = worker.f005_objective(
            short_h, long_h, targets, eligible, short_len, long_len, Head(), treatment,
            consistency_active=True, consistency_scale=.5)
        self.assertEqual(diagnostics["base_cosine_gamma"], .5)
        self.assertEqual(diagnostics["base_raw_h_mse_lambda"], .5)
        self.assertEqual(diagnostics["effective_cosine_gamma"], .25)
        self.assertEqual(diagnostics["effective_raw_h_mse_lambda"], .25)
        self.assertEqual(diagnostics["consistency_scale"], .5)

        control = {"id": "control", "kind": "dual_aam_control",
                   "cosine_gamma": 0.0, "raw_h_mse_lambda": 0.0}
        _, control_diagnostics = worker.f005_objective(
            short_h, long_h, targets, eligible, short_len, long_len, Head(), control,
            consistency_active=True, consistency_scale=1.0)
        self.assertEqual(control_diagnostics["effective_cosine_gamma"], 0.0)
        self.assertEqual(control_diagnostics["effective_raw_h_mse_lambda"], 0.0)

    def test_worker_evidence_requires_verified_audio_and_authorized_instance(self):
        base = {
            "source_verification": {},
            "readiness": {"summary": {"audio_hashes_checked": True},
                          "config": {"expected_vast_instance_id": 123}},
        }
        with patch.dict("os.environ", {"VAST_INSTANCE_ID": "123"}, clear=False):
            self.assertEqual(worker._require_worker_execution_evidence(base),
                             {"audio_hashes_checked": True, "vast_instance_id": "123"})
            invalid_audio = {**base, "readiness": {
                "summary": {"audio_hashes_checked": False},
                "config": {"expected_vast_instance_id": 123}}}
            with self.assertRaisesRegex(ValueError, "audio_hashes_checked"):
                worker._require_worker_execution_evidence(invalid_audio)
        with patch.dict("os.environ", {"VAST_INSTANCE_ID": "999"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "Vast instance"):
                worker._require_worker_execution_evidence(base)

    def test_raw_h_target_is_dynamic_stop_gradient_only_for_mse(self):
        import torch
        student = torch.randn(2, 192, requires_grad=True)
        teacher = torch.randn(2, 192, requires_grad=True)
        mask = torch.tensor([True, True]); short = torch.tensor([3, 3]); long = torch.tensor([8, 8])
        loss = worker._raw_h_mse_alignment(student, teacher, mask, short, long)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_waveform_cache_is_bounded_and_immutable_values_survive(self):
        cache = worker._WaveformCache(16)
        first = np.arange(4, dtype=np.float32); first.setflags(write=False)
        second = np.arange(4, dtype=np.float32) + 1; second.setflags(write=False)
        cache.put("first", first); cache.put("second", second)
        self.assertLessEqual(cache.bytes, 16)
        self.assertIsNone(cache.get("first"))
        self.assertFalse(cache.get("second").flags.writeable)


if __name__ == "__main__":
    unittest.main()
