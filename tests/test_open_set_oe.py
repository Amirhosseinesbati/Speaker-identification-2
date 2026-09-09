"""Focused, synthetic checks for F008's standalone open-set losses."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
import unittest

from speaker_id.adaptation.open_set_oe import (
    energy_margin_oe_loss,
    raw_cosine_logits,
    uniform_oe_loss,
)


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed in this environment")
class OpenSetOELossTests(unittest.TestCase):
    def test_raw_cosine_logits_are_pre_margin_unscaled_and_differentiable(self) -> None:
        import torch

        embeddings = torch.tensor([[3.0, 4.0], [0.0, 5.0]], requires_grad=True)
        head_weight = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-3.0, -4.0]], requires_grad=True
        )
        logits = raw_cosine_logits(embeddings, head_weight)
        expected = torch.tensor([[0.6, 0.8, -1.0], [0.0, 1.0, -0.8]])
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.allclose(logits, expected, atol=1.0e-6, rtol=0.0))
        # The calculation uses exactly the head weights, not AAM's angular
        # margin or scale, and keeps gradients for both trainable components.
        (logits.square().sum()).backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(head_weight.grad).all())

    def test_raw_cosine_is_stable_for_largest_finite_fp32_values(self) -> None:
        import torch

        largest = torch.finfo(torch.float32).max
        embeddings = torch.tensor([[largest, largest]], dtype=torch.float32, requires_grad=True)
        weights = torch.tensor([[largest, 0.0], [0.0, largest]], dtype=torch.float32, requires_grad=True)
        logits = raw_cosine_logits(embeddings, weights)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(logits, torch.full((1, 2), 2.0 ** -0.5), atol=1.0e-6))
        logits.sum().backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(torch.isfinite(weights.grad).all())

    def test_energy_margin_matches_declared_formula_and_reports_plain_diagnostics(self) -> None:
        import torch
        from torch.nn import functional as F

        logits = torch.tensor([[0.0, 0.0], [0.5, -0.5]], requires_grad=True)
        loss, diagnostics = energy_margin_oe_loss(
            logits,
            energy_temperature=0.5,
            minimum_energy=-0.1,
            softplus_temperature=0.25,
        )
        energy = -0.5 * torch.logsumexp(logits / 0.5, dim=1)
        expected = F.softplus((-0.1 - energy) / 0.25).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1.0e-7, rtol=0.0))
        self.assertEqual(diagnostics["objective"], "energy_margin_oe")
        self.assertEqual(diagnostics["batch_size"], 2)
        self.assertEqual(diagnostics["class_count"], 2)
        self.assertEqual(diagnostics["energy_temperature"], 0.5)
        self.assertEqual(diagnostics["minimum_energy"], -0.1)
        self.assertEqual(diagnostics["softplus_temperature"], 0.25)
        self.assertTrue(all(type(value) in (str, int, float) for value in diagnostics.values()))
        self.assertEqual(json.loads(json.dumps(diagnostics, allow_nan=False)), diagnostics)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_energy_margin_pushes_down_high_known_similarity(self) -> None:
        import torch

        diffuse = torch.zeros((1, 2), requires_grad=True)
        concentrated = torch.tensor([[1.0, -1.0]], requires_grad=True)
        diffuse_loss, _ = energy_margin_oe_loss(
            diffuse, energy_temperature=0.5, minimum_energy=0.0, softplus_temperature=0.5
        )
        concentrated_loss, _ = energy_margin_oe_loss(
            concentrated, energy_temperature=0.5, minimum_energy=0.0, softplus_temperature=0.5
        )
        self.assertGreater(concentrated_loss.item(), diffuse_loss.item())
        concentrated_loss.backward()
        self.assertTrue(torch.isfinite(concentrated.grad).all())

    def test_uniform_oe_matches_uniform_cross_entropy_and_penalizes_concentration(self) -> None:
        import torch

        uniform = torch.zeros((2, 4), requires_grad=True)
        concentrated = torch.tensor([[1.0, -1.0, -1.0, -1.0]], requires_grad=True)
        loss, diagnostics = uniform_oe_loss(uniform, temperature=0.25)
        concentrated_loss, _ = uniform_oe_loss(concentrated, temperature=0.25)
        self.assertTrue(math.isclose(loss.item(), math.log(4), rel_tol=0.0, abs_tol=1.0e-6))
        self.assertGreater(concentrated_loss.item(), loss.item())
        self.assertEqual(diagnostics["objective"], "uniform_oe")
        self.assertEqual(diagnostics["class_count"], 4)
        self.assertTrue(math.isclose(diagnostics["mean_uniform_kl"], 0.0, abs_tol=1.0e-6))
        self.assertTrue(math.isclose(diagnostics["mean_entropy"], math.log(4), abs_tol=1.0e-6))
        loss.backward()
        self.assertTrue(torch.isfinite(uniform.grad).all())

    def test_inputs_are_not_mutated_and_cpu_autocast_still_returns_fp32(self) -> None:
        import torch

        embeddings = torch.tensor([[3.0, 4.0], [4.0, 3.0]], requires_grad=True)
        weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        embedding_before, weight_before = embeddings.detach().clone(), weights.detach().clone()
        expected_logits = raw_cosine_logits(embeddings, weights)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            observed_logits = raw_cosine_logits(embeddings, weights)
            loss, diagnostics = uniform_oe_loss(observed_logits, temperature=0.5)
        self.assertEqual(observed_logits.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.allclose(observed_logits, expected_logits, atol=1.0e-7, rtol=0.0))
        self.assertTrue(torch.equal(embeddings.detach(), embedding_before))
        self.assertTrue(torch.equal(weights.detach(), weight_before))
        self.assertEqual(deepcopy(diagnostics), diagnostics)

    def test_raw_cosine_rejects_malformed_finite_and_zero_inputs(self) -> None:
        import torch

        valid_embeddings = torch.ones((1, 2), dtype=torch.float32)
        valid_weights = torch.ones((2, 2), dtype=torch.float32)
        invalid_cases = [
            ("not a tensor", valid_weights, TypeError),
            (valid_embeddings.double(), valid_weights, ValueError),
            (valid_embeddings, valid_weights.double(), ValueError),
            (torch.ones((1, 3)), valid_weights, ValueError),
            (torch.empty((0, 2)), valid_weights, ValueError),
            (torch.zeros((1, 2)), valid_weights, ValueError),
            (valid_embeddings, torch.zeros((2, 2)), ValueError),
            (torch.tensor([[float("nan"), 1.0]]), valid_weights, FloatingPointError),
            (valid_embeddings, torch.tensor([[1.0, float("inf")], [1.0, 1.0]]), FloatingPointError),
            (torch.ones((1, 2), device="meta"), valid_weights, ValueError),
        ]
        for embeddings, weights, error_type in invalid_cases:
            with self.subTest(embeddings=repr(embeddings), weights=repr(weights)):
                with self.assertRaises(error_type):
                    raw_cosine_logits(embeddings, weights)

    def test_losses_reject_non_cosine_malformed_and_invalid_hyperparameters(self) -> None:
        import torch

        valid = torch.zeros((1, 2), dtype=torch.float32)
        malformed = [
            torch.zeros(2),
            torch.zeros((0, 2)),
            torch.zeros((1, 1)),
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([[1.1, 0.0]]),
            torch.tensor([[float("nan"), 0.0]]),
            torch.empty((1, 2), device="meta"),
        ]
        for logits in malformed:
            with self.subTest(logits=repr(logits)):
                with self.assertRaises((ValueError, FloatingPointError)):
                    energy_margin_oe_loss(
                        logits, energy_temperature=0.5, minimum_energy=0.0, softplus_temperature=0.5
                    )
                with self.assertRaises((ValueError, FloatingPointError)):
                    uniform_oe_loss(logits, temperature=0.5)
        for bad in (True, 0.0, -1.0, float("nan"), float("inf"), "0.5"):
            with self.subTest(energy_temperature=bad):
                with self.assertRaises((TypeError, ValueError)):
                    energy_margin_oe_loss(
                        valid, energy_temperature=bad, minimum_energy=0.0, softplus_temperature=0.5
                    )
        for bad in (True, 0.0, -1.0, float("nan"), float("inf"), "0.5"):
            with self.subTest(softplus_temperature=bad):
                with self.assertRaises((TypeError, ValueError)):
                    energy_margin_oe_loss(
                        valid, energy_temperature=0.5, minimum_energy=0.0, softplus_temperature=bad
                    )
        for bad in (True, float("nan"), float("inf"), "0.0"):
            with self.subTest(minimum_energy=bad):
                with self.assertRaises((TypeError, ValueError)):
                    energy_margin_oe_loss(
                        valid, energy_temperature=0.5, minimum_energy=bad, softplus_temperature=0.5
                    )
        for bad in (True, 0.0, -1.0, float("nan"), float("inf"), "0.5"):
            with self.subTest(temperature=bad):
                with self.assertRaises((TypeError, ValueError)):
                    uniform_oe_loss(valid, temperature=bad)


if __name__ == "__main__":
    unittest.main()
