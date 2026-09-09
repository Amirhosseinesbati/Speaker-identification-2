"""CPU-only contracts for the isolated L2-SP anchor and diagnostics."""
import json
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from speaker_id.adaptation.l2sp import (
    build_l2sp_anchor,
    l2sp_gradient_norm_ratio,
    l2sp_penalty,
)


_ModuleBase = torch.nn.Module if torch is not None else object


@unittest.skipIf(torch is None, "CPU Torch is available in the separate release QA environment")
class L2SPTests(unittest.TestCase):
    class TinyEncoder(_ModuleBase):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor([10.0]))
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.frozen = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)

    class OneParameterEncoder(_ModuleBase):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def test_anchor_selects_trainable_names_and_receipt_is_deterministic_nonsecret(self):
        first = self.TinyEncoder()
        anchor = build_l2sp_anchor(first, exclude_names={"bias"})
        receipt = anchor.receipt
        self.assertEqual(anchor.names, ("weight",))
        self.assertEqual(receipt["parameter_count"], 1)
        self.assertEqual(receipt["parameter_names"], ["weight"])
        self.assertEqual(receipt["parameter_shapes"], {"weight": [2]})
        self.assertEqual(receipt["parameter_dtypes"], {"weight": "float32"})
        self.assertEqual(receipt["anchor_byte_count"], 8)
        self.assertEqual(len(receipt["anchor_bytes_sha256"]), 64)
        json.dumps(receipt, allow_nan=False)
        self.assertNotIn("1.0", json.dumps(receipt))
        self.assertEqual(receipt, build_l2sp_anchor(self.TinyEncoder(), exclude_names={"bias"}).receipt)
        changed = self.TinyEncoder()
        with torch.no_grad():
            changed.weight.add_(1.0)
        changed_receipt = build_l2sp_anchor(changed, exclude_names={"bias"}).receipt
        self.assertNotEqual(receipt["anchor_bytes_sha256"], changed_receipt["anchor_bytes_sha256"])
        self.assertFalse(anchor.references["weight"].requires_grad)

    def test_penalty_is_exact_half_sum_and_has_live_gradient(self):
        encoder = self.TinyEncoder()
        anchor = build_l2sp_anchor(encoder, exclude_names={"bias"})
        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([3.0, 4.0]))
        loss = l2sp_penalty(encoder, anchor)
        self.assertEqual(loss.item(), 4.0)
        loss.backward()
        torch.testing.assert_close(encoder.weight.grad, torch.tensor([2.0, 2.0]))
        self.assertIsNone(anchor.references["weight"].grad)

    def test_inventory_shape_and_nonfinite_changes_are_rejected(self):
        def set_nonfinite(model):
            with torch.no_grad():
                model.weight.fill_(float("nan"))

        for mutation, message, error_type in (
            (lambda model: delattr(model, "weight"), "missing anchored", ValueError),
            (lambda model: setattr(
                model, "weight", torch.nn.Parameter(torch.ones(3))
            ), "shape changed", ValueError),
            (set_nonfinite, "nonfinite", FloatingPointError),
        ):
            with self.subTest(message=message):
                encoder = self.TinyEncoder()
                anchor = build_l2sp_anchor(encoder, exclude_names={"bias"})
                mutation(encoder)
                with self.assertRaisesRegex(error_type, message):
                    l2sp_penalty(encoder, anchor)

        encoder = self.OneParameterEncoder()
        anchor = build_l2sp_anchor(encoder)
        encoder.new_weight = torch.nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(ValueError, "unexpected unanchored"):
            l2sp_penalty(encoder, anchor)

    def test_unknown_exclusions_and_nonfinite_anchor_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            build_l2sp_anchor(self.OneParameterEncoder(), exclude_names={"typo"})
        encoder = self.OneParameterEncoder()
        with torch.no_grad():
            encoder.weight.fill_(float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            build_l2sp_anchor(encoder)

    def test_gradient_ratio_preserves_grad_fields_and_autograd_graph(self):
        encoder = self.OneParameterEncoder()
        anchor = build_l2sp_anchor(encoder)
        with torch.no_grad():
            encoder.weight.fill_(2.0)
        task_loss = 0.5 * (encoder.weight - 4.0).square().sum()
        sp_loss = l2sp_penalty(encoder, anchor)
        encoder.weight.grad = torch.tensor([123.0])
        info = l2sp_gradient_norm_ratio(
            task_loss, sp_loss, encoder, anchor, lambda_sp=0.5
        )
        self.assertEqual(info["task_gradient_norm"], 2.0)
        self.assertEqual(info["l2sp_gradient_norm"], 1.0)
        self.assertEqual(info["weighted_l2sp_gradient_norm"], 0.5)
        self.assertEqual(info["task_to_weighted_l2sp_ratio"], 4.0)
        self.assertEqual(info["ratio_status"], "finite")
        torch.testing.assert_close(encoder.weight.grad, torch.tensor([123.0]))
        encoder.weight.grad = None
        (task_loss + 0.5 * sp_loss).backward()
        torch.testing.assert_close(encoder.weight.grad, torch.tensor([-1.5]))

    def test_zero_weighted_regularizer_has_explicit_noninfinite_ratio(self):
        encoder = self.OneParameterEncoder()
        anchor = build_l2sp_anchor(encoder)
        task_loss = (encoder.weight - 2.0).square().sum()
        sp_loss = l2sp_penalty(encoder, anchor)
        info = l2sp_gradient_norm_ratio(task_loss, sp_loss, encoder, anchor, lambda_sp=0.0)
        self.assertEqual(info["weighted_l2sp_gradient_norm"], 0.0)
        self.assertIsNone(info["task_to_weighted_l2sp_ratio"])
        self.assertEqual(info["ratio_status"], "weighted_l2sp_gradient_zero")


if __name__ == "__main__":
    unittest.main()
