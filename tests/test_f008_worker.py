"""Focused synthetic checks for the F008 open-set tail core.

No audio, source checkpoint, CUDA device, tracker, or server is required here.
The tests lock the two facts easiest to get wrong in an OE tail: unknowns never
receive an AAM target, and eight four-pair microbatches have exactly the same
loss/gradient as one 32-pair OE batch.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from speaker_id.training.f008_config import config_signature, load_f008_config
from speaker_id.training.f008_energy_margin_plan import build_energy_margin_plan
from speaker_id.training.f008_preflight import build_preflight_receipt
from speaker_id.training.f008_protocol import role_pools
from speaker_id.training.f008_worker import (
    F008_GRADIENT_PROBE_SCHEMA,
    F008_TAIL_CHECKPOINT_SCHEMA,
    _probe_receipt,
    _streaming_oe_contribution,
    _unknown_microbatch,
    f008_arm,
    f008_combined_objective,
    f008_oe_loss,
    f008_tail_checkpoint_metadata,
    f008_tail_identity,
    validate_f008_resume_payload,
)


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _config() -> dict:
    return load_f008_config(ROOT / "configs/train/campp_f008_unknown_oe.json")


def _fit() -> dict:
    return json.loads((ROOT / "configs/train/campp_f005_consistency.json").read_text(encoding="utf-8"))["fit"]


def _role(flag: str, index: int, *, speaker: str) -> dict:
    flags = {
        "known_enrollment": (True, True, False, False),
        "unknown_development": (True, False, False, False),
        "known_calibration_query": (False, False, True, False),
        "unknown_calibration_query": (False, False, True, False),
        "outer_validation": (False, False, False, True),
    }[flag]
    return {
        "outer_fold": 0,
        "audio_file": f"{flag}_{index:03d}.wav",
        "speaker_id": speaker,
        "group_id": f"{flag}_group_{index:03d}",
        "role": flag,
        "encoder_fit_allowed": flags[0],
        "enrollment_allowed": flags[1],
        "calibration_query": flags[2],
        "outer_evaluation_included": flags[3],
        "source_train_eligible": True,
    }


def _pools() -> dict:
    rows = [
        *[_role("known_enrollment", index, speaker=f"speaker_{index}") for index in range(32)],
        *[_role("unknown_development", index, speaker="unknown") for index in range(32)],
        _role("known_calibration_query", 0, speaker="not_read_by_pool"),
        _role("unknown_calibration_query", 0, speaker="not_read_by_pool"),
        _role("outer_validation", 0, speaker="not_read_by_pool"),
    ]
    return role_pools(rows, 0)


def _identity(*, known_hash: str = "b" * 64, unknown_hash: str = "c" * 64) -> tuple[dict, dict]:
    config = _config()
    pools = _pools()
    signature = config_signature(config)
    source = {"source": "synthetic-receipt"}
    with tempfile.TemporaryDirectory() as directory:
        shared = Path(directory) / "shared_head.pt"
        shared.write_bytes(b"synthetic-shared-head")
        preflight = build_preflight_receipt(
            f008_signature=signature,
            f005_signature=config["source_f005"]["experiment_signature"],
            f005_source_receipt=source,
            outer_fold=0,
            role_pool=pools,
            shared_head_checkpoint=shared,
            energy_margin_config=config["energy_margin"],
            known_energies=[-0.3 + index * 0.001 for index in range(64)],
            unknown_energies=[0.0 + index * 0.001 for index in range(64)],
            source_view_algorithm="synthetic-paired-views",
        )
        identity = f008_tail_identity(
            config,
            {"signature": config["source_f005"]["experiment_signature"], "config": {"fit": _fit()}},
            0,
            f008_arm(config, "energy_005"),
            shared_head_checkpoint_sha256=hashlib.sha256(shared.read_bytes()).hexdigest(),
            known_tail_plan_sha256=known_hash,
            unknown_plan_sha256=unknown_hash,
            role_pool=pools,
            preflight_receipt=preflight,
            source_receipt=source,
        )
    return identity, _fit()


class F008WorkerContractTests(unittest.TestCase):
    def test_only_declared_oe_arms_are_trainable(self) -> None:
        config = _config()
        self.assertEqual(f008_arm(config, "uniform_005")["lambda"], 0.05)
        with self.assertRaisesRegex(ValueError, "only one declared OE arm"):
            f008_arm(config, "control_f005")

    def test_identity_binds_both_tail_plans_and_preflight(self) -> None:
        first, _ = _identity()
        changed_known, _ = _identity(known_hash="d" * 64)
        changed_unknown, _ = _identity(unknown_hash="e" * 64)
        self.assertNotEqual(first["signature"], changed_known["signature"])
        self.assertNotEqual(first["signature"], changed_unknown["signature"])
        self.assertEqual(first["unknown_plan_step_range"], {"start": 600, "stop": 1100})
        self.assertEqual(first["oe_full_batch_views"], 64)

    def test_resume_metadata_rejects_changed_unknown_plan(self) -> None:
        identity, fit = _identity()
        metadata = f008_tail_checkpoint_metadata(identity, fit, 700)
        self.assertEqual(metadata["checkpoint_schema"], F008_TAIL_CHECKPOINT_SCHEMA)
        payload = {
            "metadata": metadata, "encoder": {}, "head": {}, "optimizer": {},
            "torch_rng": object(), "cuda_rng": object(),
        }
        self.assertEqual(validate_f008_resume_payload(payload, identity, fit), metadata)
        changed, _ = _identity(unknown_hash="f" * 64)
        with self.assertRaisesRegex(ValueError, "another arm, fold, source, plan, preflight or step"):
            validate_f008_resume_payload(payload, changed, fit)

    def test_unknown_microbatch_rejects_aam_target_before_audio(self) -> None:
        contract = {
            "config": {"views": {"short_seconds": 3.0, "long_seconds": 8.0, "sample_rate": 16000}},
            "readiness": {"config": {"data_dir": "unused"}},
        }
        row = {"slot": 0, "audio_file": "unknown.wav", "group_id": "g", "crop_seed": 1,
               "stream": "unknown_oe", "target": 0}
        with self.assertRaisesRegex(ValueError, "may not contain an AAM target"):
            _unknown_microbatch(contract, [row], ROOT)


@unittest.skipUnless(HAS_TORCH, "torch is not installed in this environment")
class F008WorkerTorchTests(unittest.TestCase):
    @staticmethod
    def _margin() -> tuple[dict, dict]:
        config = _config()
        return config["energy_margin"], build_energy_margin_plan(
            [-0.4, -0.35, -0.25, -0.1], [0.1, 0.2, 0.3], config["energy_margin"],
        )

    def test_energy_streaming_loss_and_gradients_match_one_full_32_pair_loss(self) -> None:
        import torch

        torch.manual_seed(91)
        config = _config()
        settings, margin = self._margin()
        arm = f008_arm(config, "energy_005")
        known = torch.tanh(torch.randn(64, 446, dtype=torch.float32)).requires_grad_()
        unknown = torch.tanh(torch.randn(64, 446, dtype=torch.float32)).requires_grad_()
        full, _ = f008_oe_loss(known, unknown, arm, energy_margin_config=settings,
                               energy_margin_plan=margin)
        full_gradients = torch.autograd.grad(full, (known, unknown), retain_graph=False)

        streamed_known = known.detach().clone().requires_grad_()
        streamed_unknown = unknown.detach().clone().requires_grad_()
        contributions = []
        for start in range(0, 64, 8):
            known_part, _ = _streaming_oe_contribution(
                streamed_known[start:start + 8], arm, stream="known", total_unknown_views=64,
                energy_margin_config=settings, energy_margin_plan=margin,
            )
            unknown_part, _ = _streaming_oe_contribution(
                streamed_unknown[start:start + 8], arm, stream="unknown", total_unknown_views=64,
                energy_margin_config=settings, energy_margin_plan=margin,
            )
            contributions.extend((known_part, unknown_part))
        streamed = sum(contributions[1:], contributions[0])
        streamed_gradients = torch.autograd.grad(streamed, (streamed_known, streamed_unknown))
        self.assertTrue(torch.allclose(full, streamed, rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.allclose(full_gradients[0], streamed_gradients[0], rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.allclose(full_gradients[1], streamed_gradients[1], rtol=1e-6, atol=1e-7))

    def test_uniform_streaming_loss_and_gradients_match_one_full_32_pair_loss(self) -> None:
        import torch

        torch.manual_seed(92)
        config = _config()
        settings, margin = self._margin()
        arm = f008_arm(config, "uniform_005")
        unknown = torch.tanh(torch.randn(64, 446, dtype=torch.float32)).requires_grad_()
        full, _ = f008_oe_loss(None, unknown, arm, energy_margin_config=settings,
                               energy_margin_plan=margin)
        (full_gradient,) = torch.autograd.grad(full, (unknown,), retain_graph=False)
        streamed_unknown = unknown.detach().clone().requires_grad_()
        parts = [
            _streaming_oe_contribution(
                streamed_unknown[start:start + 8], arm, stream="unknown", total_unknown_views=64,
                energy_margin_config=settings, energy_margin_plan=margin,
            )[0]
            for start in range(0, 64, 8)
        ]
        streamed = sum(parts[1:], parts[0])
        (streamed_gradient,) = torch.autograd.grad(streamed, (streamed_unknown,))
        self.assertTrue(torch.allclose(full, streamed, rtol=1e-6, atol=1e-7))
        self.assertTrue(torch.allclose(full_gradient, streamed_gradient, rtol=1e-6, atol=1e-7))

    def test_combined_objective_uses_both_unknown_views_without_unknown_targets(self) -> None:
        import torch
        from speaker_id.training.fit import AAMHead

        torch.manual_seed(93)
        config = _config()
        settings, margin = self._margin()
        head = AAMHead(embedding_dim=192, classes=446, margin=0.2, scale=30.0)
        known_short = torch.randn(4, 192, dtype=torch.float32, requires_grad=True)
        known_long = torch.randn(4, 192, dtype=torch.float32, requires_grad=True)
        unknown_short = torch.randn(4, 192, dtype=torch.float32, requires_grad=True)
        unknown_long = torch.randn(4, 192, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        total, diagnostics = f008_combined_objective(
            known_short, known_long, targets, unknown_short, unknown_long, head,
            f008_arm(config, "energy_005"), normalization_pairs=4,
            energy_margin_config=settings, energy_margin_plan=margin,
        )
        total.backward()
        self.assertEqual(diagnostics["unknown_aam_target_assigned"], False)
        self.assertEqual(diagnostics["oe_views"], 8)
        self.assertIsNotNone(unknown_short.grad)
        self.assertIsNotNone(unknown_long.grad)
        self.assertIsNotNone(head.weight.grad)

    def test_gradient_probe_logs_encoder_and_head_without_an_optimizer_step(self) -> None:
        import torch

        torch.manual_seed(94)
        encoder = torch.nn.Linear(3, 4)
        head = torch.nn.Linear(4, 2)
        before = copy.deepcopy(encoder.state_dict())
        values = head(encoder(torch.ones(2, 3))).square().mean()
        values.backward()
        probe = _probe_receipt(
            {"signature": "a" * 64, "outer_fold": 0}, step=600,
            encoder=encoder, head=head,
            update_metrics={"task_loss": 1.0, "oe_loss": 2.0, "combined_loss": 1.1,
                            "oe_full_batch_views": 64.0},
        )
        self.assertEqual(probe["schema_version"], F008_GRADIENT_PROBE_SCHEMA)
        self.assertEqual(probe["optimizer_steps_persisted"], 0)
        self.assertGreater(probe["raw_encoder_gradient_norm"], 0.0)
        self.assertGreater(probe["raw_head_gradient_norm"], 0.0)
        self.assertTrue(all(torch.equal(before[key], encoder.state_dict()[key]) for key in before))


if __name__ == "__main__":
    unittest.main()
