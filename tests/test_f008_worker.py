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
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from speaker_id.training.f008_config import config_signature, load_f008_config
from speaker_id.training.f008_energy_margin_plan import build_energy_margin_plan
from speaker_id.training.f008_preflight import build_preflight_receipt
from speaker_id.training.f008_protocol import role_pools
from speaker_id.training.f008_worker import (
    F008_GRADIENT_PROBE_SCHEMA,
    F008_RUNTIME_RECEIPT_SCHEMA,
    F008_TAIL_CHECKPOINT_SCHEMA,
    _probe_receipt,
    _run_tail_updates,
    _streaming_oe_contribution,
    _unknown_microbatch,
    attest_f008_runtime,
    f008_arm,
    f008_combined_objective,
    f008_oe_loss,
    f008_tail_checkpoint_metadata,
    f008_tail_identity,
    validate_f008_runtime_receipt,
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


def _identity(
    *, known_hash: str = "b" * 64, unknown_hash: str = "c" * 64,
    runtime_hash: str = "d" * 64,
) -> tuple[dict, dict]:
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
            runtime_receipt_sha256=runtime_hash,
        )
    return identity, _fit()


def _runtime_f005_contract() -> dict:
    config = _config()
    return {
        "signature": config["source_f005"]["experiment_signature"],
        "config": {
            "execution": copy.deepcopy(config["execution"]),
            "device": "cuda",
            "cpu_threads": 4,
        },
        "readiness": {"config": {"expected_vast_instance_id": 50288952}},
    }


def _fake_cuda_torch() -> object:
    """Small Torch surface for testing the one-time runtime receipt only."""
    state = {"threads": None, "deterministic": False}
    value = SimpleNamespace()
    value.float32 = object()
    value.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda index: "NVIDIA GeForce RTX 3090",
    )
    value.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=True))
    value.set_num_threads = lambda threads: state.__setitem__("threads", threads)
    value.get_num_threads = lambda: state["threads"]
    value.use_deterministic_algorithms = lambda enabled, *, warn_only: state.__setitem__(
        "deterministic", enabled,
    )
    value.are_deterministic_algorithms_enabled = lambda: state["deterministic"]
    value.get_default_dtype = lambda: value.float32
    return value


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
        changed_runtime, _ = _identity(runtime_hash="f" * 64)
        self.assertNotEqual(first["signature"], changed_known["signature"])
        self.assertNotEqual(first["signature"], changed_unknown["signature"])
        self.assertNotEqual(first["signature"], changed_runtime["signature"])
        self.assertEqual(first["unknown_plan_step_range"], {"start": 600, "stop": 1100})
        self.assertEqual(first["oe_full_batch_views"], 64)
        self.assertEqual(first["runtime_receipt_sha256"], "d" * 64)

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

    def test_runtime_receipt_binds_f008_f005_and_launch_environment(self) -> None:
        config, f005 = _config(), _runtime_f005_contract()
        environment = {
            "VAST_INSTANCE_ID": "50288952",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
        }
        fake_torch = _fake_cuda_torch()
        with patch.dict(os.environ, environment, clear=False), patch.dict(
            sys.modules, {"torch": fake_torch},
        ):
            receipt = attest_f008_runtime(config, f005)
            self.assertEqual(receipt["schema_version"], F008_RUNTIME_RECEIPT_SCHEMA)
            self.assertEqual(receipt["source_f005_signature"], f005["signature"])
            self.assertEqual(receipt["expected_vast_instance_id"], "50288952")
            self.assertTrue(receipt["cublas_checked_before_torch_import"])
            self.assertEqual(validate_f008_runtime_receipt(receipt, config, f005), receipt)
            altered = dict(receipt)
            altered["gpu_name"] = "other GPU"
            with self.assertRaisesRegex(ValueError, "runtime receipt"):
                validate_f008_runtime_receipt(altered, config, f005)
        with patch.dict(os.environ, {**environment, "CUBLAS_WORKSPACE_CONFIG": ":16:8"}, clear=False), \
                patch.dict(sys.modules, {"torch": _fake_cuda_torch()}):
            with self.assertRaisesRegex(ValueError, "before Torch import"):
                attest_f008_runtime(config, f005)


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
            {"signature": "a" * 64, "outer_fold": 0, "runtime_receipt_sha256": "b" * 64}, step=600,
            encoder=encoder, head=head,
            update_metrics={"task_loss": 1.0, "oe_loss": 2.0, "combined_loss": 1.1,
                            "oe_full_batch_views": 64.0},
        )
        self.assertEqual(probe["schema_version"], F008_GRADIENT_PROBE_SCHEMA)
        self.assertEqual(probe["optimizer_steps_persisted"], 0)
        self.assertEqual(probe["runtime_receipt_sha256"], "b" * 64)
        self.assertGreater(probe["raw_encoder_gradient_norm"], 0.0)
        self.assertGreater(probe["raw_head_gradient_norm"], 0.0)
        self.assertTrue(all(torch.equal(before[key], encoder.state_dict()[key]) for key in before))

    def test_tail_callback_receives_scalar_event_after_history_append(self) -> None:
        import torch

        encoder = torch.nn.Linear(1, 1, bias=False)
        head = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD([
            {"params": encoder.parameters(), "lr": 0.01},
            {"params": head.parameters(), "lr": 0.01},
        ])
        fit = {
            "waveform_cache_max_bytes": 1024,
            "batch_pairs": 2,
            "checkpoint_every_steps": 1,
            "gradient_clip_norm": 5.0,
        }
        aggregate = {
            "task_loss": 1.0, "oe_loss": 0.25, "combined_loss": 1.0125,
            "known_short_aam_sum": 1.0, "known_long_aam_sum": 1.5,
            "known_short_correct": 1, "known_long_correct": 2,
            "oe_full_batch_views": 64.0, "unknown_aam_target_assigned": 0.0,
            "known_energy_sum": -12.8, "unknown_energy_sum": -6.4,
            "known_oe_penalty_sum": 3.2, "unknown_oe_penalty_sum": 6.4,
        }
        saved, events = [], []

        def fake_backward(*args, **_kwargs):
            loss = args[6].weight.square().sum() + args[7].weight.square().sum()
            loss.backward()
            return dict(aggregate)

        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "fit_history.jsonl"

            def on_step(event: dict) -> None:
                # The callback must never get ahead of the recoverable history.
                self.assertIn('"step": 601', history.read_text(encoding="utf-8"))
                events.append(event)

            with patch("speaker_id.training.f008_worker._backward_f008_step", fake_backward), \
                    patch("speaker_id.training.f008_worker._set_tail_schedule", lambda *_args, **_kwargs: {
                        "encoder_lr": 0.01, "head_lr": 0.01, "margin": 0.2,
                    }), \
                    patch("torch.cuda.max_memory_allocated", return_value=0):
                _run_tail_updates(
                    {"config": {"fit": fit}}, Path(directory), 0,
                    {"id": "energy_005", "lambda": 0.05}, {}, encoder, head, optimizer,
                    energy_margin_config={}, energy_margin_plan={}, start_step=600, stop_step=601,
                    unknown_sampling_seed=1, history_path=history,
                    save_checkpoint=lambda step: saved.append(step), on_step=on_step,
                )
            recorded = json.loads(history.read_text(encoding="utf-8").strip())
        self.assertEqual(saved, [601])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], recorded)
        self.assertEqual(events[0]["step"], 601)
        self.assertEqual(events[0]["phase"], "tail")
        self.assertTrue(all(
            isinstance(value, (str, int, float)) and not isinstance(value, (dict, list))
            for value in events[0].values()
        ))


if __name__ == "__main__":
    unittest.main()
