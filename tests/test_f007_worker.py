"""Focused tests for the F007 L2-SP tail worker.

They exercise contracts and the update-order invariant without audio, a CUDA
device, F005 source artifacts, or MLflow.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from speaker_id.training.f007_worker import (
    F007_TAIL_CHECKPOINT_SCHEMA,
    _anchor_receipt,
    _run_tail_updates,
    _sha,
    batchnorm_affine_parameter_names,
    f007_dual_aam_task,
    f007_tail_checkpoint_metadata,
    f007_tail_identity,
    validate_f007_resume_payload,
)


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _f007_config() -> dict:
    return json.loads((ROOT / "configs/train/campp_f007_l2sp.json").read_text(encoding="utf-8"))


def _f005_fit() -> dict:
    return json.loads(
        (ROOT / "configs/train/campp_f005_consistency.json").read_text(encoding="utf-8")
    )["fit"]


def _identity(*, anchor_receipt: dict | None = None) -> tuple[dict, dict]:
    f007 = {"config": _f007_config(), "signature": "a" * 64}
    f005 = {"signature": "b" * 64}
    arm = f007["config"]["arms"][1]
    identity = f007_tail_identity(
        f007, f005, 0, arm,
        shared_head_checkpoint_sha256="c" * 64,
        tail_plan_sha256="d" * 64,
        anchor_receipt=anchor_receipt or {"schema": "safe", "names": ["x"]},
    )
    return identity, f005


class F007WorkerContractTests(unittest.TestCase):
    def test_tail_identity_binds_anchor_and_source(self) -> None:
        first, _ = _identity(anchor_receipt={"schema": "safe", "names": ["encoder.a"]})
        second, _ = _identity(anchor_receipt={"schema": "safe", "names": ["encoder.b"]})
        self.assertNotEqual(first["signature"], second["signature"])
        self.assertEqual(first["arm_id"], "l2sp_001")
        self.assertEqual(first["l2sp_lambda"], 0.01)
        self.assertEqual(first["anchor_scope"], "trainable_encoder_parameters_excluding_batchnorm_affine")

    def test_resume_payload_rejects_wrong_anchor(self) -> None:
        identity, _ = _identity()
        fit = _f005_fit()
        metadata = f007_tail_checkpoint_metadata(identity, fit, 700)
        self.assertEqual(metadata["checkpoint_schema"], F007_TAIL_CHECKPOINT_SCHEMA)
        payload = {
            "metadata": metadata, "encoder": {}, "head": {}, "optimizer": {},
            "torch_rng": object(), "cuda_rng": object(),
        }
        self.assertEqual(validate_f007_resume_payload(payload, identity, fit), metadata)
        altered = dict(metadata)
        altered["anchor_receipt_sha256"] = "e" * 64
        payload["metadata"] = altered
        with self.assertRaisesRegex(ValueError, "another arm, fold, source, plan, anchor or step"):
            validate_f007_resume_payload(payload, identity, fit)


@unittest.skipUnless(HAS_TORCH, "torch is not installed in this environment")
class F007WorkerTorchTests(unittest.TestCase):
    def test_batchnorm_affine_exclusions_and_safe_receipt(self) -> None:
        import torch

        encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Sequential(torch.nn.BatchNorm1d(4, affine=False)),
        )
        exclusions = batchnorm_affine_parameter_names(encoder)
        self.assertEqual(exclusions, frozenset({"1.weight", "1.bias"}))
        # A receipt may expose names and hashes but no values.
        class Anchor:
            receipt = {"schema": "l2sp-anchor-v1", "parameter_names": ["0.weight"]}
        receipt = _anchor_receipt(Anchor(), exclusions)
        self.assertEqual(receipt["excluded_batchnorm_affine_names"], ["1.bias", "1.weight"])
        self.assertNotIn("values", receipt)

    def test_dual_aam_task_matches_manual_microbatch_normalization(self) -> None:
        import torch
        from speaker_id.training.fit import AAMHead

        torch.manual_seed(31)
        head = AAMHead(embedding_dim=192, classes=446, margin=0.2, scale=30.0)
        short = torch.randn(2, 192, dtype=torch.float32, requires_grad=True)
        long = torch.randn(2, 192, dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([3, 11], dtype=torch.int64)
        loss, diagnostics = f007_dual_aam_task(
            short, long, targets, head, normalization_pairs=4,
        )
        logits_short, logits_long = head(short, targets), head(long, targets)
        expected = (
            torch.nn.functional.cross_entropy(logits_short, targets, reduction="sum")
            + torch.nn.functional.cross_entropy(logits_long, targets, reduction="sum")
        ) / 8
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics["rows"], 2)
        loss.backward()
        self.assertIsNotNone(short.grad)
        self.assertIsNotNone(long.grad)

    def test_l2sp_penalty_is_applied_once_after_two_microbatches(self) -> None:
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
        f005 = {"config": {"fit": fit}}
        calls = {"penalty": 0, "saved": 0}

        def fake_cpu_batches(*_args, **_kwargs):
            return [object(), object()]

        def fake_task(_encoder, _head, _batch, *, normalization_pairs):
            self.assertEqual(normalization_pairs, 2)
            loss = (_encoder.weight.square().sum() + _head.weight.square().sum()) / 4
            return loss, {
                "task_loss": float(loss.detach()), "short_aam_sum": 1.0,
                "long_aam_sum": 1.0, "short_correct": 1, "long_correct": 1, "rows": 1,
            }

        def fake_penalty(_encoder, _anchor):
            calls["penalty"] += 1
            return 0.5 * _encoder.weight.square().sum()

        with tempfile.TemporaryDirectory() as directory, patch(
            "speaker_id.training.f007_worker._cpu_batches", fake_cpu_batches,
        ), patch(
            "speaker_id.training.f007_worker._task_batch_loss", fake_task,
        ), patch(
            "speaker_id.training.f007_worker._set_tail_schedule",
            lambda *_args, **_kwargs: {"encoder_lr": 0.01, "head_lr": 0.01, "margin": 0.2},
        ), patch("speaker_id.training.f007_worker.l2sp_penalty", fake_penalty):
            _run_tail_updates(
                f005, Path(directory), 0, {"lambda": 0.1}, encoder, head, optimizer,
                object(), 600, 601, Path(directory) / "history.jsonl", None,
                lambda _step: calls.__setitem__("saved", calls["saved"] + 1),
            )
            history = json.loads((Path(directory) / "history.jsonl").read_text().strip())
        self.assertEqual(calls, {"penalty": 1, "saved": 1})
        self.assertEqual(history["fit/l2sp_application_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
