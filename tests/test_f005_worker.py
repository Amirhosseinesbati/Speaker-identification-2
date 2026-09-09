"""CPU tensor checks for the exact F005 objective and bounded waveform cache."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from speaker_id.training import f005_worker as worker


def _state_metadata(head_only: int, tail: int, stage: str = "tail") -> dict:
    completed = head_only + tail
    return {
        "stage": stage,
        "completed_steps": completed,
        "schedule_state": {
            "completed_steps": completed,
            "head_only_completed_steps": head_only,
            "tail_completed_steps": tail,
        },
    }


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is required by the training environment")
class F005WorkerTests(unittest.TestCase):
    def test_fixed_partial_recovery_promotes_valid_bytes_and_never_masks_bad_final(self):
        def validate(path):
            value = Path(path).read_text(encoding="ascii")
            if not value.startswith("valid:"):
                raise ValueError("invalid artifact")
            return int(value.removeprefix("valid:"))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "last.pt"
            partial = worker._fixed_partial_path(target)
            unrelated = root / "unrelated.partial"
            partial.write_text("valid:50", encoding="ascii")
            unrelated.write_text("leave-me", encoding="ascii")
            self.assertEqual(
                worker.recover_fixed_partial(target, validate, derived=False),
                "recovered",
            )
            self.assertEqual(target.read_text(encoding="ascii"), "valid:50")
            self.assertFalse(partial.exists())
            self.assertTrue(unrelated.exists())

            partial.write_text("valid:100", encoding="ascii")
            self.assertEqual(
                worker.recover_fixed_partial(target, validate, derived=False),
                "final",
            )
            self.assertEqual(target.read_text(encoding="ascii"), "valid:50")

            target.write_text("corrupt-final", encoding="ascii")
            partial.write_text("valid:150", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "invalid artifact"):
                worker.recover_fixed_partial(target, validate, derived=False)
            self.assertEqual(target.read_text(encoding="ascii"), "corrupt-final")
            self.assertEqual(partial.read_text(encoding="ascii"), "valid:150")

    def test_fixed_partial_publish_race_validates_both_candidates(self):
        validations = []

        def validate(path):
            value = Path(path).read_text(encoding="ascii")
            validations.append((Path(path).name, value))
            if not value.startswith("valid:"):
                raise ValueError("invalid artifact")
            return int(value.removeprefix("valid:"))

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "last.pt"
            partial = worker._fixed_partial_path(target)
            partial.write_text("valid:50", encoding="ascii")

            def racing_link(_source, destination):
                Path(destination).write_text("valid:40", encoding="ascii")
                raise FileExistsError

            with patch.object(worker.os, "link", side_effect=racing_link):
                self.assertEqual(
                    worker.recover_fixed_partial(target, validate, derived=True),
                    "final",
                )
            self.assertEqual(target.read_text(encoding="ascii"), "valid:40")
            self.assertFalse(partial.exists())
            self.assertEqual(validations, [("last.pt.partial", "valid:50"),
                                           ("last.pt", "valid:40")])

    def test_bad_checkpoint_state_cannot_replace_metadata_valid_final(self):
        import torch

        encoder = torch.nn.Linear(2, 2)
        head = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters()}, {"params": head.parameters()},
        ])
        (head(encoder(torch.ones(1, 2))).sum()).backward(); optimizer.step()
        metadata = _state_metadata(0, 1)
        payload = {
            "metadata": metadata,
            "encoder": encoder.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        bad_encoder = dict(payload["encoder"])
        bad_encoder["weight"] = torch.zeros((3, 2), dtype=torch.float32)
        bad_payload = {**payload, "encoder": bad_encoder}

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "last.pt"
            partial = worker._fixed_partial_path(target)
            torch.save(payload, target)
            torch.save(bad_payload, partial)
            final_bytes = target.read_bytes()

            def validate(path):
                candidate = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(candidate["metadata"]["completed_steps"], 1)
                worker._load_training_state(
                    candidate, encoder, head, optimizer, candidate["metadata"])
                return candidate["metadata"]

            self.assertEqual(
                worker.recover_fixed_partial(target, validate, derived=True),
                "discarded_invalid_partial",
            )
            self.assertEqual(target.read_bytes(), final_bytes)
            self.assertFalse(partial.exists())

    def test_checkpoint_validation_does_not_mutate_optimizer_on_bad_param_group(self):
        import copy
        import torch

        encoder = torch.nn.Linear(2, 2)
        head = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters(), "weight_decay": 0.01},
            {"params": head.parameters(), "weight_decay": 0.01},
        ])
        payload = {
            "encoder": encoder.state_dict(), "head": head.state_dict(),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        payload["optimizer"]["param_groups"][0]["weight_decay"] = 0.91
        before = copy.deepcopy(optimizer.state_dict())
        with self.assertRaisesRegex(ValueError, "hyperparameters changed"):
            worker._validate_training_state_structure(
                payload, encoder, head, optimizer, _state_metadata(0, 0, "shared_head"))
        self.assertEqual(optimizer.state_dict(), before)

    def test_wrong_length_rng_partial_is_rejected_purely_before_publish(self):
        import torch

        encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters()}, {"params": head.parameters()},
        ])
        metadata = _state_metadata(0, 0, "shared_head")
        payload = {
            "metadata": metadata, "encoder": encoder.state_dict(), "head": head.state_dict(),
            "optimizer": optimizer.state_dict(), "torch_rng": torch.zeros(1, dtype=torch.uint8),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        before_cpu = torch.get_rng_state().clone()
        before_cuda = [item.clone() for item in torch.cuda.get_rng_state_all()]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "last.pt"
            partial = worker._fixed_partial_path(target)
            torch.save(payload, partial)
            validate = lambda candidate: worker._validate_training_checkpoint(
                candidate, lambda value: value["metadata"], encoder, head, optimizer)
            self.assertEqual(
                worker.recover_fixed_partial(target, validate, derived=True),
                "discarded_invalid_partial",
            )
            self.assertFalse(target.exists()); self.assertFalse(partial.exists())
        self.assertTrue(torch.equal(torch.get_rng_state(), before_cpu))
        self.assertEqual(len(torch.cuda.get_rng_state_all()), len(before_cuda))
        self.assertTrue(all(torch.equal(actual, expected)
                            for actual, expected in zip(torch.cuda.get_rng_state_all(),
                                                       before_cuda, strict=True)))

    def test_completed_checkpoint_requires_every_adam_state_and_exact_steps(self):
        import copy
        import torch

        encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters()}, {"params": head.parameters()},
        ])
        # First update is head-only; the second is a tail update through both modules.
        head(torch.ones(1, 2)).sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        head(encoder(torch.ones(1, 2))).sum().backward(); optimizer.step()
        metadata = _state_metadata(1, 1)
        saved = copy.deepcopy(optimizer.state_dict())
        encoder_ids = set(saved["param_groups"][0]["params"])
        for identifier, state in saved["state"].items():
            state["step"].fill_(1 if identifier in encoder_ids else 2)
        payload = {
            "encoder": encoder.state_dict(), "head": head.state_dict(), "optimizer": saved,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        worker._validate_training_state_structure(payload, encoder, head, optimizer, metadata)
        for kind in ("empty", "missing"):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(payload)
                if kind == "empty":
                    changed["optimizer"]["state"] = {}
                else:
                    changed["optimizer"]["state"].pop(next(iter(changed["optimizer"]["state"])))
                with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                    worker._validate_training_state_structure(
                        changed, encoder, head, optimizer, metadata)
        wrong_step = copy.deepcopy(payload)
        wrong_step["optimizer"]["state"][next(iter(wrong_step["optimizer"]["state"]))]["step"].fill_(99)
        with self.assertRaisesRegex(ValueError, "step disagrees with metadata"):
            worker._validate_training_state_structure(
                wrong_step, encoder, head, optimizer, metadata)
        wrong_dtype = copy.deepcopy(payload)
        first_state = wrong_dtype["optimizer"]["state"][next(iter(wrong_dtype["optimizer"]["state"]))]
        first_state["exp_avg"] = first_state["exp_avg"].double()
        with self.assertRaisesRegex(ValueError, "dtype, or device changed"):
            worker._validate_training_state_structure(
                wrong_dtype, encoder, head, optimizer, metadata)

    def test_invalid_fixed_partial_is_discarded_only_when_declared_derived(self):
        def validate(path):
            if Path(path).read_bytes() != b"valid-cache":
                raise ValueError("invalid cache")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / "sample.npz"
            cache_partial = worker._fixed_partial_path(cache)
            cache_partial.write_bytes(b"torn-cache")
            self.assertEqual(
                worker.recover_fixed_partial(cache, validate, derived=True),
                "discarded_invalid_partial",
            )
            self.assertFalse(cache.exists())
            self.assertFalse(cache_partial.exists())

            checkpoint = root / "last.pt"
            checkpoint_partial = worker._fixed_partial_path(checkpoint)
            checkpoint_partial.write_bytes(b"torn-checkpoint")
            with self.assertRaisesRegex(ValueError, "invalid cache"):
                worker.recover_fixed_partial(checkpoint, validate, derived=False)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(checkpoint_partial.exists())

            self.assertEqual(
                worker.recover_fixed_partial(checkpoint, validate, derived=True),
                "discarded_invalid_partial",
            )
            self.assertFalse(checkpoint_partial.exists())

    def test_invalid_partial_checkpoint_restarts_shared_stage_from_step_zero(self):
        class EmptyState:
            def state_dict(self):
                return {}

        fit = {
            "adaptation_schedule": {"head_only_steps": 600},
        }
        arms = [{"id": "control", "kind": "dual_aam_control",
                 "cosine_gamma": 0.0, "raw_h_mse_lambda": 0.0}]
        contract = {"config": {"fit": fit, "arms": arms}}
        identity = {"signature": "a" * 64, "pairing_signature": "b" * 64}
        captured = {}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            checkpoint = output / "shared_head.pt"
            partial = worker._fixed_partial_path(checkpoint)
            partial.write_bytes(b"torn-checkpoint")
            history = output / "fit_history.jsonl"
            history.write_text('{"step":1}\n', encoding="utf-8")

            def run_updates(*args):
                captured["start"] = args[7]
                captured["history_before_run"] = args[9].read_bytes()
                checkpoint.write_bytes(b"rebuilt-checkpoint")
                return {}

            with patch.object(worker, "plan_range_sha256", return_value="c" * 64), \
                    patch.object(worker, "shared_head_identity", return_value=identity), \
                    patch.object(worker, "_components",
                                 return_value=(EmptyState(), EmptyState(), object(), {}, 7)), \
                    patch.object(worker, "_run_updates", side_effect=run_updates), \
                    patch("speaker_id.training.schedules.adaptation_checkpoint_state",
                          return_value={}):
                result = worker.fit_shared_head(contract, output, 0, output, resume=True)

            self.assertEqual(captured, {"start": 0, "history_before_run": b""})
            self.assertFalse(partial.exists())
            self.assertEqual(result["checkpoint"], checkpoint)

    def test_history_repair_drops_only_torn_tail_and_post_checkpoint_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder) / "fit_history.jsonl"
            history.write_bytes(
                b'{"step":1,"loss":1.0}\r\n'
                b'{"step":2,"loss":0.5}\n'
                b'{"step":3,"loss":0.25}\n'
                b'{"step":4,"loss":')
            worker._trim_history(history, 2)
            self.assertEqual(
                history.read_bytes(),
                b'{"step":1,"loss":1.0}\n{"step":2,"loss":0.5}\n',
            )
            self.assertFalse((history.parent / ".fit_history.jsonl.repair.partial").exists())

    def test_history_repair_rejects_malformed_complete_or_interior_rows_unchanged(self):
        cases = (
            b'{"step":1}\nnot-json\n{"step":2',
            b'{"step":1}\nnot-json\n',
        )
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder) / "fit_history.jsonl"
            for contents in cases:
                with self.subTest(contents=contents):
                    history.write_bytes(contents)
                    with self.assertRaisesRegex(ValueError, "malformed JSON"):
                        worker._trim_history(history, 1)
                    self.assertEqual(history.read_bytes(), contents)

    def test_history_repair_rejects_gaps_duplicates_and_reordering_unchanged(self):
        cases = (
            b'{"step":1}\n{"step":3}\n',
            b'{"step":1}\n{"step":1}\n',
            b'{"step":1}\n{"step":3}\n{"step":2}\n',
        )
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder) / "fit_history.jsonl"
            for contents in cases:
                with self.subTest(contents=contents):
                    history.write_bytes(contents)
                    with self.assertRaisesRegex(ValueError, "contiguous increasing"):
                        worker._trim_history(history, 3)
                    self.assertEqual(history.read_bytes(), contents)

    def test_history_repair_rejects_rows_missing_before_committed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            missing = root / "missing.jsonl"
            with self.assertRaisesRegex(ValueError, "missing rows"):
                worker._trim_history(missing, 3)
            short = root / "short.jsonl"
            short.write_text('{"step":1}\n{"step":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shorter"):
                worker._trim_history(short, 3)

    def test_resumed_tail_report_hashes_the_shared_fork_not_tail_state(self):
        import torch

        shared_encoder, shared_head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        tail_encoder, tail_head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        with torch.no_grad():
            shared_encoder.weight.zero_(); shared_encoder.bias.zero_()
            shared_head.weight.zero_(); shared_head.bias.zero_()
            tail_encoder.weight.fill_(2); tail_encoder.bias.fill_(2)
            tail_head.weight.fill_(3); tail_head.bias.fill_(3)
        source_optimizer = torch.optim.AdamW([
            {"params": shared_encoder.parameters()}, {"params": shared_head.parameters()},
        ])
        tail_optimizer = torch.optim.AdamW([
            {"params": tail_encoder.parameters()}, {"params": tail_head.parameters()},
        ])
        shared_head(torch.ones(1, 2)).sum().backward(); source_optimizer.step()
        for state in source_optimizer.state.values():
            state["step"].fill_(600)
        tail_head(tail_encoder(torch.ones(1, 2))).sum().backward(); tail_optimizer.step()
        tail_encoder_ids = {id(parameter) for parameter in tail_optimizer.param_groups[0]["params"]}
        for parameter, state in tail_optimizer.state.items():
            state["step"].fill_(100 if id(parameter) in tail_encoder_ids else 700)
        shared_metadata = _state_metadata(600, 0, "shared_head")
        tail_metadata = _state_metadata(600, 100, "tail")
        shared_payload = {
            "metadata": shared_metadata, "encoder": shared_encoder.state_dict(), "head": shared_head.state_dict(),
            "optimizer": source_optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        tail_payload = {
            "metadata": tail_metadata, "encoder": tail_encoder.state_dict(), "head": tail_head.state_dict(),
            "optimizer": tail_optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        live_encoder, live_head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        live_optimizer = torch.optim.AdamW([
            {"params": live_encoder.parameters()}, {"params": live_head.parameters()},
        ])
        fit = {"adaptation_schedule": {"head_only_steps": 600}}
        contract = {"config": {"fit": fit}}
        arm_identity = {
            "signature": "a" * 64, "pairing_signature": "b" * 64,
            "arm": {"id": "control"},
        }

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shared_path = root / "shared.pt"
            output = root / "tail"
            output.mkdir()
            torch.save(shared_payload, shared_path)
            torch.save(tail_payload, output / "last.pt")
            (output / "fit_history.jsonl").write_text(
                "".join(json.dumps({"step": step}) + "\n" for step in range(601, 701)),
                encoding="utf-8",
            )
            with patch.object(worker, "shared_head_identity", return_value={}), \
                    patch.object(worker, "arm_identity", return_value=arm_identity), \
                    patch.object(worker, "plan_range_sha256", return_value="c" * 64), \
                    patch.object(worker, "validate_shared_head_payload",
                                 return_value={**shared_metadata,
                                               "byte_identical_fork_source": True}), \
                    patch.object(worker, "validate_resume_payload",
                                 return_value=tail_metadata), \
                    patch.object(worker, "_components",
                                 return_value=(live_encoder, live_head, live_optimizer, {}, 7)), \
                    patch.object(worker, "_run_updates", return_value={}), \
                    patch("speaker_id.training.schedules.adaptation_total_steps", return_value=700), \
                    patch("speaker_id.training.schedules.adaptation_checkpoint_state",
                          return_value={}):
                result = worker.fit_tail_arm(
                    contract, root, 0, "control", shared_path, output, resume=True,
                )
        self.assertEqual(
            result["report"]["fork_encoder_state_sha256"],
            worker.state_dict_sha256(shared_payload["encoder"]),
        )
        self.assertEqual(
            result["report"]["fork_head_state_sha256"],
            worker.state_dict_sha256(shared_payload["head"]),
        )
        self.assertNotEqual(
            result["report"]["fork_encoder_state_sha256"],
            worker.state_dict_sha256(tail_payload["encoder"]),
        )

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
