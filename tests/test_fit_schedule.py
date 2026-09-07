"""CPU-only tests for actual schedule boundaries and checkpoint continuation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from speaker_id.training.contracts import validate_fit_settings
from speaker_id.training.schedules import (adaptation_checkpoint_state, adaptation_step,
                                          adaptation_total_steps, validate_adaptation_schedule)


ROOT = Path(__file__).resolve().parents[1]


class FitScheduleTests(unittest.TestCase):
    def setUp(self):
        self.fit = json.loads((ROOT / "configs/train/campp_finetune_warmup.json").read_text())["fit"]

    def test_legacy_recipe_remains_on_original_scheduler(self):
        legacy = json.loads((ROOT / "configs/train/campp_finetune.json").read_text())["fit"]
        validate_fit_settings(legacy)
        self.assertIsNone(validate_adaptation_schedule(legacy))
        self.assertEqual(adaptation_total_steps(legacy), 500)
        with self.assertRaisesRegex(ValueError, "original PyTorch scheduler"):
            adaptation_step(legacy, 0)

    def test_phase_boundaries_margin_ramp_and_used_learning_rates(self):
        steps = [adaptation_step(self.fit, step) for step in range(adaptation_total_steps(self.fit))]
        self.assertEqual(len(steps), 600)
        self.assertEqual(sum(s["phase"] == "head_only" for s in steps), 100)
        self.assertTrue(all(s["encoder_lr"] == s["margin"] == 0 for s in steps[:100]))
        self.assertTrue(all(s["tail_step"] is None for s in steps[:100]))
        self.assertAlmostEqual(steps[0]["head_lr"], 1e-4)
        self.assertAlmostEqual(steps[99]["head_lr"], 1e-3)
        self.assertEqual(steps[100]["phase"], "tail")
        self.assertEqual(steps[100]["tail_step"], 0)
        self.assertEqual(steps[100]["margin"], 0)
        self.assertAlmostEqual(steps[100]["encoder_lr"], 1e-5 / 50)
        self.assertAlmostEqual(steps[149]["encoder_lr"], 1e-5)
        self.assertLess(steps[150]["encoder_lr"], steps[149]["encoder_lr"])
        self.assertLess(steps[198]["margin"], .2)
        self.assertAlmostEqual(steps[199]["margin"], .2)
        self.assertAlmostEqual(steps[-1]["margin"], .2)
        self.assertEqual(steps[-1]["tail_step"], 499)
        self.assertEqual(steps[-1]["encoder_lr"], 0)
        self.assertEqual(steps[-1]["head_lr"], 0)

    def test_schedule_is_bounded_monotone_within_each_stage(self):
        steps = [adaptation_step(self.fit, step) for step in range(600)]
        def monotone(values, increasing=True):
            pairs = zip(values, values[1:])
            return all(a <= b if increasing else a >= b for a, b in pairs)
        self.assertTrue(monotone([s["head_lr"] for s in steps[:100]]))
        self.assertTrue(monotone([s["head_lr"] for s in steps[100:]], False))
        self.assertTrue(monotone([s["encoder_lr"] for s in steps[100:150]]))
        self.assertTrue(monotone([s["encoder_lr"] for s in steps[149:]], False))
        self.assertTrue(monotone([s["margin"] for s in steps]))
        for step in steps:
            for key, upper in (("encoder_lr", 1e-5), ("head_lr", 1e-3), ("margin", .2)):
                self.assertTrue(math.isfinite(step[key]))
                self.assertGreaterEqual(step[key], 0)
                self.assertLessEqual(step[key], upper)

    def test_resume_at_each_boundary_reconstructs_exact_remaining_schedule(self):
        full = [adaptation_step(self.fit, step) for step in range(600)]
        for completed in (0, 1, 99, 100, 101, 149, 150, 199, 200, 599, 600):
            with self.subTest(completed=completed):
                # Checkpoint serialization retains only the committed count and
                # explicit config; no scheduler object or phase counter can drift.
                restored = json.loads(json.dumps({"fit": self.fit,
                    "state": adaptation_checkpoint_state(self.fit, completed)}))
                state = restored["state"]
                self.assertEqual(state["head_only_completed_steps"], min(100, completed))
                self.assertEqual(state["tail_completed_steps"], max(0, completed - 100))
                expected_phase = "head_only" if completed < 100 else "tail" if completed < 600 else "complete"
                self.assertEqual(state["next_phase"], expected_phase)
                remaining = [adaptation_step(restored["fit"], step) for step in range(state["completed_steps"], 600)]
                self.assertEqual(remaining, full[completed:])

    def test_invalid_schedule_controls_are_rejected(self):
        changes = [("scheme", "unknown"), ("head_only_steps", -1), ("head_only_steps", 1.5),
                   ("head_only_steps", True), ("margin_ramp_tail_steps", 1),
                   ("margin_ramp_tail_steps", 501), ("encoder_lr_warmup_tail_steps", 500),
                   ("encoder_lr_warmup_tail_steps", 0), ("head_lr_warmup_start_factor", 0),
                   ("head_lr_warmup_start_factor", 1.1), ("head_lr_warmup_start_factor", float("nan")),
                   ("head_lr_warmup_start_factor", float("inf"))]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                fit = deepcopy(self.fit)
                fit["adaptation_schedule"][key] = value
                with self.assertRaises(ValueError):
                    validate_fit_settings(fit)
        fit = deepcopy(self.fit)
        fit["adaptation_schedule"]["unimplemented_margin_mode"] = "automatic"
        with self.assertRaises(ValueError):
            validate_fit_settings(fit)
        fit["adaptation_schedule"] = None
        with self.assertRaises(ValueError):
            validate_fit_settings(fit)

    def test_contract_rejects_silent_typos_and_nonfinite_optimizer_inputs(self):
        for key, value in (("head_warmup_steps", 100), ("encoder_lr", float("nan")),
                           ("head_lr", -1), ("weight_decay", -1), ("margin", math.pi / 2),
                           ("scale", 0), ("batch_size", 32.5), ("epochs", True),
                           ("freeze_batchnorm", False), ("mixed_precision", "true")):
            with self.subTest(key=key):
                fit = deepcopy(self.fit)
                fit[key] = value
                with self.assertRaises(ValueError):
                    validate_fit_settings(fit)

    def test_out_of_range_checkpoint_and_update_indices_fail(self):
        for completed in (-1, 601, True, 100.0):
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                adaptation_checkpoint_state(self.fit, completed)
        for step in (-1, 600, True, 100.0):
            with self.subTest(step=step), self.assertRaises(ValueError):
                adaptation_step(self.fit, step)


if __name__ == "__main__":
    unittest.main()
